from . import _init_paths

import h5py
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import lava.lib.dl.slayer as slayer

from lava.lib.dl import netx
from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps

from consts.exp_consts import EXC
from utils.base_utils.exp_utils import ExpUtils
from utils.network_utils.lava_processes_utils.lava_processes_utils import (
    InpSigToLdn, PyInpSigToLdnModel, OutSpkToCls, PyOutSpkToClsModel,
    LdnEncToSpk, PyLdnEncToSpkOnCpuModel, CLdnEncToSpkOnLmtModel,
    InputAdapter, PyInputAdapterModel, NxInputAdapterModel,
    OutputAdapter, PyOutputAdapterModel, NxOutputAdapterModel
    )
from utils.base_utils import log

###############################################################################
########################### S L A Y E R     S N N #############################
###############################################################################

class SlayerSNN(torch.nn.Module):
  """
  Spiking network to be trained and eventually deployed on Loihi-2. It accepts
  spikes and outputs spikes (corresponding to the classes).
  """
  def __init__(self, rtc):
    super(SlayerSNN, self).__init__()
    self._rtc = rtc

    neuron_params = {
      "threshold": rtc.SLAYERSNN_PARAMS.V_THR,
      "current_decay": rtc.SLAYERSNN_PARAMS.C_DECAY,
      "voltage_decay": rtc.SLAYERSNN_PARAMS.V_DECAY,
      # If `requires_grad` is False, then NO learning on Neuron Parameters.
      "requires_grad": rtc.IS_TRNBL_NRN_PARAMS
    }

    # Rate Encoding to spike either happens on host CPU or on the embedded LMT
    # cores; therefore, the Encoding layer is NOT defined here.
    self.blocks = torch.nn.ModuleList([
      # 1st Hidden Layer -- 2*ORDER spikes input with 3*ORDER hidden neurons.
      slayer.block.cuba.Dense(
          neuron_params,
          2*rtc.DCFG["n_dim"]*rtc.ORDER, 3*rtc.DCFG["n_dim"]*rtc.ORDER,
          weight_norm=False,
          delay=False
          ),

      # Output Layer -- 3*ORDER spikes input, n_classes neurons.
      slayer.block.cuba.Dense(
          neuron_params,
          3*rtc.DCFG["n_dim"]*rtc.ORDER, rtc.DCFG["n_classes"],
          weight_norm=False,
          delay=False
          )
    ])

  def forward(self, spikes):
    for block in self.blocks:
      spikes = block(spikes)

    return spikes

  def export_hdf5(self, file_name):
    h = h5py.File(file_name, "w") # Export the network to hdf5 format.
    layer = h.create_group("layer")
    for i, block in enumerate(self.blocks):
      block.export_hdf5(layer.create_group(f'{i}'))

###############################################################################
############################# L A V A     S N N ###############################
###############################################################################

class LavaLSNN(object):
  def __init__(self, rtc, n_test_sigs, start_test_idx):
    """
    Args:
      rtc <class RTC>: runtime_consts class.
      n_test_sigs <int>: Number of test samples to inference upon.
      start_test_idx <int>: Index of the test sample at which inference starts.
    """
    assert rtc.BACKEND in ["L2Sim", "L2Hw"]
    if not rtc.IS_SCALED_LDN:
      log.WARN("Training LDN is not scaled and quantized, although scaled and "
               "quantized LDN is used in LavaLSNN for inference.")
    self._rtc = rtc
    self._n_test_sigs = n_test_sigs
    # Instantiate the network components.
    self.sig_to_ldn = InpSigToLdn(crnt_sig_idx=start_test_idx, rtc=rtc)
    self.ldn_to_spk = LdnEncToSpk(
        in_shape=(rtc.DCFG["n_dim"], ), order=rtc.ORDER, theta=rtc.THETA,
        gain=rtc.GAIN, bias=rtc.BIAS, v_thr=rtc.ENC_V_THR,
        scale_factor=EXC.SCALE_FACTOR, per_smpl_tsteps=rtc.DCFG["n_ts"])
    self.net = netx.hdf5.Network(
        net_config=rtc.OUTPUT_PATH + "/trained_network.net",
        reset_interval=rtc.RESET_INTERVAL, # Reset the slayer-net after 1 sample
        reset_offset=1, # Offset the resetting of slayer-net by 1 time-step.
        )
    self.spk_to_cls = OutSpkToCls(
        per_smpl_tsteps=rtc.DCFG["n_ts"],
        n_test_sigs=self._n_test_sigs, n_cls_shape=(rtc.DCFG["n_classes"], ))

    # Instantiate the spike adapters to-and-fro the netx-obtained network.
    # NOTE: You don't need the self.inp_adp to transfer spikes here if the input
    # signal is sent to the LMT and spikes are generated from LMT to Neuro-Cores
    #self.inp_adp = InputAdapter(shape=self.net.inp.shape)
    self.out_adp = OutputAdapter(shape=self.net.out.shape)

  def get_loihi2_run_config(self):
    """
    Returns the run_config corresponding to either Loihi2SimCfg or Loihi2HwCfg.
    """
    if self._rtc.BACKEND == "L2Sim": # ALl the Processes run on host CPU.
      run_config = Loihi2SimCfg(
          select_tag="fixed_pt",
          exception_proc_model_map={
            InpSigToLdn: PyInpSigToLdnModel, # On host CPU.
            OutSpkToCls: PyOutSpkToClsModel, # On host CPU.
            LdnEncToSpk: PyLdnEncToSpkOnCpuModel, # On host CPU.
            #InputAdapter: PyInputAdapterModel, # On host CPU.
            OutputAdapter: PyOutputAdapterModel # On host CPU.
            }
          )
    elif self._rtc.BACKEND == "L2Hw": # Mix of CPU, LMT, & Neuro-core Processes.
      run_config = Loihi2HwCfg(
          select_sub_proc_model=True,
          exception_proc_model_map={
            InpSigToLdn: PyInpSigToLdnModel, # On host CPU.
            OutSpkToCls: PyOutSpkToClsModel, # On host CPU.
            LdnEncToSpk: CLdnEncToSpkOnLmtModel, # On embedded LMT.
            #InputAdapter: NxInputAdapterModel, # On Loihi2NeuroCore.
            OutputAdapter: NxOutputAdapterModel # On Loihi2NeuroCore.
            }
          )

    return run_config

  def create_lava_lsnn(self):
    # Connect Processes.
    self.sig_to_ldn.sig_out.connect(self.ldn_to_spk.sig_inp)

    # NOTE: You don't necessarily need an Input adapter here if spikes are sent
    # from LMT to NeuroCores.
    self.ldn_to_spk.spk_out.connect(self.net.inp)
    #self.ldn_to_spk.spk_out.connect(self.inp_adp.inp)
    #self.inp_adp.out.connect(self.net.inp)

    self.net.out.connect(self.out_adp.inp)
    self.out_adp.out.connect(self.spk_to_cls.spk_in)

    self.sig_to_ldn.lbl_out.connect(self.spk_to_cls.lbl_in)

  def eval_lava_lsnn(self):
    """Runs inference on Loihi2SimCfg or Loihi2HwCfg depending on RTC.BACKEND"""
    # Create the LavaLSNN.
    self.create_lava_lsnn()
    # Get RunConfig based on the backend.
    run_config = self.get_loihi2_run_config()

    for _ in range(self._n_test_sigs):
      self.sig_to_ldn.run(
          condition=RunSteps(num_steps=self._rtc.DCFG["n_ts"]),
          run_cfg=run_config
          )

    all_true_lbls = self.spk_to_cls.true_lbls.get().astype(np.int32)
    all_pred_lbls = self.spk_to_cls.pred_lbls.get().astype(np.int32)
    self.sig_to_ldn.stop()

    log.INFO("*"*80)
    log.INFO("Ground Truth Labels: {0}".format(all_true_lbls))
    log.INFO("Predicted Labels: {0}".format(all_pred_lbls))
    log.INFO("Confusion Matrix: {0}".format(
             confusion_matrix(all_true_lbls, all_pred_lbls)))
    log.INFO("Accuracy on Loihi {0}: {1}".format(
             "Simulation" if self._rtc.BACKEND=="L2Sim" else "Board",
             np.mean(np.array(all_true_lbls) == np.array(all_pred_lbls))))
    log.INFO("*"*80)
################################################################################
