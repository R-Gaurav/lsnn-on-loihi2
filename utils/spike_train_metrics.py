import _init_paths
import os, sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


from consts.exp_consts import EXC
from consts.runtime_consts import RTC
from utils.base_utils.exp_utils import ExpUtils
from utils.base_utils.data_prep_utils import DataPrepUtils
from utils.network_utils.slayer_lsnn_utils import SpkEncoderLayer
from utils.network_utils.net_builder_utils import LDNOps


from utils.network_utils.lava_processes_utils.lava_processes_utils import (
    LdnEncToSpk, CLdnEncToSpkOnLmtModel, InpSigToLdn, PyInpSigToLdnModel,
    PyLdnEncToSpkOnCpuModel)
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps


# Disable Printing.
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore Printing.
def enablePrint():
    sys.stdout = sys.__stdout__

################################################################################
# Real Valued LDN on CPU via Python
################################################################################
def get_spikes_from_RV_LDN_CPU_Py(test_x_sig, order, theta):
  RTC.IS_SCALED_LDN = False
  RTC.ORDER = order
  RTC.THETA = theta/1000

  ldn_lyr = LDNOps(RTC)
  ldn_sigs = ldn_lyr.get_ldn_extracted_signals_from_1dim_input(
      test_x_sig.reshape(1, -1))

  spk_enc = SpkEncoderLayer(RTC)
  spikes = []
  for t in range(RTC.DCFG["n_ts"]):
    spikes.append(spk_enc.encode_ldn_stvecs(ldn_sigs[t]))
  spikes = np.array(spikes)

  return spikes
################################################################################


################################################################################
# Quantized LDN on CPU via Python
################################################################################
def get_spikes_from_QT_LDN_CPU_Py(scaled_test_x_sig, order, theta):
  RTC.IS_SCALED_LDN = True
  RTC.ORDER = order
  RTC.THETA = theta/1000

  scaled_ldn_lyr = LDNOps(RTC)
  scaled_ldn_sigs = scaled_ldn_lyr.get_ldn_extracted_signals_from_1dim_input(
      scaled_test_x_sig.reshape(1, -1))

  # RTC.IS_SCALED_LDN = True already takes care of scaling V_THR.
  scaled_spk_enc = SpkEncoderLayer(RTC)
  spikes_for_scaled_inp = []
  for t in range(RTC.DCFG["n_ts"]):
    spikes_for_scaled_inp.append(
        scaled_spk_enc.encode_ldn_stvecs(scaled_ldn_sigs[t]))
  spikes_for_scaled_inp = np.array(spikes_for_scaled_inp)

  return spikes_for_scaled_inp
################################################################################


################################################################################
# Quantized LDN on CPU via Lava (`PyLdnEncToSpkOnCpuModel`)
################################################################################
def get_spikes_from_QT_LDN_CPU_Lv(idx, order, theta):
  RTC.IS_SCALED_LDN = True
  RTC.ORDER = order
  RTC.THETA = theta/1000

  sig_on_cpu = InpSigToLdn(rtc=RTC, crnt_sig_idx=idx)
  enc_on_cpu = LdnEncToSpk(
    in_shape=(1, ), order=RTC.ORDER, theta=RTC.THETA, gain=RTC.GAIN,
    bias=RTC.BIAS, v_thr=RTC.ENC_V_THR,
    scale_factor=EXC.SCALE_FACTOR, per_smpl_tsteps=RTC.DCFG["n_ts"]
  )

  sig_on_cpu.sig_out.connect(enc_on_cpu.sig_inp)

  run_config = Loihi2SimCfg(
    exception_proc_model_map={
      InpSigToLdn: PyInpSigToLdnModel,
      LdnEncToSpk: PyLdnEncToSpkOnCpuModel
    }
  )

  spk_monitor = Monitor()
  spk_monitor.probe(enc_on_cpu.spk_out, RTC.DCFG["n_ts"])

  sig_on_cpu.run(
      condition=RunSteps(num_steps=RTC.DCFG["n_ts"]), run_cfg=run_config)
  spk_data = spk_monitor.get_data()
  keys = list(spk_data.keys())
  sig_on_cpu.stop()

  return spk_data[keys[0]]["spk_out"]
################################################################################


################################################################################
# Quantized LDN on LMT via Lava (`CLdnEncToSpkOnLmtModel`)
################################################################################
def get_spikes_from_QT_LDN_LMT_Lv(idx, order, theta):
  RTC.IS_SCALED_LDN = True
  RTC.ORDER = order
  RTC.THETA = theta/1000

  sig_on_cpu = InpSigToLdn(rtc=RTC, crnt_sig_idx=idx)
  enc_on_lmt = LdnEncToSpk(
    in_shape=(1, ), order=RTC.ORDER, theta=RTC.THETA, gain=RTC.GAIN,
    bias=RTC.BIAS, v_thr=RTC.ENC_V_THR,
    scale_factor=EXC.SCALE_FACTOR, per_smpl_tsteps=RTC.DCFG["n_ts"])

  sig_on_cpu.sig_out.connect(enc_on_lmt.sig_inp)

  run_config = Loihi2HwCfg(
    exception_proc_model_map={
      LdnEncToSpk: CLdnEncToSpkOnLmtModel,
      InpSigToLdn: PyInpSigToLdnModel
    }
  )

  spk_monitor = Monitor()
  spk_monitor.probe(enc_on_lmt.spk_out, RTC.DCFG["n_ts"])

  sig_on_cpu.run(
    condition=RunSteps(num_steps=RTC.DCFG["n_ts"]), run_cfg=run_config)
  spk_data = spk_monitor.get_data()
  keys = list(spk_data.keys())
  sig_on_cpu.stop()

  return spk_data[keys[0]]["spk_out"]
################################################################################


################################################################################
# Get the Spike-Train Synchrony Scores
################################################################################
def get_spike_train_synchrony_scores(tst_x_sig, scld_tst_x_sig, idx, order,
                                     theta):
  blockPrint()
  spikes_rv_cpu_py = get_spikes_from_RV_LDN_CPU_Py(tst_x_sig, order, theta)
  spikes_qt_cpu_py = get_spikes_from_QT_LDN_CPU_Py(scld_tst_x_sig, order, theta)
  spikes_qt_cpu_lv = get_spikes_from_QT_LDN_CPU_Lv(idx, order, theta)
  spikes_qt_lmt_lv = get_spikes_from_QT_LDN_LMT_Lv(idx, order, theta)
  enablePrint()

  print(spikes_rv_cpu_py.shape, spikes_qt_cpu_py.shape,
        spikes_qt_cpu_lv.shape, spikes_qt_lmt_lv.shape)

  rv_cpu_py_vs_qt_cpu_py = ExpUtils.get_dist_metric_bwn_ldn_spk_trains(
      RTC.ORDER, spikes_rv_cpu_py.T, spikes_qt_cpu_py.T)

  qt_cpu_py_vs_qt_cpu_lv = ExpUtils.get_dist_metric_bwn_ldn_spk_trains(
      RTC.ORDER, spikes_qt_cpu_py.T, spikes_qt_cpu_lv.T)

  qt_cpu_py_vs_qt_lmt_lv = ExpUtils.get_dist_metric_bwn_ldn_spk_trains(
      RTC.ORDER, spikes_qt_cpu_py.T, spikes_qt_lmt_lv.T)

  rv_cpu_py_vs_qt_cpu_lv = ExpUtils.get_dist_metric_bwn_ldn_spk_trains(
      RTC.ORDER, spikes_rv_cpu_py.T, spikes_qt_cpu_lv.T)

  rv_cpu_py_vs_qt_lmt_lv = ExpUtils.get_dist_metric_bwn_ldn_spk_trains(
      RTC.ORDER, spikes_rv_cpu_py.T, spikes_qt_lmt_lv.T)

  qt_cpu_lv_vs_qt_lmt_lv = ExpUtils.get_dist_metric_bwn_ldn_spk_trains(
      RTC.ORDER, spikes_qt_cpu_lv.T, spikes_qt_lmt_lv.T)

  return rv_cpu_py_vs_qt_cpu_py, qt_cpu_py_vs_qt_cpu_lv, qt_cpu_py_vs_qt_lmt_lv, \
        rv_cpu_py_vs_qt_cpu_lv, rv_cpu_py_vs_qt_lmt_lv, qt_cpu_lv_vs_qt_lmt_lv
################################################################################

def get_one_row(idx, order, theta, comparison):
  return [
      idx, order, theta,
      comparison["isi_dist"][0], comparison["isi_dist"][1],
      comparison["spk_dist"][0], comparison["spk_dist"][1],
      comparison["spk_sync"][0], comparison["spk_sync"][1],
      comparison["vpr_dist"][0], comparison["vpr_dist"][1],
      comparison["vrm_dist"][0], comparison["vrm_dist"][1]
      ]
################################################################################

def get_one_dataset_spk_train_sim_dissim_socres(dataset, n_sigs):
  # `n_sigs` = Randomly pick 10 sample signals for each dataset.

  RTC.DATASET = dataset
  RTC.DCFG = EXC.DATASETS_CFG[RTC.DATASET]
  RTC.IS_SLAYER_TRNG = False
  # Disable shulffling of test data as this set is used for 5 sample signals.
  # Note that Lava Processes use test signals in their InpSigToLdn Process.
  # This ensures that same 5 random signals are used for all three
  # implementations LDN --  RV-LDN-CPU-Py, QT-LDN-CPU-Lv, and QT-LDN-LMT-Lv
  RTC.DO_SHUFFLE_TEST_DATA = False

  RTC.GAIN = 1
  RTC.BIAS = 0
  RTC.ENC_V_THR = 1

  dpu = DataPrepUtils(RTC)
  _, _, test_x, _ = dpu.get_exp_compatible_train_test_x_y()
  _, _, scaled_test_x, _ = dpu.get_scaled_train_and_test_x_y_data()

  ALL_IDCS = np.random.choice(RTC.DCFG["tst_size"], size=n_sigs, replace=False)

  print("Dataset: ", dataset, "Test_X shape: ", test_x.shape)
  print("Test Indices: ", ALL_IDCS)

  # Initialize empty rows of the six comparisons datarame.
  rv_cpu_py_vs_qt_cpu_py_rows = []
  qt_cpu_py_vs_qt_cpu_lv_rows = []
  qt_cpu_py_vs_qt_lmt_lv_rows = []
  rv_cpu_py_vs_qt_cpu_lv_rows = []
  rv_cpu_py_vs_qt_lmt_lv_rows = []
  qt_cpu_lv_vs_qt_lmt_lv_rows = []

  for idx in ALL_IDCS:
    for order in ALL_ORDERS:
      for theta in ALL_THETAE:
        print(idx, order, theta)

        (rv_cpu_py_vs_qt_cpu_py, qt_cpu_py_vs_qt_cpu_lv,
         qt_cpu_py_vs_qt_lmt_lv, rv_cpu_py_vs_qt_cpu_lv,
         rv_cpu_py_vs_qt_lmt_lv, qt_cpu_lv_vs_qt_lmt_lv) = (
         get_spike_train_synchrony_scores(test_x[idx], scaled_test_x[idx],
                                          idx, order, theta)
        )

        rv_cpu_py_vs_qt_cpu_py_rows.append(
            get_one_row(idx, order, theta, rv_cpu_py_vs_qt_cpu_py))
        qt_cpu_py_vs_qt_cpu_lv_rows.append(
            get_one_row(idx, order, theta, qt_cpu_py_vs_qt_cpu_lv))
        qt_cpu_py_vs_qt_lmt_lv_rows.append(
            get_one_row(idx, order, theta, qt_cpu_py_vs_qt_lmt_lv))
        rv_cpu_py_vs_qt_cpu_lv_rows.append(
            get_one_row(idx, order, theta, rv_cpu_py_vs_qt_cpu_lv))
        rv_cpu_py_vs_qt_lmt_lv_rows.append(
            get_one_row(idx, order, theta, rv_cpu_py_vs_qt_lmt_lv))
        qt_cpu_lv_vs_qt_lmt_lv_rows.append(
            get_one_row(idx, order, theta, qt_cpu_lv_vs_qt_lmt_lv))

    print("Dataset: %s, IDX: %s done!" % (dataset, idx))

  df_rv_cpu_py_vs_qt_cpu_py = pd.DataFrame(
      rv_cpu_py_vs_qt_cpu_py_rows, columns=columns)
  df_qt_cpu_py_vs_qt_cpu_lv = pd.DataFrame(
      qt_cpu_py_vs_qt_cpu_lv_rows, columns=columns)
  df_qt_cpu_py_vs_qt_lmt_lv = pd.DataFrame(
      qt_cpu_py_vs_qt_lmt_lv_rows, columns=columns)
  df_rv_cpu_py_vs_qt_cpu_lv = pd.DataFrame(
      rv_cpu_py_vs_qt_cpu_lv_rows, columns=columns)
  df_rv_cpu_py_vs_qt_lmt_lv = pd.DataFrame(
      rv_cpu_py_vs_qt_lmt_lv_rows, columns=columns)
  df_qt_cpu_lv_vs_qt_lmt_lv = pd.DataFrame(
      qt_cpu_lv_vs_qt_lmt_lv_rows, columns=columns)

  path = "./spk-train-sync-scores/"
  df_rv_cpu_py_vs_qt_cpu_py.to_csv(
      path+"/%s_rv_cpu_py_vs_qt_cpu_py.csv" % dataset)
  df_qt_cpu_py_vs_qt_cpu_lv.to_csv(
      path+"/%s_qt_cpu_py_vs_qt_cpu_lv.csv" % dataset)
  df_qt_cpu_py_vs_qt_lmt_lv.to_csv(
      path+"/%s_qt_cpu_py_vs_qt_lmt_lv.csv" % dataset)
  df_rv_cpu_py_vs_qt_cpu_lv.to_csv(
      path+"/%s_rv_cpu_py_vs_qt_cpu_lv.csv" % dataset)
  df_rv_cpu_py_vs_qt_lmt_lv.to_csv(
      path+"/%s_rv_cpu_py_vs_qt_lmt_lv.csv" % dataset)
  df_qt_cpu_lv_vs_qt_lmt_lv.to_csv(
      path+"/%s_qt_cpu_lv_vs_qt_lmt_lv.csv" % dataset)


if __name__ == "__main__":
  columns = ["TEST_IDX", "ORDER", "THETA",
           "ISI_DIST_MEAN", "ISI_DIST_STD",
           "SPK_DIST_MEAN", "SPK_DIST_STD",
           "SPK_SYNC_MEAN", "SPK_SYNC_STD",
           "VPR_DIST_MEAN", "VPR_DIST_STD",
           "VRM_DIST_MEAN", "VRM_DIST_STD"
          ]

  ALL_DATASETS = ["ECG5000", "FordA",
                  "FordB", "Wafer", "Earthquakes", "Coffee",
                  "ToeSegmentation1", "ToeSegmentation2", "Lightning2", "Worms",
                  "WormsTwoClass", "OliveOil", "Meat", "Computers",
                  "RefrigerationDevices"]
  ALL_ORDERS = [4, 6, 8, 10, 12, 14, 16, 24]
  ALL_THETAE = [110, 130, 150]

  timestamp = ExpUtils.get_timestamp()

  for dataset in ALL_DATASETS:
    get_one_dataset_spk_train_sim_dissim_socres(dataset, n_sigs=5)
