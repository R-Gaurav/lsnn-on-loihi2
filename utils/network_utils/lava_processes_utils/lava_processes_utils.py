import numpy as np

from lava.magma.core.decorator import implements, requires
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.model.c.model import CLoihiProcessModel
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.c.ports import CInPort, COutPort
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.c.type import LavaCDataType, LavaCType
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import LMT, CPU, Loihi2NeuroCore
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.utils.system import Loihi2
if Loihi2.is_loihi2_available:
  from lava.proc import embedded_io as eio

from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils
from utils.base_utils.data_prep_utils import DataPrepUtils

###############################################################################
################# I N P U T    S I G N A L     T O     L D N ##################
###############################################################################

class InpSigToLdn(AbstractProcess):
  """
  Inputs a univarate time-series signal to the LDN (on either CPU or LMT).
  """
  def __init__(self, crnt_sig_idx: int, **kwargs):
    super().__init__(**kwargs) # NOTE: Pass kwargs to access in ProcessModels.
    rtc = kwargs.get("rtc", None)
    self.sig_out = OutPort(shape=(rtc.DCFG["n_dim"], ))
    self.lbl_out = OutPort(shape=(1, ))

    self.crnt_idx = Var(shape=(1, ), init=crnt_sig_idx)
    self.ps_ts = Var(shape=(1, ), init=rtc.DCFG["n_ts"]) # per_smpl_tsteps.
    self.sig_inp = Var(shape=(rtc.DCFG["n_ts"], ), init=0)
    self.ground_truth = Var(shape=(1,))

# -----------------------------------------------------------------------------
# Input Signal to LDN on CPU.
# -----------------------------------------------------------------------------

@implements(proc=InpSigToLdn, protocol=LoihiProtocol)
@requires(CPU)
class PyInpSigToLdnModel(PyLoihiProcessModel):
  sig_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=32)
  lbl_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=32)

  crnt_idx: int = LavaPyType(int, int, precision=32)
  ps_ts: int = LavaPyType(int, int, precision=32)
  sig_inp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
  ground_truth: int = LavaPyType(int, int, precision=32)

  def __init__(self, proc_params):
    super().__init__(proc_params=proc_params)
    self._rtc = proc_params.get("rtc")
    self.dpu = DataPrepUtils(self._rtc)
    _, _, self.test_x, self.test_y = (
        self.dpu.get_scaled_train_and_test_x_y_data()
        )
    self.test_x = self.test_x.squeeze(axis=1) # Squeeze the Univariate dim.

  def post_guard(self):
    """
    Guard function to the Post-Management Phase.
    """
    if self.time_step % self.ps_ts == 1:
      return True

    return False

  def run_post_mgmt(self):
    """
    Post-Management phase executed only when the post_guard() returns True.
    Update/reset the variables and send ground_truth to the receiving Process.
    """
    self.sig_inp = self.test_x[self.crnt_idx]
    self.ground_truth = self.test_y[self.crnt_idx]
    self.lbl_out.send(np.array([self.ground_truth]))
    self.crnt_idx += 1

  def run_spk(self):
    """
    This function is called every time-step. Note that the time-steps in Lava
    start with the index 1, and NOT 0.
    """
    if self.time_step % self.ps_ts == 1:
      self.sig_inp = np.zeros(self.sig_inp.shape, dtype=np.int32)

    sig_element = self.sig_inp[(self.time_step-1)%self.ps_ts].reshape(1, )
    self.sig_out.send(sig_element)

###############################################################################
###### L D N     S I G N A L     E N C O D I N G     T O     S P I K E ########
###############################################################################

class LdnEncToSpk(AbstractProcess):
  """
  Implements LDN encoding and conversion to binary spikes (on CPU or on LMT).
  The spikes are fed to the netx-obtained network on Loihi-2 neuro-cores.
  """
  def __init__(self, in_shape: tuple, order: int, theta: float,
               gain: int, bias: int, v_thr: int, scale_factor: int,
               per_smpl_tsteps: int) -> None:
    super().__init__()
    assert scale_factor is not None and scale_factor > 0
    # Get the A-prime and B-prime matrices, as well as Encoders.
    Ap, Bp = ExpUtils.get_scaled_Ap_Bp_matrices(
        scale_factor=scale_factor, order=order, theta=theta, is_c2d=True)
    log.INFO("*"*80)
    log.INFO("Scaled and quantized A-prime matrix:")
    log.INFO(Ap)
    log.INFO("Scaled and quantized B-prime matrix:")
    log.INFO(Bp)
    log.INFO("*"*80)
    enc = ExpUtils.get_encoders(num_neurons=2*order)
    t_mat = ExpUtils.get_duplicating_transform_matrix(
        n_dim=1, order=order, n_neurons=2*order) # n_dim = 1 for Univ. signal.

    self.scale_factor = Var(shape=(1, ), init=scale_factor)
    self.ORDER = Var(shape=(1, ), init=order)
    # Set global gain value, where the scale_factor is subsumed in the scaled
    # LDN signals already, so NO need to scale up the `gain` value below.
    self.g_gain = Var(shape=(1, ), init=gain)
    # bias is set to 0 in experiments, so scaling has no effect here.
    self.g_bias = Var(shape=(1, ), init=scale_factor * bias) # Global bias.
    self.g_v_thr = Var(shape=(1, ), init=scale_factor * v_thr) # Global v_thr.
    self.Ap = Var(shape=(order, order), init=Ap) # Initialize the A-prime mat.
    self.Bp = Var(shape=(order, 1), init=Bp) # Initialize the B-prime matrix.
    self.E = Var(shape=(2*order, ), init=enc) # Initiliaze the Encoder matrix.
    self.ps_ts = Var(shape=(1, ), init=per_smpl_tsteps) # Per-sample pres tstep.
    self.t_mat = Var(shape=(order, 2*order), init=t_mat)
    self.ldn_state = Var(shape=(order, ), init=0)
    self.volt = Var(shape=(2*order, ), init=0) # Number of neurons = 2 * order.

    self.sig_inp = InPort(shape=in_shape)
    self.spk_out = OutPort(shape=(2*order, ))

# -----------------------------------------------------------------------------
# LDN Signals Encoding to Spike on LMT
# -----------------------------------------------------------------------------

@implements(proc=LdnEncToSpk, protocol=LoihiProtocol)
@requires(LMT)
class CLdnEncToSpkOnLmtModel(CLoihiProcessModel):
  sig_inp: CInPort = LavaCType(cls=CInPort, d_type=LavaCDataType.INT32)
  spk_out: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)

  Ap: np.ndarray = LavaCType(cls=np.ndarray, d_type=LavaCDataType.INT32)
  Bp: np.ndarray = LavaCType(cls=np.ndarray, d_type=LavaCDataType.INT32)
  E: np.ndarray = LavaCType(cls=np.ndarray, d_type=LavaCDataType.INT32)
  ldn_state: np.ndarray = LavaCType(cls=np.ndarray, d_type=LavaCDataType.INT32)
  volt: np.ndarray = LavaCType(cls=np.ndarray, d_type=LavaCDataType.INT32)
  t_mat: np.ndarray = LavaCType(cls=np.ndarray, d_type=LavaCDataType.INT32)

  ORDER: int = LavaCType(cls=int, d_type=LavaCDataType.INT32)
  scale_factor: int = LavaCType(cls=int, d_type=LavaCDataType.INT32)
  g_gain: int = LavaCType(cls=int, d_type=LavaCDataType.INT32)
  g_bias: int = LavaCType(cls=int, d_type=LavaCDataType.INT32)
  g_v_thr: int = LavaCType(cls=int, d_type=LavaCDataType.INT32)
  ps_ts: int = LavaCType(cls=int, d_type=LavaCDataType.INT32)

  @property
  def source_file_name(self):
    return "ldn_enc_to_spk_on_lmt.c"

# -----------------------------------------------------------------------------
# LDN Signals Encoding to Spike on CPU
# -----------------------------------------------------------------------------

@implements(proc=LdnEncToSpk, protocol=LoihiProtocol)
@requires(CPU)
class PyLdnEncToSpkOnCpuModel(PyLoihiProcessModel):
  """
  Generates LDN signals from the Input signals and encodes them to binary
  spikes, all done on CPU.
  """
  sig_inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32, precision=32)
  spk_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)

  Ap: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
  Bp: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
  E: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
  ldn_state: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
  volt: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
  t_mat: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)

  ORDER: int = LavaPyType(int, int, precision=32)
  scale_factor: int = LavaPyType(int, int, precision=32)
  g_gain: int = LavaPyType(int, int, precision=32)
  g_bias: int = LavaPyType(int, int, precision=32)
  g_v_thr: int = LavaPyType(int, int, precision=32)
  ps_ts: int = LavaPyType(int, int, precision=32)

  def __init__(self, proc_params):
    super().__init__(proc_params=proc_params)

  def post_guard(self):
    """
    Guard function to the Post-Management Phase.
    """
    if self.time_step % self.ps_ts == 1:
      return True

    return False

  def run_post_mgmt(self):
    """
    Post-Management phase executed only when the post_guard() returns True.
    """
    self.ldn_state = np.zeros(self.ORDER, dtype=np.int32)
    self.volt = np.zeros(2*self.ORDER, dtype=np.int32) # num_neurons = 2 * order

  def run_spk(self):
    """
    Spiking phase, executed every time-step and first in order of all phases.
    """
    if self.time_step % self.ps_ts == 1:
      self.ldn_state = np.zeros(self.ldn_state.shape, dtype=np.int32)
      self.volt = np.zeros(self.volt.shape, dtype=np.int32)

    inp = self.sig_inp.recv() # Receive scaled signal input.
    # Compute Ap * x and B * u.
    state = self.Ap@self.ldn_state + self.Bp@inp
    # Divide and take ceil op.
    self.ldn_state = np.ceil(state/self.scale_factor).astype(np.int32)
    # Rate Encode the current time-step's `self.ldn_state`
    J = self.g_gain * (
            self.E * np.matmul(self.ldn_state, self.t_mat)) + self.g_bias
    self.volt = ExpUtils.update_voltage(self.volt, J)
    spikes = ExpUtils.spike_func(self.volt - self.g_v_thr).astype(bool)
    self.volt = ExpUtils.reset_voltage(self.volt, spikes)
    self.spk_out.send(spikes)

###############################################################################
############# O U T P U T     S P I K E S     T O     C L A S S ###############
###############################################################################

class OutSpkToCls(AbstractProcess):
  """
  Implements inferring classes from the output spikes (on CPU). The spikes are
  obtained from the netx network.
  """
  def __init__(self, per_smpl_tsteps: int, n_test_sigs: int, n_cls_shape: tuple):
    super().__init__()
    self.spk_in = InPort(shape=n_cls_shape)
    self.lbl_in = InPort(shape=(1, ))

    self.spks_accum = Var(shape=n_cls_shape, init=0)
    self.ps_ts = Var(shape=(1, ), init=per_smpl_tsteps)
    self.pred_lbls = Var(shape=(n_test_sigs, ), init=0)
    self.true_lbls = Var(shape=(n_test_sigs, ), init=0)

# ------------------------------------------------------------------------------
# Output Spikes to Class on CPU.
# ------------------------------------------------------------------------------
@implements(proc=OutSpkToCls, protocol=LoihiProtocol)
@requires(CPU)
class PyOutSpkToClsModel(PyLoihiProcessModel):
  spk_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
  lbl_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int, precision=32)

  spks_accum: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=32)
  ps_ts: int = LavaPyType(int, int, precision=32)
  pred_lbls: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
  true_lbls: np.ndarray = LavaPyType(np.ndarray, int, precision=32)

  def __init__(self, proc_params):
    super().__init__(proc_params=proc_params)
    self.crnt_idx = 0

  def post_guard(self):
    """
    Guard function to the Post-Management Phase.
    """
    if self.time_step % self.ps_ts == 0: # Presentation of one sample is over.
      return True

    return False

  def run_post_mgmt(self):
    """
    Post-Management phase: executed only when the post_guard() returns True.
    """
    true_lbl = self.lbl_in.recv()
    pred_lbl = np.argmax(self.spks_accum)
    self.true_lbls[self.crnt_idx] = true_lbl[0]
    self.pred_lbls[self.crnt_idx] = pred_lbl
    self.crnt_idx += 1
    self.spks_accum = np.zeros_like(self.spks_accum)

  def run_spk(self):
    """
    Spiking phase, executed every time-step and first in order of all phases.
    """
    spks = self.spk_in.recv()
    self.spks_accum = self.spks_accum + spks

###############################################################################
######################### I N P U T     A D A P T E R #########################
###############################################################################

class InputAdapter(AbstractProcess):
  """
  Input adapter Process. Simply relays the input based on different backends.

  Args:
    shape <(int)>: Shape of input.
  """
  def __init__(self, shape: tuple):
    super().__init__(shape=shape)
    self.inp = InPort(shape=shape)
    self.out = OutPort(shape=shape)

# ------------------------------------------------------------------------------
# Input Adapter from host CPU to host CPU.
# ------------------------------------------------------------------------------
@implements(proc=InputAdapter, protocol=LoihiProtocol)
@requires(CPU)
class PyInputAdapterModel(PyLoihiProcessModel):
  """ Input adapter ProcessModel for CPU. """
  inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
  out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

  def run_spk(self):
    self.out.send(self.inp.recv())

# ------------------------------------------------------------------------------
# Input Adapter from host CPU to Loihi-2.
# ------------------------------------------------------------------------------
@implements(proc=InputAdapter, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class NxInputAdapterModel(AbstractSubProcessModel):
  """ Input Adapter model to send signal to LDN on LMT (i.e., on Loihi2)"""
  def __init__(self, proc: AbstractProcess):
    self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
    self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)

    shape = proc.proc_params.get("shape")
    self.adapter = eio.spike.PyToNxAdapter(shape=shape)
    proc.inp.connect(self.adapter.inp)
    self.adapter.out.connect(proc.out)

###############################################################################
################### O U T P U T     A D A P T E R ####################
###############################################################################

class OutputAdapter(AbstractProcess):
  """
  Output adapter Process, simply relays the output based on different backends.

  Args:
    shape <(int)>: Shape of output.
  """
  def __init__(self, shape: tuple):
    super().__init__(shape=shape)
    self.inp = InPort(shape=shape)
    self.out = OutPort(shape=shape)

@implements(proc=OutputAdapter, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputAdapterModel(PyLoihiProcessModel):
  """ Output adapter ProcessModel for CPU. """
  inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
  out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

  def run_spk(self):
    self.out.send(self.inp.recv())

@implements(proc=OutputAdapter, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class NxOutputAdapterModel(AbstractSubProcessModel):
  """ Output Adapter model to send spikes from Loihi-2 neuro-cores to CPU. """
  def __init__(self, proc: AbstractProcess):
    self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
    self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)

    shape = proc.proc_params.get("shape")
    self.adapter = eio.spike.NxToPyAdapter(shape=shape)
    proc.inp.connect(self.adapter.inp)
    self.adapter.out.connect(proc.out)

###############################################################################
###############################################################################
############### F O R     H A R D - C O D E D     S I G N A L #################
###############################################################################
###############################################################################

class LdnEncToSpkOnLmtHCSig(AbstractProcess):
  """
  Process for Hard-Coded signal.
  """
  def __init__(self, in_shape: tuple, order: int, theta: float,
               gain: int, bias: int, v_thr: int, scale_factor: int) -> None:
    super().__init__()
    assert scale_factor is not None and scale_factor > 0
    # Get the A-prime and B-prime matrices, as well as Encoders.
    Ap, Bp = ExpUtils.get_scaled_Ap_Bp_matrices(
        scale_factor=scale_factor, order=order, theta=theta, is_c2d=True)
    enc = ExpUtils.get_encoders(num_neurons=2*order)

    self.scale_factor = Var(shape=(1, ), init=scale_factor)
    self.ORDER = Var(shape=(1, ), init=order)
    # Set global gain value, where the scale_factor is subsumed in the scaled
    # LDN signals already, so no need to scale up the `gain` value below.
    self.g_gain = Var(shape=(1, ), init=gain)
    # bias is set to 0 in experiments, so scaling has no effect here.
    self.g_bias = Var(shape=(1, ), init=scale_factor * bias) # Global bias.
    self.g_v_thr = Var(shape=(1, ), init=scale_factor * v_thr) # Global v_thr.
    self.Ap = Var(shape=(order, order), init=Ap) # Initialize the A-prime mat.
    self.Bp = Var(shape=(order, ), init=Bp) # Initialize the B-prime matrix.
    self.E = Var(shape=(2*order, ), init=enc) # Initiliaze the Encoder matrix.

    self.spk_out = OutPort(shape=(2*order, ))

# -----------------------------------------------------------------------------

@implements(LdnEncToSpkOnLmtHCSig, protocol=LoihiProtocol)
@requires(LMT)
class CLdnEncToSpkOnLmtHCSigModel(CLoihiProcessModel):
  """
  ProcessModel for Hard-Coded signal.
  """
  spk_out: COutPort = LavaCType(cls=COutPort, d_type=LavaCDataType.INT32)
  Ap: np.ndarray = LavaCType(cls=np.ndarray, d_type=LavaCDataType.INT32)
  Bp: np.ndarray = LavaCType(cls=np.ndarray, d_type=LavaCDataType.INT32)
  E: np.ndarray = LavaCType(cls=np.ndarray, d_type=LavaCDataType.INT32)
  ORDER: int = LavaCType(cls=int, d_type=LavaCDataType.INT32)
  scale_factor: int = LavaCType(cls=int, d_type=LavaCDataType.INT32)
  g_gain: int = LavaCType(cls=int, d_type=LavaCDataType.INT32)
  g_bias: int = LavaCType(cls=int, d_type=LavaCDataType.INT32)
  g_v_thr: int = LavaCType(cls=int, d_type=LavaCDataType.INT32)

  @property
  def source_file_name(self):
    return "ldn_enc_to_spk_on_lmt_hard_coded_sig.c"

###############################################################################
###############################################################################
###############################################################################
###############################################################################
