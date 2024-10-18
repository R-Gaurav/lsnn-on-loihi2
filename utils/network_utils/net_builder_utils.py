from . import _init_paths
import numpy as np

from abc import ABC, abstractmethod

from consts.exp_consts import EXC
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils

###############################################################################
###############################################################################

class BaseSpkNeuron(object): # No spiking function implemented here.
  def __init__(self, tensor_size, v_thr):
    """
    Args:
      tensor_size <tuple>: (Batch Size, Number of Neurons).
      v_thr <float>: Voltage Threshold.
    """
    self._v = np.zeros(tensor_size, dtype=EXC.NP_DTYPE)
    self._v_thr = np.int32(v_thr)

  def update_voltage(self, c):
    self._v = ExpUtils.update_voltage(self._v, c)

  def re_initialize_voltage(self):
    self._v = np.zeros_like(self._v)

  @abstractmethod
  def spike_and_reset_voltage(self):
    pass

###############################################################################
###############################################################################

class TensorTwoNeuronEncoder(BaseSpkNeuron):
  def __init__(self, tensor_size, rtc):
    """
    Args:
      tensor_size <tuple>: (Batch Size, Number of Neurons).
      rtc <RTC>: The runtime constants class.
    """
    super().__init__(
        tensor_size,
        rtc.ENC_V_THR*EXC.SCALE_FACTOR if rtc.IS_SCALED_LDN else rtc.ENC_V_THR
        )
    self._rtc = rtc
    self._tensor_size = tensor_size
    self._gain = rtc.GAIN
    self._bias = rtc.BIAS # BIAS is set to 0, so scaling has no effect here.
    self._e = self._get_e_values()

  def _get_e_values(self):
    assert len(self._tensor_size) == 1 # (Number of Neurons, ).
    enc = ExpUtils.get_encoders(self._tensor_size[0]) # Encoders for one sample.
    e = np.empty(self._tensor_size, dtype=EXC.NP_DTYPE)
    # Iterate over neurons, each having alternate +1 and -1 value from `enc`.
    for i in range(self._tensor_size[0]):
      e[i] = enc[i]

    assert e.sum() == 0
    return e

  # Implement the abstract method of the parent class.
  def spike_and_reset_voltage(self):
    delta_v = self._v - self._v_thr
    spikes = ExpUtils.spike_func(delta_v)
    self._v = ExpUtils.reset_voltage(self._v, spikes)

    return spikes

  def encode(self, x_t):
    """
    Encodes the vector x_t at time t, note that it is composed of two repeated
    values of each scalar in the extracted LDN signal at time t. That is, for
    the LDN signal [a, b, c, ..., n], `x_t` is [a, a, b, b, c, c, ..., n, n].

    Args:
      x_t <Tensor>: A vector input at time t.
    """
    J = self._gain*self._e*x_t + self._bias
    self.update_voltage(J)
    spikes = self.spike_and_reset_voltage()
    return spikes

###############################################################################
###############################################################################

class BaseLDN(object):
  """
  Create a non-spiking LDN.
  """
  def __init__(self, rtc, is_c2d=True):
    """
    Args:
      rtc <RTC>: RunTime constants.
      is_c2d <bool>: Use continuous to discrete transformation if True, else
                     don't.
    """
    self._scale_factor = EXC.SCALE_FACTOR if rtc.IS_SCALED_LDN else None
    self._is_c2d = is_c2d

    self._order = rtc.ORDER
    self._theta = rtc.THETA

    self._init_A_p_and_B_p_matrices()

  def _print_Ap_and_Bp_matrices(self):
    log.INFO("*"*80)
    log.INFO("Obtained A-prime matrix:")
    log.INFO(self.A_p)
    log.INFO("Obtained B-prime matrix:")
    log.INFO(self.B_p)
    log.INFO("*"*80)

  def _init_A_p_and_B_p_matrices(self):
    if self._scale_factor == None:
      log.INFO("Getting unscaled original A-prime and B-prime matrices...")
      self.A_p, self.B_p = ExpUtils.get_Ap_Bp_matrices(
          order=self._order, theta=self._theta, is_c2d=self._is_c2d)
    else:
      log.INFO("Getting scaled quantized A-prime and B-prime matrices...")
      self.A_p, self.B_p = ExpUtils.get_scaled_Ap_Bp_matrices(
          scale_factor=self._scale_factor,
          order=self._order, theta=self._theta, is_c2d=self._is_c2d)

    self._print_Ap_and_Bp_matrices()

###############################################################################
###############################################################################

class LDNOps(BaseLDN):
  def __init__(self, rtc, is_c2d=True):
    """
    Args:
      rtc <class>: Run Time Constants class.
      is_c2d <bool>: Use contiuous to discrete transformation if True else
                     don't.
    """
    # Note that BaseLDN creates the attribute self._scale_factor.
    BaseLDN.__init__(self, rtc, is_c2d=is_c2d)

    if rtc.IS_SCALED_LDN == True:
      assert self._scale_factor > 0
      log.INFO("Using the scale factor: %s to obtain quantized A-prime and "
               "B-prime matrices." % self._scale_factor)
    else:
      log.INFO("Using original unscaled A-prime and B-prime matrices.")

  def get_ldn_transform(self, u_t, x_t):
    """
    Returns the LDN transform of the scalar input u_t, i.e., the next
    time-step's (t+1)^{th} state vector.

    Args:
      u_t <float>: Scalar input at time t.
      x_t <Tensor>: State-vector tensor at time t.
    """
    return np.matmul(self.A_p, x_t) + np.matmul(self.B_p, u_t)

  def get_ldn_extracted_signals_from_1dim_input(self, u):
    """
    Returns the LDN extracted signals, i.e., the state-vector throughout time.
    Note that `u` is has only one dimension here, i.e., univariate signal.

    Args:
      u <Tensor>: 1D Input signal vector (all time-steps), shape: (1, sig_size).
    """
    # Initialize the state-vector `x`.
    x = np.zeros((self._order, 1), dtype=EXC.NP_DTYPE)
    out = np.zeros((u.shape[1], self._order), dtype=EXC.NP_DTYPE)

    for i, u_t in enumerate(u[0, :]):
      u_t = u_t.reshape(1, 1)
      out[i, :] = self.get_ldn_transform(u_t, x).reshape(-1)
      if self._scale_factor is not None:
        # Since the input `u_t` is an integer and `A_p` and `B_p` are also
        # integers, the multiplication `out` is also an integer.
        out[i, :] = np.ceil(out[i, :]/self._scale_factor).astype(np.int32)

      x[:, 0] = out[i, :] # Update the state vector `x`.

    return out

###############################################################################
###############################################################################
