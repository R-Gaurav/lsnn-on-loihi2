from . import _init_paths
import torch
import numpy as np

from torch.utils.data import Dataset

from consts.exp_consts import EXC
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils
from utils.network_utils.net_builder_utils import TensorTwoNeuronEncoder, LDNOps

###############################################################################
###############################################################################

class LDNLayer(object):
  def __init__(self, rtc):
    """
    Args:
      rtc <class>: RunTime Constants class.
    """
    self._rtc = rtc
    self._ldn = LDNOps(rtc)

  def get_all_sigs_all_ts_ldn_state_vectors(self, inputs):
    """
    Executes the LDN and returns the LDN state-vectors for all the time-steps of
    the signal, i.e., for the entire duration of each input signal in a dataset.

    Args:
      inputs <np.ndarray>: A 3D matrix of input signals, shape: (all n_samples,
                           all n_dimensions, entire sig_duration).
    """
    log.INFO(
        "Extracting the LDN state-vectors of: %s signals" % self._rtc.DATASET)
    n_samples, n_dim, n_ts = inputs.shape
    assert self._rtc.DCFG["n_dim"] == n_dim and self._rtc.DCFG["n_ts"] == n_ts
    ldn_stvecs = np.zeros(
        (n_samples, n_dim*self._rtc.ORDER, n_ts), dtype=EXC.NP_DTYPE)

    for dim in range(n_dim):
      for i, inp in enumerate(inputs[:, dim, :]):
        inp =  inp.reshape(1, -1)
        ldn_stvecs[i, dim*self._rtc.ORDER : (dim+1)*self._rtc.ORDER, :] = (
            self._ldn.get_ldn_extracted_signals_from_1dim_input(inp).T)

    log.INFO("LDN state-vectors obtained, shape: {0}".format(ldn_stvecs.shape))
    return ldn_stvecs

  def extract_and_save_ldn_state_vectors(self, inputs, file_name):
    """
    Saves the extracted LDN state-vectors of all the signals in the `inputs`.

    Args:
      inputs <np.ndarray>: A 3D matrix of input signals, shape: (all n_samples,
                           all n_dimensions, entire sig_duration).
      file_name <str>: File name of the extracted state-vectors to be saved.
    """
    # First remove the extracted files if they exist.
    log.INFO("Checking if already exists, if so then removing it...")
    ExpUtils.if_file_exists_then_remove_it(
        self._rtc.OUTPUT_PATH + "/" + file_name)
    # Now, obtain the LDN state-vectors and save them.
    ldn_stvecs = self.get_all_sigs_all_ts_ldn_state_vectors(inputs)
    ExpUtils.if_file_not_exists_then_save_it(ldn_stvecs,
        self._rtc.OUTPUT_PATH + "/" + file_name)
    log.INFO("Extracted LDN state-vectors saved!")

###############################################################################
###############################################################################

class SpkEncoderLayer(TensorTwoNeuronEncoder):
  def __init__(self, rtc):
    self._n_neurons = 2*rtc.DCFG["n_dim"]*rtc.ORDER # Neurons in Encoder Layer.
    TensorTwoNeuronEncoder.__init__(self, (self._n_neurons, ), rtc)

    # Create a transformation matrix which transforms input of `n` dimensions to
    # its doubled copy, such that each dimension is copied twice. E.g., an input
    # of dimesion 4, i.e., [a, b, c, d] is transformed to [a, a, b, b, c, c, d,
    # d] after multiplication with the matrix below.
    self._t_mat = ExpUtils.get_duplicating_transform_matrix(
        rtc.DCFG["n_dim"], rtc.ORDER, self._n_neurons
        )

  def re_initialize_neuron_states(self):
    """
    Re-initializes the neurons' voltage. Note that the encoding neuron does not
    have a corresponding "current" state.
    """
    self.re_initialize_voltage()

  def encode_ldn_stvecs(self, x_t):
    """
    Encodes the input x_t to spikes. Note that each scalar of the one sample
    vector x_t (in the batch input) is either positive or negative. Therefore,
    each scalar is connected to two neurons in the EncodingLayer, one neuron
    with positive encoder and other neuron with negative encoder.

    Args:
      x_t <torch.Tensor>: Real valued input of shape (batch_size, rtc.DCFG[
                          "n_dim"]*RTC.ORDER) where each row sample is a 1-D
                          vector at time t.
   """
    x_t = np.matmul(x_t, self._t_mat)
    return self.encode(x_t)

###############################################################################
###############################################################################

class ExpDataset(Dataset):
  """
  The ExpDataset object will be passed to a DataLoader class, such that it will
  be called every epoch for all the data samples. It should output binary spikes
  which are obtained by encoding the LDN state-vectors. The output spikes from
  this class are of shape: (2*ORDER, sig_duration). The spikes output from the
  DataLoader should be (batch_size, 2*ORDER, sig_duration). Note that the LDN
  state-vectors are extracted separate from this class, where they remain as is
  for a certain number of epochs, after which the state-vectors are extracted
  again from the reshuffled dataset; this is done to avoid extracting LDN
  state-vectors every epoch and save time.
  """
  def __init__(self, rtc, is_train):
    """
    Args:
      is_train <bool>: Instantiates the class with Training data if True else
                       Test data.
      rtc <Runtime Constants class>: Object of RuntimeConstants class.
    """
    super(ExpDataset, self).__init__()
    self._rtc = rtc
    self._spk_enc = SpkEncoderLayer(rtc)
    if is_train:
      ldn_file_name = "training_X_ldn_stvecs.p"
      labels_file_name = "training_Y.p"
    else:
      ldn_file_name = "test_X_ldn_stvecs.p"
      labels_file_name = "test_Y.p"
    self.ldn_stvecs = ExpUtils.load_file(rtc.OUTPUT_PATH + "/" +  ldn_file_name)
    self.labels = ExpUtils.load_file(rtc.OUTPUT_PATH + "/" + labels_file_name)

  def __len__(self):
    """
    Returns the number of samples in the dataset.
    """
    return len(self.labels)

  def __getitem__(self, idx):
    """
    Returns the binary spikes matrix for the data sample corresponding to the
    index `idx`. The matrix is of shape (2*ORDER, sig_duration).
    """
    # Encode the LDN state-vectors of shape (ORDER, sig_duration) to spikes of
    # shape (2*ORDER, sig_duration) via the Two-Neuron encoder. For each call of
    # this function, reset the neuron states in the SpkEncoderLayer.
    self._spk_enc.re_initialize_neuron_states()
    spikes = np.zeros( # Last dimension should be total number of time-steps.
        (2*self._rtc.DCFG["n_dim"]*self._rtc.ORDER, self._rtc.DCFG["n_ts"]),
        dtype=np.float32)

    for t in range(self._rtc.DCFG["n_ts"]):
      spikes[:, t] = self._spk_enc.encode_ldn_stvecs(self.ldn_stvecs[idx, :, t])

    return spikes, self.labels[idx]
###############################################################################
