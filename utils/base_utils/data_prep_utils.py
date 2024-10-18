from . import _init_paths

import numpy as np

from aeon import datasets
from sklearn.utils import shuffle

from consts.exp_consts import EXC
from consts.dir_consts import DRC
from utils.base_utils import log

class DataPrepUtils(DRC):
  def __init__(self, rtc):
    """
    Args:
      rtc <RTC>: Run Time Constants class.
    """
    super().__init__(rtc.DATASET)
    self._rtc = rtc

  def _load_ts_dataset(self, dataset):
    """
    Loads the *.ts formatted dataset.

    Args:
      dataset <str>: Train or Test dataset file name.
    """
    log.INFO("Loading dataset: {}".format(dataset))
    data = datasets.load_from_tsfile(self.data_path+"/%s" % dataset) # Tuple.
    log.INFO("ts format dataset loaded, it's shape: {}".format(data[0].shape))
    return data[0], data[1] # First tuple entry is X, Last tuple entry is Y.

  def get_x_y_from_dataset(self):
    """
    Returns the train_x, train_y, test_x, test_y data from the chosen dataset in
    `self._data_path` in DRC.

    Returns:
      np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """
    train_x, train_y = self._load_ts_dataset(self.train_set)
    test_x, test_y = self._load_ts_dataset(self.test_set)

    _, n_dim, n_ts = train_x.shape
    assert self._rtc.DCFG["n_dim"] == n_dim
    if self._rtc.IS_SLAYER_TRNG:
      assert self._rtc.DCFG["n_ts"] == n_ts
    assert (n_dim, n_ts) == (test_x.shape[1], test_x.shape[2])
    assert self._rtc.DCFG["tst_size"] == test_x.shape[0]

    # Shuffle the training and test dataset.
    train_x, train_y = shuffle(train_x, train_y)
    if self._rtc.DO_SHUFFLE_TEST_DATA:
      test_x, test_y = shuffle(test_x, test_y)

    return train_x, train_y, test_x, test_y

  def make_dataset_n_ary_classification(self, train_y, test_y):
    """
    In Worms, OliveOil, and Meat datasets, the number of classes are: 5, 4, 3
    respectively, which are all numbered from '1' onwards (as characters). This
    function makes them numeral, starting from 1 onwards.

    Args:
      train_y <np.ndarray>: The training lables.
      test_y <np.ndarray>: The test labels.
    """
    return train_y.astype(np.int32), test_y.astype(np.int32)

  def make_dataset_binary_classification(self, train_y, test_y):
    """
    In ECG5000 dataset, the normal class is 1 and all the other classes are
    abnormal classes. In FordA, FordB, and Wafer datasets, the binary classes
    are -1 and 1. In Earthquakes dataset, the classes are 1 and 0. This function
    binarizes the dataset with classes as 1 and 2 only.

    Args:
      train_y <np.ndarray>: The training lables.
      test_y <np.ndarray>: The test labels.
    """
    train_y = train_y.astype(np.int32)
    test_y = test_y.astype(np.int32)
    # To binarize, make all the classes which are NOT 1, as class 2.
    train_y[np.where(train_y != 1)] = 2
    test_y[np.where(test_y != 1)] = 2

    return train_y, test_y

  def get_exp_compatible_train_test_x_y(self):
    """
    Returns experiment compatible train_x, train_y, test_x, test_y dataset.
    """
    train_x, train_y, test_x, test_y = self.get_x_y_from_dataset()
    n_samples, _, _ = train_x.shape # Shape: (n_samples, n_dim, n_ts).

    # Relabel the class labels to 1, 2, ... num_classes.
    if self._rtc.DATASET in [
        "Worms", "OliveOil", "Meat", "RefrigerationDevices"]:
      train_y, test_y = self.make_dataset_n_ary_classification(train_y, test_y)
    else:
      train_y, test_y = self.make_dataset_binary_classification(train_y, test_y)
    # Convert the labels to int, and subtract 1 since the labels start at 1.
    train_y, test_y = train_y.astype(int)-1, test_y.astype(int)-1
    assert n_samples == train_y.shape[0]
    assert self._rtc.DCFG["n_classes"] == np.unique(train_y).size
    assert self._rtc.DCFG["n_classes"] == np.unique(test_y).size
    log.INFO("Returning original train_x: {0}, train_y: {1}, test_x: {2}, "
             "test_y: {3} data shapes".format(
             train_x.shape, train_y.shape, test_x.shape, test_y.shape))
    return train_x, train_y, test_x, test_y

  def get_scaled_train_and_test_x_y_data(self):
    """
    Returns the scaled up (with fixed-point precision) train and test data.
    """
    log.INFO("Scaling up the train_x and test_x input signals...")
    train_x, train_y, test_x, test_y = self.get_exp_compatible_train_test_x_y()
    scaled_train_x, scaled_test_x = (
        np.round(train_x * EXC.SCALE_FACTOR).astype(np.int32),
        np.round(test_x * EXC.SCALE_FACTOR).astype(np.int32)
        )
    log.INFO("Returning scaled train_x: {0}, train_y: {1}, test_x: {2}, "
             "test_y: {3} data shapes".format(scaled_train_x.shape,
             train_y.shape, scaled_test_x.shape, test_y.shape))
    return scaled_train_x, train_y, scaled_test_x, test_y
################################################################################
