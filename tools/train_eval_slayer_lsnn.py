# This file only train and evaluates the SLAYER LSNN. Its conversion to Lava
# LSNN and subsequent evaluation on Loihi2Sim or Loihi2Hw is done in a separate
# class LavaLSNN in the file tools/networks/lava_networks.py.

from . import _init_paths

import torch
import lava.lib.dl.slayer as slayer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from utils.network_utils.slayer_lsnn_utils import ExpDataset, LDNLayer
from utils.base_utils.data_prep_utils import DataPrepUtils
from tools.networks.lava_networks import SlayerSNN
from consts.runtime_consts import RTC
from consts.exp_consts import EXC
from consts.dir_consts import DRC, OUTPUT_DIR
from utils.base_utils.exp_utils import ExpUtils
from utils.base_utils import log

class TrEvSlayerLSNN(object):
  def __init__(self, rtc):
    self._rtc = rtc
    self._dpu = DataPrepUtils(rtc)
    self._model = SlayerSNN(rtc).to(rtc.DEVICE)
    self._ldn_lyr = LDNLayer(rtc)

  def get_experiment_train_test_x_y(self):
    if self._rtc.IS_SCALED_LDN:
      train_x, train_y, test_x, test_y = (
          self._dpu.get_scaled_train_and_test_x_y_data())
      log.INFO("Scaled and quantized train_x and test_x obtained.")
    else:
      train_x, train_y, test_x, test_y = (
          self._dpu.get_exp_compatible_train_test_x_y())
      log.INFO("Original unscaled train_x and test_x obtained.")

    log.INFO("Shapes of train_x: {0}, train_y: {1}, test_x: {2}, test_y: {3}" \
             .format(train_x.shape, train_y.shape, test_x.shape, test_y.shape))

    return train_x, train_y, test_x, test_y

  def extract_stvecs_and_save_training_eval_files(self):
    train_x, train_y, test_x, test_y = self.get_experiment_train_test_x_y()
    log.INFO("Extracting and saving train/test LDN state-vectors...")
    ldn_trn_x = self._ldn_lyr.extract_and_save_ldn_state_vectors(
        train_x, "training_X_ldn_stvecs.p")
    ldn_tst_x = self._ldn_lyr.extract_and_save_ldn_state_vectors(
        test_x, "test_X_ldn_stvecs.p")
    log.INFO("Saving the train/test labels...")
    ExpUtils.check_file_and_remove_and_save(
        train_y, self._rtc.OUTPUT_PATH+"/training_Y.p")
    ExpUtils.check_file_and_remove_and_save(
        test_y, self._rtc.OUTPUT_PATH+"/test_Y.p")

  def train_eval_slayer_lsnn(self):
    loss = slayer.loss.SpikeRate(
        true_rate=0.99, false_rate=0.01, reduction="sum").to(self._rtc.DEVICE)
    stats = slayer.utils.LearningStats()
    optimizer = torch.optim.Adam(self._model.parameters(), lr=EXC.LEARNING_RATE)
    assistant = slayer.utils.Assistant(
        self._model, loss, optimizer, stats,
        classifier=slayer.classifier.Rate.predict)
    # Halve the learning rate every `step_size`, i.e., twice of SHUFFLE_EPOCHS.
    scheduler = StepLR(optimizer, step_size=2*EXC.SHUFFLE_EPOCHS, gamma=0.5)

    # Get the dataset.
    log.INFO("Extracting and saving the training/test file for the first time.")
    self.extract_stvecs_and_save_training_eval_files()

    for epoch in range(1, self._rtc.EPOCHS+1):
      if epoch%EXC.SHUFFLE_EPOCHS == 0:
        log.INFO(
            "Epoch: %s, extracting and saving the training/test files." % epoch)
        self.extract_stvecs_and_save_training_eval_files()

      # Train the model.
      self._model.train()
      train_data = ExpDataset(self._rtc, is_train=True)
      train_loader = DataLoader(
          train_data, batch_size=self._rtc.DCFG["batch_size"], num_workers=2)
      for inp, lbl in train_loader:
        inp, lbl = inp.to(self._rtc.DEVICE), lbl.to(self._rtc.DEVICE)
        # NOTE: The `inp` has to be spikes with dtype=torch.float32, and `lbl`
        # should be an integer, i.e., NOT one-hot encoded.
        output = assistant.train(inp, lbl)

      # Evaluate the model.
      self._model.eval()
      test_data = ExpDataset(self._rtc, is_train=False)
      test_loader = DataLoader(
          test_data, batch_size=self._rtc.DCFG["batch_size"], num_workers=2)
      for inp, lbl in test_loader:
        inp, lbl = inp.to(self._rtc.DEVICE), lbl.to(self._rtc.DEVICE)
        output = assistant.test(inp, lbl)

      log.INFO("Epoch: {0}, Stats: {1}".format(epoch, stats))

      if stats.testing.best_accuracy:
        torch.save(self._model.state_dict(),
                   self._rtc.OUTPUT_PATH + "/trained_network.pt")

      stats.update()
      scheduler.step()

    # Now load the saved (best) model and export it in HDF5 format.
    self._model.load_state_dict(
        torch.load(self._rtc.OUTPUT_PATH + "/trained_network.pt"))
    self._model.export_hdf5(self._rtc.OUTPUT_PATH + "/trained_network.net")
################################################################################
