# This file execute the LavaLSNN on physical Loihi-2 chip/hardware.

import _init_paths

import argparse
import numpy as np
import os
import sys
from sklearn.metrics import confusion_matrix

from lava.magma.core.run_conditions import RunSteps

from consts.runtime_consts import RTC
from consts.exp_consts import EXC
from consts.dir_consts import OUTPUT_DIR
from tools.networks.lava_networks import LavaLSNN
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils

class L2HwLavaLSNN(LavaLSNN):
  def __init__(self, rtc, n_test_sigs, start_test_idx):
    assert rtc.BACKEND == "L2Hw" # LavaLSNN is hard-coded here to run with
                                 # SCALED and QUANTIZED version of LDN on LMT.
    super().__init__(rtc, n_test_sigs, start_test_idx)

  def eval_lava_lsnn(self, is_profiler_print=True):
    """
    Override the eval_lava_lsnn() of LavaLSNN class, and runs the LavaLSNN on
    the physical Loihi2 Hardware.

    Args:
      is_profiler_print <bool>: Print profiling results if True else do not.
    """
    # Create the LavaLSNN.
    self.create_lava_lsnn()
    # Get RunConfig based on the backend.
    run_config = self.get_loihi2_run_config()

    self.sig_to_ldn.run(
        condition=RunSteps(num_steps=self._rtc.DCFG["n_ts"]*self._n_test_sigs),
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
    log.INFO("Accuracy on Loihi Hardware: {0}".format(
        np.mean(np.array(all_true_lbls) == np.array(all_pred_lbls))))
    log.INFO("*"*80)

def setup_rtc(rtc, args):
  rtc.DATASET = args.dataset
  rtc.DCFG = EXC.DATASETS_CFG[args.dataset]
  # Set RESET_INTERVAL to a power of 2 for execution on physical Loihi-2 chip.
  if args.batch_size == 1:
    # For each sample, a fresh instantiation of LavaLSNN will be done, which
    # implies that there is no need to reset the network OR reset it way later
    # than the input signal's duration, thus, before that reset time-step is
    # reached, the signal ends, the class is inferred, and the LavaLSNN stops.
    rtc.RESET_INTERVAL = 1024 # 1024 is longer than any experimented dataset.
  elif args.batch_size > 1:
    sys.exit("Batch Size for Loihi-2 hardware inference should be 1. Exiting..")

  rtc.ORDER = args.order
  rtc.THETA = args.theta

  # Set hyper-params for Encoding Layer Neurons.
  # Only for the Encoding Layer (scaled later in LDN/LMT code)
  rtc.ENC_V_THR = EXC.ENC_V_THR_LIST[0] # Only 1 encoding V_THR is experimented.
  rtc.GAIN = EXC.ENC_GAIN_LIST[0]
  rtc.BIAS = EXC.ENC_BIAS_LIST[0]

  rtc.BACKEND = "L2Hw" # Loihi2HwCfg for inference on physical Loihi2 Board.
  rtc.OUTPUT_PATH = OUTPUT_DIR + \
                    "SCALED_QUANTIZED_LDN/%s/" % args.dataset + \
                    "ldn_order_%s_theta_%s/" % (args.order, args.theta) + \
                    "slayer_c_decay_%s_v_decay_%s/SEED_%s/" \
                    % (args.c_decay, args.v_decay, args.seed)
  rtc.DO_SHUFFLE_TEST_DATA = False # Do not shuffle the test dataset.
  # You need IS_SCALED_LDN only to meet the attribute check criteria in
  # tools/networks/lava_networks.py in the LavaLSNN class. NOTE: While obtaining
  # the Ap and Bp matrices in LavaLSNN, I have hard-coded to get the scaled ones
  rtc.IS_SCALED_LDN = True # Doesn't matter bcs Lava LDN is hardcoded quantized.
  rtc.IS_SLAYER_TRNG = False

  return rtc

def setup_logging(rtc, start_test_idx, end_test_idx,
                  log_file_name="loihi2_hw_inference"):
  """
  Sets up logging file to log inference info.

  Args:
    rtc <RTC>: RunTime Constants class.
    start_test_idx<int>: Test Index (including) onwards which inference is done.
    end_test_idx <int>: Test Index (excluding) uptil which inference is done.
    log_file_name <str>: Log file name.
  """
  log_path = rtc.OUTPUT_PATH + (
      "/per_sample_inference/l2hw_inference_start_idx_%s_end_idx_%s/"
      % (start_test_idx, end_test_idx))
  os.makedirs(log_path, exist_ok=True)
  files = os.listdir(log_path)
  for f in files:
    if f.startswith(log_file_name):
      os.rename(log_path + "/" + f, log_path + "/old_" + f)

  logger = log.configure_log_handler(
      "%s/%s_TS_%s.log"
      % (log_path, log_file_name, ExpUtils.get_timestamp())
  )
  keys = list(vars(rtc).keys())
  log.INFO("#"*20 + " L2HW      R T C     C O N F I G " + "#"*20)
  for key in keys:
    log.INFO("{0}: {1}".format(key, getattr(rtc, key)))
  log.INFO("#"*80)

  return logger

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, required=True)
  parser.add_argument("--start_test_idx", type=int, required=True)
  parser.add_argument("--order", type=int, required=True)
  parser.add_argument("--theta", type=float, required=True)
  parser.add_argument("--c_decay", type=float, required=True)
  parser.add_argument("--v_decay", type=float, required=True)
  parser.add_argument("--seed", type=int, required=True)

  parser.add_argument("--batch_size", type=int, default=1, required=False)
  parser.add_argument("--end_test_idx", type=int, default=0, required=False)

  args = parser.parse_args()
  end_test_idx = args.end_test_idx
  if end_test_idx == 0:
    end_test_idx = EXC.DATASETS_CFG[args.dataset]["tst_size"]

  RTC = setup_rtc(RTC, args)
  logger = setup_logging(RTC, args.start_test_idx, end_test_idx)

  for end_idx in range(
      args.start_test_idx+args.batch_size, end_test_idx+1, args.batch_size):
    if (end_idx-args.batch_size) > 0:
      logger.disabled = True
    L2H_lsnn = L2HwLavaLSNN(RTC, args.batch_size, end_idx-args.batch_size)
    logger.disabled = False
    L2H_lsnn.eval_lava_lsnn()
    log.INFO("Inference upto (not including) %s index done!" % end_idx)
