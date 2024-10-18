# This file executes training and evaluation on CPU/GPU, as well as evaluation
# on Loihi-2's simulation backend on the CPU. For execution on physical Loihi-2
# look into the file `tools/l2hw_lava_lsnn.py`.

from consts.runtime_consts import RTC
from consts.exp_consts import EXC, SLAYERSNN_PARAMS
from consts.dir_consts import OUTPUT_DIR
from utils.base_utils import log
from utils.base_utils.exp_utils import ExpUtils
from tools.train_eval_slayer_lsnn import TrEvSlayerLSNN
from tools.networks.lava_networks import LavaLSNN

from functools import partial
from ray import tune

import argparse
import os
import ray
import numpy as np
import random
import torch

def setup_logging(rtc, log_file_name="experiment"):
  """
  Sets up logging file to log experiment info.

  Args:
    log_file_name <str>: Log file name.
  """
  files = os.listdir(rtc.OUTPUT_PATH)
  for f in files:
    if f.startswith(log_file_name):
      os.rename(rtc.OUTPUT_PATH + "/" + f, rtc.OUTPUT_PATH + "/old_" + f)

  log.configure_log_handler(
      "%s/%s_TS_%s.log"
      % (rtc.OUTPUT_PATH, log_file_name, ExpUtils.get_timestamp())
  )
  keys = list(vars(rtc).keys())
  log.INFO("#"*20 + " E X P     R T C     C O N F I G " + "#"*20)
  for key in keys:
    log.INFO("{0}: {1}".format(key, getattr(rtc, key)))
  log.INFO("#"*80)

def setup_output_path(rtc, exp_type):
  """
  Sets up the output path to store log file and intermediate experiment data.

  Args:
    rtc <RTC>: Runtime Constants Class.
    exp_type <str>: "CONTINUOUS_VALUED_LDN" or "SCALED_QUANTIZED_LDN"
  """
  output_path = (
    "%s/%s/%s/" % (OUTPUT_DIR, exp_type, rtc.DATASET) +
    "ldn_order_%s_theta_%s/" % (rtc.ORDER, rtc.THETA) +
    "slayer_c_decay_%s_v_decay_%s/"
    % (rtc.SLAYERSNN_PARAMS.C_DECAY, rtc.SLAYERSNN_PARAMS.V_DECAY) +
    "SEED_%s/" % rtc.SEED
  )

  os.makedirs(output_path, exist_ok=True)
  return output_path

def setup_rtc(rtc, config, args):
  rtc.SEED = args.seed
  rtc.DATASET = args.dataset
  rtc.DCFG = EXC.DATASETS_CFG[args.dataset]
  rtc.EPOCHS = EXC.DATASETS_CFG[args.dataset]["trn_epochs"]
  rtc.DO_SHUFFLE_TEST_DATA = True # Not needed, but shuffle test data anyways.

  rtc.SLAYERSNN_PARAMS = SLAYERSNN_PARAMS
  rtc.SLAYERSNN_PARAMS.V_THR = EXC.SLYR_V_THR_LIST[0] # Only one SLYR_V_THR.
  rtc.SLAYERSNN_PARAMS.C_DECAY = config["SLYR_C_DECAY"]
  rtc.SLAYERSNN_PARAMS.V_DECAY = config["SLYR_V_DECAY"]
  rtc.IS_TRNBL_NRN_PARAMS = True if args.is_trnbl_nrn_params == 1 else False
  rtc.ORDER = config["ORDER"]
  rtc.THETA = config["THETA"]/1000 # Time-steps to seconds (1 time-step = 1 ms).

  rtc.ENC_V_THR = EXC.ENC_V_THR_LIST[0] # Only one ENC_V_THR.
  rtc.GAIN = EXC.ENC_GAIN_LIST[0] # Only one ENC_GAIN.
  rtc.BIAS = EXC.ENC_BIAS_LIST[0] # Only one ENC_BIAS.

  return rtc

def run_experiment(rtc):
  # Train Slayer LSNN
  rtc.IS_SLAYER_TRNG = True
  tel_lsnn = TrEvSlayerLSNN(rtc)
  log.INFO("Starting Training and Evaluating SlayerLSNN...")
  tel_lsnn.train_eval_slayer_lsnn()

  # Evaluate Lava LSNN on CPU.
  rtc.BACKEND = "L2Sim"
  # RESET_INTERVAL can be a number other than power of 2 for BACKEND = "L2Sim".
  rtc.RESET_INTERVAL = rtc.DCFG["n_ts"]
  lava_lsnn = LavaLSNN(rtc, n_test_sigs=rtc.DCFG["tst_size"], start_test_idx=0)
  log.INFO("Evaluating the LavaLSNN on CPU...")
  lava_lsnn.eval_lava_lsnn()

  log.INFO("Experiment done, now removing pickle files...")
  ExpUtils.if_file_exists_then_remove_it(
      rtc.OUTPUT_PATH + "/training_X_ldn_stvecs.p")
  ExpUtils.if_file_exists_then_remove_it(rtc.OUTPUT_PATH + "training_Y.p")
  ExpUtils.if_file_exists_then_remove_it(
      rtc.OUTPUT_PATH + "test_X_ldn_stvecs.p")
  ExpUtils.if_file_exists_then_remove_it(rtc.OUTPUT_PATH + "test_Y.p")

def execute_one_hyperparam_combination(config, rtc, args):
  rtc.IS_SCALED_LDN = True if args.is_scaled_ldn == 1 else False
  rtc.DEVICE = torch.device("cuda" if args.gpu_per_trial > 0 else "cpu")
  exp_type = (
      "SCALED_QUANTIZED_LDN" if rtc.IS_SCALED_LDN else "CONTINUOUS_VALUED_LDN"
  )

  rtc = setup_rtc(rtc, config, args)
  output_path = setup_output_path(rtc, exp_type)
  rtc.OUTPUT_PATH = output_path

  setup_logging(rtc)
  run_experiment(rtc)
  log.RESET()

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, required=True)
  parser.add_argument(
      "--is_scaled_ldn", type=int, choices=[0, 1], required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--is_trnbl_nrn_params", type=int, choices=[0, 1],
                      default=0, required=False)
  parser.add_argument("--gpu_per_trial", type=float, default=0, required=False)

  args = parser.parse_args()
  if args.gpu_per_trial == 0:
    ray.init(num_cpus=
             int(os.environ.get("SLURM_CPUS_PER_TASK")) *
             int(os.environ.get("SLURM_NTASKS"))
    )
  elif args.gpu_per_trial > 0:
    ray.init(num_cpus=8) # NOTE: Set 8 parallel processes on GPU systems.

  config = {
    "ORDER": tune.grid_search(
        EXC.LDN_N_DIM_TO_ORDER_LIST[EXC.DATASETS_CFG[args.dataset]["n_dim"]]
    ),
    "THETA": tune.grid_search(
      ExpUtils.get_valid_thetas(EXC.DATASETS_CFG[args.dataset]["n_ts"])
    ),
    "SLYR_C_DECAY": tune.grid_search(EXC.SLYR_C_DECAY_LIST),
    "SLYR_V_DECAY": tune.grid_search(EXC.SLYR_V_DECAY_LIST),
  }

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.use_deterministic_algorithms(True)

  tune_analysis = tune.run(
      partial(execute_one_hyperparam_combination, rtc=RTC, args=args),
      config=config,
      resources_per_trial={
          "cpu": 1 if args.gpu_per_trial > 0 else int(
              os.environ.get("SLURM_CPUS_PER_TASK")
          ),
          "gpu": args.gpu_per_trial
      },
      verbose=1,
  )
  print("Dataset: {0} and SEED: {1} Experiment DONE!!!".format(
        args.dataset, args.seed))
