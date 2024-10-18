import numpy as np
import torch
from collections import namedtuple

SLAYERSNN_PARAMS = namedtuple("SLAYERSNN_PARAMS", "V_THR C_DECAY V_DECAY")

class EXC(object):
  DT = 1e-3
  NP_DTYPE = np.float32
  LEARNING_RATE = 1e-3 # For optimizer.
  SCALE_FACTOR = 4096 # For quantization of LDN.
  SHUFFLE_EPOCHS = 20 # For shuffling dataset.

  ##############################################################################
  # Parallelize tuning on all the following parameters.
  LDN_N_DIM_TO_ORDER_LIST = {
      1: [4, 6, 8, 10, 12, 14, 16, 24],
      }
  LDN_THETA_LIST = [110, 130, 150] # time-steps.

  # Signal to Spike ENCODING layer hyper-params.
  ENC_GAIN_LIST = [1] # Higher values of neuron gain could he helpful?
  ENC_V_THR_LIST = [1] # NOTE: set it Integer, otherwise Lava complains on LMT.
  ENC_BIAS_LIST = [0]

  # SLAYERSNN_PARAMS
  SLYR_V_THR_LIST = [1.0]
  SLYR_C_DECAY_LIST = [0.00, 0.10, 0.20]
  SLYR_V_DECAY_LIST = [0.00, 0.10, 0.20]
  #############################################################################
  ############ D A T A S E T    C O N F I G S #############
  ##############################################################################
  DATASETS_CFG = {
    "ECG5000": {
      "n_dim": 1,
      "n_ts": 140,
      "n_classes": 2, # Originally 5, but binarized for this project.
      "batch_size": 50,
      "trn_epochs": 100,
      "tst_size": 4500,
    },

    "Earthquakes": {
      "n_dim": 1,
      "n_ts": 512,
      "n_classes": 2,
      "batch_size": 23,
      "trn_epochs": 250,
      "tst_size": 139,
    },

    "Wafer": {
      "n_dim": 1,
      "n_ts": 152,
      "n_classes": 2,
      "batch_size": 50,
      "trn_epochs": 50,
      "tst_size": 6164,
    },

    "FordA": {
      "n_dim": 1,
      "n_ts": 500,
      "n_classes": 2,
      "batch_size": 40,
      "trn_epochs": 250,
      "tst_size": 1320,
    },

    "FordB": {
      "n_dim": 1,
      "n_ts": 500,
      "n_classes": 2,
      "batch_size": 18,
      "trn_epochs": 250,
      "tst_size": 810,
    },


    "Coffee": {
      "n_dim": 1,
      "n_ts": 286,
      "n_classes": 2,
      "batch_size": 28,
      "trn_epochs": 250,
      "tst_size": 28,
      },

    "ToeSegmentation1": {
      "n_dim": 1,
      "n_ts": 277,
      "n_classes": 2,
      "batch_size": 20,
      "trn_epochs": 250,
      "tst_size": 228,
      },

    "ToeSegmentation2": {
      "n_dim": 1,
      "n_ts": 343,
      "n_classes": 2,
      "batch_size": 18,
      "trn_epochs": 250,
      "tst_size": 130,
    },

    "Lightning2": {
      "n_dim": 1,
      "n_ts": 637,
      "n_classes": 2,
      "batch_size": 20,
      "trn_epochs": 250,
      "tst_size": 61,
    },

    "Worms": {
      "n_dim": 1,
      "n_ts": 900,
      "n_classes": 5,
      "batch_size": 25,
      "trn_epochs": 250,
      "tst_size": 77,
    },

    "WormsTwoClass": {
      "n_dim": 1,
      "n_ts": 900,
      "n_classes": 2,
      "batch_size": 25,
      "trn_epochs": 250,
      "tst_size": 77,
    },

    "OliveOil": {
      "n_dim": 1,
      "n_ts": 570,
      "n_classes": 4,
      "batch_size": 30,
      "trn_epochs": 250,
      "tst_size": 30,
    },

    "Meat": {
      "n_dim": 1,
      "n_ts": 448,
      "n_classes": 3,
      "batch_size": 20,
      "trn_epochs": 250,
      "tst_size": 60,
    },

    "RefrigerationDevices": {
      "n_dim": 1,
      "n_ts": 720,
      "n_classes": 3,
      "batch_size": 25,
      "trn_epochs": 250,
      "tst_size": 375,
    },

    "Computers": {
      "n_dim": 1,
      "n_ts": 720,
      "n_classes": 2,
      "batch_size": 25,
      "trn_epochs": 250,
      "tst_size": 250,
    }
  }
