from . import _init_paths

import datetime
import elephant as elp
import itertools
import matplotlib.pyplot as plt
import neo
import numpy as np
import os
import pickle
import pyspike as psp
import sys

from collections import defaultdict
from lava.utils.profiler import Profiler
from nengo.utils.filter_design import cont2discrete

from consts.exp_consts import EXC
from utils.base_utils import log

class ExpUtils(object):
  def __init__(self):
    pass

  @staticmethod
  def plot_spikes(spikes_matrix):
    fig, ax = plt.subplots(figsize=(14, 4))
    n_steps, n_neurons = spikes_matrix.shape
    timesteps = np.arange(n_steps)
    for i in range(n_neurons):
      for t in timesteps[np.where(spikes_matrix[:, i] != 0)]:
        ax.plot([t, t], [i+0.5, i+1.5])

    ax.set_ylim(0.5, n_neurons+0.5)
    ax.set_yticks(list(range(1, n_neurons+1, int(np.ceil(n_neurons/25)))))
    ax.set_xticks(list(range(0, n_steps)))
    ax.set_ylabel("Neuron Index")
    ax.set_xlabel("Time in $ms$")

  @staticmethod
  def get_distance_metric_between_1D_spike_trains(spk_train_1, spk_train_2):
    """
    Both spk_train_1 and spk_train_2 should be numpy objects, not torch objects.
    """
    assert spk_train_1.size == spk_train_2.size
    # Convert spike trains to spike_times, starting with time-step 1.
    ts_of_spks_1 = np.where(spk_train_1==1)[0] # Spike time-steps.
    ts_of_spks_2 = np.where(spk_train_2==1)[0] # Spike time-steps.
    np.put(spk_train_1, ts_of_spks_1, ts_of_spks_1+1) # Set the spike timesteps.
    np.put(spk_train_2, ts_of_spks_2, ts_of_spks_2+1) # Set the spike timesteps.
    ret = {}

    st_1 = psp.SpikeTrain(spk_train_1, edges=[1, spk_train_1.size])
    st_2 = psp.SpikeTrain(spk_train_2, edges=[1, spk_train_2.size])
    # Compute ISI distance.
    ret["isi_dist"] = psp.isi_distance(st_1, st_2)
    # Compute Spike Distance.
    ret["spk_dist"] = psp.spike_distance(st_1, st_2)
    # Compute Spike Synchronization.
    ret["spk_sync"] = psp.spike_sync(st_1, st_2)

    st_1 = neo.SpikeTrain(spk_train_1, units="ms", t_stop=spk_train_1.size)
    st_2 = neo.SpikeTrain(spk_train_2, units="ms", t_stop=spk_train_2.size)
    # Compute Victor-Purpura disance.
    ret["vpr_dist"] = elp.spike_train_dissimilarity.victor_purpura_distance(
        [st_1, st_2])[0, 1] # By default, the `cost_factor` here is 1 Hz.
    # Compute van-Rossum distance.
    ret["vrm_dist"] = elp.spike_train_dissimilarity.van_rossum_distance(
        [st_1, st_2])[0, 1]

    return ret

  @staticmethod
  def get_dist_metric_bwn_ldn_spk_trains(order, ldn_spktrns_1, ldn_spktrns_2):
    assert (ldn_spktrns_1.shape == ldn_spktrns_2.shape and
            ldn_spktrns_1.shape[0] == 2*order)
    ret = {}
    all_dms =  defaultdict(list) # All the distance metrics.
    for i in range(2*order): # Number of neurons = 2*ORDER (Two Neuron Encoder).
      st_dist = ExpUtils.get_distance_metric_between_1D_spike_trains(
          ldn_spktrns_1[i], ldn_spktrns_2[i])
      for dm, val in st_dist.items():
        all_dms[dm].append(val)

    for dm, vals in all_dms.items():
      assert len(vals) == 2*order
      ret[dm] = (np.mean(vals), np.std(vals))

    return ret

  @staticmethod
  def spike_func(x): # Heaviside Step Function.
    spikes = np.zeros_like(x)
    spikes[x>0] = 1.0
    return spikes

  @staticmethod
  def reset_voltage(v, spikes):
    mask = spikes > 0
    v[mask] = 0 # Hard Reset.
    return v

  @staticmethod
  def update_voltage(volt, current):
    volt += current
    rctfy_mask = volt < 0 # Rectify negative voltage to zero.
    volt[rctfy_mask] = 0

    return volt

  @staticmethod
  def get_Ap_Bp_matrices(order: int, theta: float, is_c2d: bool, tau=None):
    assert is_c2d # Hard check here to ensure only cont2discrete vals are used.
    Q = np.arange(order, dtype=EXC.NP_DTYPE)
    R = (2*Q + 1)[:, None]/theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i<j, -1, (-1.0)**(i-j+1)) * R
    B = (-1.0)**Q[:, None]*R

    if is_c2d:
      log.INFO("Approximating A-prime and B-prime matrices via cont2discrete.")
      C = np.ones((1, order), dtype=EXC.NP_DTYPE)
      D = np.zeros((1,), dtype=EXC.NP_DTYPE)
      Ap, Bp, _, _, _ = cont2discrete(
          (A, B, C, D), dt=EXC.DT, method="zoh")
    else:
      log.INFO("Computing A-prime and B-prime matrices for Spiking LDN.")
      Ap = A*tau + np.eye(order)
      Bp = B*tau

    return Ap, Bp

  @staticmethod
  def get_scaled_Ap_Bp_matrices(scale_factor: int,
                                order: int, theta: float, is_c2d: bool):
    assert scale_factor is not None and scale_factor > 0
    Ap, Bp = ExpUtils.get_Ap_Bp_matrices(order=order, theta=theta, is_c2d=is_c2d)
    log.INFO("Computing the scaled and quantized Ap and Bp matrices...")
    scaled_Ap = np.round(scale_factor * Ap).astype(np.int32)
    scaled_Bp = np.round(scale_factor * Bp).astype(np.int32)

    return scaled_Ap, scaled_Bp

  @staticmethod
  def get_duplicating_transform_matrix(n_dim, order, n_neurons):
    t_mat = np.zeros((n_dim*order, n_neurons), dtype=np.int32)
    for i in range(n_dim*order):
      t_mat[i, 2*i] = 1
      t_mat[i, 2*i+1] = 1

    return t_mat

  @staticmethod
  def get_encoders(num_neurons):
    enc = np.empty(num_neurons, dtype=np.int32)
    for i in range(num_neurons):
      enc[i] = -1 if i%2 else 1

    assert enc.sum() == 0
    return enc

  @staticmethod
  def get_valid_thetas(sig_n_ts):
    ret = []
    for theta in EXC.LDN_THETA_LIST:
      if theta <= sig_n_ts:
        ret.append(theta)

    return ret

  @staticmethod
  def get_all_hyperparams_combinations(dcfg):
    all_lists = [
      EXC.LDN_N_DIM_TO_ORDER_LIST[dcfg["n_dim"]], # LDN Orders.
      ExpUtils.get_valid_thetas(dcfg["n_ts"]), # LDN Thetas.
      EXC.SLYR_C_DECAY_LIST, # Slayer network current decays.
      EXC.SLYR_V_DECAY_LIST, # Slayer network voltage decays.
    ]

    all_combs = itertools.product(*all_lists)
    return all_combs

  @staticmethod
  def get_timestamp():
    now = datetime.datetime.now()
    now = "%s" % now
    now = "T".join(now.split())
    return "-".join(now.split(":"))

  @staticmethod
  def if_file_exists_then_remove_it(file_path):
    if os.path.isfile(file_path):
      log.INFO("File: %s exists, now removing it..." % file_path)
      os.remove(file_path)

  @staticmethod
  def if_file_not_exists_then_save_it(file, file_path):
    if os.path.isfile(file_path):
      sys.exit("File: %s already exists, why saving again?" % file_path)
    log.INFO("File: %s does not exists, saving it..." % file_path)
    pickle.dump(file, open(file_path, "wb"))

  @staticmethod
  def load_file(file_path):
    return pickle.load(open(file_path, "rb"))

  @staticmethod
  def check_file_and_remove_and_save(file, file_path):
    ExpUtils.if_file_exists_then_remove_it(file_path)
    ExpUtils.if_file_not_exists_then_save_it(file, file_path)
###############################################################################
