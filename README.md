# LSNN-on-Loihi2

Paper Title: [**Legendre-SNN on Loihi-2: Evaluation and
Insights**](https://openreview.net/pdf?id=wUUvWjdE0K)

Accepted at: [NeurIPS 2024 Workshop](https://blog.neurips.cc/2024/08/02/announcing-the-neurips-2024-workshops/) - [ML with New Compute Paradigms (MLNCP)](https://www.mlwithnewcompute.com/)


## Description:

In this work, I have implemented my designed _reservoir_-based univariate- Time
Series Classification (TSC) spiking network called [Legendre Spiking Neural
Network (Legendre-SNN)](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2023.1148284/full) (a.k.a. -- LSNN) on Intel's Loihi-2 neuromorphic chip using [Lava](https://lava-nc.org/).

### More details:

The _reservoir_ in the Legendre-SNN is _non-spiking_, in fact, it does _not_
constitute any kind of neuron, and is rather implemented with basic matrix
operations. The reservoir used is the **Legendre Delay Network (LDN)** [[1](https://proceedings.neurips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf), [2](https://ieeexplore.ieee.org/abstract/document/8294070)] --
an example of a _State-Space Model_.

Intel's Loihi-2 chip has _two_ types of On-chip computational resources: the embedded
**Lakemont (LMT)** x86 low-power processors (6 in total) and the **Neurocores** (128 in total). The LMT
processors support only INT32-bit operations, i.e., _no_ floating point
operations, and the Neurocores support deployment of spiking neurons/networks.
On one hand, where extensive technical-documentation to program Neurocores (via
Lava) is available, on the other, technical resources to program LMT cores are
severely limited (as of this writing); this often prompts the researchers to implement their
_non-spiking_ operations on _less_ power-efficient CPUs (than LMTs).

Herein, in a first, I have implemented a _quantized_ State-Space Model (i.e.,
a _quantized_ version of LDN) on the LMT cores -- this implementation adds to
the _scarce_ technical-documentation to program LMTs, and can be used as a
reference by other interested researchers to deploy their _non-spiking_ operations (in an SNN) on
the Loihi-2 chip; please refer Appendix F in my paper above. Note that the
_spiking_ network following the LDN (in Legendre-SNN) is implemented on the Neurocores; thus,
my abovementioned paper presents a detailed pipeline / technical specifications to
_implement_ and _profile_ SNNs (with _spiking_ and _non-spiking_ components) on the Loihi-2 chip.

#### References:
<sub>
[1]: Voelker, Aaron, Ivana Kajić, and Chris Eliasmith. "Legendre memory units: Continuous-time representation in recurrent neural networks." Advances in neural information processing systems 32 (2019).
</sub>
<br/>
<sub>
[2]: Voelker, Aaron R., and Chris Eliasmith. "Improving spiking dynamical networks: Accurate delays, higher-order synapses, and time cells." Neural computation 30.3 (2018): 569-609.
</sub>

## Steps to use this repository:

To deploy the code in this repository, you need a CPU/GPU for training and
evaluating the Legendre-SNN, as well as access to the Loihi-2 boards (on [INRC
cloud](https://intel-ncl.atlassian.net/wiki/spaces/INRC/overview)) to evaluate 
the Legendre-SNN on _physical_ Loihi-2 chips.

### Directory setup

Set up the `USER_DIR` macro in `consts/dir_consts.py` to point to the directory where you have downloaded this repo, followed by downloading the desired datasets from the [Time Series Classification website](https://timeseriesclassification.com/dataset.php) and placing them under the `all_datasets/` parent directory (you will have to create `all_datasets/` directory under `USER_DIR`). Next, make sure that the `all_datasets/` directory is correctly referred via the macro `DATA_DIR` in `consts/dir_consts.py`. Note that the experiment outputs will be created/logged in the `exp_outputs/` directory (referred by `OUTPUT_DIR` in `consts/dir_consts.py`), which will be automatically created.

### Environment setup

The code in this repository is executed in a Python (3.10.2) environment with the following libraries installed. NOTE: To install `lava-loihi` library and subsequently execute Legendre-SNN on a _physical_ Loihi-2 chip, you need to have access to the [INRC Cloud](https://intel-ncl.atlassian.net/wiki/spaces/INRC/overview).

* For training and evaluation on a GPU (and evaluation on Loihi-2's _simulation_ on a CPU):
```
aeon==0.7.1
elephant==1.0.0
lava-dl==0.5.0
lava-nc==0.9.0
nengo==4.0.0
nengo-extras==0.5.0
neo==0.13.0
pandas==2.0.3
pyspike==0.8.0
ray==2.12.0
scikit-learn==1.4.1
scipy==1.10.1
torch==2.0.0
torchvision==0.15.1
```

* For evaluation on a _physical_ Loihi-2 chip on INRC VM (apart from the abovementioned libraries):
```
lava-loihi==0.6.0
```

### Experiment setup

The tunable hyperparameters (LDN's $d$ & $\theta$, and CUBA neurons' $\tau_{cur}$ & $\tau_{vol}$) over which the experiments for different datasets can be run are mentioned in the file `consts/exp_consts.py`. Feel free to modify them to experiment on a smaller subset of their values. Note that I have used the `ray` library to parallelize experiments over all the hyperparameter combinations, i.e., one can choose to execute -- say $N$ number of experiments in _parallel_ (on a GPU) for $N$ combinations of all $4$ hyperparameters.

### Commands to run

* To train/evaluate on a GPU (and evaluate on Loihi-2's _simulation_ on a CPU):

  `python execute_experiment.py --dataset ECG5000 --is_scaled_ldn 1 --seed 0 --is_trnbl_nrn_params 0 --gpu_per_trial 0.125` where,

    - `--dataset` accepts the name of a dataset (datasets' metadata can be found in `consts/exp_consts.py` in `DATASETS_CFG`).
    - `--is_scaled_ldn` accepts either $0$ or $1$, where $0$ and $1$ denote the usage of _continuous_ and _quantized_ valued LDN respectively.
    - `--seed` accepts a seed value
    - `--is_trnbl_nrn_params` accepts either $0$ or $1$, where $0$ denotes _no_ training of neuron parameters (of the spiking network following the LDN) and $1$ denotes training them as well (apart from the weights).
    - `--gpu_per_trial` accepts a non-negative floating point value $\leq 1$ denoting the number ($N$) of parallel experiments to run (pertaining to the hyperparameter combinations), where $N$=1/`gpu_per_trial`. Note that $N$ is also capped by the number of CPUs available in your system (my code caps $N$ to $8$ in the line `ray.init(num_cpus=8)` in `execute_experiment.py` file); for more details, look into [ray.tune.run](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.run.html#ray.tune.run) API.

Once the individual experiments complete, look into the `exp_outputs/` directory; there, you can navigate all the way to the end of a directory pertaining to a hyperparameter combination and check the newly created log file for test accuracies. Within the log file (starting with `experiment_TS_*.log`), you will find a line `INFO - Accuracy on Loihi Simulation:` that carries the test accuracy obtained on Loihi-2's _simulation_ on CPU. Upon inspection of the parent directory of the log file, you will also find other files: `trained_network.net` and `trained_network.pt`. To evaluate the trained LSNN on a _physical_ Loihi-2 chip, you need to transfer these files (i.e., `trained_network.*` files) to your INRC VM; therefore, transfer the entirety of the `exp_outputs/` directory (and place it) under the `USER_DIR` on your INRC VM. You can also transfer `trained_network.*` files of your selected hyperparameter combination directories to your INRC VM, however, make sure to maintain the directory structure from the `exp_outputs/` directory onwards. Note that you may also have to accordingly update the `USER_DIR` (in `consts/dir_consts.py`) to reflect your INRC VM's directory structure.

* To evaluate trained Legendre-SNN (LSNN) on a _physical_ Loihi-2 chip:

  `SLURM=1 BOARD=ncl-ext-og-02 python tools/l2hw_lava_lsnn.py --dataset=ECG5000 --start_test_idx 0 --end_test_idx 20 --order 10 --theta 0.11 --c_decay 0.0 --v_decay 0.0 --seed 0` where,

    - `SLURM=1` is the INRC VM's shell-environment setting to execute Lava programs.
    - `BOARD` accepts the Loihi-2 board on which you want to run your inference on.
    - `--dataset` accepts the name of the dataset you want to run your inference for.
    - `--start_test_idx` accepts the index of the test sample you want to start your inference from.
    - `--end_test_idx` accept the index of the test sample up to which you want to run your inference.
    - `--order` accepts the hyperparameter $d$ of the LDN.
    - `--theta` accepts the hyperparameter $\theta$ of the LDN.
    - `--c_decay` accepts the hyperparameter $\tau_{cur}$ of the CUBA neurons.
    - `--v_decay` accepts the hyperparameter $\tau_{vol}$ of the CUBA neurons.
    - `--seed` accepts the seed value.

NOTE: Make sure that the directory path (and files) corresponding to the `--dataset`, `--order`, `--theta`, `--c_decay`, `--v_decay`, and `--seed`  is valid and present in the `exp_outputs/` on your INRC VM. Once the inference experiment (for the chosen `--start_test_idx` and `--end_test_idx`) completes, you will see a log file named `loihi2_hw_inference_TS_*.log` created within the (newly created) directory `l2hw_inference_start_idx_*_end_idx_*/` under the (newly created) `per_sample_inference/` in the directory path corresponding to the passed hyperparameters. In the `loihi2_hw_inference_TS_*.log` file, you will find the lines `Ground Truth Labels:` and `Predicted Labels:` that carry the ground truth and predicted labels respectively for each inferred test sample (i.e., the inference process on _physical_ Loihi-2 chip is per-sample). You can then parse these lines and obtain the ground truth and predicted labels for all of the test samples within the chosen range. Needless to say, to compute the Loihi-2 hardware inference accuracy for the chosen dataset and the hyperparameters (and seed), you need to obtain ground truth and predicted labels for all the samples in the test set of the dataset.
