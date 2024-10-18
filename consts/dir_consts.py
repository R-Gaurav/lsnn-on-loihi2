# Place this repo under $HOME/Documents/Projects/.
USER_DIR = "/home/rgaurav/Documents/Projects/" # Local Machine.
#USER_DIR = "/homes/rgaurav/projects/" # INRC VM.

# Experiment outputs/logs will be automatically created in the exp_outputs/ dir.
OUTPUT_DIR = USER_DIR+"/exp_outputs/"
# Place the downloaded datasets under $HOME/Documents/Projects/all_datasets/.
DATA_DIR = USER_DIR+"/all_datasets/"

class DRC(object):
  def __init__(self, dataset):
    """
    Initializes the dataset paths.

    Args:
      dataset <str>: The dataset to work with e.g. ECG5000
    """
    self.dataset = dataset
    self.data_path = DATA_DIR + "/" + dataset + "/"
    self.train_set = dataset + "_TRAIN.ts"
    self.test_set = dataset + "_TEST.ts"
