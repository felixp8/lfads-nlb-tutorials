"""Example script for training LFADS

Taken from https://github.com/snel-repo/autolfads-tf2/blob/main/example_scripts/train_lfads.py
"""

from lfads_tf2.utils import restrict_gpu_usage
restrict_gpu_usage(gpu_ix=1)
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from os import path
import os

os.chdir('../') # unnecessary step so models can be loaded from notebooks

from lfads_tf2.models import LFADS
from lfads_tf2.utils import load_data, merge_chops, load_posterior_averages
from lfads_tf2.defaults import get_cfg_defaults

# create and train the LFADS model
cfg_path = path.join('models/config', 'lorenz.yaml') # switch to 'lorenz_underfit.yaml' and 'lorenz_overfit.yaml, DATA_DIR may require adjustment
model = LFADS(cfg_path=cfg_path)
model.train()

# Read config to load data for evalution
cfg = get_cfg_defaults()
cfg.merge_from_file(cfg_path)
cfg.freeze()

# Load the spikes and the true rates
train_truth, valid_truth = load_data(
    cfg.TRAIN.DATA.DIR, 
    prefix=cfg.TRAIN.DATA.PREFIX, 
    signal='truth')[0]

# perform posterior sampling, then merge the chopped segments
model.sample_and_average()