"""Example script for training LFADS

Taken from https://github.com/snel-repo/autolfads-tf2/blob/main/example_scripts/train_lfads.py
"""

from lfads_tf2.utils import restrict_gpu_usage
restrict_gpu_usage(gpu_ix=1)
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from os import path

from lfads_tf2.models import LFADS
from lfads_tf2.utils import load_data, merge_chops, load_posterior_averages
from lfads_tf2.defaults import get_cfg_defaults

# create and train the LFADS model
cfg_path = path.join('config', 'lorenz.yaml') # switch to 'lorenz_underfit.yaml' and 'lorenz_overfit.yaml, DATA_DIR may require adjustment
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
# model_dir = path.realpath(path.expanduser(cfg.TRAIN.MODEL_DIR))
# train_output, valid_output = load_posterior_averages(model_dir)
# train_lfads_rates, *_ = train_output
# valid_lfads_rates, *_ = valid_output

# # define how to compute r2
# def compute_r2(truth, lfads_rates):
#     n_trials, seg_len, _ = truth.shape
#     truth_merged = merge_chops(truth, 0, n_trials * seg_len)
#     lfads_rates_merged = merge_chops(lfads_rates, 0, n_trials * seg_len)
#     r2 = r2_score(
#         truth_merged, 
#         lfads_rates_merged, 
#         multioutput='uniform_average')
#     return r2

# # compute R2 for training and validation data
# train_r2 = compute_r2(train_truth, train_lfads_rates)
# valid_r2 = compute_r2(valid_truth, valid_lfads_rates)
# print(f"Train R2: {train_r2}   Valid R2: {valid_r2}")

# # plot and pause to allow the model to be examined
# plt.plot(valid_lfads_rates[0,:,:])
# fig_path = path.join(model_dir, 'rates.png')
# plt.savefig(fig_path)