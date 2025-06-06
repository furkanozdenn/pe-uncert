"""
Test script for crispAI and crispAI-pi models
    - test_data_path contains .csv file with test data required fields 
    - 
"""
import os 
import sys
import time
import warnings
import sys

from sklearn.preprocessing import PowerTransformer
from sklearn.isotonic import IsotonicRegression
from CnnCrispr_final.otscore import calcCnnCrisprScore
from CFD.otscore import calcCfdScore
from MIT.otscore import calcMitScore

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

import pdb
from tqdm import tqdm

from model import CrispAI, CrispAI_pi, ModelConfig, two_armed_ZINB_multitask_physical_features, CrispAI_pi_ZIP, CrispAI_pi_Poisson, CrispAI_pi_NB
from utils import preprocess_features
from loss_functions import MyZeroInflatedNegativeBinomialLoss
from openpyxl import load_workbook

def get_MIT_score_df(df_string1, df_string2):
    # get MIT score for each pair of target and off-target sequences
    mit_scores = []
    for x,y in zip(df_string1, df_string2):
        mit_scores.append(calcMitScore(x, y))
    return np.asarray(mit_scores)

def get_CFD_score_df(df_string1, df_string2):
    # get CFD score for each pair of target and off-target sequences
    cfd_scores = []
    for x,y in zip(df_string1, df_string2):
        cfd_scores.append(calcCfdScore(x, y))
    return np.asarray(cfd_scores)

# test parameters
changeseq_test_data_path = '/home/furkan/dphil/crispr_offtarget/crispAI/data/changeseq/changeseq_offtarget_data_flank73_filtered_nupop_gc_bdm_preprocessed_test.csv'
test_data_path = changeseq_test_data_path
checkpoint_path = '/home/furkan/dphil/crispr_offtarget/crispAI/checkpoints/crispAI-pi_conv[256, 64]_lstm256_dense[256, 64, 32]/epoch:19-best_valid_loss:0.270.pt'

# system parameters
device = 'cuda:2'
warnings.filterwarnings('ignore')

# load test data
df_test = pd.read_csv(test_data_path)
df_test = preprocess_features(df = df_test,
                            reads = 'CHANGEseq_reads',
                            target = 'target',
                            offtarget_sequence= 'offtarget_sequence',
                            distance= 'distance',
                            read_cutoff= 10,
                            max_reads = 1e4,
                            nupop_occupancy_col= 'NuPoP occupancy',
                            nupop_affinity_col= 'NuPoP affinity',
                            gc_content_col= 'GC flank73',
                            nucleotide_bdm_col= 'nucleotide BDM')
# get 100 rows for debugging
# df_test = df_test.iloc[:10000]


X = np.stack([x.astype(np.float32) for x in df_test['interface_encoding'].values], axis=0)
X_pi = np.stack([x.astype(np.float32) for x in df_test['physical_features'].values], axis=0)
y = np.stack([x.astype(np.float32) for x in df_test['CHANGEseq_reads_adjusted'].values], axis=0)
# get Target_N as string not float32
X_target_n = np.stack([x for x in df_test['target_N'].values], axis=0)
X_strand = np.stack([x for x in df_test['strand'].values], axis=0)
X_chromstart = np.stack([x for x in df_test['chromStart'].values], axis=0)
X_chromend = np.stack([x for x in df_test['chromEnd'].values], axis=0)
X_chrom = np.stack([x for x in df_test['chrom'].values], axis=0)
X_offtarget_sequence = np.stack([x for x in df_test['offtarget_sequence'].values], axis=0)


# concat X and X_pi if pi_features
X = np.concatenate([X, X_pi], axis=2)

# test loader 
test_dataset = TensorDataset(torch.tensor(X), torch.tensor(y))

# load config and model from checkpoint
checkpoint = torch.load(checkpoint_path)
config = checkpoint['config']
model = CrispAI_pi(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()



preds = []
pbar = tqdm(DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4), desc='predicting test set')
with torch.no_grad():
    for x, _ in pbar:
        x = x.to(device)
        samples = model.draw_samples(x).T
        preds.append(samples)

# concat 128 dim arrays in preds to get 1d array
preds = np.concatenate(preds, axis=0)
preds_mean = np.mean(preds, axis=1)



pt = PowerTransformer(method='box-cox', standardize=True) # maps to normal distribution with zero mean and unit variance

changeseqtest_preds_scores = pt.fit_transform(preds_mean.reshape(-1,1)+ model.eps).reshape(-1)
changeseqtest_preds_median_scores = pt.fit_transform(np.median(preds, axis=1).reshape(-1,1)+ model.eps).reshape(-1)

y_test = pt.fit_transform(y.reshape(-1,1)+ model.eps).reshape(-1)



df_test['crispAI'] = preds_mean


"""
Diagnostics plot with calibration model 
    - Accurate uncertainties for deep learning using calibrated regression
      (Kuleshov et al., 2018)
"""

# positive portion of the test dataset
X_test_pos = X[y > 0]
y_test_pos = y[y > 0]

# target_N, strand, chromStart, chromEnd, chrom, off_target_sequence
X_test_pos_target_n = X_target_n[y > 0]
X_test_pos_strand = X_strand[y > 0]
X_test_pos_chromstart = X_chromstart[y > 0]
X_test_pos_chromend = X_chromend[y > 0]
X_test_pos_chrom = X_chrom[y > 0]
X_test_pos_offtarget_sequence = X_offtarget_sequence[y > 0]

y_test_pos_posterior = model.draw_samples(torch.tensor(X_test_pos).to(device), n_samples=2000).T

predicted_cdf = np.mean(y_test_pos_posterior <= y_test_pos.reshape(-1,1), axis=1)
empirical_cdf = np.zeros(len(predicted_cdf))
for i, p in enumerate(predicted_cdf):
    empirical_cdf[i] = np.sum(predicted_cdf <= p) / len(predicted_cdf)


'''
Train baseline uncertainty models on the same training data
'''

# changeseq_test_data_path = '/home/furkan/dphil/crispr_offtarget/crispAI/data/changeseq/changeseq_offtarget_data_flank73_filtered_nupop_gc_bdm_preprocessed_test.csv'
changeseq_train_data_path = '/home/furkan/dphil/crispr_offtarget/crispAI/data/changeseq/changeseq_offtarget_data_flank73_filtered_nupop_gc_bdm_preprocessed_train.csv'

df_train = pd.read_csv(changeseq_train_data_path)
df_train = preprocess_features(df = df_train,
                            reads = 'CHANGEseq_reads',
                            target = 'target',
                            offtarget_sequence= 'offtarget_sequence',
                            distance= 'distance',
                            read_cutoff= 10,
                            max_reads = 1e4,
                            nupop_occupancy_col= 'NuPoP occupancy',
                            nupop_affinity_col= 'NuPoP affinity',
                            gc_content_col= 'GC flank73',
                            nucleotide_bdm_col= 'nucleotide BDM')

# X = np.stack([x.astype(np.float32) for x in df_test['interface_encoding'].values], axis=0)
# X_pi = np.stack([x.astype(np.float32) for x in df_test['physical_features'].values], axis=0)
# y = np.stack([x.astype(np.float32) for x in df_test['CHANGEseq_reads_adjusted'].values], axis=0)

X_train = np.stack([x.astype(np.float32) for x in df_train['interface_encoding'].values], axis=0)
X_pi_train = np.stack([x.astype(np.float32) for x in df_train['physical_features'].values], axis=0)
y_train = np.stack([x.astype(np.float32) for x in df_train['CHANGEseq_reads_adjusted'].values], axis=0)

# quantile regression
from sklearn.ensemble import RandomForestRegressor

# Create RandomForestRegressor
n_trees = 1000
mpg_forest = RandomForestRegressor(n_estimators=n_trees, random_state=42)
X_train_r = X_train.reshape(X_train.shape[0], -1)
X_train_r_pos = X_train_r[y_train > 0]
y_train_pos = y_train[y_train > 0]

# mpg_forest.fit(X_train_r_pos[:50000], y_train_pos[:50000]) # TODO: increase to 120k?
mpg_forest.fit(X_train_r_pos[:5000], y_train_pos[:5000])
X_test = np.stack([x.astype(np.float32) for x in df_test['interface_encoding'].values], axis=0)
y_test = np.stack([x.astype(np.float32) for x in df_test['CHANGEseq_reads_adjusted'].values], axis=0)

X_test_r = X_test.reshape(X_test.shape[0], -1)
X_test_r_pos = X_test_r[y_test > 0]
y_test_pos = y_test[y_test > 0]

X_test_pos_random_forest_all_preds = np.stack([tree.predict(X_test_r_pos) for tree in mpg_forest.estimators_], axis=0)
X_test_pos_random_forest_all_preds = X_test_pos_random_forest_all_preds.T

from sklearn.linear_model import QuantileRegressor

quantiles = np.arange(0.005, 1.00, 0.005)
quantile_models = []

for q in tqdm(quantiles):
    q_model = QuantileRegressor(quantile=q)
    # random 1000 indices from X_train_r_pos
    time1 = time.time()
    idx = np.random.choice(X_train_r_pos.shape[0], 10, replace=False) # TODO: increase to 1000 takes too long
    # fit model on random 1000 indices
    q_model.fit(X_train_r_pos[idx], y_train_pos[idx])
    time2 = time.time()
    # model.fit(X_train_r_pos[:5000], y_train_pos[:5000])
    quantile_models.append(q_model)


# get predictions for each quantile model
X_test_pos_quantile_preds = np.stack([q_model.predict(X_test_r_pos) for q_model in quantile_models], axis=0).T


# calibration model 
calibration_model = IsotonicRegression()
calibration_model.fit(empirical_cdf, predicted_cdf)

# diagnostic plot
y_test_pos_posterior_means = np.mean(y_test_pos_posterior, axis=1)
conf_level_lower_bounds = np.arange(start=0.025, stop=0.5, step=0.025)
conf_levels = 1-2*conf_level_lower_bounds
unc_pcts = []
cal_pcts = []
cnncrispr_pcts = []
cfd_pcts = []
mit_pcts = []
random_forest_pcts = []
quantile_pcts = []
zip_pcts = []
poisson_pcts = []
nb_pcts = []



# Linearly scale each predicted distribution to [0,1]
for i in tqdm(range(y_test_pos_posterior.shape[0])):
    #y_test_pos_posterior[i,:] = pt_preds.transform(y_test_pos_posterior[i,:].reshape(-1,1) + model.eps).reshape(-1)
    break

# y_test_pos_posterior = y_test_pos_posterior / np.max(y)
# y_test_pos = y[y > 0] / np.max(y)

y_test_pos_posterior = y_test_pos_posterior / 10000
y_test_pos = y[y > 0] / 10000

X_test_pos_random_forest_all_preds = X_test_pos_random_forest_all_preds / 10000

X_test_pos_quantile_preds = X_test_pos_quantile_preds / 10000



for cl_lower in conf_level_lower_bounds:
    quants = [cl_lower, 1-cl_lower]
    new_quantiles = calibration_model.transform(quants)
    # replace Nan values with 0 in new_quantiles
    new_quantiles[np.isnan(new_quantiles)] = 0
    
    cal_lower, cal_upper = np.quantile(y_test_pos_posterior, new_quantiles, axis=1)
    unc_lower, unc_upper = np.quantile(y_test_pos_posterior, quants, axis=1)
    unc_lower_random_forest, unc_upper_random_forest = np.quantile(X_test_pos_random_forest_all_preds, quants, axis=1)
    unc_lower_quantile, unc_upper_quantile = np.quantile(X_test_pos_quantile_preds, quants, axis=1)

    

    perc_within_unc = np.mean((y_test_pos <= unc_upper)&(y_test_pos >= unc_lower))
    perc_within_random_forest = np.mean((y_test_pos <= unc_upper_random_forest)&(y_test_pos >= unc_lower_random_forest))
    perc_within_quantile = np.mean((y_test_pos <= unc_upper_quantile)&(y_test_pos >= unc_lower_quantile))
    perc_within_cal = np.mean((y_test_pos <= cal_upper)&(y_test_pos >= cal_lower))


    unc_pcts.append(perc_within_unc)
    random_forest_pcts.append(perc_within_random_forest)
    quantile_pcts.append(perc_within_quantile)
    cal_pcts.append(perc_within_cal)

    print(f'cl_lower: {cl_lower}')

sns.set_style("ticks")

plt.clf()
plt.close()

fig, ax = plt.subplots(constrained_layout=True)
# pdb.set_trace()
sns.lineplot([0, 1], [0, 1], color="grey", ax=ax, linestyle='dashed', label='Perfect Scorer')
# sns.lineplot(conf_levels, unc_pcts, color="purple", label="CRISPAI", ax=ax, linestyle='solid')
sns.lineplot(conf_levels, cal_pcts, color="green", label="crispAI", ax=ax, linestyle='solid')
sns.lineplot(conf_levels, random_forest_pcts, color="blue", label="Random Forest", ax=ax, linestyle='solid')
sns.lineplot(conf_levels, quantile_pcts, color="red", label="Quantile Regression", ax=ax, linestyle='solid')


ax.set_xlabel("Predicted Confidence Level", fontsize=14)
ax.set_ylabel("Observed Confidence Level", fontsize=14)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# x and y ticks fontsize
ax.tick_params(axis='both', which='major', labelsize=12)


# Remove 0.0 at the bottom of y-axis and left of x-axis
ax.spines["left"].set_bounds(0, 1)
ax.spines["bottom"].set_bounds(0, 1)


# remove x-ticks and y-ticks at 0 
ax.tick_params(axis="x", which="both", bottom=False, top=False)
ax.tick_params(axis="y", which="both", left=False, right=False)

# despine
sns.despine()

# add legend 
# ax.legend()
# to top left
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('../test_results/diagnostics_plot_extended_last.png')
#pdf
plt.savefig('../test_results/diagnostics_plot_extended_last.pdf')

pdb.set_trace()