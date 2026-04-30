import os

import numpy as np
import torch
import gpytorch as gp
from scipy.io import loadmat
import pandas as pd

np.random.seed(seed=0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

from mogp_icm import MultitaskGPModel
from mogp_var import VariationalMultitaskGPModel
from mogp_plmc import ProjectedGPModel

from test_bench import train_and_eval_model

## Data preprocessing
root = 'data/SARCOS/'
exp_name = 'exp_sarcos'
train_data = loadmat(root + 'sarcos_inv.mat')['sarcos_inv'].astype(np.float64)
test_data = loadmat(root + 'sarcos_inv_test.mat')['sarcos_inv_test'].astype(np.float64)

## Data formatting for model training
X, Y = train_data[:, :21], train_data[:, 21:]
X_test, Y_test = test_data[:, :21], test_data[:, 21:]
Mean, Std = Y.mean(axis=0), Y.std(axis=0)
Y, Y_test = (Y - Mean) / Std, (Y_test - Mean) / Std
n_points, n_tasks = Y.shape
X, Y, X_test, Y_test = torch.as_tensor(X), torch.as_tensor(Y), torch.as_tensor(X_test), torch.as_tensor(Y_test)

## Model parameters
v_default = {
    'n_latents': n_tasks, 
    'lik_rank': 0,
    'noise_thresh': 1e-3,
    'void' : [0.]}
v_vals = {
    'n_latents' : range(1, n_tasks+1), 
    'lik_rank' : [0, n_tasks],
    'noise_thresh': np.logspace(-5, -3, 3),
    'void' : [0.]}
v_test = 'void'
v_test_2 = 'void'
n_ind_points = 500
if n_ind_points is not None:
    train_ind_rat = n_points / n_ind_points
else:
    train_ind_rat = 1. # In this case VariationalMultitaskGPModel fixes inducing points locations at the true inputs
kernel_type = gp.kernels.MaternKernel
mean_type = gp.means.ZeroMean
ker_kwargs = {}

## Training settings
gpu = True
lr_min = 1e-3
lr_max = 1e-0
n_iters = 1#00000
use_stop = True
loss_thresh = 1e-3
patience_sched = 100
patience_crit = patience_sched * 10


## Experiment
PLMC_models_to_run = ['PLMC', 'PLMC_fast', 'oilmm']
other_models_to_run = ['var']
results = {}

## Results export
export_results = True
appendix = '' # to further customize experiment model_name
all_models = PLMC_models_to_run + other_models_to_run
if len(all_models) == 1:
    appendix += all_models[0]
if n_ind_points is not None:
    appendix += '{0}ind'.format(n_ind_points)
results_dir = 'results'
path = os.path.join(results_dir, '_'.join([exp_name, appendix, v_test, v_test_2, '.csv']))
print("Results path: ", path + '\n')

for model_name in other_models_to_run:
    for i_v, vval in enumerate(v_vals[v_test]):
        for i_v2, vval2 in enumerate(v_vals[v_test_2]):
            v = v_default.copy()
            v[v_test] = vval
            v[v_test_2] = vval2
            n_latents, lik_rank, noise_thresh = v['n_latents'], v['lik_rank'], v['noise_thresh']
            run_key = '_'.join([model_name, v_test, v_test_2, str(i_v), str(i_v2)])
            print('\n', run_key, '\n')

            if model_name == 'ICM':
                model = MultitaskGPModel(X, Y, n_latents=n_latents, mean_type=mean_type, kernel_type=kernel_type, ker_kwargs=ker_kwargs,
                                                lik_mat_rank=lik_rank, noise_thresh=noise_thresh, n_inducing_points=n_ind_points)
                mll = None

            if model_name == 'var':
                model = VariationalMultitaskGPModel(X, Y, n_latents=n_latents, mean_type=mean_type, kernel_type=kernel_type, ker_kwargs=ker_kwargs,
                                    noise_thresh=noise_thresh, lik_mat_rank=lik_rank, train_ind_ratio=train_ind_rat)
                mll = gp.mlls.PredictiveLogLikelihood(likelihood=model.likelihood, model=model, num_data=n_points)
                
            metrics, model = train_and_eval_model(model=model,
                                                X=X, Y=Y, X_test=X_test, Y_test=Y_test,
                                                lr_min=lr_min, lr_max=lr_max, loss_thresh=loss_thresh,
                                                patience_sched=patience_sched, patience_crit=patience_crit,
                                                use_stop=use_stop, gpu=gpu, n_iters=n_iters, mll=mll,
                                                )
            metrics.update(v)
            metrics['model'] = model_name
            results[run_key] = metrics
            if export_results:
                df = pd.DataFrame.from_dict(results, orient='index')
                df.to_csv(path)

# lik_rank is not a parameter for PLMC models
if v_test == 'lik_rank':
    v_test = 'void'
if v_test_2 == 'lik_rank':
    v_test_2 = 'void'
for model_name in PLMC_models_to_run:
    for i_v, vval in enumerate(v_vals[v_test]):
        for i_v2, vval2 in enumerate(v_vals[v_test_2]):
            v = v_default.copy()
            v[v_test] = vval
            v[v_test_2] = vval2
            n_latents, lik_rank, noise_thresh = v['n_latents'], v['lik_rank'], v['noise_thresh']
            run_key = '_'.join([model_name, v_test, v_test_2, str(i_v), str(i_v2)])
            print('\n', run_key, '\n')

            if model_name == 'PLMC':
                model = ProjectedGPModel(X, Y, n_latents=n_latents, mean_type=mean_type,  kernel_type=kernel_type, ker_kwargs=ker_kwargs,
                                                n_inducing_points=n_ind_points, noise_thresh=noise_thresh, zero_M=False, diagonal_R=False, scalar_Sig_orth=False)

            if model_name == 'PLMC_fast':
                model = ProjectedGPModel(X, Y, n_latents=n_latents, mean_type=mean_type, kernel_type=kernel_type, ker_kwargs=ker_kwargs, 
                                                n_inducing_points=n_ind_points, noise_thresh=noise_thresh, zero_M=True, diagonal_R=False, scalar_Sig_orth=True)
                
            if model_name == 'oilmm':
                model = ProjectedGPModel(X, Y, n_latents=n_latents, mean_type=mean_type, kernel_type=kernel_type, ker_kwargs=ker_kwargs,
                                                n_inducing_points=n_ind_points, noise_thresh=noise_thresh, zero_M=True, diagonal_R=True, scalar_Sig_orth=True, use_QR_decomp=False)

            metrics, model = train_and_eval_model(model=model,
                                                X=X, Y=Y, X_test=X_test, Y_test=Y_test,
                                                lr_min=lr_min, lr_max=lr_max, loss_thresh=loss_thresh,
                                                patience_sched=patience_sched, patience_crit=patience_crit,
                                                use_stop=use_stop, gpu=gpu, n_iters=n_iters,
                                                )
            metrics.update(v)
            metrics['model'] = model_name
            results[run_key] = metrics
            if export_results:
                df = pd.DataFrame.from_dict(results, orient='index')
                df.to_csv(path)
