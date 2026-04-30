import os
from datetime import datetime

import numpy as np
import torch
import gpytorch as gp
import pandas as pd
from scipy.interpolate import interp1d

np.random.seed(seed=0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

from mogp_icm import MultitaskGPModel
from mogp_var import VariationalMultitaskGPModel
from mogp_plmc import ProjectedGPModel

from test_bench import train_and_eval_model

def detrend_data(x, y, degree=1):
    coef = np.polyfit(x, y, degree)
    return y - np.polyval(coef, x)

root = 'data/bramblemet/'
exp_name = 'exp_bramblemet'
degree = 2 # degree of the polynomial detrending
ndiv = 2 # subsampling factor
start_date = '2020-06-01'
end_date = '2020-06-16'
start_date = datetime.strptime(start_date, '%Y-%m-%d')
end_date = datetime.strptime(end_date, '%Y-%m-%d')
num_days = (end_date - start_date).days

## Data preprocessing
dico = {}
stations = ['bramblemet', 'cambermet', 'chimet', 'sotonmet']
for station in stations:
    df = pd.read_csv(os.path.join(root,'{0}.csv.gz'.format(station)), compression='gzip', low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M')
    df = df.loc[(df['Date'] >= start_date) & (df['Date'] < end_date)]
    df['time_num'] = df['Date'].apply(lambda x: x.timestamp())
    values = df['DEPTH'].values
    if 'time_num' not in dico:  # create a reference time vector with values between 0 and 1
        ref_time = df['time_num'].values
        ref_time_norm = ref_time / ref_time.max()
        ref_time_norm = ref_time_norm - ref_time_norm[0]
        dico['time_num'] = ref_time_norm
        dico['Date'] = df['Date'].values
    else:
        f = interp1d(df['time_num'].values, values) # align all time series on the same reference time vector
        values = f(ref_time)
    dico[station] = detrend_data(ref_time_norm, values, degree=degree)
df = pd.DataFrame(dico).set_index('Date').astype(np.float64)

## Data formatting for model use
df = df.iloc[::ndiv] # subsampling the time series by a factor ndiv
X, Y = df['time_num'].values[:,None], df.drop('time_num', axis=1).values
X = X * 1e5 # scaling X up for numerics precision
test_indices = np.arange(len(df)//2, len(df)//2 + 2 * len(df)//num_days) # test set is two days in the middle of the time series
X, X_test = np.delete(X, test_indices, axis=0), X[test_indices]
Y, Y_test = np.delete(Y, test_indices, axis=0), Y[test_indices]
Mean, Std = Y.mean(axis=0), Y.std(axis=0)
Y, Y_test = (Y - Mean) / Std, (Y_test - Mean) / Std
X, Y, X_test, Y_test = torch.as_tensor(X), torch.as_tensor(Y), torch.as_tensor(X_test), torch.as_tensor(Y_test)
n_points, n_tasks = Y.shape

## Model parameters
v_default = {
    'n_latents': 2, 
    'lik_rank': n_tasks,
    'noise_thresh': 1e-3,
    'n_mix':5,
    'void' : [0.]}
v_vals = {
    'n_latents' : range(1, n_tasks+1), 
    'lik_rank' : [0, n_tasks],
    'n_mix': range(4,9),
    'noise_thresh': np.logspace(-4, -2, 3),
    'void' : [0.]}
v_test = 'void'
v_test_2 = 'void'
n_ind_points = None
if n_ind_points is not None:
    train_ind_rat = n_points / n_ind_points
else:
    train_ind_rat = 1. # In this case VariationalMultitaskGPModel fixes inducing points locations at the true inputs
kernel_type = gp.kernels.SpectralMixtureKernel
mean_type = gp.means.ZeroMean

## Training settings
gpu = True
lr_min = 1e-3
lr_max = 1e-0
n_iters = 20000
use_stop = True
loss_thresh = 1e-3
patience_sched = 200
patience_crit = patience_sched * 5

## Experiment
PLMC_models_to_run = ['PLMC', 'PLMC_fast', 'oilmm']
other_models_to_run = ['var', 'ICM']
results = {}
min_err = 1e6

## Results export
export_results = True
appendix = '' # to further customize experiment model_name
all_models = PLMC_models_to_run + other_models_to_run
if len(all_models) == 1:
    appendix += all_models[0]
if n_ind_points is not None:
    appendix += '_{0}ind'.format(n_ind_points)
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
            ker_kwargs = {'num_mixtures':v['n_mix']}
            run_key = '_'.join([model_name, v_test, v_test_2, str(i_v), str(i_v2)])
            print('\n', run_key, '\n')

            if model_name == 'ICM':
                model = MultitaskGPModel(X, Y, n_latents=n_latents, mean_type=mean_type, kernel_type=kernel_type, ker_kwargs=ker_kwargs,
                                                lik_mat_rank=lik_rank, noise_thresh=noise_thresh, n_inducing_points=n_ind_points)

            if model_name == 'var':
                model = VariationalMultitaskGPModel(X, Y, n_latents=n_latents, mean_type=mean_type, kernel_type=kernel_type, ker_kwargs=ker_kwargs,
                                    noise_thresh=noise_thresh, lik_mat_rank=lik_rank, train_ind_ratio=train_ind_rat)
            
            attribute_string = 'covar_module'
            if model_name=='ICM':
                attribute_string += '.data_covar_module'
            if n_ind_points is not None and model_name!='var':
                attribute_string += '.base_kernel'
            attributes = attribute_string.split('.')
            obj = model
            for attr in attributes:
                obj = getattr(obj, attr)
            obj.initialize_from_data(X, Y)  # Spectral Mixture Kernel has to be carefully initialized

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
                df_results = pd.DataFrame.from_dict(results, orient='index')
                df_results.to_csv(path)
                if metrics['RMSE'] < min_err:
                    best_model = model
                    best_key = run_key
                    min_err = metrics['RMSE']

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

            attribute_string = 'covar_module'
            if n_ind_points is not None and model_name!='var':
                attribute_string += '.base_kernel'
            attributes = attribute_string.split('.')
            obj = model
            for attr in attributes:
                obj = getattr(obj, attr)
            obj.initialize_from_data(X, Y)  # Spectral Mixture Kernel has to be carefully initialized

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
                df_results = pd.DataFrame.from_dict(results, orient='index')
                df_results.to_csv(path)
                if metrics['RMSE'] < min_err:
                    best_model = model
                    best_key = run_key
                    min_err = metrics['RMSE']

## Illustrating predictions
if export_results:
    df[['pred'+str(i) for i in range(n_tasks)]] = np.zeros((len(df), n_tasks))
    df[['lower'+str(i) for i in range(n_tasks)]] = np.zeros((len(df), n_tasks))
    df[['upper'+str(i) for i in range(n_tasks)]] = np.zeros((len(df), n_tasks))
    best_model.eval()
    full_likelihood = best_model.full_likelihood() if hasattr(best_model, 'full_likelihood') else best_model.likelihood
    observed_pred = full_likelihood(best_model(X_test))
    pred_y = observed_pred.mean
    lower, upper = observed_pred.confidence_region()
    test_indices_pd = df.index[test_indices]
    df['is_test_idx'] = np.zeros(len(df)).astype(bool)
    df.loc[test_indices_pd, 'is_test_idx'] = np.ones(len(test_indices)).astype(bool)
    for i in range(n_tasks):
        df.loc[test_indices_pd, 'pred'+str(i)] = pred_y[:,i].detach().cpu().numpy()
        df.loc[test_indices_pd, 'lower'+str(i)] = lower[:,i].detach().cpu().numpy()
        df.loc[test_indices_pd, 'upper'+str(i)] = upper[:,i].detach().cpu().numpy()
        df.to_csv(path.replace('exp', 'preds'))
