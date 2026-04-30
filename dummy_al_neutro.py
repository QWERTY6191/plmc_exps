import time
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import gpytorch as gp

from active_sampler import ActiveSampler, prod_func
from mogp_plmc import ProjectedGPModel
from test_bench import compute_metrics, evaluate_model

np.random.seed(seed=0)
torch.manual_seed(0)
dtype = torch.float64
torch.set_default_dtype(dtype)

# -------------------- Experiment setup ------------------
print_metrics = True
export_results = True
appendix = ''
run_key = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
run_key += appendix
path = os.path.join('results', 'al_neutro_{run_key}.csv')

# -------------------- Load ship data --------------------
root = 'data/neutro/'
exp_name = 'exp_neutro'
X_test = torch.load(root + 'train_x_sobol256.pt').to(dtype)
X = torch.load(root + 'test_x_lhs512.pt').to(dtype)
Y_test = torch.load(root + 'train_y_sobol256.pt').to(dtype)
Y = torch.load(root + 'test_y_lhs512.pt').to(dtype)
n_points, n_tasks = Y.shape

# -------------------- Split --------------------
n_init = 20
n_sample = 200
perm = np.random.permutation(len(X))
train_idx = perm[:n_init]
cand_idx = perm[n_init:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_cand, Y_cand = X[cand_idx], Y[cand_idx]

# -------------------- Model params --------------------
v = {'n_latents': 16, 'lik_rank': 0}
n_ind_points = None
noise_thresh = 1e-5
kernel_type = gp.kernels.MaternKernel
# -------------------- Active sampling --------------------

# Choose active learning model (can be any model defined in realdata_experiments)
active_model_name = 'PLMC_fast'  # change as needed

if active_model_name == 'PLMC':
    sampler_model = ProjectedGPModel(
        X_train, Y_train,
        n_latents=v['n_latents'],
        n_ind_points=n_ind_points,
        noise_thresh=noise_thresh,
        kernel_type=kernel_type,
        zero_M=False, diagonal_R=False, scalar_Sig_orth=False
    )
elif active_model_name == 'PLMC_fast':
    sampler_model = ProjectedGPModel(
        X_train, Y_train,
        n_latents=v['n_latents'],
        n_ind_points=n_ind_points,
        noise_thresh=noise_thresh,
        zero_M=True, diagonal_R=False, scalar_Sig_orth=True
    )
elif active_model_name == 'oilmm':
    sampler_model = ProjectedGPModel(
        X_train, Y_train,
        n_latents=v['n_latents'],
        n_ind_points=n_ind_points,
        noise_thresh=noise_thresh,
        zero_M=True, diagonal_R=True, scalar_Sig_orth=True, use_QR_decomp=False
    )
elif active_model_name == 'ICM':
    from mogp_icm import MultitaskGPModel
    sampler_model = MultitaskGPModel(
        X_train, Y_train,
        n_latents=v['n_latents'],
        lik_mat_rank=v['lik_rank'],
        n_ind_points=n_ind_points,
        noise_thresh=noise_thresh
    )
else:
    raise ValueError(f"Unsupported model {active_model_name}")

sampling_domain = 'latent' if active_model_name != 'ICM' else 'task'
# Initialise ActiveSampler with candidate set
sampler = ActiveSampler(
    model=sampler_model,
    strategy='variance',
    sampling_domain=sampling_domain,
    aggr_func=prod_func,
    current_data=Y_train,
    current_X=X_train,
    candidate_X=X_cand,
)
# -------------------- Training settings -----------------
n_iters = int(1e5)
lr_min = 1e-3
lr_max = 1e-1
use_stop = True
loss_thresh = 1e-3
patience_sched = 1000
patience_crit = 5 * patience_sched
training_settings = (lr_min, lr_max, n_iters, use_stop, loss_thresh, patience_crit, patience_sched)

# -------------------- Initial fit ---------------------------
last_loss, n_iters_eff = sampler.train_model(training_settings, gpu=True)
print("Stats after initial training step: last loss = {0}, n_iters = {1}".format(last_loss, n_iters_eff))

# --------------------- Sampling ----------------------------
sampling_duration = 0
start = time.time()
for i_iter in range(n_sample):
    with torch.no_grad():
        nxt, best_score = sampler.find_next_points(n_samples=1, only_idx=True, verbose=False)
        idx = int(nxt.squeeze())
        print("Id of selected point at iteration {0} : {1}".format(i_iter, idx))
        new_x = torch.as_tensor(X_cand[idx:idx+1])
        new_y = torch.as_tensor(Y_cand[idx:idx+1])
        sampler.modify_train_set(new_x, new_y, update_inducing_points=(active_model_name == 'var'))
    print("Best score at iteration {0} : {1}. Last loss : {2}".format(i_iter, best_score, last_loss))
sampling_duration = time.time() - start

# # -------------------- Final fit ---------------------------
n_iters_final_fit = int(1e5)
final_patience_crit = 5 * patience_sched
final_lr_max = 1e-3
final_training_settings = (lr_min, final_lr_max, n_iters_final_fit, use_stop, loss_thresh, final_patience_crit, patience_sched)

last_loss, n_iters_eff = sampler.train_model(final_training_settings, reset_optimizer=True)

# -------------------- Final evaluation --------------------
# Normalize test set using sampler's statistics
Y_test = (Y_test - sampler.train_mean) / sampler.train_std
n_test_points, n_tasks = Y_test.shape

results = {}
start = time.time()
pred_y, std_pred, negative_log_lik, Sigma_guess, condition_numbers = evaluate_model(model=sampler_model, X_test=X_test, Y_test=Y_test,
                                                                                    n_tasks=n_tasks, n_points=n_points, on_cpu=True,
                                                                                    has_many_tasks=True, has_many_points=False)
test_duration = time.time() - start
##------------------------------------------------------------------
## Computing, displaying and storing performance metrics
metrics = compute_metrics(y_test=Y_test, y_pred=pred_y, std_pred=std_pred, loss=last_loss, Sigma_guess=Sigma_guess,
                        n_iter=n_iters_eff, train_time=sampling_duration, pred_time=test_duration,
                        print_metrics=print_metrics, nll=negative_log_lik, kernel_cond=condition_numbers)
metrics['model'] = active_model_name

if export_results:
    results[active_model_name] = metrics
    pd.DataFrame.from_dict(results, orient='index').to_csv(path)
