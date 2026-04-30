import time
import warnings
import psutil

import numpy as np
import torch
import gpytorch as gp

from utilities import CorrectReduceLROnPlateau

def compute_metrics(y_test, y_pred, std_pred, loss, Sigma_guess, n_iter, train_time, pred_time, nll, kernel_cond, print_metrics=True):
    delta = y_test - y_pred
    errs_abs = torch.abs(delta).squeeze()
    alpha_CI = torch.mean((errs_abs < 2 * std_pred).float())
    err2 = errs_abs ** 2
    R2_list = 1 - torch.mean(err2, dim=0) / torch.var(y_test, dim=0)
    PVA_list = torch.log(torch.mean(err2 / std_pred ** 2, dim=0))
    noise_full = Sigma_guess.diag().mean()  if len(Sigma_guess.shape) > 1 else Sigma_guess.mean() # mean of the diagonal coefficients
    n_tasks = y_test.shape[-1]

    errs_abs = errs_abs.cpu().numpy()
    metrics = {}
    metrics['n_iter'] = n_iter
    metrics['train_time'] = train_time
    metrics['pred_time'] = pred_time
    metrics['loss'] = loss
    metrics['R2'] = R2_list.mean().cpu().numpy()
    metrics['RMSE'] = torch.sqrt(err2.mean()).cpu().numpy()
    metrics['mean_err_abs'], metrics['max_err_abs'] = errs_abs.mean(), errs_abs.max()
    metrics['mean_err_quant05'], metrics['mean_err_quant95'], metrics['mean_err_quant99'] = np.quantile(errs_abs, np.array([0.05, 0.95, 0.99]))
    metrics['mean_sigma'] = std_pred.mean().cpu().numpy()
    metrics['PVA'] = PVA_list.mean().cpu().numpy()
    metrics['NLL'] = nll.mean().cpu().numpy() / n_tasks
    metrics['alpha_CI'] = alpha_CI.mean().cpu().numpy()  # proportion of points within 2 sigma (should be around 0.95)
    metrics['noise'] = noise_full.cpu().numpy()
    metrics['max_ker_cond'] = kernel_cond.max().cpu().numpy()
    if print_metrics:
        for key, value in metrics.items():
            print(key, value)
    return metrics


def train_and_eval_model(model, X, Y, X_test, Y_test, lr_min, lr_max, gpu, n_iters, use_stop,
                         loss_thresh, patience_crit, patience_sched, print_loss=True, freq_print=1000, preds_on_cpu=True, mll=None):
    """
    model : model to train and evaluate
    X : train inputs
    Y : train outputs
    X_test : test inputs
    Y_test : test outputs
    lr_min : learning rate at the end of scheduling
    lr_max : learning rate at the beginning of scheduling
    gpu : whether to use the gpu for training
    n_iters : iteration budget (maximal number of iterations)
    use_stop : whether to use the stopping criterion
    loss_thresh : if the loss doesn't improve by more than loss_thresh in patience_sched iterations, the learning rate is halved.
    | If the stopping criterion is used, if the loss doesn't improve by more than loss_thresh in patience_crit iterations, training is stopped
    run_key : id of the run for storing results
    results : dictionary for storing results
    """
    n_points = len(X)
    n_tasks = Y.shape[-1] if len(Y.shape) > 1 else 1 
    has_many_tasks = (n_tasks > 1000)
    has_many_points = (n_points > 1000)

    print('Num training points: {0}, num tasks: {1}'.format(n_points, n_tasks))
    if gpu:
        X = X.cuda()
        Y = Y.cuda()
        model = model.cuda()

    model.train()
    if mll is None:
        mll = model.default_mll()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max)
    scheduler = CorrectReduceLROnPlateau(optimizer, factor=0.5, patience=patience_sched,
                            threshold=loss_thresh, threshold_mode='rel', mode='min', min_lr=lr_min)

    ##------------------------------------------------------------------
    ## Training model

    start = time.time()
    best_loss = 1e9
    last_lr = 1.
    no_improve_count = 0
    effective_n_iters = n_iters
    for i in range(n_iters):
        optimizer.zero_grad()
        with gp.settings.cholesky_max_tries(8):
            output_train = model(X)
            loss = -mll(output_train, Y).squeeze()
            loss.backward()
            optimizer.step()
            new_loss = loss.item() / n_tasks

        if print_loss and i%freq_print==0:
            print("Loss at iteration {0} : {1}".format(i, new_loss))

        scheduler.step(new_loss)
        current_lr = optimizer.param_groups[0]['lr']
        # If the learning rate has changed, we display it, and restart the "no improvement" counter
        if current_lr < last_lr:
            print('LR changed to {0} at iteration {1}'.format(current_lr, i))
            last_lr = current_lr
            no_improve_count = 0

        loss_better_than_best = (new_loss >= 0. and new_loss < best_loss * (1 - loss_thresh)) \
                                or (new_loss < 0. and new_loss < best_loss * (1 + loss_thresh))
        loss_almost_as_good_as_best = (new_loss >= 0. and new_loss < best_loss * (1 + loss_thresh)) \
                                or (new_loss < 0. and new_loss < best_loss * (1 - loss_thresh))

        if loss_better_than_best :
            best_loss = new_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        last_loss = new_loss
        if use_stop and (no_improve_count > patience_crit) and loss_almost_as_good_as_best:
            effective_n_iters = i
            break

    train_duration = time.time() - start

    ##------------------------------------------------------------------
    ## Making predictions

    model.eval()
    if gpu:
        if preds_on_cpu:
            model = model.cpu() # Predictions on the CPU for avoiding memory overflows
        else:
            X_test = X_test.cuda()
            Y_test = Y_test.cuda()

    print("\n Making predictions... \n")
    start = time.time()

    pred_y, std_pred, negative_log_lik, Sigma_guess, condition_numbers = evaluate_model(model=model, X_test=X_test, Y_test=Y_test, n_tasks=n_tasks, n_points=n_points,
                                                                                        has_many_tasks=has_many_tasks, has_many_points=has_many_points)

    pred_time = time.time() - start
    ##------------------------------------------------------------------
    ## Computing, displaying and storing performance metrics
    with torch.no_grad():
        metrics = compute_metrics(y_test=Y_test, y_pred=pred_y, std_pred=std_pred, loss=last_loss, Sigma_guess=Sigma_guess,
                                n_iter=effective_n_iters, train_time=train_duration, pred_time=pred_time, print_metrics=True,
                                nll=negative_log_lik, kernel_cond=condition_numbers)
    return metrics, model


def evaluate_model(model, X_test, Y_test, n_tasks, n_points, has_many_tasks=False, has_many_points=False, on_cpu=True):
    with torch.no_grad(),\
        gp.settings.eval_cg_tolerance(1e-2),\
        gp.settings.cholesky_max_tries(8):

        if hasattr(model, 'full_likelihood'):  # we have to compute the full likelihood of projected models
            full_likelihood = model.full_likelihood(diag=has_many_tasks)
        else:
            full_likelihood = model.likelihood

        ## Predictions computation
        if has_many_tasks:
            # Predictions are made in batches
            free_mem = psutil.virtual_memory().available if on_cpu else torch.cuda.mem_get_info()[0]
            num_bytes = X_test.element_size()
            batch_size = int(free_mem / (16 * n_points * n_tasks * num_bytes))
            preds, variances = [], [] 
            for i in range(0, len(X_test), batch_size):
                x_batch = X_test[i:i+batch_size]
                observed_pred = full_likelihood(model(x_batch))
                pred_y = observed_pred.mean
                preds.append(pred_y)
                var_pred = observed_pred.variance
                variances.append(var_pred)
            pred_y = torch.cat(preds)
            var_pred = torch.cat(variances)
        else:
            observed_pred = full_likelihood(model(X_test))
            pred_y = observed_pred.mean
            var_pred = observed_pred.variance
        std_pred = var_pred.sqrt().squeeze()

        ## Negative log likelihood computation
        if has_many_tasks:
            warnings.warn("Negative log-likelihood will not be computed to avoid memory overflows")
            negative_log_lik = torch.ones(n_tasks)
        else:
            negative_log_lik =  gp.metrics.negative_log_predictive_density(pred_dist=observed_pred, test_y=Y_test)

        ## Noise covariance computation
        if hasattr(full_likelihood, 'task_noise_covar'):
            Sigma_guess = full_likelihood.task_noise_covar
        elif hasattr(full_likelihood, 'task_noises'):
            Sigma_guess = full_likelihood.task_noises
        else:
            Sigma_guess = torch.ones(n_tasks) * full_likelihood.noise

        ## Conditioning number computation
        if has_many_points:
            warnings.warn("Conditioning numbers will not be computed to avoid memory overflows")
            condition_numbers = torch.ones(n_tasks)
        else:
            condition_numbers = model.kernel_cond()

    return pred_y, std_pred, negative_log_lik, Sigma_guess, condition_numbers 
