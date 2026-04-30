import numpy as np
import torch
import gpytorch as gp
from scipy.stats import qmc
from scipy.spatial.distance import cdist
from utilities import CorrectReduceLROnPlateau

## Utilitaires

def prod_func(x, dim=None):
    return torch.sum(torch.log(x), dim=dim)

def max_func(x, dim=None):
    return torch.max(x, dim=dim).values

def sum_func(x, dim=None):
    return torch.sum(x, dim=dim)

class ActiveSampler:
    def __init__(self, model, strategy, sampling_domain, aggr_func, current_data=None, current_X=None, candidate_X=None, **kwargs):
        self.model = model
        self.strategy = strategy
        self.sampling_domain = sampling_domain
        self.aggr_func = aggr_func
        self.current_X = current_X
        
        # Persistent mask: True means "available", False means "already visited"
        self.mask = None 

        if candidate_X is not None:
            self.X_candidates = candidate_X
            self.mask = torch.ones(len(self.X_candidates), dtype=torch.bool)
        elif self.strategy == 'downsampling' and self.current_X is not None:
            self.mask = torch.ones(len(self.current_X), dtype=torch.bool)

        if current_data is not None:
            mean = current_data.mean(dim=0)
            std = torch.std(current_data, dim=0)
            self.current_data = (current_data - mean) / std
            self.train_mean = mean
            self.train_std = std
            self.model.set_train_data(self.current_X, self.current_data, strict=False)

        self.optimizer = None
        self.scheduler = None

    def gen_candidate_set(self, n_points, dim, algo='sobol', seed=0):
        if algo == 'sobol':
            sampler = qmc.Sobol(d=dim, seed=seed)
            m = int(np.ceil(np.log2(n_points)))
            points = 2 * sampler.random_base2(m=m) - 1
        elif algo == 'LHS':
            sampler = qmc.LatinHypercube(d=dim, seed=seed)
            points = 2 * sampler.random(n=n_points) - 1
        else:
            points = 2 * np.random.rand(n_points, dim) - 1
            
        self.X_candidates = torch.as_tensor(points, dtype=torch.get_default_dtype())
        self.mask = torch.ones(len(self.X_candidates), dtype=torch.bool)
        return points


    def add_data(self, X, Y):
        self.current_X = torch.cat([self.current_X, X])

        # De-normalize existing data if statistics exist
        if hasattr(self, 'train_mean') and self.train_mean is not None:
            existing = self.current_data * self.train_std + self.train_mean
        else:
            existing = self.current_data

        combined = torch.cat([existing, Y])
        mean = combined.mean(dim=0)
        std = torch.std(combined, dim=0)
        self.current_data = (combined - mean) / std
        self.train_mean = mean
        self.train_std = std


    def modify_train_set(self, new_X=None, new_Y=None, update_inducing_points=False):
        if new_X is not None and new_Y is not None:
            self.add_data(new_X, new_Y)
            if self.strategy == "downsampling":
                new_entries = torch.ones(len(new_X), dtype=torch.bool, device=self.mask.device)
                self.mask = torch.cat([self.mask, new_entries])

        self.model.set_train_data(self.current_X, self.current_data, strict=False)

        if update_inducing_points:
            self.model.variational_strategy.base_variational_strategy.inducing_points = self.current_X
            var_dist_class = self.model.variational_strategy.base_variational_strategy._variational_distribution.__class__
            n_points = len(self.current_X)
            variational_dist = var_dist_class(n_points, batch_shape=torch.Size([self.model.n_latents]))
            self.model.variational_strategy.base_variational_strategy._variational_distribution = variational_dist


    def find_next_points(self, n_samples, output_train=None, only_idx=False, verbose=True, **kwargs):
        # 1. Compute Scores for ALL points
        if self.strategy == 'downsampling':
            var_loo, delta_loo = self.model.compute_loo(output_train, latent=(self.sampling_domain == 'latent'))
            scores = self.aggr_func(torch.abs(delta_loo), dim=1)
            eval_points = self.current_X

        else:
            if self.sampling_domain == 'latent':
                var_values = self.model.latent_variance(self.X_candidates)
            else:
                var_values = self.model.task_variance(self.X_candidates)

            if self.strategy == 'variance':
                scores = self.aggr_func(var_values, dim=-1)
            elif self.strategy == 'loo':
                var_loo, delta_loo = self.model.compute_loo(output_train, latent=(self.sampling_domain == 'latent'))
                scores = self.compute_loo_scores(var_values=var_values, lscales_mat=self.model.lscales(),
                e_loo2=delta_loo**2, s_loo2=var_loo)
            else:
                raise ValueError('Strategy not recognized : {}'.format(self.strategy))
            
            eval_points = self.X_candidates

        # 2. APPLY THE MASK
        # We fill "False" (visited) positions with a value that won't be picked by topk
        # Downsampling: we want smallest (use +inf), Discovery: we want largest (use -inf)
        fill_value = float('inf') if self.strategy == 'downsampling' else float('-inf')
        
        # Ensure mask is on same device as scores
        masked_scores = scores.clone().detach().masked_fill_(~self.mask.to(scores.device), fill_value)

        # 3. Retrieve Top K
        # largest=False for downsampling (remove least useful), True for active learning (pick most uncertain)
        top_values, top_ind = torch.topk(masked_scores, n_samples, largest=(not self.strategy == 'downsampling'))

        # 4. Update the persistent mask (Mark as visited)
        self.mask[top_ind] = False

        if verbose:
            print(f'Score at current iteration: {top_values[0].item():.4f}')

        new_points = top_ind if only_idx else eval_points[top_ind]
        
        # Match your original return format (handling singleton dims)
        if n_samples == 1 and not only_idx:
            new_points = new_points.unsqueeze(0) if torch.is_tensor(new_points) else np.expand_dims(new_points, 0)

        return new_points, top_values[0].item()


    def train_model(self, training_settings, gpu=True, params_to_train=None, reset_optimizer=False, freq_print=1000):
        """Train the sampler's model using the exact stopping logic from realdata_experiments.
        training_settings: (lr_min, lr_max, n_iters, use_stop, loss_thresh, patience_crit, patience_sched)
        Returns (best_loss, effective_iters).
        """
        lr_min, lr_max, n_iters, use_stop, loss_thresh, patience_crit, patience_sched = training_settings
        # Setup model and self.optimizer
        if gpu:
            self.model = self.model.cuda()
            self.current_X = self.current_X.cuda()
            self.current_data = self.current_data.cuda()

        self.model.train()
        mll = self.model.default_mll()
        if params_to_train is not None:
            self.optimizer = torch.optim.AdamW(params_to_train, lr=lr_max)
        elif self.optimizer is None or reset_optimizer:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr_max)
        if self.scheduler is None or reset_optimizer:
            self.scheduler = CorrectReduceLROnPlateau(self.optimizer, factor=0.5, patience=patience_sched,
                                            threshold=loss_thresh, threshold_mode='rel', mode='min', min_lr=lr_min)
        best_loss = 1e9
        last_lr = 1.
        no_improve_count = 0
        effective_iters = n_iters
        n_tasks = self.current_data.shape[-1]
        for i in range(n_iters):
            self.optimizer.zero_grad()
            with gp.settings.cholesky_max_tries(8):
                output = self.model(self.current_X)
                loss = -mll(output, self.current_data).squeeze()
                loss.backward()
                self.optimizer.step()
                new_loss = loss.item() / n_tasks

            self.scheduler.step(new_loss)
            if i % freq_print == 0:
                print("Loss at iter {0} : {1}".format(i, new_loss))

            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr < last_lr:
                print('LR changed to {0}'.format(current_lr))
                last_lr = current_lr
                no_improve_count = 0
                
            # Improvement checks
            loss_better_than_best = (new_loss >= 0. and new_loss < best_loss * (1 - loss_thresh)) \
                                    or (new_loss < 0. and new_loss < best_loss * (1 + loss_thresh))
            loss_almost_as_good_as_best = (new_loss >= 0. and new_loss < best_loss * (1 + loss_thresh)) \
                                    or (new_loss < 0. and new_loss < best_loss * (1 - loss_thresh))
            if loss_better_than_best:
                best_loss = new_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if use_stop and (no_improve_count > patience_crit) and loss_almost_as_good_as_best:
                effective_iters = i
                break

        if gpu:
            self.model = self.model.cpu()
            self.current_X = self.current_X.cpu()
            self.current_data = self.current_data.cpu()
        self.model.eval()
        return new_loss, effective_iters


    def compute_loo_scores(self, var_values, lscales_mat, e_loo2, s_loo2, aggregated=True):
        space_size = lscales_mat.shape[0]   
        weights = torch.zeros((len(self.X_candidates), space_size))
        for i in range(space_size):
            dist_mat = cdist(self.X_candidates.cpu().numpy() / lscales_mat[i,:].cpu().numpy(), self.current_X.values / lscales_mat[i,:].cpu().numpy())
            nn_index = np.argmin(dist_mat, axis=1)
            weights[:,i] = e_loo2[nn_index, i] / s_loo2[nn_index, i]
        if var_values.is_cuda:
          weights = weights.cuda()
        objective_function = var_values * (1 + weights)
        scores = objective_function
        if aggregated:
            return self.aggr_func(scores, dim=1)
        else:
            return scores
