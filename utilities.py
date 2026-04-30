from typing import Union, List, Tuple
from gpytorch.constraints import Interval
import numpy as np
import torch
from torch import Tensor
import gpytorch as gp
from gpytorch.kernels.kernel import Kernel
from gpytorch.priors import Prior
from sklearn.cluster import KMeans
from scipy.stats import qmc

## Custom means and kernels    

class SplineKernel(gp.kernels.Kernel):
    is_stationary = False

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return (1 + x1**2 + x1**3 / 3).prod(dim=-1)
        mins = torch.min(x1.unsqueeze(-2), x2.unsqueeze(-3))
        maxes = torch.max(x1.unsqueeze(-2), x2.unsqueeze(-3))
        oned_vals = 1 + mins*maxes + 0.5 * mins**2 * (maxes - mins/3)
        res = oned_vals.prod(dim=-1)
        ## !! The following block is a workaround to handle batched inputs for nonstationary kernels. It should be removed when gpytorch is fixed
        if hasattr(self, 'batch_shape') and self.batch_shape and self.batch_shape != x1.shape[:-2]:
            batch_dim = -1 if params.get("last_dim_is_batch", False) else 0
            res = res.unsqueeze(batch_dim).expand(*self.batch_shape, *res.shape)
        return res


class FixedRQKernel(gp.kernels.Kernel):
    has_lengthscale = True

    def forward(self, x1, x2, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        diff = self.covar_dist(x1_, x2_, **params)
        alpha = 8
        res = (1 + 0.5 * diff / alpha)
        for i in range(3):
            res = res*res
        return 1/res


class PolynomialMean(gp.means.mean.Mean):
    def __init__( self, input_size, batch_shape=torch.Size(), bias=True, degree=3):
        super().__init__()
        for i in range(degree+1):
            self.register_parameter(name="weights_{0}".format(i),
                                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None
        self.degree = degree

    def forward( self, x: Tensor)-> Tensor:
        """

        Args:
            x: input data to be evaluated at

        Returns:
            A tensor of the values of the mean function at evaluation points.
        """
        res = 0
        for i in range(1, self.degree + 1):
            res += (x ** i).matmul(getattr(self, 'weights_{0}'.format(i))).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res


class LinearMean(gp.means.mean.Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None

    def forward(self, x):
        res = x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res

    def basis_matrix( self, x ):
        return torch.hstack([x, torch.ones((len(x), 1), device=x.device)])

##----------------------------------------------------------------------------------------------------------------------
    
## Model definition and initialization
    
def handle_covar_( kernel: Kernel, dim: int, decomp: Union[List[List[int]], None, dict]=None, batch_shape:torch.Size=torch.Size(),
                   prior_scales:Union[Tensor,None]=None, prior_width:Union[Tensor,None]=None, outputscales:bool=True,
                   disc_ranks:Tuple[int,...]=(), ker_kwargs:Union[dict, None]=None )-> Kernel:

    """ An utilitary to create and initialize covariance functions.

    Args:
        kernel: basis kernel type
        dim: dimension of the data (number of variables)
        decomp: instructions to create a composite kernel with subgroups of variables. Defaults to None
        | Ex : decomp = [[0,1],[1,2]] --> k(x0,x1,x2) = k1(x0,x1) + k2(x1,x2)
        batch_shape: batch dimensions (number of tasks or latent functions depending on the case, plus number of models to train in parallel)
        prior_scales: mean values of the prior for characteristic lengthscales. Defaults to None
        prior_width: deviation_to_mean ratio of the prior for characteristic lengthscales. Defaults to None
        outputscales: whether or not the full kernel has a learned scaling factor, i.e k(x) = a* k'(x). 
        | If decomp is nontrivial, each subkernel is automatically granted an outputscale. Defaults to True
        disc_ranks: optional tuple of integers indicating the correlation rank of each discrete kernel.
        | Used (and required) only if decomp contains a non-empty 'cont' entry (continuous variables)
        ker_kwargs: additional arguments to pass to the underlying gp kernel. Defaults to None

    Returns:
        A gp-compatible kernel
    """
    if ker_kwargs is None:
        ker_kwargs = {}

    if decomp is None:
        decomp = [list(range(dim))]
    if isinstance(decomp, dict):
        if not ('cont' in decomp and 'disc' in decomp):
            raise KeyError('Invalid format for variable grouping : {0}'.format(decomp))
        else:
            disc_vars = decomp['disc']
            decomp = decomp['cont']
            if len(disc_vars) != len(disc_ranks):
                raise(ValueError('Provided decomposition contains {0} discrete variables, but {1} ' \
                'discrete correlation ranks were specified'.format(len(disc_vars), len(disc_ranks))))
    else:
        disc_vars = []

    l_priors = [None] * len(decomp)
    if prior_scales is not None and prior_width is not None:
        if type(prior_scales) is not list:  # 2 possible formats: an array with one length per variable, or a list with one array per kernel
            prior_scales = [prior_scales[idx_list] for idx_list in decomp]
        if type(prior_width) is not list:   # 2 possible formats: an array with one length per variable, or a list with one array per kernel
            prior_width = [prior_width[idx_list] for idx_list in decomp]

        for i_ker, idx_list in enumerate(decomp):
            if len(idx_list) > 1:
                l_priors[i_ker] = gp.priors.MultivariateNormalPrior(loc=prior_scales[i_ker],
                                            covariance_matrix=torch.diag_embed(prior_scales[i_ker]*prior_width[i_ker]))
            else:
                l_priors[i_ker] = gp.priors.NormalPrior(loc=prior_scales[i_ker],
                                                              scale=prior_scales[i_ker]*prior_width[i_ker])

    kernels_args = [{'ard_num_dims': len(idx_list), 'active_dims': idx_list, 'lengthscale_prior': l_priors[i_ker],
                         'batch_shape': batch_shape} for i_ker, idx_list in enumerate(decomp)]

    kernels = []
    for i_ker, ker_args in enumerate(kernels_args):
        ker = kernel(**ker_args, **ker_kwargs)
        kernels.append(ker)

    if len(decomp) > 1 :
        covar_module = gp.kernels.ScaleKernel(kernels[0], batch_shape=batch_shape)
        for ker in kernels[1:]:
            covar_module += gp.kernels.ScaleKernel(ker, batch_shape=batch_shape)
    else:
        if outputscales:
            covar_module = gp.kernels.ScaleKernel(kernels[0], batch_shape=batch_shape)
        else:
            covar_module = kernels[0]
    
    for i_disc, el in enumerate(disc_vars):
        var_location, n_vals = el
        covar_module *= gp.kernels.IndexKernel(num_tasks=n_vals, active_dims=var_location, rank=disc_ranks[i_disc], batch_shape=batch_shape)

    if prior_scales is not None and kernels[0].has_lengthscale:
        if len(decomp) > 1 :
            for i_ker in range(len(kernels)):
                covar_module.kernels[i_ker].base_kernel.lengthscale = prior_scales[i_ker]
        elif outputscales:
            covar_module.base_kernel.lengthscale = prior_scales[0]
        else:
            covar_module.lengthscale = prior_scales[0]

    return covar_module


def compute_truncated_svd(Y: Tensor, n_latents: int, last_target_dim_is_datapoint:bool=False):
    """
    Input shape: (n_batch x) n_tasks x n_points if last_target_dim_is_datapoint, else (n_batch x) n_points x n_tasks
    Return shapes:
    U: (n_batch x) n_tasks x n_lat
    S: (n_batch x) n_lat
    V: (n_batch x) n_points x n_lat
    """
    Y_reshaped = Y if last_target_dim_is_datapoint else Y.mT
    n_points = Y_reshaped.shape[-1]
    if n_points >= n_latents:
        U, S, V = torch.svd_lowrank(Y_reshaped, q=n_latents)
    else:
        # very specific case, V is meaningless then. Only for initialization, is not a real SVD
        Q, R = torch.linalg.qr(Y_reshaped, mode='complete') # shapes (minus batch) : Q -> n_tasks x n_tasks, R -> n_tasks x n_points
        S = 1e-3 * torch.ones(R.shape[:-1])
        S[..., :n_points] = torch.diag(R)
        U = Q[..., :n_latents]
        V = torch.ones_like(R.mT)
        V = V[..., :n_latents]
    return U, S, V


def initialize_inducing_points(X:Tensor, M:int, with_qmc:bool, seed:int=0):
    """
    Initializes M inducing point locations using K-means clustering.
    
    Parameters:
    X (ndarray or tensor): Training inputs of shape (n_points, n_dims)
    M (int): Number of inducing points (clusters)
    
    Returns:
    Z (ndarray): Initialized inducing point locations of shape (M, n_dims)
    """
    if hasattr(X, "detach"):
        X_np = X.detach().cpu().numpy()
    elif hasattr(X, "numpy"):
        X_np = X.numpy()
    else:
        X_np = np.asarray(X)

    if with_qmc:
        dim = X.shape[-1]
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        locations = 2 * sampler.random(n=M) - 1
    else:
        # n_init='auto' is recommended for newer sklearn versions
        # Use k-means++ for better initial centroid placement
        kmeans = KMeans(n_clusters=M, n_init='auto', init='k-means++')
        kmeans.fit(X_np)
        locations = kmeans.cluster_centers_

    if hasattr(X, "numpy"):
        res = torch.as_tensor(locations, dtype=X.dtype, device=X.device)
    else:
        res = locations
    return res


def get_median_heuristic_ard(X, num_samples=1000):
    """
    Calculates the median distance between points along each dimension.
    
    Args:
        X (np.ndarray or torch.Tensor): Training inputs of shape (n_points, n_dims).
        num_samples (int): Max points to use for the calculation (prevents memory errors).
        
    Returns:
        np.ndarray: Median distances of shape (n_dims,).
    """
    if torch.is_tensor(X):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.atleast_2d(X)
        
    n_points, n_dims = X_np.shape
    
    if n_points > num_samples:
        indices = np.random.choice(n_points, num_samples, replace=False)
        X_np = X_np[indices]
        n_points = num_samples

    median_dists = np.zeros(n_dims)

    for d in range(n_dims):
        x_d = X_np[:, d][:, np.newaxis]
        dists = np.abs(x_d - x_d.T)
        triu_indices = np.triu_indices(n_points, k=1)
        pairwise_dists = dists[triu_indices]
        m_val = np.median(pairwise_dists)
        median_dists[d] = m_val if m_val > 0 else 1.0
        
    return median_dists

##----------------------------------------------------------------------------------------------------------------------

## Parametrizations

class PositiveDiagonalParam(torch.nn.Module):
    """
    Torch parametrization for a positive diagonal matrix.
    """
    def forward( self, X: Tensor)-> Tensor:
        return torch.diag_embed(torch.exp(torch.diagonal(X, dim1=-2, dim2=-1)))
    def right_inverse( self, A: Tensor)-> Tensor:
        return torch.diag_embed(torch.log(torch.diagonal(A, dim1=-2, dim2=-1)))

class UpperTriangularParam(torch.nn.Module):
    """
    Torch parametrization for an upper triangular matrix.
    """
    def forward( self, X: Tensor)-> Tensor:
        upper =  X.triu()
        mat_side = X.shape[-1]
        upper[..., range(mat_side), range(mat_side)] = torch.exp(upper[..., range(mat_side), range(mat_side)])
        return upper
    def right_inverse( self, A: Tensor)-> Tensor: 
        res = A
        mat_side = A.shape[-1]
        res[..., range(mat_side), range(mat_side)] = torch.log(res[..., range(mat_side), range(mat_side)])
        return res

class LowerTriangularParam(torch.nn.Module):
    """
    Torch parametrization for a Cholesky factor matrix (lower triangular with positive diagonal).
    """
    def __init__(self, bounds: List[float] = [1e-16, 1e16]):
        super().__init__()
        self.bounds = bounds

    def forward( self, X: Tensor )-> Tensor:
        lower = X.tril()
        mat_side = X.shape[-1]
        lower[..., range(mat_side), range(mat_side)] = torch.exp(torch.clamp(lower[..., range(mat_side), range(mat_side)], *self.bounds))
        return lower
    def right_inverse( self, A: Tensor)-> Tensor:
        res = A
        mat_side = A.shape[-1]
        res[..., range(mat_side), range(mat_side)] = torch.log(res[..., range(mat_side), range(mat_side)])
        return res

##----------------------------------------------------------------------------
# Custom multitask likelihood
from typing import Optional, Any
from gpytorch.likelihoods.multitask_gaussian_likelihood import MultitaskGaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.priors.prior import Prior
from gpytorch.constraints import GreaterThan
from linear_operator import to_linear_operator
from linear_operator.operators import (
    ConstantDiagLinearOperator,
    DiagLinearOperator,
    KroneckerProductDiagLinearOperator,
    KroneckerProductLinearOperator,
    LinearOperator,
    RootLinearOperator,
)

class CustomMultitaskGaussianLikelihood(MultitaskGaussianLikelihood):
    """
    Contrary to the MultitaskGaussianLikelihood of gpytorch, this one can have both a task_covar_factor and additional on-diagonal task_noises
    """

    def __init__(
        self,
        num_tasks: int,
        rank: int = 0,
        batch_shape: torch.Size = torch.Size(),
        task_prior: Optional[Prior] = None,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[Interval] = None,
        has_global_noise: bool = True,
        has_task_noise: bool = True,
    ) -> None:
        super(Likelihood, self).__init__()  # pyre-ignore[20]
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        if not has_task_noise and not has_global_noise:
            raise ValueError(
                "At least one of has_task_noise or has_global_noise must be specified. "
                "Attempting to specify a likelihood that has no noise terms."
            )

        if has_task_noise:
            self.register_parameter(
                name="raw_task_noises", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, num_tasks))
            )
            self.register_constraint("raw_task_noises", noise_constraint)
            if noise_prior is not None:
                self.register_prior("raw_task_noises_prior", noise_prior, lambda m: m.task_noises)

            if rank == 0:
                if task_prior is not None:
                    raise RuntimeError("Cannot set a `task_prior` if rank=0")
            else:
                self.register_parameter(
                    name="task_noise_covar_factor",
                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, num_tasks, rank)),
                )
                if task_prior is not None:
                    self.register_prior("MultitaskErrorCovariancePrior", task_prior, lambda m: m._eval_covar_matrix)
        self.num_tasks = num_tasks
        self.rank = rank

        if has_global_noise:
            self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
            self.register_constraint("raw_noise", noise_constraint)
            if noise_prior is not None:
                self.register_prior("raw_noise_prior", noise_prior, lambda m: m.noise)

        self.has_global_noise = has_global_noise
        self.has_task_noise = has_task_noise

    @property
    def task_noises(self) -> Optional[Tensor]:
        if self.has_task_noise:
            return self.raw_task_noises_constraint.transform(self.raw_task_noises)
        else:
            raise AttributeError("Cannot set diagonal task noises if the likelihood only has a global noise")

    @task_noises.setter
    def task_noises(self, value: Union[float, Tensor]) -> None:
        self._set_task_noises(value)

    @property
    def task_noise_covar(self) -> Tensor:
        if self.rank > 0:
            D = torch.diag_embed(self.task_noises)
            return self.task_noise_covar_factor.matmul(self.task_noise_covar_factor.transpose(-1, -2)) + D
        else:
            raise AttributeError("Cannot retrieve task noises when covariance is diagonal.")

    @task_noise_covar.setter
    def task_noise_covar(self, value: Tensor) -> None:
        # internally uses a pivoted cholesky decomposition to construct a low rank
        # approximation of the covariance
        if not self.has_task_noise:
            raise AttributeError("Cannot set task noise covar if the likelihood only has a global noise")
        if self.rank > 0:
            with torch.no_grad():
                value_minus_diag = value - torch.diag_embed(self.task_noises)
                self.task_noise_covar_factor.data = to_linear_operator(value_minus_diag).pivoted_cholesky(rank=self.rank)
        else:
            raise AttributeError("Cannot set non-diagonal task noises when covariance is diagonal.")

    def _eval_covar_matrix(self) -> Tensor:
        covar_factor = self.task_noise_covar_factor
        noise = self.noise
        D = noise * torch.eye(self.num_tasks, dtype=noise.dtype, device=noise.device)  # pyre-fixme[16]
        return covar_factor.matmul(covar_factor.transpose(-1, -2)) + D + torch.diag_embed(self.task_noises)

    def _shaped_noise_covar(
        self, shape: torch.Size, add_noise: Optional[bool] = True, interleaved: bool = True, *params: Any, **kwargs: Any
    ) -> LinearOperator:
        if not self.has_task_noise:
            noise = ConstantDiagLinearOperator(self.noise, diag_shape=shape[-2] * self.num_tasks)
            return noise

        task_noises = self.raw_task_noises_constraint.transform(self.raw_task_noises)
        task_var_lt = DiagLinearOperator(task_noises)
        dtype, device = task_noises.dtype, task_noises.device
        if self.rank > 0:
            task_noise_covar_factor = self.task_noise_covar_factor
            task_var_lt += RootLinearOperator(task_noise_covar_factor)
            ckl_init = KroneckerProductLinearOperator
        else:
            ckl_init = KroneckerProductDiagLinearOperator

        eye_lt = ConstantDiagLinearOperator(
            torch.ones(*shape[:-2], 1, dtype=dtype, device=device), diag_shape=shape[-2]
        )
        task_var_lt = task_var_lt.expand(*shape[:-2], *task_var_lt.matrix_shape)  # pyre-ignore[6]

        # to add the latent noise we exploit the fact that
        # I \kron D_T + \sigma^2 I_{NT} = I \kron (D_T + \sigma^2 I)
        # which allows us to move the latent noise inside the task dependent noise
        # thereby allowing exploitation of Kronecker structure in this likelihood.
        if add_noise and self.has_global_noise:
            noise = ConstantDiagLinearOperator(self.noise, diag_shape=task_var_lt.shape[-1])
            task_var_lt = task_var_lt + noise

        if interleaved:
            covar_kron_lt = ckl_init(eye_lt, task_var_lt)
        else:
            covar_kron_lt = ckl_init(task_var_lt, eye_lt)

        return covar_kron_lt
    
##-----------------------------------------------------------------------------------------------
class CorrectReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    torch's ReduceLROnPlateau has incorrect behavior in relative mode when the loss is negative : the sign in 1.0 +/- self.threshold is wrong
    (see https://github.com/pytorch/pytorch/issues/47513)
    """
    def _is_better(self, a, best, return_thresh=False):  # noqa: D102
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold if best > 0. else 1.0 + self.threshold
            thresh = best * rel_epsilon
            res = a < thresh

        elif self.mode == "min" and self.threshold_mode == "abs":
            thresh = best - self.threshold
            res = a < thresh

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0 if best > 0. else 1.0 - self.threshold
            thresh = best * rel_epsilon
            res = a > thresh

        else:  # mode == 'max' and epsilon_mode == 'abs':
            thresh = best + self.threshold
            res = a > thresh

        return thresh if return_thresh else res

    def step(self, metrics) -> None:  # type: ignore[override]
        """Perform a step."""
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self._is_better(current, self.best):
            # print('Loss has improved enough ! Current: {0}, thresh: {2}, best: {1}', current, self.best, self._is_better(current, self.best, return_thresh=True))
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


