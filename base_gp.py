from functools import reduce #, lru_cache
from typing import Union, List, Tuple
import torch
import gpytorch as gp
from gpytorch.means.mean import Mean
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from torch import Tensor

from utilities import handle_covar_ , get_median_heuristic_ard, CustomMultitaskGaussianLikelihood, initialize_inducing_points

class ExactGPModel(gp.models.ExactGP):
    """
    Standard exact GP model. Can handle independant multitasking via batch dimensions
    """
    def __init__( self,
                  train_x:Tensor,
                  train_y:Tensor,
                  likelihood:Union[Likelihood,None]=None,
                  kernel_type:Kernel=gp.kernels.RBFKernel,
                  mean_type:Mean=gp.means.ConstantMean,
                  decomp:Union[List[List[int]], None]=None,
                  disc_ranks:Tuple[int,...]=(),
                  outputscales:bool=False,
                  noise_thresh:float=1e-4,
                  noise_init:float=1e-1,
                  n_inducing_points:Union[int,None]=None,
                  batch_lik:Union[bool,None]=None,
                  lik_mat_rank:int=0,
                  last_target_dim_is_datapoint:bool=False,
                  ignore_n_tasks:bool=False,
                  prior_scales:Union[Tensor, None]=None,
                  prior_width:Union[Tensor, None]=None,
                  init_induc_with_qmc:bool=False,
                  ker_kwargs:Union[dict,None]=None,
                  jitter_val:float=1e-6,
                  seed:int=0,
                  **kwargs ):
        """
        Args:
            train_x: training input data
            train_y: training data labels
            likelihood: likelihood function for the model. If None, a Gaussian likelihood is used. Defaults to None.
            kernel_type: . gp kernel function for latent processes. Defaults to gp.kernels.RBFKernel.
            mean_type: gp mean function for the outputs. Defaults to gp.means.ConstantMean.
            decomp: instructions to create a composite kernel with subgroups of variables. Ex : decomp = [[0,1],[1,2]] --> k(x0,x1,x2) = k1(x0,x1) + k2(x1,x2). Defaults to None.
            disc_ranks: optional tuple of integers indicating the correlation rank of each discrete kernel.
            | Used (and required) only if decomp contains a non-empty 'cont' entry (continuous variables)
            outputscales: whether to endow the kernel with a learned scaling factor, k(.) = a*k_base(.). Defaults to True
            noise_thresh: minimum value for the noise parameter. Has a large impact for ill-conditioned kernel matrices, which is the case of the HXS application. Defaults to 1e-6.
            n_inducing_points: if an integer is provided, the model will use the sparse GP approximation of Titsias (2009) with this many inducing points. Defaults to None.
            batch_lik: whether to use a batch Gaussian likelihood, or a MultitaskGaussianLikelihood able to model cross-tasks correlatins. Only meaningful if n_tasks > 1. Defaults to False.
            ignore_n_tasks: whether to neglect the tasks axis when building covariance. Useful for inheritance, when forming an ICM
            prior_scales: Prior mean for characteristic lengthscales of the kernel. Defaults to None.
            prior_width: Prior deviation-to-mean ratio for characteristic lengthscales of the kernel. Defaults to None.
            ker_kwargs: Additional arguments to pass to the gp kernel function. Defaults to None.
            jitter_val: jitter value for the Cholesky decomposition of the kernel matrix in specific leave-one-out computations. Defaults to 1e-6.
        """
        if len(train_y.shape) == 1:
            train_y = train_y.view(1,-1) # add a task axis

        if last_target_dim_is_datapoint:
            *batch_shape, n_tasks, n_points = train_y.shape
        else:
            *batch_shape, n_points, n_tasks = train_y.shape
            
        if ignore_n_tasks:
            output_batch_shape = torch.Size(batch_shape)
        else:
            output_batch_shape = torch.Size([*batch_shape, n_tasks])
        lik_batch_shape = torch.Size([*batch_shape, n_tasks])
        multilik_batch_shape = torch.Size(batch_shape)

        # Initialization of the likelihood
        if batch_lik is None:
            batch_lik = ( len(batch_shape) < 2 or batch_shape == (1,1) )
        if likelihood is None:
            if batch_lik :
                likelihood = gp.likelihoods.GaussianLikelihood(batch_shape=lik_batch_shape, noise_constraint=gp.constraints.GreaterThan(noise_thresh))
                likelihood.noise = noise_init * torch.ones_like(likelihood.noise)
            else:
                likelihood = CustomMultitaskGaussianLikelihood(num_tasks=n_tasks, batch_shape=multilik_batch_shape,
                                                                        has_global_noise=False,
                                                                        rank=lik_mat_rank,
                                                                        noise_constraint=gp.constraints.GreaterThan(noise_thresh))
                likelihood.task_noises = torch.ones_like(likelihood.task_noises) * noise_init
                if lik_mat_rank > 0:
                    likelihood.task_noise_covar_factor = torch.nn.Parameter(torch.zeros_like(likelihood.task_noise_covar_factor))
                
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        # Initialization of the lenghtscales
        if prior_scales is None:
            prior_scales = get_median_heuristic_ard(X=train_x)

        if ker_kwargs is None:
            ker_kwargs = {}
        self.dim = train_x.shape[-1]
        self.n_tasks = n_tasks
        self.batch_lik = batch_lik
        self.mean_module = mean_type(input_size=self.dim, batch_shape=output_batch_shape)
        self.covar_module = handle_covar_(kernel_type, dim=self.dim, decomp=decomp, disc_ranks=disc_ranks,
                                          prior_scales=prior_scales, prior_width=prior_width, outputscales=outputscales,
                                          batch_shape=output_batch_shape, ker_kwargs=ker_kwargs)
        
        if n_inducing_points is not None:
            self.covar_module = gp.kernels.InducingPointKernel(self.covar_module, torch.randn(n_inducing_points, self.dim), likelihood)
            induc_locations = initialize_inducing_points(X=train_x, M=n_inducing_points, with_qmc=init_induc_with_qmc, seed=seed)
            self.covar_module.inducing_points = torch.nn.Parameter(induc_locations)
        
        if jitter_val is None:
            self.jitter_val = gp.settings.cholesky_jitter.value(train_x.dtype)
        else:
            self.jitter_val = jitter_val


    def forward( self, x:Tensor )-> Union[gp.distributions.MultivariateNormal, gp.distributions.MultitaskMultivariateNormal]:
        """
        Defines the computation performed at every call.
        Args:
            x: Data to evaluate the model at

        Returns:
            Prior distribution of the model output at the input locations. Can be a multitask multivariate normal if batch dimension is >1 
            and the model was instanciated with batch_lik = False, or a (batch) multivariate normal otherwise
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if self.batch_lik:
            return gp.distributions.MultivariateNormal(mean_x, covar_x)
        else:
            return gp.distributions.MultitaskMultivariateNormal.from_batch_mvn(gp.distributions.MultivariateNormal(mean_x, covar_x))


    def lscales( self, unpacked:bool=True )-> Union[List[Tensor], Tensor]:  # returned format : n_kernels x n_dims
        """
        Displays the learned characteric lengthscales of the kernel(s).
        Args:
            unpacked: whether to unpack the output list and trim useless dimensions of the tensor. 
            Applies only if the model kernel is not composite. Defaults to True

        Returns:
            A list of tensors representing the learned characteristic lengthscales of each subkernel and each task (or a single tensor if the kernel is non-composite and unpacked=True).
            Each one has shape n_tasks x n_dims (n_dim number of dimensions of the subkernel)
        """
        if hasattr(self.covar_module, 'kernels'):
            n_kernels = len(self.covar_module.kernels)
            ref_kernel = self.covar_module.kernels[0]
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            scales = [reduce(getattr, attr_name.split('.'), ker).squeeze() for ker in self.covar_module.kernels]
        else:
            n_kernels = 1
            ref_kernel = self.covar_module
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            scales = reduce(getattr, attr_name.split('.'), self.covar_module).squeeze()

        return [scales] if (n_kernels==1 and not unpacked) else scales


    def outputscale( self, unpacked:bool=False)-> Tensor:
        """
        Displays the outputscale(s) of the kernel(s).
        Args:
            unpacked: whether to trim useless dimensions of the tensor. Defaults to False

        Returns:
            A tensor representing the learned outputscales of each subkernel and each task (shape n_tasks x n_kernels)
        """
        # TODO : adapt to the batch case
        n_kernels = len(self.covar_module.kernels) if hasattr(self.covar_module, 'kernels') else 1
        n_funcs = self.n_latents if hasattr(self, 'n_latents') else self.n_tasks  ## to distinguish between the projected and batch-exact cases
        res = torch.zeros((n_funcs, n_kernels))
        if n_kernels > 1 :
            for i_ker in range(n_kernels):
                res[:, i_ker] = self.covar_module.kernels[i_ker].outputscale.data.squeeze()
        else:
            res[:,0] = self.covar_module.outputscale.data.squeeze()
        return res.squeeze() if (n_kernels==1 and unpacked) else res
    

    def kernel_cond( self ) -> Tensor:
        """
        Computes the condition number of the training kernel matrix.
        Returns:
            The condition number of the training kernel matrix
        """
        with torch.no_grad():
            if not self.prediction_strategy:
                self.eval()
                __ = self(torch.zeros_like(self.train_inputs[0])) # to initialize the prediction strategy
            K_plus = self.prediction_strategy.lik_train_train_covar.evaluate_kernel().to_dense()
        return torch.linalg.cond(K_plus)
    

    def default_mll(self) -> MarginalLogLikelihood:
        """
        Returns the default marginal log likelihood (loss function) object for the model.
        Returns:
            A MarginalLogLikelihood object for the model
        """
        return gp.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
