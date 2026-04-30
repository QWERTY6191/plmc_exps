from functools import reduce
from typing import Union, List
import psutil
import numpy as np
import torch
from torch import Tensor
import gpytorch as gp
from gpytorch.likelihoods.likelihood import Likelihood
from linear_operator.operators import PsdSumLinearOperator, RootLinearOperator

from utilities import compute_truncated_svd
from base_gp import ExactGPModel

class MultitaskGPModel(ExactGPModel):
    """
    A multitask GP model with exact GP treatment. This class encompasses both ICM and naive LMC models.
    """
    def __init__( self,
                  train_x: Tensor,
                  train_y: Tensor,
                  n_latents: int, 
                  likelihood:Union[Likelihood,None]=None,
                  init_lmc_coeffs:bool=True,
                  last_target_dim_is_datapoint:bool=False,
                  init_task_vars:Tensor=torch.tensor([1.]),
                  **kwargs):
        """Initialization of the model. Note that the optional arguments of the ExactGPModel (in particular the choice of 
        mean and kernel function) also apply here thanks to the inheritance.

        Args:
            train_x: Input data
            train_y: Input labels
            n_latents: number of latent functions
            likelihood: gpytorch likelihood function for the outputs. If none is provided, a default MultitaskGaussianLikelihood is used. Defaults to None.
            init_lmc_coeffs: whether to initialize LMC coefficients with SVD of the training labels. If False, these coefficients are sampled from a normal distribution. Defaults to True.
        """
        if last_target_dim_is_datapoint:
            *batch_shape, n_tasks, n_points = train_y.shape
        else:
            *batch_shape, n_points, n_tasks = train_y.shape

        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood, ignore_n_tasks=True, batch_lik=False, **kwargs)
        # we build upon a single-task GP, created by calling parent class
            
        self.mean_module = gp.means.MultitaskMean(self.mean_module, num_tasks=n_tasks)
        self.covar_module = gp.kernels.MultitaskKernel(self.covar_module, num_tasks=n_tasks, rank=n_latents)
        self.covar_module.task_covar_module.var = init_task_vars # initialization

        if init_lmc_coeffs:
            U, S, V = compute_truncated_svd(Y=train_y, n_latents=n_latents, last_target_dim_is_datapoint=last_target_dim_is_datapoint)
            S = S / np.sqrt(n_points) # because of Marchenko-Pastur Law
            lmc_coeffs = (U * S.unsqueeze(-2))
            self.covar_module.task_covar_module.covar_factor = torch.nn.Parameter(lmc_coeffs)
                
        self.n_tasks, self.n_latents = n_tasks, n_latents


    def lmc_coefficients( self )-> Tensor:
        """

        Returns:
            tensor of shape n_latents x n_tasks representing the LMC/ICM coefficients of the model.
        """
        res = self.covar_module.task_covar_module.covar_factor.data.squeeze().mT
        return res


    def lscales( self, unpacked:bool=True)-> Union[List[Tensor], Tensor] :
        """
        Displays the learned characteric lengthscales of the kernel(s).
        Args:
            unpacked: whether to unpack the output list and trim useless dimensions of the tensor. 
            Applies only if the model kernel is not composite. Defaults to True

        Returns:
            A list of tensors representing the learned characteristic lengthscales of each subkernel and each task (or a single tensor if the kernel is non-composite and unpacked=True).
            Each one has shape n_latents x n_dims (n_dim number of dimensions of the subkernel)
        """
        data_covar = self.covar_module.data_covar_module

        if hasattr(data_covar, 'kernels'):
            n_kernels = len(data_covar.kernels)
            ref_kernel = data_covar.kernels[0]
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            scales = [reduce(getattr, attr_name.split('.'), ker).squeeze().repeat(self.n_latents, 1) for ker in data_covar.kernels]
        else:
            n_kernels = 1
            ref_kernel = data_covar
            attr_name = 'base_kernel.lengthscale.data' if hasattr(ref_kernel, 'base_kernel') else 'lengthscale.data'
            scales = reduce(getattr, attr_name.split('.'), data_covar).squeeze().repeat(self.n_latents, 1)
      
        return [scales] if (n_kernels==1 and not unpacked) else scales    
        

    def outputscale( self, unpacked:bool=False)-> Tensor:
        """
        Displays the outputscale(s) of the kernel(s).
        Args:
            unpacked: whether to trim useless dimensions of the tensor. Defaults to False

        Returns:
            A tensor representing the learned outputscales of each subkernel and each task (shape n_latents x n_kernels)
        """
        data_covar = self.covar_module.data_covar_module

        n_kernels = len(data_covar.kernels) if hasattr(data_covar, 'kernels') else 1
        res = torch.zeros((self.n_latents, n_kernels))
        if n_kernels > 1:
            for i_ker in range(n_kernels):
                res[:, i_ker] = data_covar.kernels[i_ker].outputscale.data.squeeze()
        else:
            res[:, 0] = data_covar.outputscale.data.squeeze()

        return res.squeeze() if (n_kernels==1 and unpacked) else res


    def forward( self, x ):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    

    def compute_var(self, x):
        """
        Computes the variance of the model at input locations.
        Args:
            x: input data to evaluate the model at

        Returns:
            A tensor of the variances of the model at input locations.
        """
        linop = self.covar_module.forward(x,x) + self.likelihood._shaped_noise_covar((len(x), len(x)), add_noise=True)
        first_term = linop.diagonal(dim1=-2, dim2=-1).reshape((len(x), self.n_tasks))

        x_train = self.train_inputs[0]
        ker_op = self.covar_module.forward(x_train,x_train)
        noise_op = self.likelihood._shaped_noise_covar((len(x_train), len(x_train)), add_noise=True)
        i_task = 1 if isinstance(ker_op.linear_ops[1], (PsdSumLinearOperator, RootLinearOperator)) else 0 #TODO : clean this

        data_ker = ker_op.linear_ops[(i_task + 1) % 2]
        k_evals, k_evecs = data_ker._symeig(eigenvectors=True)

        noise_inv_root = noise_op.linear_ops[i_task].root_inv_decomposition()
        C = ker_op.linear_ops[i_task]
        C_tilde = noise_inv_root.matmul(C.matmul(noise_inv_root))
        C_evals, C_evecs = C_tilde._symeig(eigenvectors=True)
        C_hat = C.matmul(noise_inv_root).matmul(C_evecs).to_dense().squeeze()
        C_square = C_hat**2

        S = torch.kron(k_evals, C_evals) + 1.0
        if x.is_cuda:
            free_mem = torch.cuda.mem_get_info()[0]
        else:
            free_mem = psutil.virtual_memory()[1]

        num_bytes = x.element_size()
        batch_size = int(free_mem / (16 * len(x_train) * self.n_tasks**2 * num_bytes))
        n = x.shape[0]  # Total number of samples
        second_term_results = []  # List to store the results
        for i in range(0, n, batch_size):
            x_batch = x[i:i+batch_size]
            k_hat = self.covar_module.data_covar_module(x_batch, x_train).matmul(k_evecs).to_dense().squeeze()
            k_square = k_hat**2
            second_term = torch.kron(k_square, C_square) @ S.pow(-1).squeeze()
            second_term_results.append(second_term.reshape((len(x_batch), self.n_tasks)))

        # Convert the list of results to a tensor
        second_term = torch.cat(second_term_results)
        return torch.clamp(first_term - second_term, min=1e-6)

    
    def kernel_cond( self ) -> Tensor:
        """
        Computes the condition number of the training data kernel matrix.
        Returns:
            The condition number of the training data kernel matrix
        """
        with torch.no_grad():
            K = self.covar_module.data_covar_module(self.train_inputs[0]).to_dense()
            K_plus_noise = K  + torch.eye(len(self.train_inputs[0]), device=self.train_inputs[0].device)
            # In the efficient implementation of the ICM, used in particular in gpytorch, the noise "seen" by the data kernel is always 1.
            # The amplitude of this noise is reported onto the task kernel instead.
            # See : 
            # Rakitsch, B., Lippert, C., Borgwardt, K., & Stegle, O. (2013).
            # It is all in the noise: Efficient multi-task Gaussian process inference with structured residuals.
            #Advances in neural information processing systems, 26.
        return torch.linalg.cond(K_plus_noise)


    def default_mll(self):
        return gp.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

