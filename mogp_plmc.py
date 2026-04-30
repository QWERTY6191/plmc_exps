from typing import Union, List, Tuple, Any, Optional
import numpy as np
import torch
from torch import Tensor
import gpytorch as gp
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.means.mean import Mean
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.multitask_gaussian_likelihood import _MultitaskGaussianLikelihoodBase
from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator, ConstantDiagLinearOperator, DenseLinearOperator, LinearOperator

from utilities import compute_truncated_svd, PositiveDiagonalParam, LowerTriangularParam, UpperTriangularParam
from base_gp import ExactGPModel


class NonfactoredMultitaskGaussianLikelihood(_MultitaskGaussianLikelihoodBase):
    """
    This class represents a multitask likelihood very similar to gpytorch's MultitaskGaussianLikelihood, but the task part of the covariance
    | is stored as a full matrix rather than a root. Global noise and taskwise diagonal noise are not available here.
    """
    has_global_noise = False
    
    def __init__(
        self,
        num_tasks: int,
        task_noise_covar: Optional[Tensor] = None,
        batch_shape: torch.Size = torch.Size(),
        noise_constraint: Optional[Interval] = None,
    ) -> None:
        super(Likelihood, self).__init__()  # pyre-ignore[20]
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        if task_noise_covar is None:
            task_noise_covar = torch.randn(*batch_shape, num_tasks, num_tasks)
        self.register_parameter(
            name="task_noise_covar",
            parameter=torch.nn.Parameter(task_noise_covar),
        )
        self.num_tasks = num_tasks


    def _shaped_noise_covar(
        self, shape: torch.Size, interleaved: bool = True, *params: Any, **kwargs: Any
    ) -> LinearOperator:
        
        task_var_lt = DenseLinearOperator(self.task_noise_covar)
        dtype, device = self.task_noise_covar.dtype, self.task_noise_covar.device
        ckl_init = KroneckerProductLinearOperator

        eye_lt = ConstantDiagLinearOperator(
            torch.ones(*shape[:-2], 1, dtype=dtype, device=device), diag_shape=shape[-2]
        )
        task_var_lt = task_var_lt.expand(*shape[:-2], *task_var_lt.matrix_shape)  # pyre-ignore[6]

        if interleaved:
            covar_kron_lt = ckl_init(eye_lt, task_var_lt)
        else:
            covar_kron_lt = ckl_init(task_var_lt, eye_lt)

        return covar_kron_lt
    

## making the mixing matrix a separate class allows to call torch.nn.utils.parametrizations.orthogonal
## onto it during instanciation of a ProjectedGPModel
class LMCMixingMatrix(torch.nn.Module):
    """
    Class for the parametrized mixing matrix of projected models. Making it a separate class allows to call 
    torch.nn.utils.parametrizations.orthogonal onto it during instanciation of a ProjectedGPModel
    """
    def __init__( self, Q_plus:Tensor, R:Tensor, use_QR_decomp:bool=True, diagonal_R:bool=False):
        """
        Args:
            Q_plus: (augmented) orthonormal part of the mixing matrix, of shape n_tasks x n_latents or n_tasks x n_tasks
            R: upper triangular part of the mixing matrix, of shape n_latents x n_latents
            use_QR_decomp: whether to parametrize the mixing matrix as a unique n_tasks x n_latents (or n_tasks x n_tasks) matrix with no
            | specific property, or as a product of its Q and R factors. The first option is only possible if no constraint is put on these factors.
            | It is generally faster and more stable, but can be less so in the "augmented" case (general PLMC) where a n_tasks x n_tasks matrix must be parametrized.
            | Defaults to True.
        """
        super().__init__()
        if len(Q_plus.shape) != len(R.shape):
            raise ValueError("Q_plus and R have different number of axes: {0} and {1}".format(Q_plus.shape, R.shape))
        else:
            self.shape_batch = Q_plus.shape[:-2]
            self.n_batch_dims = len(self.shape_batch)

        if Q_plus.shape[self.n_batch_dims + 1] == Q_plus.shape[self.n_batch_dims + 0]:
            ## If the inputed Q matrix is of shape n_tasks x n_tasks, we assume that it is the augmented Q_plus matrix
            self.mode = 'Q_plus'
        elif Q_plus.shape[self.n_batch_dims + 1] == R.shape[self.n_batch_dims + 0]:
            ## If the inputed Q matrix is of shape n_tasks x n_latents, we assume that it is the regular Q matrix (Q factor of the QR decomposition of the mixing matrix)
            self.mode = 'Q'
        else:
            raise ValueError('Wrong dimensions for Q_plus : should be (n_batch x) n_tasks x n_tasks or (n_batch x) n_tasks x n_latents,' \
            'got {0}. n_latents has been infered from R to be {1}'.format(Q_plus.shape, R.shape[self.n_batch_dims + 0]))
        
        self.n_latents = R.shape[self.n_batch_dims]
        self.n_tasks = Q_plus.shape[self.n_batch_dims]
        self._size = torch.Size([*self.shape_batch, self.n_latents, self.n_tasks])
        self.use_QR_decomp = use_QR_decomp
        self.diagonal_R = diagonal_R
        if use_QR_decomp:
            if self.mode=='Q_plus':
                R_padded = torch.broadcast_to(torch.eye(self.n_tasks), (*self.shape_batch, self.n_tasks, self.n_tasks)).clone()
                R_padded[..., :self.n_latents, :self.n_latents] = R
                H = Q_plus @ R_padded
            else:
                H = Q_plus @ R
            self.register_parameter("H", torch.nn.Parameter(H, requires_grad=True))
        else:
            self.register_parameter("Q_plus", torch.nn.Parameter(Q_plus, requires_grad=True))
            self.register_parameter("R", torch.nn.Parameter(R, requires_grad=True))


    def Q( self ) -> Tensor:
        """
        Outputs the Q factor of the QR decomposition of the mixing matrix.
        Returns:
            Q factor of the mixing matrix, of shape n_tasks x n_latents.
        """
        if self.mode=='Q_plus':
            return self.Q_plus[..., :, :self.n_latents]
        else:
            return self.Q_plus


    def Q_orth( self ) -> Tensor:
        """
        Outputs the orthonormal complement of the Q factor of the QR decomposition of the mixing matrix.
        Returns:
            Orthonormal complement of Q, of shape n_tasks x (n_tasks - n_latents).
        """
        return self.Q_plus[..., :, self.n_latents:]


    def QR(self) -> Tuple[Tensor, Tensor, Union[Tensor,None]]:
        """
        Outputs the Q and R factors of the mixing matrix, and the orthonormal complement Q_orth of Q.
        Returns:
            Q factor of the mixing matrix, of shape n_tasks x n_latents.
            R factor of the mixing matrix, of shape n_latents x n_latents.
            Orthonormal complement of Q, of shape n_tasks x (n_tasks - n_latents) or None if self.model == 'Q_plus' (general PLMC model).
        """
        if self.use_QR_decomp:
            Q_plus, R_padded = torch.linalg.qr(self.H)
            if self.mode=='Q_plus':
                Q = Q_plus[..., :, :self.n_latents]
                Q_orth = Q_plus[..., :, self.n_latents:]
                R = R_padded[..., :self.n_latents, :self.n_latents]
            else:
                Q, Q_orth, R = Q_plus, None, R_padded
        else:
            Q, Q_orth, R = self.Q(), self.Q_orth(), self.R
        if self.diagonal_R:
            R = torch.diag_embed(torch.diag(R))
        return Q, R, Q_orth


    def forward( self ) -> Tensor:
        """
        Outputs the full (batch) mixing matrix H, in transposed form in order to match the standard storage format of data labels.
        Returns:
            Transposed mixing matrix H, of shape (n_batch x) n_tasks x n_latents.
        """
        if self.use_QR_decomp:
            if self.diagonal_R:
                Q, R, Q_orth = self.QR()
                return (Q @ torch.diag_embed(torch.diag(R))).mT
            if self.mode == 'Q':
                return self.H.mT
            else:
                return self.H[..., :, :self.n_latents].mT
        else:
            return (self.Q() @ self.R).mT #format : (n_batch x) n_latents x n_tasks


    def size( self, int=None ) -> Union[int, torch.Size]:
        if int:
            return self._size[int]
        else:
            return self._size


class ProjectedGPModel(ExactGPModel):
    """
    The projected LMC model. Reference : https://arxiv.org/abs/2310.12032
    """
    def __init__( self,
                  train_x:Tensor,
                  train_y:Tensor,
                  n_latents:int,
                  proj_likelihood:Union[None,Likelihood]=None, 
                  zero_M:bool=True,
                  diag_Sig_orth:bool=True,
                  scalar_Sig_orth:bool=False,
                  diagonal_R:bool=False,
                  use_QR_decomp=True,
                  ortho_param='matrix_exp',
                  mean_type:Mean=gp.means.ZeroMean,
                  noise_thresh:float=1e-4,
                  noise_init:float=1e-1,
                  last_target_dim_is_datapoint:bool=False,
                  **kwargs):
        """Initialization of the model. Note that the optional arguments of the ExactGPModel (in particular the choice of 
        mean and kernel function) also apply here thanks to the inheritance.
        
        Args:
            train_x: training input data
            train_y: training input labels
            n_latents: number of latent processes
            proj_likelihood: batched independant likelihood of size n_latents for latent processes. Defaults to None.
            zero_M: whether to enforce the Block Diagonal Noise approximation (see reference article), making for a block-diagonal task noise matrix. Defaults to True.
            diag_Sig_orth: whether to parametrize a diagonal noise factor Sigma_orth (see reference article), a simplification which theoretically causes no loss of generality. Defaults to False.
            scalar_Sig_orth: whether to parametrize a scalar noise factor Sigma_orth (see reference article). Overrides diag_Sig_orth=False if set to True. Defaults to False.
            diagonal_R: whether to parametrize a diagonal scale component for the mixing matrix (see reference article).
            If set to True and scalar_Sig_orth=True and zero_M=True, this results in the OILMM model (see reference in the article). Defaults to False.
            use_QR_decomp: whether to parametrize the mixing matrix as a unique n_tasks x n_latents (or n_tasks x n_tasks) matrix with no
            specific property, or as a product of its Q and R factors. The first option is only possible if no constraint is put on these factors.
            It is generally faster and more stable, but can be less so in the "augmented" case (general PLMC) where a n_tasks x n_tasks matrix must be parametrized.
            Defaults to True.
            ortho_param: orthonormal parametrizzation for the mixing matrix, in the case where use_QR_decomp=False and diag_Sig_orth=False.
            Can be 'matrix_exp' (default, the only one proved stable in previous studies), 'cayley' or 'householder'. Defaults to 'matrix_exp'. 
            mean_type: gp mean function for task-level processes. At the moment, only a zero mean is implemented ; every other choice will throw an error.
            Defaults to gp.means.ZeroMean.
            noise_thresh: minimum value for the noise parameter. Has a large impact for ill-conditioned kernel matrices, which is the case of the HXS application. Defaults to 1e-6.
        """
        if mean_type is not gp.means.ZeroMean:
            raise NotImplementedError('Projected GP model does not support non-zero output-wise means for now !')

        if last_target_dim_is_datapoint:
            *batch_shape, n_tasks, n_points = train_y.shape
        else:
            *batch_shape, n_points, n_tasks = train_y.shape
        latent_batch_shape = torch.Size([*batch_shape, n_latents])
        discarded_noise_shape = torch.Size([*batch_shape, n_tasks - n_latents])
        batch_shape = torch.Size(batch_shape)

        # Likelihood (noise model) initialization
        if proj_likelihood is not None and proj_likelihood.noise.shape[-1] != n_latents:
            raise ValueError("In projected GP model the dimension of the likelihood is the number of latent processes. "
                  "Provided likelihood has length {0} while n_latents is {1}".format(proj_likelihood.noise.shape[-1], n_latents))
        elif proj_likelihood is None:
            proj_likelihood = gp.likelihoods.GaussianLikelihood(batch_shape=latent_batch_shape,
                                        noise_constraint=gp.constraints.GreaterThan(noise_thresh))
            proj_likelihood.noise = noise_init * torch.ones_like(proj_likelihood.noise)
            
        # Initialization of LMC coefficients and projected data
        if scalar_Sig_orth and zero_M: # !!! Very structuring choice, corresponding to PLMC-fast ; see PLMC article
            U, S, V = compute_truncated_svd(Y=train_y, n_latents=n_latents, last_target_dim_is_datapoint=last_target_dim_is_datapoint)
            S = S / np.sqrt(n_points) # because of Marchenko-Pastur Law
            R = S
        else:
            U, S, V = compute_truncated_svd(Y=train_y, n_latents=n_tasks, last_target_dim_is_datapoint=last_target_dim_is_datapoint)
            S = S / np.sqrt(n_points) # because of Marchenko-Pastur Law
            R, V = S[..., :n_latents], V[..., :n_latents]
        Q_plus = U
        proj_y = V.mT
        R = torch.diag_embed(R)
        lmc_coefficients = LMCMixingMatrix(Q_plus, R, use_QR_decomp=use_QR_decomp, diagonal_R=diagonal_R)
        if not use_QR_decomp:
            lmc_coefficients = torch.nn.utils.parametrizations.orthogonal(lmc_coefficients, name="Q_plus",
                                    orthogonal_map=ortho_param, use_trivialization=(ortho_param!='householder'))  # parametrizes Q_plus as orthogonal
            if diagonal_R:
                torch.nn.utils.parametrize.register_parametrization(lmc_coefficients, "R", PositiveDiagonalParam())
            else:
                torch.nn.utils.parametrize.register_parametrization(lmc_coefficients, "R", UpperTriangularParam())

        # Initialization of the latent processes
        super().__init__(train_x=train_x, train_y=proj_y, likelihood=proj_likelihood, last_target_dim_is_datapoint=True,
                         mean_type=gp.means.ZeroMean, batch_lik=True, **kwargs)
        # !! The latent processes always have the datapoint axis as last dim
        # !! The projected likelihood will only be named likelihood in the model. The task-level one is self.full_likelihood()
        self.register_buffer('train_y', train_y)
        self.lmc_coefficients = lmc_coefficients

        # Initialization of the discarded noise terms ; see PLMC article
        discarded_noise_tens = torch.ones(discarded_noise_shape)
        log_noise_thresh = np.log(noise_thresh)
        log_init_noise = np.log(noise_init)
        if scalar_Sig_orth:
            diag_Sig_orth = True
            self.register_parameter("log_Sigma_orth", torch.nn.Parameter(log_init_noise * discarded_noise_tens[..., :1]))
            self.register_constraint("log_Sigma_orth", gp.constraints.GreaterThan(log_noise_thresh))
            if zero_M:
                self.register_buffer('Y_squared_norm', torch.linalg.matrix_norm(train_y) ** 2) # case of the PLMC_fast (term for MLL computation)
        elif diag_Sig_orth:
            self.register_parameter("log_Sigma_orth", torch.nn.Parameter(log_init_noise * discarded_noise_tens))
            self.register_constraint("log_Sigma_orth", gp.constraints.GreaterThan(log_noise_thresh))
        else:
            self.register_parameter("Sigma_orth_inv_chol", torch.nn.Parameter(torch.diag_embed(1 / noise_init * discarded_noise_tens)))
            torch.nn.utils.parametrize.register_parametrization(self, "Sigma_orth_inv_chol",
                                                        LowerTriangularParam(bounds=(log_noise_thresh, -log_noise_thresh)))
        if not zero_M and n_tasks > n_latents:
            self.register_parameter("M", torch.nn.Parameter(torch.zeros([*batch_shape, n_latents, n_tasks - n_latents])))
            self._has_M_term = True
        else:
            self._has_M_term = False

        self.diag_Sig_orth, self.scalar_Sig_orth = diag_Sig_orth, scalar_Sig_orth
        self.n_tasks = n_tasks
        self.n_latents = n_latents
        self.shape_batch = batch_shape
        self.latent_dim = -1
        self.last_target_dim_is_datapoint = last_target_dim_is_datapoint
        self.noise_thresh = noise_thresh


    def projected_noise( self )-> Tensor:
        """
        Returns a vector containing the modeled noises of latent processes. Its diagonal embedding is the matrix Sigma_P from the article.
        Returns:
            Modeled noise vector of size (n_batch x) n_latents. 
        """
        return self.likelihood.noise.squeeze(-1)
    

    def projection_matrix( self )-> Tensor:
        """
        Returns matrix T from the reference article, such that YT is the "projected data" seen by latent processes
        Returns:
            Projection matrix T, of shape (n_batch x) n_tasks x n_latents. 
        """
        Q, R, Q_orth = self.lmc_coefficients.QR()
        H_pinv = torch.linalg.solve_triangular(R.mT, Q, upper=False, left=False)  # shape (n_batch x) n_tasks x n_latents
        if self._has_M_term:
            return H_pinv + Q_orth @ self.M.mT * self.projected_noise()
        else:
            return H_pinv


    def project_data( self, data:Tensor, last_data_dim_is_datapoint:bool=False ) -> Tensor:
        """
        Projects some data labels onto the latent space.
        Args:
            data: data tensor of shape (n_batch x) n_tasks x n_points
        Returns:
            Projected data tensor of shape (n_batch x) n_latents x n_points.
            This shape convention corresponds to the batch treatment in gpytorch, not to the usual convention.
        """
        Q, R, Q_orth = self.lmc_coefficients.QR()
        if last_data_dim_is_datapoint:
            unscaled_proj = Q.mT @ data
        else:
            unscaled_proj = (data @ Q).mT
        Hpinv_times_Y = torch.linalg.solve_triangular(R, unscaled_proj, upper=True)

        if self._has_M_term:
            if last_data_dim_is_datapoint:
                orthog_proj = Q_orth.mT @ data
            else:
                orthog_proj = (data @ Q_orth).mT
            res = Hpinv_times_Y + self.projected_noise().unsqueeze(-1) * self.M @ orthog_proj
        else:
            res = Hpinv_times_Y
        return res  # (n_batch x) shape n_latents x n_points


    def full_likelihood( self, diag=False ) -> Union[gp.likelihoods.MultitaskGaussianLikelihood, NonfactoredMultitaskGaussianLikelihood] :
        """
        Outputs the task-level likelihood of the model (Sigma matrix from the reference article), including the noise of the latent processes and the discarded noise.
        Returns:
            Task-level likelihood of the model, with a multitask gaussian likelihood of size n_tasks.
        """
        Q, R, Q_orth = self.lmc_coefficients.QR()
        QR = Q @ R
        sigma_p = self.projected_noise()
        Sigma_orth_root = self.Sigma_orth_root
        
        if self.n_latents < self.n_tasks:
            if self.scalar_Sig_orth:
                Sigma_orth = Sigma_orth_root ** 2 # Only taske the scalar value, not the full vector (but same number of dimensions) 
                if diag:
                    B_term = Sigma_orth * (1 - (Q**2).sum(dim=-1))
                else:
                    identities = torch.broadcast_to(
                        torch.eye(self.n_tasks, device=Sigma_orth.device),
                        (*self.shape_batch, self.n_tasks, self.n_tasks))
                    B_term = Sigma_orth.unsqueeze(-1) * (identities - Q @ Q.mT)
            else:
                if self.diag_Sig_orth:
                    B_term_root = Q_orth * Sigma_orth_root
                    if self._has_M_term:
                        Sigma_orth = (Sigma_orth_root ** 2)# shape (*batch_shape) x (n_tasks - n_lat)
                else:
                    B_term_root = Q_orth @ Sigma_orth_root
                    if self._has_M_term:
                        Sigma_orth = Sigma_orth_root @ Sigma_orth_root.mT
                B_term = B_term_root @ B_term_root.mT if not diag else (B_term_root**2).sum(dim=-1)
        else:
            B_term = 0.

        if self._has_M_term:
            if self.diag_Sig_orth:
                M_term = - QR @ ((sigma_p.unsqueeze(-1) * self.M) * Sigma_orth) @ Q_orth.mT
                extra_D_term_root = sigma_p.unsqueeze(-1) * self.M * Sigma_orth_root
            else:
                M_term = - QR @ (sigma_p.unsqueeze(-1) * self.M) @ Sigma_orth @ Q_orth.mT
                extra_D_term_root = sigma_p.unsqueeze(-1) * self.M @ Sigma_orth_root
            extra_D_term = extra_D_term_root @ extra_D_term_root.mT
            D_term_rotated = torch.diag_embed(sigma_p) + extra_D_term
            D_term = QR @ D_term_rotated @ QR.mT
            if diag:
                D_term = torch.diagonal(D_term, dim1=-2, dim2=-1)
        else:
            M_term = 0.
            D_term_root = QR * torch.sqrt(sigma_p.unsqueeze(-2))
            D_term = D_term_root @ D_term_root.mT if not diag else (D_term_root**2).sum(dim=-1)

        if diag:
            res = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.n_tasks, batch_shape=self.shape_batch,
                                                             rank=0, has_global_noise=False)
            if sigma_p.is_cuda:
                res.cuda()
            diag_M_term = torch.diagonal(M_term, dim1=-2, dim2=-1) if self._has_M_term else 0.
            diag_noises = B_term + D_term + 2 * diag_M_term
            res.task_noises = torch.clamp(diag_noises, min= 2 * self.noise_thresh)
        else:
            Mt_term = M_term.mT if self._has_M_term else 0.
            Sigma = D_term + M_term + Mt_term + B_term
            res = NonfactoredMultitaskGaussianLikelihood(num_tasks=self.n_tasks, batch_shape=self.shape_batch, task_noise_covar=Sigma)
            if sigma_p.is_cuda:
                res.cuda()

        return res


    @property
    def Sigma_orth_root( self )-> Tensor:
        """
        Outputs the root of the discarded noise factor Sigma_orth from the reference paper. 
        Returns:
            Root of discarded noise factor Sigma_orth (see reference paper), symmetric or diagonal matrix of size (n_tasks - n_latents).
            In the diagonal case, only the diagonal is returned (not its square embedding)
        """
        if self.n_latents < self.n_tasks:
            if self.diag_Sig_orth:
                res = torch.exp(self.log_Sigma_orth / 2)
            else:
                discarded_noise_size = self.n_tasks - self.n_latents
                identities = torch.broadcast_to(torch.eye(discarded_noise_size, device=self.Sigma_orth_inv_chol.device),
                                    (*self.shape_batch, discarded_noise_size, discarded_noise_size))
                res = torch.linalg.solve_triangular(self.Sigma_orth_inv_chol, identities, upper=False).mT # TOSEE : is this mT legit ?
        else:
            res = torch.Tensor()
        return res


    def forward( self, x:Tensor )-> gp.distributions.MultivariateNormal:  # ! forward only returns values of the latent processes !
        """
        Computes the prior distribution of the latent processes at the input locations. ! This does not return task-level values !
        Args:
            x: input data tensor
        Returns:
            A batched gp multivariate normal distribution representing latent processes values, which mean has shape n_latents x n_points.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


    def compute_latent_distrib( self, x:Tensor, **kwargs )-> gp.distributions.MultivariateNormal:
        """
        Outputs (distributional) posterior values of the latent processes at the input locations. This is the function which is called to compute
        the loss during training.
        Args:
            x: input data tensor

        Returns:
            A batched gp multivariate normal distribution representing latent processes values, which mean has shape n_latents x n_points.
        """
        proj_targets = self.project_data(self.train_y, last_data_dim_is_datapoint=self.last_target_dim_is_datapoint)
        super().set_train_data(inputs=self.train_inputs, targets=proj_targets, strict=False)
        batch_distrib = ExactGPModel.__call__(self, x, **kwargs)
        return batch_distrib  # shape (n_batch x) n_latents x n_points
    

    def compute_loo(self, output=None, latent=False) -> Tuple[Tensor, Tensor]:
        """
        Computes the leave-one-out (LOO) variance and error gaps (y_true - y_loo) values for the model.
        Args:
            output: the latent distribution of the model at the training points. If None, it is computed.
            latent: whether to compute the leave-one-out errors at the latent level (True) or at the task level (False). Default is False.
            train_y: the training labels. If None and latent=False, they must be stored in the model. Default is None.
        Returns:
            A tuple containing the LOO variances and error gaps for each task (each of size n_points x n_tasks, or n_points x n_latents if latent=True).
        """
        # TODO : adapt to batch case
        train_x, train_y = self.train_inputs[0], self.train_y
        with torch.no_grad():
            if output is None:
                output = self.compute_latent_distrib(train_x)
            K = self.likelihood(output).lazy_covariance_matrix
            y_proj = self.project_data(train_y)
            identity = torch.eye(*K.shape[-2:], dtype=K.dtype, device=K.device)
            L = K.cholesky(upper=False)
            loo_var = 1.0 / L._cholesky_solve(identity[None,:], upper=False).diagonal(dim1=-1, dim2=-2)
            loo_delta = L._cholesky_solve(y_proj.unsqueeze(-1), upper=False).squeeze(-1) * loo_var
            loo_var, loo_delta = loo_var.mT, loo_delta.mT
            if not latent:
                lmc_coeffs = self.lmc_coefficients()
                e_loo_raw = (loo_delta @ lmc_coeffs)
                diff = (self.train_y - y_proj.mT @ lmc_coeffs)
                loo_delta = e_loo_raw + diff
                loo_var = loo_var @ lmc_coeffs**2
        return loo_var, loo_delta


    def latent_variance( self, X:Tensor ) -> Tensor :
        output = self.likelihood(self.compute_latent_distrib(X))
        return output.variance.mT


    def task_variance (self, X:Tensor ) -> Tensor :
        output = self.full_likelihood()(self.__call__(X))
        return output.variance


    def set_train_data( self, inputs:Tensor, targets:Tensor, strict:bool=True, last_target_dim_is_datapoint:Union[bool, None]=None):
        """
        Replaces the current training data of the model. Overrides the parent method to store the training labels in the model.
        """
        if last_target_dim_is_datapoint is None:
            last_target_dim_is_datapoint = self.last_target_dim_is_datapoint
        projected_data = self.project_data(targets, last_data_dim_is_datapoint=last_target_dim_is_datapoint)
        super().set_train_data(inputs=inputs, targets=projected_data, strict=strict)
        self.train_y = targets if self.last_target_dim_is_datapoint == last_target_dim_is_datapoint else targets.mT
        if self.scalar_Sig_orth and not self._has_M_term :
            self.Y_squared_norm = torch.linalg.matrix_norm(self.train_y) ** 2


    def __call__(self, x:Tensor, **kwargs)-> gp.distributions.MultitaskMultivariateNormal:
        """
        Outputs the full posterior distribution of the model at input locations. This is used to make predictions.
        Args:
            x: input data tensor

        Returns:
            A multitask multivariate gp normal distribution representing task processes values, which mean has shape n_points x n_tasks.
        """
        if self.training: # in training mode, we just compute the prior distribution of latent processes
            return super().__call__(x, **kwargs)
        
        projected_data = self.project_data(self.train_y, last_data_dim_is_datapoint=self.last_target_dim_is_datapoint)
        super().set_train_data(inputs=self.train_inputs, targets=projected_data, strict=False)
        latent_dist = ExactGPModel.__call__(self, x, **kwargs)

        num_batch = len(latent_dist.batch_shape)
        latent_dim = num_batch + self.latent_dim

        num_dim = num_batch + len(latent_dist.event_shape)
        lmc_coefficients = self.lmc_coefficients().expand(*latent_dist.batch_shape, self.lmc_coefficients.size(-1))

        # Mean: ... x N x n_tasks
        latent_mean = latent_dist.mean.permute(*range(0, latent_dim), *range(latent_dim + 1, num_dim), latent_dim)
        mean = latent_mean @ lmc_coefficients.permute(
            *range(0, latent_dim), *range(latent_dim + 1, num_dim - 1), latent_dim, -1
        )

        # Covar: ... x (N x n_tasks) x (N x n_tasks)
        latent_covar = latent_dist.lazy_covariance_matrix
        lmc_factor = RootLinearOperator(lmc_coefficients.unsqueeze(-1))
        # latent_covar = to_linear_operator(latent_covar.evaluate())
        covar = KroneckerProductLinearOperator(latent_covar, lmc_factor).sum(latent_dim)
        covar = covar.add_jitter(self.jitter_val)

        return gp.distributions.MultitaskMultivariateNormal(mean, covar)
    
    def default_mll(self):
        return ProjectedLMCmll(self.likelihood, self)
    

class ProjectedLMCmll(gp.mlls.ExactMarginalLogLikelihood):
    """
    The loss function for the ProjectedGPModel. 
    """
    def __init__(self, latent_likelihood:Likelihood, model:ProjectedGPModel, last_target_dim_is_datapoint:Union[bool, None]=None):
        """

        Args:
            latent_likelihood: the likelihood of a ProjectedGPModel (batched gaussian likelihood of size n_latents)
            model: any ProjectedGPModel.

        Raises:
            RuntimeError: rejects non-gaussian likelihoods.
        """        
        if not isinstance(latent_likelihood, gp.likelihoods.gaussian_likelihood._GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super(ProjectedLMCmll, self).__init__(latent_likelihood, model)
        self.previous_lat = None
        if last_target_dim_is_datapoint is None:
            last_target_dim_is_datapoint = model.last_target_dim_is_datapoint
        self.last_target_dim_is_datapoint = last_target_dim_is_datapoint


    def forward(self, latent_function_dist:gp.distributions.Distribution, target:Tensor, inputs=None, *params) -> Tensor:
        """
        Computes the value of the loss (MLL) given the model predictions and the observed values at training locations. 
        Args:
            latent_function_dist: gp batched gaussian distribution of size n_latents x n_points representing the values of latent processes.
            target: training labels Y of shape n_points x n_tasks

        Raises:
            RuntimeError: rejects non-gaussian latent distributions.

        Returns:
            The (scalar) value of the MLL loss for this model and data.
        """        
        if not isinstance(latent_function_dist, gp.distributions.multivariate_normal.MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

        if self.last_target_dim_is_datapoint:
            target = target.mT
        num_data = latent_function_dist.event_shape.numel()
        
        # project the targets
        proj_target = self.model.project_data(target, last_data_dim_is_datapoint=False) # shape (n_batch x) n_latents x n_points

        # Get the log prob of the marginal distribution of latent processes
        latent_output = self.likelihood(latent_function_dist, *params) # shape (n_batch x) n_latents x n_points
        latent_res = latent_output.log_prob(proj_target)
        latent_res = self._add_other_terms(latent_res, params).sum() / num_data  # Scale by the amount of data we have

        # compute the part of likelihood lost by projection
        p, q = self.model.n_tasks, self.model.n_latents
        self.proj_term_list = [0]*3
        ## We store the additional terms in a list attribute in order to be able to plot them individually for testing
        Q, R, Q_orth = self.model.lmc_coefficients.QR()
        
        if not self.model._has_M_term and self.model.scalar_Sig_orth: # case of the PLMC-fast
            if self.model.log_Sigma_orth.numel() > 0:
                log_Sigma_orth = self.model.log_Sigma_orth
                scalar_Sigma_orth_inv = torch.exp(- log_Sigma_orth)
                log_Sigma_orth_root_diag = log_Sigma_orth / 2 * (p - q)
                self.proj_term_list[1] = ( scalar_Sigma_orth_inv * (self.model.Y_squared_norm - torch.linalg.matrix_norm(target @ Q) ** 2) ).sum()/ num_data
                # the parenthesis is the squared norm of the projection of target onto the space orthogonal to span(Q)
            else:
                self.proj_term_list[1] = 0.
                log_Sigma_orth_root_diag = torch.tensor([0.])
        else:
            if self.model.diag_Sig_orth:
                log_Sigma_orth_root_diag = self.model.log_Sigma_orth / 2
                rot_proj_scaled_target = target @ Q_orth * torch.exp(- log_Sigma_orth_root_diag)
            else:
                Sigma_orth_inv_root_diag = self.model.Sigma_orth_inv_chol[..., range(p-q), range(p-q)]
                log_Sigma_orth_root_diag = -torch.log(Sigma_orth_inv_root_diag)
                rot_proj_scaled_target = target @ Q_orth @ self.model.Sigma_orth_inv_chol
            self.proj_term_list[1] = torch.linalg.matrix_norm(rot_proj_scaled_target) ** 2 / num_data

        # All terms are implicitly or explicitly divided by the number of datapoints
        self.proj_term_list[0] = 2 * torch.sum(log_Sigma_orth_root_diag) # factor 2 because of the use of a root

        if self.model.lmc_coefficients.use_QR_decomp:
            self.proj_term_list[2] = torch.log(R[..., range(q), range(q)]**2).sum() # keep the square in the log because the quantity can be negative
        else:
            self.proj_term_list[2] = 2 * self.model.lmc_coefficients.parametrizations.R.original[..., range(q), range(q)].sum()
            # The diagonal of R is already parametrized with its logarithm (UpperTriangularParam or PositiveDiagParam)
        projection_term = sum(self.proj_term_list) + (p - q) * np.log(2*np.pi)

        self.latent_res = latent_res
        res = latent_res - 0.5 * projection_term
        return res
