import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Tuple

blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
)

blku = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.Conv2d(oc, oc, 5, padding=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(oc, oc, 2, stride=2),
    nn.GroupNorm(oc // 8, oc),
    nn.LeakyReLU(),
)

class DummyX0Model(nn.Module):
    def __init__(self, n_channel: int, N: int = 16, num_timesteps: int = 1000) -> None:
        super(DummyX0Model, self).__init__()
        self.down1 = blk(n_channel, 16)
        self.down2 = blk(16, 32)
        self.down3 = blk(32, 64)
        self.down4 = blk(64, 512)
        self.down5 = blk(512, 512)
        self.up1 = blku(512, 512)
        self.up2 = blku(512 + 512, 64)
        self.up3 = blku(64, 32)
        self.up4 = blku(32, 16)
        self.convlast = blk(16, 16)
        self.final = nn.Conv2d(16, N * n_channel, 1, bias=False)

        self.tr1 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tr2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tr3 = nn.TransformerEncoderLayer(d_model=64, nhead=8)

        self.cond_embedding_1 = nn.Embedding(10, 16)
        self.cond_embedding_2 = nn.Embedding(10, 32)
        self.cond_embedding_3 = nn.Embedding(10, 64)
        self.cond_embedding_4 = nn.Embedding(10, 512)
        self.cond_embedding_5 = nn.Embedding(10, 512)
        self.cond_embedding_6 = nn.Embedding(10, 64)

        self.temb_1 = nn.Linear(32, 16)
        self.temb_2 = nn.Linear(32, 32)
        self.temb_3 = nn.Linear(32, 64)
        self.temb_4 = nn.Linear(32, 512)
        self.N = N
        self.num_timesteps = num_timesteps

    def forward(self, x, t, cond) -> torch.Tensor:
        # Handle Soft Samples from FLDD Warmup (Shape: B, C, H, W, N)
        # If x has 5 dimensions, it is a soft probability distribution.
        # We collapse it to a scalar "pixel intensity" using expected value.
        if x.dim() == 5:
            # Create indices [0, 1, ..., N-1]
            indices = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)
            # Sum(Probability * Index) -> Weighted Average (Expected Value)
            x = (x * indices).sum(dim=-1)

        # Note: x can be float (soft sample) or int (hard sample)
        # We assume x is roughly in range [0, N-1]
        if x.dtype == torch.long or x.dtype == torch.int:
            x = (2 * x.float() / (self.N - 1)) - 1.0
        else:
            x = (2 * x) - 1.0
            
        if x.min() < -1.0 or x.max() > 1.0:
            print("Warning: x out of range [-1, 1]")
        t = t.float().reshape(-1, 1) / self.num_timesteps
        t_features = [torch.sin(t * 3.1415 * 2**i) for i in range(16)] + [
            torch.cos(t * 3.1415 * 2**i) for i in range(16)
        ]
        tx = torch.cat(t_features, dim=1).to(x.device)

        t_emb_1 = self.temb_1(tx).unsqueeze(-1).unsqueeze(-1)
        t_emb_2 = self.temb_2(tx).unsqueeze(-1).unsqueeze(-1)
        t_emb_3 = self.temb_3(tx).unsqueeze(-1).unsqueeze(-1)
        t_emb_4 = self.temb_4(tx).unsqueeze(-1).unsqueeze(-1)

        cond_emb_1 = self.cond_embedding_1(cond).unsqueeze(-1).unsqueeze(-1)
        cond_emb_2 = self.cond_embedding_2(cond).unsqueeze(-1).unsqueeze(-1)
        cond_emb_3 = self.cond_embedding_3(cond).unsqueeze(-1).unsqueeze(-1)
        cond_emb_4 = self.cond_embedding_4(cond).unsqueeze(-1).unsqueeze(-1)
        cond_emb_5 = self.cond_embedding_5(cond).unsqueeze(-1).unsqueeze(-1)
        cond_emb_6 = self.cond_embedding_6(cond).unsqueeze(-1).unsqueeze(-1)

        x1 = self.down1(x) + t_emb_1 + cond_emb_1
        x2 = self.down2(nn.functional.avg_pool2d(x1, 2)) + t_emb_2 + cond_emb_2
        x3 = self.down3(nn.functional.avg_pool2d(x2, 2)) + t_emb_3 + cond_emb_3
        x4 = self.down4(nn.functional.avg_pool2d(x3, 2)) + t_emb_4 + cond_emb_4
        x5 = self.down5(nn.functional.avg_pool2d(x4, 2))

        x5 = (
            self.tr1(x5.reshape(x5.shape[0], x5.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(x5.shape)
        )

        y = self.up1(x5) + cond_emb_5

        y = (
            self.tr2(y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(y.shape)
        )

        y = self.up2(torch.cat([x4, y], dim=1)) + cond_emb_6

        y = (
            self.tr3(y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2))
            .transpose(1, 2)
            .reshape(y.shape)
        )
        y = self.up3(y)
        y = self.up4(y)
        y = self.convlast(y)
        y = self.final(y)

        # reshape to B, C, H, W, N
        y = (
            y.reshape(y.shape[0], -1, self.N, *x.shape[2:])
            .transpose(2, -1)
            .contiguous()
        )

        return y


class ForwardProcess(nn.Module, ABC):
    def __init__(self, num_classes: int, num_timesteps: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

    @abstractmethod
    def sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples x_t given x_0 and t."""
        pass

    @abstractmethod
    def compute_posterior_logits(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns logits for q(x_{t-1} | x_t, x_0)."""
        pass

    def get_auxiliary_loss(self, x_0, x_t, t, kl_div=None, log_probs=None):
        """Returns 0.0 by default or specific auxiliary losses (e.g. for FLDD)."""
        return 0.0

    def step_schedule(self, global_step: int):
        """Updates internal state like temperature based on training step."""
        pass

    @property
    def input_num_classes(self) -> int:
        """Returns the number of classes the model should expect (e.g., K or K+1)."""
        return self.num_classes


class MarkovianForwardProcess(ForwardProcess):
    def __init__(self, num_classes: int, num_timesteps: int):
        super().__init__(num_classes, num_timesteps)
        self.eps = 1e-6
        # These buffers must be registered by subclasses after construction
        # self.register_buffer("q_mats", ...)
        # self.register_buffer("q_one_step_transposed", ...)

    def _at(self, a, t, x):
        # t is 1-d, x is integer value of 0 to num_classes - 1
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
        return a[t - 1, x, :]

    def sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # forward process, x_0 is the clean input.
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1), torch.Tensor(())

    def compute_posterior_logits(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # if x_0 is integer, we convert it to one-hot.
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.input_num_classes) + self.eps
            )
        else:
            x_0_logits = x_0.clone()
            if x_0_logits.shape[-1] != self.input_num_classes:
                diff = self.input_num_classes - x_0_logits.shape[-1]
                padding = torch.ones((*x_0_logits.shape[:-1], diff), device=x_0_logits.device) * -1e9
                x_0_logits = torch.cat([x_0_logits, padding], dim=-1)

        # fact1 is "guess of x_{t-1}" from x_t
        # fact2 is "guess of x_{t-1}" from x_0
        fact1 = self._at(self.q_one_step_transposed, t, x_t)

        softmaxed = torch.softmax(x_0_logits, dim=-1)
        qmats2 = self.q_mats[t - 2].to(dtype=softmaxed.dtype)
        # bs, num_classes, num_classes
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        
        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        
        bc = torch.where(t_broadcast == 1, x_0_logits, out)
        return bc

class Uniform(MarkovianForwardProcess):
    def __init__(self, num_classes: int, num_timesteps: int):
        super().__init__(num_classes, num_timesteps)
        
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        self.beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        )

        q_onestep_mats = []
        for beta in self.beta_t:
            mat = torch.ones(num_classes, num_classes) * beta / num_classes
            mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
            q_onestep_mats.append(mat)
            
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)
        q_one_step_transposed = q_one_step_mats.transpose(1, 2)

        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, num_timesteps):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)

        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)


class Masking(MarkovianForwardProcess):
    def __init__(self, num_classes: int, num_timesteps: int):
        super().__init__(num_classes, num_timesteps)
        
        # Masking increases vocabulary size by 1 (last index is MASK)
        self.mask_token_idx = num_classes
        
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        self.beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        )

        # Dimension is K+1 x K+1
        dim = num_classes + 1
        q_onestep_mats = []
        
        for beta in self.beta_t:
            mat = torch.zeros(dim, dim)
            # Transition to mask with prob beta
            mat[:, self.mask_token_idx] = beta
            # Self loop with prob 1 - beta
            mat.diagonal().fill_(1 - beta)
            # Mask token stays mask token
            mat[self.mask_token_idx, :] = 0.0
            mat[self.mask_token_idx, self.mask_token_idx] = 1.0
            
            q_onestep_mats.append(mat)

        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)
        q_one_step_transposed = q_one_step_mats.transpose(1, 2)

        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, num_timesteps):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)

        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

    @property
    def input_num_classes(self) -> int:
        return self.num_classes + 1

class FLDD(ForwardProcess):
    """
    Forward-Learned Discrete Diffusion (https://openreview.net/pdf?id=45EtKUdgbJ)
    Implements Learnable Forward Process (Sec 3.1), Maximum Coupling (Sec 3.2),
    and REINFORCE/Relaxation training (Sec 3.3).
    """
    def __init__(self, num_classes: int, num_timesteps: int, forward_net: nn.Module):
        super().__init__(num_classes, num_timesteps)
        self.forward_net = forward_net
        self.eps = 1e-6
        
        # Training state
        self.warmup_steps = 0
        self.tau_start = 1.0
        self.tau_end = 1e-3
        self.current_tau = self.tau_start
        self.use_reinforce = False
        
        # Baseline for REINFORCE
        self.moving_avg_baseline = 0.0
        self.baseline_decay = 0.95

    def step_schedule(self, global_step: int):
        if global_step < self.warmup_steps:
            self.use_reinforce = False
            frac = global_step / self.warmup_steps
            # Exponential decay
            self.current_tau = self.tau_start * (self.tau_end / self.tau_start) ** frac
        else:
            self.use_reinforce = True
            self.current_tau = self.tau_end

    def _log_linear_mix(self, logits_a: torch.Tensor, logits_b: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Computes log( (1 - w) * exp(a) + w * exp(b) ).
        Handles normalization, broadcasting, and boundary conditions (w=0, w=1).
        """
        # Ensure weight matches logits dimension structure
        if weight.ndim < logits_a.ndim:
            weight = weight.view(*weight.shape, *([1] * (logits_a.ndim - weight.ndim)))

        # We use clamping to avoid log(0) -> -inf producing NaNs in gradients, 
        w_safe = torch.clamp(weight, 1e-6, 1.0 - 1e-6)
        log_w = torch.log(w_safe)
        log_1_minus_w = torch.log(1.0 - w_safe)
        
        mix_logits = torch.logaddexp(
            logits_a + log_1_minus_w,
            logits_b + log_w
        )
        mix_logits = torch.where(weight < 1e-6, logits_a, mix_logits)
        mix_logits = torch.where(weight > 1 - 1e-6, logits_b, mix_logits)
            
        return mix_logits

    def _get_marginals(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Computes q_phi(z_t | x) using the forward network. Expects x (B, ...) as indices."""
        if t.dim() == 0:
            t = t.reshape(1)
        
        # The forward net might need dummy conditioning if it's the same class as reverse model
        bs = x.shape[0]
        dummy_cond = torch.zeros(bs, dtype=torch.long, device=x.device)
        raw_net_logits = self.forward_net(x, t, dummy_cond)
        log_net_probs = F.log_softmax(raw_net_logits, dim=-1)
        t_b = t.view(-1, *[1]*(log_net_probs.dim() - 1))
        t_norm = t_b.float() / self.num_timesteps
        control_ratio = (2*t_norm -1).pow(2) # The closer we are to the edges, the more similar we are to uniform or delta
        
        if log_net_probs.dim() != x.dim():
            x_0_log_probs = torch.log(F.one_hot(x, num_classes=self.num_classes).float() + self.eps)
        else:
            # x must already be a probability distribution
            x_0_log_probs = torch.log(x)
        uniform_log_probs = torch.log(torch.ones_like(log_net_probs) / self.num_classes) # We assert that num_classes > 0
        control_log_probs = self._log_linear_mix( x_0_log_probs, uniform_log_probs, t_norm)
        return self._log_linear_mix(log_net_probs, control_log_probs, control_ratio)
    
    def sample(self, x_0, t, noise):
        """
        Samples z_t from q_phi(z_t | x).
        Returns z_t and auxiliary info (log_prob).
        """
        assert (t > 0).all(), "t must be >= 1"
        logits_t = self._get_marginals(x_0, t)
        
        # Explicit Boundary Conditions 
        # q(z_0|x) = delta(x). This is handled by t=1 logic in posteriors,
        # but for sampling z_0 specifically we would just return x_0.
        # Here we sample for general t.

        if self.training and not self.use_reinforce:
            # Warm-up: Relaxed samples (Concrete Distribution) 
            # Returns Soft samples (B, ..., K)
            z_t_soft = F.gumbel_softmax(logits_t, tau=self.current_tau, hard=False, dim=-1)
            return z_t_soft, None
        else:
            # Standard Discrete Sampling (or REINFORCE phase)
            noise = torch.clip(noise, self.eps, 1.0)
            gumbel_noise = -torch.log(-torch.log(noise))
            z_t_hard = torch.argmax(logits_t + gumbel_noise, dim=-1)
            
            # Calculate log_prob for REINFORCE 
            log_probs = F.log_softmax(logits_t, dim=-1)
            selected_log_prob = torch.gather(log_probs, -1, z_t_hard.unsqueeze(-1)).squeeze(-1)
            
            return z_t_hard, selected_log_prob

    def _compute_coupling(self, u_s, u_t):
        """
        Implements Maximum Coupling Transport (Eq 11).
        u_s, u_t: Probability vectors (B, ..., K)
        Returns: Transition matrices Q such that Q[b, j, k] = q(z_s=j | z_t=k)
        """
        B_dims = u_t.shape[:-1]
        K = u_t.shape[-1]
        
        # m_{s|t} (Deficit distribution) 
        diff = F.relu(u_s - u_t)
        norm = diff.sum(dim=-1, keepdim=True) + self.eps
        m_st = diff / norm # (B, ..., K)
        
        # Prob stay: min(u_s, u_t) / u_t 
        prob_stay = torch.minimum(u_s, u_t) / (u_t + self.eps) # (B, ..., K)
        
        # Prob move: (u_t - u_s)+ / u_t 
        # Note: If u_t > u_s, then u_t - u_s is positive.
        prob_move_source = F.relu(u_t - u_s) / (u_t + self.eps) # (B, ..., K)
        
        # Construct full transition matrix for Eq 14 (Warmup) or general use
        # T[k, j] = P(z_s = j | z_t = k)
        # Using broadcasting:
        # T[..., k, j] = delta_kj * prob_stay[..., k] + prob_move_source[..., k] * m_st[..., j]
        
        # Diagonal part
        eye = torch.eye(K, device=u_t.device).view(*([1]*len(B_dims)), K, K)
        term1 = eye * prob_stay.unsqueeze(-1) # (B, ..., K, K) where last dim is j
        
        # Off-diagonal redistribution part
        term2 = prob_move_source.unsqueeze(-1) * m_st.unsqueeze(-2) # (B, ..., K_from, K_to)
        
        Q = term1 + term2
        Q = Q / (Q.sum(dim=-1, keepdim=True) + self.eps)
        return Q

    def compute_posterior_logits(self, x_0, x_t, t):
        """
        Computes log q(z_s | z_t, x).
        Handles both discrete x_t (REINFORCE) and soft x_t (Warmup/Eq 14).
        """
        if x_0.dtype == torch.long or x_0.dtype == torch.int:
            # Case 1: Ground Truth. Convert indices to one-hot distribution.
            x_0_input = x_0
            x_0_dist = F.one_hot(x_0, self.num_classes).float()
        else:
            # Case 2: Prediction. Convert logits to probability distribution.
            # We must Softmax here so that _get_marginals receives valid probs
            # and the t=1 boundary condition gets a valid distribution.
            x_0_dist = torch.softmax(x_0, dim=-1)
            x_0_input = x_0_dist
        
        # 1. Get Marginals u_t and u_s 
        logits_t = self._get_marginals(x_0_input, t)
        
        # For s = t-1. If t=1, q(z_0|x) is delta(x). 
        # We simulate this by making u_s very sharp around x_0.
        # However, to be differentiable for phi, we calculate u_s normally for s>0.
        # For t=1, we enforce boundary condition manually later.
        logits_s = self._get_marginals(x_0_input, t - 1)
        
        u_t = torch.softmax(logits_t, dim=-1)
        u_s = torch.softmax(logits_s, dim=-1)

        # 2. Compute Full Coupling Matrix Q (z_s | z_t)
        Q = self._compute_coupling(u_s, u_t) # (B, ..., K_from, K_to)
        
        # 3. Select posterior based on x_t
        if x_t.dim() == logits_t.dim(): # Soft Sample (Warmup)
            # x_t is (B, ..., K). We perform matrix multiplication.
            # Q is (B, ..., K_t, K_s)
            # x_t is (B, ..., K_t) -> unsqueeze to (B, ..., 1, K_t)
            # Result: (B, ..., K_s)
            
            # Weighted average of posteriors
            posterior_dist = torch.einsum("...k,...kj->...j", x_t, Q)
            
        else: # Discrete Sample (REINFORCE) 
            # x_t is indices (B, ...). Gather rows from Q.
            # Q shape: (B, H, W, K, K)
            # x_t shape: (B, H, W) -> unsqueeze to (B, H, W, 1, 1) and expand
            
            # Flatten spatial dims for gather.
            flat_Q = Q.view(-1, self.num_classes, self.num_classes)
            flat_xt = x_t.view(-1)
            
            # Select the row corresponding to x_t for each batch element
            # shape (B*..., K_s)
            posterior_dist = flat_Q[torch.arange(flat_Q.shape[0]), flat_xt]
            
            # Reshape back
            posterior_dist = posterior_dist.view(*x_t.shape, self.num_classes)

        return torch.log(posterior_dist + self.eps)

    def get_auxiliary_loss(self, x_0, x_t, t, kl_div=None, log_probs=None):
        """
        Computes gradients for phi.
        Warmup: Standard backprop through soft samples (handled by autograd).
        REINFORCE: Eq 13 estimator.
        """
        if self.use_reinforce:
            # REINFORCE estimator: grad approx (KL - b) * grad_log_q(z_t|x)
            # But we also need the gradient through the posterior parameters inside KL.
            # The "Magic Box" trick (prob / prob.detach()) * cost handles both:
            # Differentiate E_q[C] -> grad_q * C + q * grad_C.
            
            if log_probs is None or kl_div is None:
                return 0.0

            # Aggregate log_prob and KL to match shapes (sum over spatial dims)
            # z_t_log_prob is (B, H, W), kl_per_element is (B*H*W)
            # We assume x_0 corresponds to 'x' from forward call, used for shape inference if needed
            
            flat_log_prob = log_probs.flatten(start_dim=1).mean(dim=1) # (B,)
            
            # For variance reduction, we treat each pixel/dimension? 
            # Standard REINFORCE usually sums rewards per sample.
            # KL per sample:
            kl_per_sample = kl_div.view(x_0.shape[0], -1).mean(dim=1) # (B,)
            
            # Update baseline
            with torch.no_grad():
                self.moving_avg_baseline = (
                    self.baseline_decay * self.moving_avg_baseline + 
                    (1 - self.baseline_decay) * kl_per_sample.mean()
                )
                baseline = self.moving_avg_baseline

            # REINFORCE Gradient Estimator
            reinforce_loss = (flat_log_prob * (kl_per_sample.detach() - baseline)).mean()
            
            return reinforce_loss
            
        return 0.0

# --- Training Loop & Model Wrapper ---

class DiscreteDiffusion(nn.Module):
    def __init__(
        self,
        x0_model: nn.Module,
        num_timesteps: int,
        num_classes: int = 10,
        forward_process: ForwardProcess = None,
        hybrid_loss_coeff=0.001,
    ) -> None:
        super(DiscreteDiffusion, self).__init__()
        self.x0_model = x0_model
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.hybrid_loss_coeff = hybrid_loss_coeff
        
        # Use provided forward process or default to Uniform
        if forward_process is None:
            self.forward_process = Uniform(num_classes, num_timesteps)
        else:
            self.forward_process = forward_process

        self.eps = 1e-6

    def vb(self, dist1, dist2):
        # flatten dist1 and dist2
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)

        out = torch.softmax(dist1 + self.eps, dim=-1) * (
            torch.log_softmax(dist1 + self.eps, dim=-1)
            - torch.log_softmax(dist2 + self.eps, dim=-1)
        )
        return out.sum(dim=-1) # Return per-pixel KL, not mean yet

    def model_predict(self, x_0, t, cond):
        predicted_x0_logits = self.x0_model(x_0, t, cond)
        return predicted_x0_logits

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, global_step: int = 0) -> torch.Tensor:
        """
        Forward pass with support for FLDD REINFORCE and Warmup.
        """
        self.forward_process.step_schedule(global_step)
        
        t = torch.randint(1, self.num_timesteps, (x.shape[0],), device=x.device)
        
        # 1. Sample z_t
        # forward_process.sample returns (sample, log_prob) or (soft_sample, None)
        x_t, z_t_log_prob = self.forward_process.sample(
            x, t, torch.rand((*x.shape, self.forward_process.input_num_classes), device=x.device)
        )
        
        # 2. Predict x_0
        predicted_x0_logits = self.model_predict(x_t, t, cond)

        # 3. Calculate True and Predicted Posteriors
        true_q_posterior_logits = self.forward_process.compute_posterior_logits(x, x_t, t)
        pred_q_posterior_logits = self.forward_process.compute_posterior_logits(predicted_x0_logits, x_t, t)

        # 4. Variational Bound (Diffusion Loss) 
        kl_per_element = self.vb(true_q_posterior_logits, pred_q_posterior_logits)
        vb_loss = kl_per_element.mean()
        
        # 5. Auxiliary / REINFORCE Loss 
        aux_loss = self.forward_process.get_auxiliary_loss(
            x, x_t, t, kl_div=kl_per_element, log_probs=z_t_log_prob
        )

        # Reconstruction Loss (Cross Entropy)
        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x_flat = x.flatten(start_dim=0, end_dim=-1)

        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x_flat)

        total_loss = self.hybrid_loss_coeff * vb_loss + ce_loss + aux_loss
        
        return total_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
            "aux_loss": aux_loss if isinstance(aux_loss, float) else aux_loss.detach().item()
        }

    def p_sample(self, x, t, cond, noise):
        predicted_x0_logits = self.model_predict(x, t, cond)
        # Use the forward process's logic to compute the transition probability p_theta(x_{t-1}|x_t)
        # This works because p_theta is parameterized to match q(x_{t-1}|x_t, \hat{x}_0)
        pred_q_posterior_logits = self.forward_process.compute_posterior_logits(predicted_x0_logits, x, t)

        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))
        gumbel_noise = -torch.log(-torch.log(noise))
        
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample

    def sample(self, x, cond=None):
        for t in reversed(range(1, self.num_timesteps)):
            t_tensor = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(
                x, t_tensor, cond, torch.rand((*x.shape, self.forward_process.input_num_classes), device=x.device)
            )
        return x

    def sample_with_image_sequence(self, x, cond=None, stride=10):
        steps = 0
        images = []
        for t in reversed(range(1, self.num_timesteps)):
            t_tensor = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(
                x, t_tensor, cond, torch.rand((*x.shape, self.forward_process.input_num_classes), device=x.device)
            )
            steps += 1
            if steps % stride == 0:
                images.append(x)

        if steps % stride != 0:
            images.append(x)
        return images


if __name__ == "__main__":
    N = 2  # number of classes per pixel
    T = 4

    EXP_DIR = "runs/jan/22/fldd_gemini_fix"
    
    # Example 1: Uniform (Equivalent to original D3PM)
    # forward_proc = Uniform(num_classes=N, num_timesteps=T)
    
    # Example 2: Masking
    # forward_proc = Masking(num_classes=N, num_timesteps=T)
    
    # Example 3: FLDD
    # We need a separate network for FLDD. We reuse DummyX0Model structure.
    # Note: DummyX0Model outputs (N * n_channel) channels. 
    # For FLDD marginals q(z_t|x), we need output size N.
    fldd_net = DummyX0Model(n_channel=1, N=N, num_timesteps=T)
    forward_proc = FLDD(num_classes=N, num_timesteps=T, forward_net=fldd_net)

    # Initialize Model
    # Note: If using Masking, num_classes for x0_model should be N+1 potentially?
    # The x0_model needs to accept 'input_num_classes'. 
    # The DummyX0Model implementation provided hardcodes convolutions and doesn't explicitly 
    # check input channels against N, but it does `x = (2 * x.float() / self.N)`.
    # We pass `forward_proc.input_num_classes` to the model init if we want it to handle resizing,
    # but the provided DummyX0Model takes `n_channel` (input image channels, usually 1 for MNIST)
    # and `N` (quantization levels).
    
    x0_model = DummyX0Model(n_channel=1, N=N, num_timesteps=T)
    
    model = DiscreteDiffusion(
        x0_model=x0_model,
        num_timesteps=T,
        num_classes=N,
        forward_process=forward_proc,
        hybrid_loss_coeff=0.01
    ).cuda()
    
    # If FLDD, we also need to optimize the forward_net.
    # We collect parameters from both if FLDD.
    params = list(model.x0_model.parameters())
    if isinstance(forward_proc, FLDD):
        params += list(forward_proc.forward_net.parameters())
        model.forward_process.forward_net.cuda()

    print(f"Total Param Count: {sum([p.numel() for p in params])}")
    
    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    optim = torch.optim.AdamW(params, lr=2e-4) # LR from paper
    model.train()

    n_epoch = 5
    device = "cuda"

    global_step = 0
    for i in range(n_epoch):
        pbar = tqdm(dataloader)
        loss_ema = None
        for x, cond in pbar:
            optim.zero_grad()
            x = x.to(device)
            cond = cond.to(device)

            # discretize x to N bins
            x = (x * (N - 1)).round().long().clamp(0, N - 1)
            
            # Pass global_step for scheduling
            loss, info = model(x, cond, global_step=global_step)

            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 0.1)
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()
            
            pbar.set_description(
                f"loss: {loss_ema:.4f}, norm: {norm:.4f}, vb: {info['vb_loss']:.4f}, ce: {info['ce_loss']:.4f}, aux: {info['aux_loss']:.4f}"
            )
            optim.step()
            global_step += 1

            if global_step % 300 == 1:
                model.eval()
                with torch.no_grad():
                    # Sampling logic
                    cond_sample = torch.arange(0, 4).cuda() % 10
                    # Initial noise depends on input_num_classes (N for Uniform, N+1 for Masking)
                    init_dim = model.forward_process.input_num_classes
                    init_noise = torch.randint(0, init_dim, (4, 1, 32, 32)).cuda()

                    images = model.sample_with_image_sequence(
                        init_noise, cond_sample, stride=1
                    )
                    
                    # image sequences to gif
                    gif = []
                    for image in images:
                        # Normalize for visualization (handle mask token if exists by clamping)
                        viz_img = image.float().clamp(0, N-1) / (N - 1)
                        x_as_image = make_grid(viz_img, nrow=2)
                        img = x_as_image.permute(1, 2, 0).cpu().numpy()
                        img = (img * 255).astype(np.uint8)
                        gif.append(Image.fromarray(img))

                    # Ensure directory exists or handle path
                    import os
                    os.makedirs(f"{EXP_DIR}/results", exist_ok=True)
                    
                    gif[0].save(
                        f"{EXP_DIR}/results/sample_{global_step}.gif",
                        save_all=True,
                        append_images=gif[1:],
                        duration=100,
                        loop=0,
                    )
                    gif[-1].save(f"{EXP_DIR}/results/sample_{global_step}_last.png")

                model.train()