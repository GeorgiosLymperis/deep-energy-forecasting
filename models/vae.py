"""
Variational Autoencoder (VAE) models and training utilities for conditional
probabilistic energy forecasting.

This module provides a complete implementation of several VAE components:
    - ELBO loss module with β-annealing
    - Gaussian Encoder and Decoder networks
    - Conditional Prior network
    - BetaScheduler for KL warm-up (linear, cosine, sigmoid, constant)
    - Generic training loop (train_vae)
    - Fully-connected conditional VAE implementation (VAElinear)
    - Training routine with early stopping (train_vae_linear)
    - Optuna objective for hyperparameter optimization (vae_objective)

The architecture supports conditional generative modeling:
    x ~ pθ(x | z, c)
    z ~ pψ(z | c)
    where c represents exogenous features (e.g., weather conditions).

All models follow the PyTorch interface and expose:
    - loss(x, c, beta): compute ELBO components
    - forward(x, c): reconstruct inputs
    - sample(n, c): generate n probabilistic forecasts per context
    - quantiles(c, q, n): estimate predictive quantiles from samples

Expected shapes:
    x:          (B, x_dim)
    c:          (B, c_dim)
    latent z:   (B, latent_dim)
    samples:    (S, B, x_dim)
    quantiles:  (len(q), B, x_dim)

Dependencies:
    - PyTorch
    - Zuko for LazyDistribution and flow abstractions
    - NumPy
    - Optuna (optional, for hyperparameter tuning)

Use cases:
    - Probabilistic time-series forecasting (wind, solar, load)
    - Conditional density modeling
    - Generative scenario simulation
    - Benchmarking against GANs and Normalizing Flows

This module is part of a unified probabilistic forecasting framework
containing GAN, NF, and VAE models, sharing a consistent interface for
evaluation, visualization, and scoring.
"""

import math
import copy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.distributions import Independent, Normal
import torch.nn.functional as F
import zuko
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna

from evaluation.metrics import crps_batch_per_marginal


def ensure_batch_context(x: Tensor, c: Union[Tensor, np.ndarray, list, float, int]) -> Tensor:
    """Ensure context `c` matches the batch dimension of `x`.

    This normalizes context into shape ``(batch, c_dim)`` depending on input:

    Accepted shapes:
        - ``(batch,)``      → ``(batch, 1)``
        - ``(batch, 1)``    → unchanged
        - ``(1, c_dim)``    → expand to ``(batch, c_dim)``
        - ``(batch, c_dim)``→ unchanged
        - ``(batch, *, *)`` → flattened to ``(batch, c_dim)``

    Args:
        x (Tensor): Input batch whose batch dimension is used.
        c (Tensor or array-like): Context tensor or array-like input.

    Returns:
        Tensor: Context of shape ``(batch, c_dim)``.

    Raises:
        ValueError: If context shape is incompatible with batch size.
    """
    b = x.size(0)
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c, dtype=x.dtype, device=x.device)

    if c.dim() == 1:
        if c.size(0) != b:
            raise ValueError(f"c has shape {tuple(c.shape)} but batch is {b}")
        return c.view(b, 1).to(x.device, dtype=x.dtype)

    if c.dim() == 2:
        if c.size(0) == 1 and b > 1:
            return c.expand(b, c.size(1)).to(x.device, dtype=x.dtype)
        if c.size(0) != b:
            raise ValueError(f"c batch {c.size(0)} != {b}")
        return c.to(x.device, dtype=x.dtype)

    if c.dim() == 3:
        return c.view(b, -1).to(x.device, dtype=x.dtype)

    raise ValueError(f"Unsupported c shape: {tuple(c.shape)}")


class ELBO(nn.Module):
    """Evidence Lower Bound (ELBO) module for conditional VAEs.

    This wraps an encoder, decoder, and prior into a trainable ELBO objective
    using a configurable beta-scheduling strategy.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        prior: nn.Module,
        warmup_steps: int = 50_000,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        mode: str = "linear",
    ) -> None:
        """Initialize ELBO module.

        Args:
            encoder (nn.Module): Encoder returning q(z|x,c).
            decoder (nn.Module): Decoder returning p(x|z,c).
            prior (nn.Module): Prior distribution p(z|c).
            warmup_steps (int): Beta scheduler warmup length.
            beta_start (float): Initial KL weight.
            beta_end (float): Final KL weight.
            mode (str): One of {"linear", "cosine", "sigmoid", "constant"}.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.beta_sched = BetaScheduler(
            warmup_steps=warmup_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            mode=mode,
        )

    @property
    def beta(self) -> float:
        """float: Current beta in KL annealing schedule."""
        return self.beta_sched.beta()

    def step_beta(self) -> None:
        """Increment beta scheduler by one step."""
        self.beta_sched.update()

    def reset_beta(self) -> None:
        """Reset beta scheduler to step 0."""
        self.beta_sched.reset()

    def forward(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Compute ELBO loss and metrics.

        Args:
            x (Tensor): Observed data of shape (B, D).
            c (Tensor): Context of shape (B, C).

        Returns:
            Tuple[Tensor, Dict[str, float]]:  
                - Loss scalar (Tensor)  
                - Logging dict with keys {"beta", "elbo", "recon", "kl"}.
        """
        q = self.encoder(x, c)
        z = q.rsample()

        log_px = self.decoder(z, c).log_prob(x)
        log_pz = self.prior(c).log_prob(z)
        log_qz = q.log_prob(z)

        beta = self.beta
        elbo = log_px + beta * (log_pz - log_qz)
        loss = -elbo.mean()

        return loss, {
            "beta": beta,
            "elbo": float(elbo.mean().item()),
            "recon": float(log_px.mean().item()),
            "kl": float((log_qz - log_pz).mean().item()),
        }


class Encoder(zuko.lazy.LazyDistribution):
    """Gaussian encoder q(z|x,c) implemented with an MLP."""

    def __init__(
        self,
        z_dim: int,
        c_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize Gaussian encoder.

        Args:
            z_dim (int): Latent dimension.
            c_dim (int): Context dimension.
            x_dim (int): Input dimension.
            hidden_dim (int): Hidden layer size.
            device (torch.device, optional): Device for parameters.
        """
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(x_dim + c_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * z_dim),
        )

    def forward(self, x: Tensor, c: Union[Tensor, Sequence, float, int]) -> Independent:
        """Return q(z|x,c) as a Normal distribution.

        Args:
            x (Tensor): Input of shape (B, x_dim).
            c (Tensor or array-like): Context.

        Returns:
            Independent: A multivariate diagonal Normal distribution.
        """
        c = ensure_batch_context(x, c)
        h = self.net(torch.cat([x, c], dim=1))
        mu, raw = h.chunk(2, dim=-1)
        std = F.softplus(raw) + 1e-6
        return Independent(Normal(mu, std), 1)


class Decoder(zuko.lazy.LazyDistribution):
    """Gaussian decoder p(x|z,c) with heteroscedastic or fixed variance."""

    def __init__(
        self,
        z_dim: int,
        c_dim: int,
        x_dim: int,
        hidden_dim: int = 128,
        fixed_logvar: Optional[Union[bool, float, Sequence[float]]] = None,
        var_mode: str = "tanh",
        sigma_min: float = 1e-3,
        sigma_max: Optional[float] = 1.0,
    ) -> None:
        """Initialize the decoder.

        Args:
            z_dim (int): Latent dimension.
            c_dim (int): Context dimension.
            x_dim (int): Output dimension.
            hidden_dim (int): Hidden layer size.
            fixed_logvar (bool or float or list, optional):  
                - None: learn per-sample logvar  
                - True: learn global logvar per dimension  
                - float/list: fixed logvar  
            var_mode (str): {"tanh", "softplus"} mapping for variance.
            sigma_min (float): Minimum standard deviation.
            sigma_max (float, optional): Maximum standard deviation (for tanh).
        """
        super().__init__()
        self.fixed_logvar = fixed_logvar
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.var_mode = var_mode
        self.sigma_min = float(sigma_min)
        self.sigma_max = None if sigma_max is None else float(sigma_max)

        self.net = nn.Sequential(
            nn.Linear(z_dim + c_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * x_dim if fixed_logvar is None else x_dim),
        )

        if fixed_logvar is True:
            self.logvar_param = nn.Parameter(torch.zeros(x_dim))
        else:
            self.logvar_param = None

    def _pos_std(self, raw: Tensor) -> Tensor:
        """Map unconstrained raw parameter to positive standard deviation.

        Args:
            raw (Tensor): Raw variance parameter.

        Returns:
            Tensor: Positive standard deviation.
        """
        if self.var_mode == "softplus":
            std = F.softplus(raw) + self.sigma_min
            return std if self.sigma_max is None else torch.clamp(std, max=self.sigma_max)

        if self.var_mode == "tanh":
            assert self.sigma_max is not None and self.sigma_max > self.sigma_min
            lo = math.log(self.sigma_min)
            hi = math.log(self.sigma_max)
            s = 0.5 * (torch.tanh(raw) + 1.0)
            log_std = lo + s * (hi - lo)
            return torch.exp(log_std)

        raise ValueError(f"Unknown var_mode: {self.var_mode}")

    def forward(self, z: Tensor, c: Union[Tensor, Sequence, float, int]) -> Independent:
        """Return p(x|z,c) as a Normal distribution.

        Args:
            z (Tensor): Latent variable tensor of shape (B, z_dim).
            c (Tensor or array-like): Context.

        Returns:
            Independent: A diagonal Normal distribution.
        """
        c = ensure_batch_context(z, c)
        h = self.net(torch.cat([z, c], dim=1))

        if self.fixed_logvar is None:
            mu, raw = h.chunk(2, dim=-1)
            std = self._pos_std(raw)
        elif self.fixed_logvar is True:
            mu = h
            std = torch.exp(0.5 * self.logvar_param).unsqueeze(0)
            std = torch.clamp(std, min=self.sigma_min)
            std = std.expand_as(mu)
        else:
            mu = h
            logvar = torch.as_tensor(self.fixed_logvar, device=mu.device, dtype=mu.dtype)
            if logvar.dim() == 0:
                logvar = logvar.expand(self.x_dim)
            std = torch.exp(0.5 * logvar).unsqueeze(0).expand_as(mu)
            std = torch.clamp(std, min=self.sigma_min)

        return Independent(Normal(mu, std), 1)


class Prior(zuko.lazy.LazyDistribution):
    """Conditional prior p(z|c) parameterized by an MLP."""

    def __init__(
        self,
        c_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize prior network.

        Args:
            c_dim (int): Context dimension.
            z_dim (int): Latent dimension.
            hidden_dim (int): Hidden layer size.
            device (torch.device, optional): Device.
        """
        super().__init__()
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() and device is None else "cpu")

        self.net = nn.Sequential(
            nn.Linear(c_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * z_dim),
        )

    def forward(self, c: Union[Tensor, Sequence, float, int]) -> Independent:
        """Return p(z|c) distribution.

        Args:
            c (Tensor or array-like): Context.

        Returns:
            Independent: Diagonal Normal distribution p(z|c).
        """
        if not isinstance(c, torch.Tensor):
            c = torch.as_tensor(
                c,
                device=self.net[0].weight.device,
                dtype=self.net[0].weight.dtype,
            )

        if c.dim() == 1:
            c = c.view(-1, 1)
        elif c.dim() >= 3:
            c = c.view(c.size(0), -1)

        h = self.net(c)
        mu, raw = h.split(self.z_dim, dim=1)
        std = F.softplus(raw) + 1e-6
        return Independent(Normal(mu, std), 1)


class BetaScheduler:
    """Scheduler controlling KL weight β during VAE warmup.

    Modes:
        - `"linear"`: Linear warmup 0 → 1.
        - `"cosine"`: Cosine easing.
        - `"sigmoid"`: Smooth S-shaped warmup.
        - `"constant"`: Always β_end.
    """

    def __init__(
        self,
        warmup_steps: int,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        mode: str = "linear",
    ) -> None:
        """Initialize the beta scheduler.

        Args:
            warmup_steps (int): Number of warmup steps.
            beta_start (float): Starting β value.
            beta_end (float): Target β value.
            mode (str): Warmup mode.
        """
        self.warmup_steps = max(int(warmup_steps), 1)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.mode = mode
        self._step = 0

    @property
    def step(self) -> int:
        """int: Current scheduler step."""
        return self._step

    def reset(self) -> None:
        """Reset scheduler step."""
        self._step = 0

    def update(self) -> None:
        """Advance scheduler by one step."""
        self._step += 1

    def beta(self) -> float:
        """Compute current beta according to schedule.

        Returns:
            float: Weight for KL term.
        """
        if self.mode == "constant":
            return self.beta_end

        t = min(self._step / self.warmup_steps, 1.0)

        if self.mode == "linear":
            s = t
        elif self.mode == "cosine":
            s = 0.5 - 0.5 * math.cos(math.pi * t)
        elif self.mode == "sigmoid":
            k = 10.0
            s = 1.0 / (1.0 + math.exp(-k * (t - 0.5)))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return (1 - s) * self.beta_start + s * self.beta_end


def train_vae(
    elbo: ELBO,
    trainloader: DataLoader,
    lr: float = 1e-4,
    epochs: int = 10,
    device: Optional[torch.device] = None,
) -> ELBO:
    """Train ELBO-based VAE.

    Args:
        elbo (ELBO): ELBO module.
        trainloader (DataLoader): Training data loader (c, x).
        lr (float): Learning rate.
        epochs (int): Number of epochs.
        device (torch.device, optional): Training device.

    Returns:
        ELBO: Trained ELBO module.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elbo.to(device)
    elbo.train()

    optimizer = torch.optim.Adam(elbo.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        for c, x in trainloader:
            x = x.to(device)
            c = c.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = elbo(x, c)
            loss.backward()
            optimizer.step()

            elbo.step_beta()

        print(
            f"Epoch {epoch:03d} | "
            f"loss={metrics['elbo']:.3f}  "
            f"recon={metrics['recon']:.3f}  "
            f"kl={metrics['kl']:.3f}  "
            f"beta={metrics['beta']:.3f}"
        )

    return elbo


class VAElinear(nn.Module):
    """Conditional VAE with MLP encoder/decoder and heteroscedastic output."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize VAElinear.

        Required keyword args:
            latent_s (int): Dimensionality of latent variable z.
            in_size  (int): Input dimension of x.
            cond_in  (int): Conditional context dimension.

        Optional:
            enc_w (int): Encoder width.
            enc_l (int): Encoder depth.
            dec_w (int): Decoder width.
            dec_l (int): Decoder depth.
            gpu   (bool): Whether to use CUDA if available.
        """
        super().__init__()
        self.latent_s = kwargs["latent_s"]
        self.in_size = kwargs["in_size"]
        self.cond_in = kwargs["cond_in"]

        enc_w = kwargs.get("enc_w", 128)
        enc_l = kwargs.get("enc_l", 2)
        dec_w = kwargs.get("dec_w", 128)
        dec_l = kwargs.get("dec_l", 2)
        gpu = kwargs.get("gpu", True)

        self.device = torch.device("cuda:0" if (gpu and torch.cuda.is_available()) else "cpu")

        # Encoder network
        enc_sizes = [self.in_size + self.cond_in] + [enc_w] * enc_l + [2 * self.latent_s]
        enc_layers: List[nn.Module] = []
        for a, b in zip(enc_sizes[:-1], enc_sizes[1:]):
            enc_layers += [nn.Linear(a, b), nn.ReLU()]
        enc_layers.pop()  # remove last ReLU
        self.enc = nn.Sequential(*enc_layers)

        # Decoder network
        dec_sizes = [self.latent_s + self.cond_in] + [dec_w] * dec_l + [2 * self.in_size]
        dec_layers: List[nn.Module] = []
        for a, b in zip(dec_sizes[:-1], dec_sizes[1:]):
            dec_layers += [nn.Linear(a, b), nn.ReLU()]
        dec_layers.pop()
        self.dec = nn.Sequential(*dec_layers)

        self.to(self.device)

    def _ensure_2d(self, t: Tensor) -> Tensor:
        """Flatten inputs to (B, -1) if necessary."""
        return t.view(t.size(0), -1) if t.dim() > 2 else t

    def encode(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode q(z|x,y)."""
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        h = torch.cat([x, y], dim=1)
        mu_z, log_var_z = torch.split(self.enc(h), self.latent_s, dim=1)
        return mu_z, log_var_z

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample z using reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Decode p(x|z,y)."""
        zy = torch.cat([z, y], dim=1)
        mu_x, log_var_x = torch.split(self.dec(zy), self.in_size, dim=1)
        log_var_x = torch.clamp(log_var_x, min=-6.0, max=4.0)
        return mu_x, log_var_x

    def loss(
        self,
        x0: Tensor,
        cond_in: Optional[Tensor] = None,
        beta: float = 1.0,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute VAE loss = reconstruction + β * KL.

        Args:
            x0 (Tensor): Data batch (B, in_size).
            cond_in (Tensor, optional): Context batch.
            beta (float): KL weight.

        Returns:
            Tuple[Tensor, Dict[str, Tensor]]:
                Loss scalar and dict {"recon", "kl"}.
        """
        x0 = self._ensure_2d(x0).to(self.device)
        bs = x0.size(0)

        if cond_in is None:
            cond_in = torch.zeros(bs, self.cond_in, device=self.device, dtype=x0.dtype)
        else:
            cond_in = self._ensure_2d(cond_in).to(x0.device, dtype=x0.dtype)

        mu_z, log_var_z = self.encode(x0, cond_in)
        z = self.reparameterize(mu_z, log_var_z)
        mu_x, log_var_x = self.decode(z, cond_in)

        inv_var_x = torch.exp(-log_var_x)
        recon_nll = 0.5 * (log_var_x + (x0 - mu_x).pow(2) * inv_var_x)
        recon = recon_nll.sum(dim=1).mean()

        kl = 0.5 * (
            torch.exp(log_var_z) + mu_z.pow(2) - 1.0 - log_var_z
        ).sum(dim=1).mean()

        return recon + beta * kl, {"recon": recon.detach(), "kl": kl.detach()}

    @torch.no_grad()
    def forward(
        self,
        x0: Tensor,
        cond_in: Optional[Tensor] = None,
        return_latent: bool = False,
    ) -> Union[Tensor, Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]]:
        """Compute mean reconstruction (optionally return latent variables)."""
        self.eval()
        x0 = self._ensure_2d(x0).to(self.device)
        bs = x0.size(0)

        if cond_in is None:
            cond_in = torch.zeros(bs, self.cond_in, device=self.device, dtype=x0.dtype)
        else:
            cond_in = self._ensure_2d(cond_in).to(self.device)

        mu_z, log_var_z = self.encode(x0, cond_in)
        z = self.reparameterize(mu_z, log_var_z)
        mu_x, log_var_x = self.decode(z, cond_in)

        if return_latent:
            return (mu_x, log_var_x), (mu_z, log_var_z, z)
        return mu_x

    def to(self, device: Union[str, torch.device]) -> "VAElinear":
        """Move module to device and update internal device reference."""
        ret = super().to(device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        return ret

    @torch.no_grad()
    def sample(
        self,
        n: int,
        c: Union[Tensor, Sequence, float, int],
        sample_mean: bool = False,
        var_scale: float = 1.0,
    ) -> Tensor:
        """Sample from VAE generative distribution p(x|c).

        Args:
            n (int): Number of samples per context.
            c (Tensor or array-like): Context batch.
            sample_mean (bool): Use mean instead of random sampling.
            var_scale (float): Scale decoder variance.

        Returns:
            Tensor: Samples of shape (n, B, in_size).
        """
        self.eval()
        device = self.device

        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float32, device=device)
        else:
            c = c.to(device=device, dtype=torch.float32)

        if c.dim() == 1:
            if c.numel() == self.cond_in:
                c = c.view(1, -1)
            else:
                c = c.view(-1, 1)
        elif c.dim() > 2:
            c = c.view(c.size(0), -1)

        B = c.size(0)
        assert c.size(1) == self.cond_in

        z = torch.randn(n * B, self.latent_s, device=device)
        c_rep = c.repeat(n, 1)

        mu_x, log_var_x = self.decode(z, c_rep)

        if sample_mean:
            x = mu_x
        else:
            std = torch.exp(0.5 * log_var_x) * (var_scale**0.5)
            eps = torch.randn_like(std)
            x = mu_x + std * eps

        return x.view(n, B, self.in_size)

    @torch.no_grad()
    def quantiles(
        self,
        c: Union[Tensor, Sequence, float, int],
        q: Iterable[float] = (0.25, 0.5, 0.75),
        n: int = 100,
        sample_mean: bool = False,
        var_scale: float = 1.0,
    ) -> np.ndarray:
        """Estimate conditional quantiles via sampling.

        Args:
            c (Tensor or array-like): Context.
            q (Iterable[float]): Quantiles to compute.
            n (int): Number of Monte Carlo samples.
            sample_mean (bool): Whether to sample from mean only.
            var_scale (float): Variance scaling.

        Returns:
            np.ndarray: Array of shape (len(q), B, in_size).
        """
        y = self.sample(n=n, c=c, sample_mean=sample_mean, var_scale=var_scale)
        return np.quantile(y.cpu().numpy(), q, axis=0)


def plot_vae_training(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Learning curve",
) -> Tuple[Any, Any]:
    """Plot VAE training curves (loss, reconstruction, KL divergence).

    This function visualizes the learning progress of the VAE by plotting:
    - training loss
    - training reconstruction loss
    - training KL divergence

    Args:
        history (Dict[str, List[float]]):
            A dictionary containing logged training metrics. Expected keys:
            ``"train_loss"``, ``"train_recon"``, ``"train_kl"``.
        save_path (str, optional):
            If provided, the figure is saved to this path.
        title (str):
            Title of the plot.

    Returns:
        Tuple[Figure, Axes]:
            A tuple containing the Matplotlib figure and axis objects.

    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="loss")
    ax.plot(history["train_recon"], label="recon")
    ax.plot(history["train_kl"], label="kl")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path)
    return fig, ax


def train_vae_linear(
    vae: VAElinear,
    train_dataloader: DataLoader,
    validation_dataloader: Optional[DataLoader] = None,
    lr: float = 1e-3,
    save_path: Optional[str] = None,
    epochs: int = 100,
) -> Tuple[VAElinear, Dict[str, List[float]]]:
    """Train a VAElinear model with optional validation and early stopping.

    This training loop supports:
    - minibatch stochastic gradient descent,
    - KL annealing via a β schedule,
    - validation monitoring,
    - early stopping based on best validation loss,
    - saving the best model to disk.

    The function logs:
        - training loss
        - reconstruction loss
        - KL divergence
        - validation versions of the above (if validation loader provided)
        - current β value

    Args:
        vae (VAElinear):
            The conditional VAE model to train.
        train_dataloader (DataLoader):
            DataLoader yielding batches of (context, x).
        validation_dataloader (DataLoader, optional):
            DataLoader for validation. If ``None``, validation and early stopping
            are disabled.
        lr (float):
            Learning rate for Adam optimizer.
        save_path (str, optional):
            If provided, saves the weights of the best validation model.
        epochs (int):
            Maximum number of epochs to train.

    Returns:
        Tuple[VAElinear, Dict[str, List[float]]]:
            - The trained VAE model (restored to best state if validation used).
            - A history dictionary containing lists of metrics over epochs.

    """
    device = next(vae.parameters()).device
    opt = torch.optim.Adam(vae.parameters(), lr=lr)

    vae_history = {
        "train_loss": [],
        "train_recon": [],
        "train_kl": [],
        "val_loss": [],
        "val_recon": [],
        "val_kl": [],
        "beta": [],
    }

    best = float("inf")
    patience = 20
    bad_epochs = 0
    best_state = copy.deepcopy(vae.state_dict())

    for epoch in range(epochs):
        beta = min(1.0, (epoch + 1) / 50.0)

        # Training loop
        vae.train()
        t_loss = t_recon = t_kl = 0.0
        train_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{epochs} [train] β={beta:.2f}",
            leave=False,
        )

        for i, (c_batch, x) in enumerate(train_bar, start=1):
            c_batch = c_batch.to(device, dtype=torch.float32)
            x = x.to(device, dtype=torch.float32)

            loss, logs = vae.loss(x, c_batch, beta=beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 5.0)
            opt.step()

            t_loss += loss.item()
            t_recon += logs["recon"].item()
            t_kl += (beta * logs["kl"]).item()

            train_bar.set_postfix(loss=t_loss / i, recon=t_recon / i, kl=t_kl / i)

        vae_history["train_loss"].append(t_loss / i)
        vae_history["train_recon"].append(t_recon / i)
        vae_history["train_kl"].append(t_kl / i)

        # Validation loop
        if validation_dataloader is not None:
            vae.eval()
            v_loss = v_recon = v_kl = 0.0

            with torch.no_grad():
                val_bar = tqdm(
                    validation_dataloader,
                    desc=f"Epoch {epoch+1}/{epochs} [val]",
                    leave=False,
                )
                for j, (c_batch, x) in enumerate(val_bar, start=1):
                    c_batch = c_batch.to(device, dtype=torch.float32)
                    x = x.to(device, dtype=torch.float32)

                    loss, logs = vae.loss(x, c_batch, beta=beta)
                    v_loss += loss.item()
                    v_recon += logs["recon"].item()
                    v_kl += (beta * logs["kl"]).item()

                    val_bar.set_postfix(
                        loss=v_loss / j,
                        recon=v_recon / j,
                        kl=v_kl / j,
                    )

            vae_history["val_loss"].append(v_loss / j)
            vae_history["val_recon"].append(v_recon / j)
            vae_history["val_kl"].append(v_kl / j)
            vae_history["beta"].append(beta)

            tqdm.write(
                f"Epoch {epoch+1:03d} | "
                f"train {vae_history['train_loss'][-1]:.4f} "
                f"(recon={vae_history['train_recon'][-1]:.4f}, kl={vae_history['train_kl'][-1]:.4f}) | "
                f"val {vae_history['val_loss'][-1]:.4f} "
                f"(recon={vae_history['val_recon'][-1]:.4f}, kl={vae_history['val_kl'][-1]:.4f}) | "
                f"β={beta:.2f}"
            )

            # Early stopping
            if vae_history["val_loss"][-1] < best - 1e-6:
                best = vae_history["val_loss"][-1]
                best_state = copy.deepcopy(vae.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    vae.load_state_dict(best_state)
                    tqdm.write(f"Early stopping at epoch {epoch+1}. Best val_loss={best:.4f}")
                    if save_path is not None:
                        torch.save(vae.state_dict(), save_path)
                    return vae, vae_history

    if save_path is not None:
        torch.save(vae.state_dict(), save_path)

    return vae, vae_history


def vae_objective(
    trial: optuna.Trial,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    c_dim: int,
    x_dim: int,
    device: torch.device,
    obj_epochs: int = 15,
) -> float:
    """Optuna objective function for VAE hyperparameter optimization.

    This function runs short training cycles inside an Optuna trial and evaluates
    the model using CRPS on the validation dataset.

    Hyperparameters optimized include:
    - latent size,
    - encoder width & depth,
    - decoder width & depth,
    - learning rate.

    Args:
        trial (optuna.Trial):
            The active Optuna trial.
        train_dataloader (DataLoader):
            Training dataset.
        validation_dataloader (DataLoader):
            Validation dataset for CRPS evaluation.
        c_dim (int):
            Dimensionality of conditional input.
        x_dim (int):
            Dimensionality of target variable.
        device (torch.device):
            Device on which to run evaluation.
        obj_epochs (int):
            Number of training-evaluation cycles per trial.

    Returns:
        float:
            Best CRPS value achieved during the trial.

    Raises:
        optuna.TrialPruned:
            If the pruning condition is met.

    """
    latent_s = trial.suggest_categorical("latent_s", [8, 16, 32])
    enc_w = trial.suggest_categorical("enc_w", [32, 64, 128])
    enc_l = trial.suggest_categorical("enc_l", [2, 3, 4])
    dec_w = trial.suggest_categorical("dec_w", [32, 64, 128])
    dec_l = trial.suggest_categorical("dec_l", [2, 3, 4])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    vae = VAElinear(
        latent_s=latent_s,
        cond_in=c_dim,
        in_size=x_dim,
        gpu=True,
        enc_w=enc_w,
        enc_l=enc_l,
        dec_w=dec_w,
        dec_l=dec_l,
    )

    best_crps = np.inf

    for epoch in range(obj_epochs):
        vae, _ = train_vae_linear(
            vae,
            train_dataloader,
            validation_dataloader=None,
            epochs=1,
            lr=lr,
        )

        vae.eval()
        all_crps = []

        with torch.no_grad():
            for x, label in validation_dataloader:
                x = x.to(device)
                label = label.to(device)

                c_batch = x.reshape(x.size(0), -1)
                x_batch = label

                y_samps = vae.sample(100, c_batch)
                y_np = y_samps.cpu().numpy()
                x_np = x_batch.cpu().numpy()

                all_crps.extend(crps_batch_per_marginal(y_np, x_np))

        mean_crps = float(np.mean(all_crps))
        trial.report(mean_crps, epoch)

        if mean_crps < best_crps:
            best_crps = mean_crps

        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch} with crps={mean_crps}")

    return best_crps
