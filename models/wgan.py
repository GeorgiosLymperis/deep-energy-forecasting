"""
Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP)
used for multi-step probabilistic forecasting.

Includes:
    - Generator
    - Discriminator
    - Training loop with GP and drift penalty
    - Optuna objective for hyperparameter tuning

The GAN conditions on context vectors and generates multi-dimensional outputs.
"""

from typing import Iterable, Union

import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna

from evaluation.metrics import crps_batch_per_marginal


def ensure_batch_context(x: Tensor,
                         c: Union[Tensor, np.ndarray, list, float, int]) -> Tensor:
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
        # per-sample scalar context
        if c.size(0) != b:
            raise ValueError(f"c has shape {tuple(c.shape)} but batch is {b}")
        c = c.view(b, 1)
        return c.to(x.device, dtype=x.dtype)

    if c.dim() == 2:
        # (batch, c_dim) or (1, c_dim)
        if c.size(0) == 1 and b > 1:
            c = c.expand(b, c.size(1))
        elif c.size(0) != b:
            raise ValueError(f"c batch {c.size(0)} != {b}")
        return c.to(x.device, dtype=x.dtype)

    if c.dim() == 3:
        c = c.view(b, -1)
        return c.to(x.device, dtype=x.dtype)
    raise ValueError(f"Unsupported c shape: {tuple(c.shape)}")


class Generator(nn.Module):
    def __init__(self, z_dim: int, c_dim: int,
                 y_dim: int, hidden_dim: int = 128, device=None):
        """
        Conditional Generator for WGAN-GP.

        The generator receives latent noise `z` and context `c`, concatenates
        them, and outputs a sample vector `y` of dimension `y_dim`.

        Args:
            z_dim (int): Dimensionality of latent noise input.
            c_dim (int): Dimensionality of context conditioning.
            y_dim (int): Dimensionality of output sample.
            hidden_dim (int): Width of hidden layers.
            device (str or torch.device, optional):
                Device on which to place the module.
        """
        super(Generator, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + c_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim)
        )
        self._weights_initialize(mean=0.0, std=0.02)
        self.to(self.device)

    def forward(self, z: Tensor,
                c: Union[Tensor, np.ndarray, list, float, int]) -> Tensor:
        """
        Forward pass of the generator.

        Args:
            z (torch.Tensor): Latent noise of shape (B, z_dim).
            c (torch.Tensor or array-like): Context input.

        Returns:
            torch.Tensor: Generated samples of shape (B, y_dim).
        """
        c = ensure_batch_context(z, c)
        x = torch.cat([z, c], dim=1)
        return self.net(x)

    def _weights_initialize(self, mean: float, std: float):
        """
        Initialize module weights with a normal distribution.

        Args:
            mean (float): Mean of Gaussian initialization.
            std (float): Standard deviation of Gaussian initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=mean, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def sample(self, n: int,
               c: Union[Tensor, np.ndarray, list, float, int]) -> Tensor:
        """
        Generate multiple samples per context.

        Args:
            n (int):
                Number of samples per context.
            c (torch.Tensor or array-like):
                Context. Shape (B, c_dim) or broadcastable.

        Returns:
            torch.Tensor:
                Generated samples of shape (n, B, y_dim).
        """
        device = next(self.parameters()).device
        # normalize c to (B, c_dim)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, device=device)
        c = c.to(device)
        if c.dim() == 1:
            c = c.view(1, -1)
        B = c.size(0)

        z = torch.randn(n * B, self.z_dim, device=device)
        c_rep = c.repeat(n, 1)  # (S*B, c_dim)
        y = self.forward(z, c_rep)  # (S*B, y_dim)
        y = y.view(n, B, self.y_dim)
        return y

    @torch.no_grad()
    def quantiles(self, c: Tensor,
                  q: Iterable[float] = (0.25, 0.5, 0.75), n: int = 100):
        """
        Compute empirical output quantiles for a given context.

        Args:
            c (torch.Tensor or array-like):
                Context.
            q (sequence of float):
                Quantiles to compute.
            n (int):
                Number of Monte Carlo samples to estimate quantiles.

        Returns:
            np.ndarray:
                Array of shape (len(q), B, y_dim) with quantile values.
        """
        y = self.sample(n, c)                      # (S, B, D)
        y = y.detach().cpu().numpy()
        qv = np.quantile(y, q, axis=0)            # (len(q), B, D)
        return qv

    def get_config(self):
        """
        Return the constructor configuration of the generator.

        Returns:
            dict: Serializable configuration dictionary.
        """
        return {
            "z_dim": self.z_dim,
            "c_dim": self.c_dim,
            "y_dim": self.y_dim,
            "hidden_dim": self.hidden_dim,
            "device": str(self.device),
        }

    def save(self, path: str):
        """
        Save model state and configuration to disk.

        Args:
            path (str): Output file path.
        """
        payload = {
            "state_dict": self.state_dict(),
            "config": self.get_config(),
            "class": self.__class__.__name__,
        }
        torch.save(payload, path)

    @classmethod
    def load_from(cls, path: str, map_location=None):
        """
        Load a saved generator from disk.

        Args:
            path (str): Path to saved model.
            map_location: Device mapping.

        Returns:
            Generator: Loaded model instance.
        """
        payload = torch.load(path, map_location=map_location)
        # Recreate the instance with the saved config
        cfg = payload.get("config", {})
        model = cls(**cfg) if cfg else cls()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


class Discriminator(nn.Module):
    def __init__(self, c_dim: int, y_dim: int,
                 hidden_dim: int = 128, alpha: float = 0.01,
                 device=None):
        """
        Conditional Discriminator for WGAN-GP.

        Receives a target sample `y` and a context `c`, and outputs a scalar
        critic score.

        Args:
            c_dim (int): Context dimensionality.
            y_dim (int): Sample dimensionality.
            hidden_dim (int): Width of hidden layers.
            alpha (float): Negative slope for LeakyReLU.
            device (str or torch.device, optional):
                Device allocation.
        """
        super(Discriminator, self).__init__()
        self.c_dim = c_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.net = nn.Sequential(
            nn.Linear(y_dim + c_dim, hidden_dim),
            nn.LeakyReLU(alpha),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(alpha),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(alpha),
            nn.Linear(hidden_dim, 1)
        )
        self._weights_initialize(mean=0.0, std=0.02)
        self.to(self.device)

    def forward(self, z: Tensor,
                c: Union[Tensor, np.ndarray, list, float, int]) -> Tensor:
        """
        Forward pass of the discriminator.

        Args:
            z (torch.Tensor):
                Sample (real or fake) of shape (B, y_dim).
            c (torch.Tensor or array-like):
                Context.

        Returns:
            torch.Tensor:
                Critic scores of shape (B, 1).
        """
        c = ensure_batch_context(z, c)
        x = torch.cat([z, c], dim=1)
        return self.net(x)

    def _weights_initialize(self, mean: float, std: float):
        """
        Initialize weights for all Linear layers.

        Args:
            mean (float): Mean of normal distribution.
            std (float): Standard deviation.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=mean, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_config(self):
        """
        Return configuration dictionary.

        Returns:
            dict: Serializable discriminator config.
        """
        return {
            "c_dim": self.c_dim,
            "y_dim": self.y_dim,
            "hidden_dim": self.hidden_dim,
            "alpha": self.alpha,
            "device": str(self.device),
        }

    def save(self, path: str):
        """
        Save discriminator to disk.

        Args:
            path (str): Output file path.
        """
        payload = {
            "state_dict": self.state_dict(),
            "config": self.get_config(),
            "class": self.__class__.__name__,
        }
        torch.save(payload, path)

    @classmethod
    def load_from(cls, path: str, map_location=None):
        """
        Load discriminator from disk.

        Args:
            path (str): Path to saved model.
            map_location: Device mapping.

        Returns:
            Discriminator: Loaded instance.
        """
        payload = torch.load(path, map_location=map_location)
        # Recreate the instance with the saved config
        cfg = payload.get("config", {})
        model = cls(**cfg) if cfg else cls()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def gradient_penalty(discriminator: nn.Module,
                     real_data: Tensor, fake_data: Tensor,
                     c: Tensor) -> Tensor:
    """
    Compute the gradient penalty term for WGAN-GP.

    Implements the gradient penalty from:
      Gulrajani et al., "Improved Training of Wasserstein GANs" (2017).

    Args:
        discriminator (nn.Module):
            The critic network.
        real_data (torch.Tensor):
            Real samples (B, y_dim).
        fake_data (torch.Tensor):
            Generated samples (B, y_dim).
        c (torch.Tensor):
            Context matching the batch.

    Returns:
        torch.Tensor:
            Scalar gradient penalty value.
    """
    device = next(discriminator.parameters()).device
    batch_size = real_data.shape[0]
    alpha = torch.rand((batch_size, 1)).to(device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated = interpolated.to(device)
    interpolated.requires_grad_(True)
    disc_interpolated = discriminator(interpolated, c)
    gradients = torch.autograd.grad(
        outputs=disc_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(disc_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # (batch, y_dim)

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_wgan_gp_(
    generator, discriminator, trainloader, validationloader=None,
    g_lr=1e-4, d_lr=1e-4, gp_lambda=10, n_critic=5, epochs=20,
    save_path=None, patience=20
):
    """
    Train a conditional WGAN-GP model.

    Implements the critic-only loop (n_critic steps per generator update),
    gradient penalty, drift regularization, early stopping, validation based
    on critic score, and optional scheduler.

    Args:
        generator (Generator): The generator module.
        discriminator (Discriminator): The discriminator module.
        trainloader (torch.utils.data.DataLoader):
            Training dataloader yielding (context, target).
        validationloader (DataLoader, optional):
            Optional validation set.
        g_lr (float): Generator learning rate.
        d_lr (float): Discriminator learning rate.
        gp_lambda (float): Weight of gradient penalty term.
        n_critic (int): Number of critic steps per generator step.
        epochs (int): Number of training epochs.
        save_path (str, optional): Base path to save best models.
        patience (int): Epochs without improvement before early stop.

    Returns:
        (Discriminator, Generator, dict):
            The trained discriminator, generator, and history dictionary.
    """
    g_optimizer = torch.optim.Adam(generator.parameters(),
                                   lr=g_lr, betas=(0.0, 0.9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr,
                                   betas=(0.0, 0.9))

    g_sched = torch.optim.lr_scheduler.\
        ReduceLROnPlateau(g_optimizer, mode='min', factor=0.5, patience=3)

    d_sched = torch.optim.lr_scheduler.\
        ReduceLROnPlateau(d_optimizer, mode='min', factor=0.5, patience=3)

    device = next(generator.parameters()).device
    z_dim = generator.z_dim

    best_val = float('inf')
    best_state = None
    bad_epochs = 0
    stop = False

    history = {"d_loss": [], "g_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        if stop:
            break

        generator.train()
        discriminator.train()

        d_losses_epoch = []
        g_losses_epoch = []

        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{epochs} [train]",
                    leave=False)
        for c_batch, y_real in pbar:
            c_batch = c_batch.to(device)
            y_real = y_real.to(device)
            batch_size = y_real.size(0)

            # --- Train D n_critic times ---
            d_losses = []
            for _ in range(n_critic):
                z = torch.randn(batch_size, z_dim, device=device)
                y_fake = generator.forward(z, c_batch).detach()

                d_real = discriminator(y_real, c_batch)
                d_fake = discriminator(y_fake, c_batch)

                gp = gradient_penalty(discriminator, y_real, y_fake, c_batch)
                epsilon_drift = 1e-3   # 1e-3 is standard
                d_loss = -(d_real.mean() - d_fake.mean())\
                    + gp_lambda * gp + epsilon_drift * (d_real**2).mean()

                d_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                d_optimizer.step()

                d_losses.append(d_loss.detach().item())

            d_loss = float(np.mean(d_losses))
            d_losses_epoch.append(d_loss)

            # --- Train G once ---
            z = torch.randn(batch_size, z_dim, device=device)
            y_fake = generator(z, c_batch)
            d_fake = discriminator(y_fake, c_batch)
            g_loss = -d_fake.mean()

            g_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            g_optimizer.step()

            g_losses_epoch.append(g_loss.detach().item())

        # --- End epoch: aggregate ---
        d_epoch = float(np.mean(d_losses_epoch))
        g_epoch = float(np.mean(g_losses_epoch))

        # Validation once per epoch (generator quality proxy: -E[D(fake)])
        if validationloader is not None:
            generator.eval()
            discriminator.eval()
            val_losses = []
            with torch.no_grad():
                for val_c, val_y in validationloader:
                    val_c = val_c.to(device)
                    B = val_c.size(0)
                    z = torch.randn(B, z_dim, device=device)
                    y_fake = generator(z, val_c)
                    val_d_fake = discriminator(y_fake, val_c)
                    val_losses.append((-val_d_fake.mean()).item())
            val_loss = float(np.mean(val_losses))
            history["val_loss"].append(val_loss)

            # schedulers on epoch metrics
            g_sched.step(val_loss)
            d_sched.step(d_epoch)

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                best_state = (generator.state_dict(),
                              discriminator.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    if best_state is not None:
                        generator.load_state_dict(best_state[0])
                        discriminator.load_state_dict(best_state[1])
                    stop = True
        else:
            # still step schedulers to avoid stagnation
            g_sched.step(g_epoch)
            d_sched.step(d_epoch)

        history["d_loss"].append(d_epoch)
        history["g_loss"].append(g_epoch)

        tqdm.write(
            f"Epoch {epoch}/{epochs} | D: {d_epoch:.4f} | G: {g_epoch:.4f}" +
            (f" | Val: {history['val_loss'][-1]:.4f}"
             if validationloader is not None else "")
        )

    if best_state is not None:
        generator.load_state_dict(best_state[0])
        discriminator.load_state_dict(best_state[1])

    if save_path is not None:
        torch.save(generator.state_dict(), f"gen_{save_path}")
        torch.save(discriminator.state_dict(), f"disc_{save_path}")

    return discriminator, generator, history


def plot_gan_training(history, save_path=None, title='Learning losses'):
    """
    Plot GAN training curves for generator and discriminator losses.

    Args:
        history (dict):
            Dictionary containing 'd_loss' and 'g_loss'.
        save_path (str, optional):
            If provided, save figure to this location.
        title (str):
            Title of the plot.

    Returns:
        (matplotlib.figure.Figure, matplotlib.axes.Axes):
            The created figure and axes instances.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['d_loss'], label='discriminator loss')
    ax.plot(history['g_loss'], label='generator loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path)

    return fig, ax


def gan_objective(trial, train_dataloader, validation_dataloader,
                  c_dim, x_dim, device, obj_epochs=15):
    """Optuna objective function for tuning WGAN-GP hyperparameters.

    For each Optuna trial:
      - Suggest hyperparameters
      - Create Generator/Discriminator
      - Train a few epochs
      - Evaluate CRPS on validation set
      - Report metric to Optuna
      - Potentially prune

    Args:
        trial (optuna.Trial):
            Active Optuna trial.
        train_dataloader (DataLoader):
            Training dataset.
        validation_dataloader (DataLoader):
            Validation dataset.
        c_dim (int):
            Context dimensionality.
        x_dim (int):
            Output sample dimensionality.
        device (str or torch.device):
            Target device.
        obj_epochs (int):
            Training epochs per trial.

    Returns:
        float:
            Best mean CRPS achieved over the validation set.

    Raises:
        optuna.TrialPruned:
            When the trial is pruned.
    """
    n_critic = trial.suggest_categorical("n_critic", [1, 3, 5, 7, 9, 13, 15])
    gp_lambda = trial.suggest_float("gp_lambda", 1.0, 15.0)
    d_lr = trial.suggest_float("d_lr", 1e-4, 1e-2, log=True)
    g_lr = trial.suggest_float("g_lr", 1e-4, 1e-2, log=True)
    z_dim = trial.suggest_categorical("z_dim", [8, 16, 32, 64, 128, 256])
    hidden_dim = trial.suggest_categorical("hidden_dim",
                                           [8, 16, 32, 64, 128, 256, 512, 1024])
    generator = Generator(z_dim=z_dim, c_dim=c_dim, y_dim=x_dim,
                          hidden_dim=hidden_dim, device=device)
    discriminator = Discriminator(c_dim=c_dim, y_dim=x_dim,
                                  hidden_dim=hidden_dim, device=device)

    best_crps = np.inf
    for epoch in range(obj_epochs):
        discriminator, generator, _ = \
            train_wgan_gp_(generator, discriminator, train_dataloader,
                           validationloader=None, d_lr=d_lr, g_lr=g_lr,
                           epochs=1, gp_lambda=gp_lambda, n_critic=n_critic)

        generator.eval()
        all_crps = []
        with torch.no_grad():
            for x, label in validation_dataloader:
                x = x.to(device)
                label = label.to(device)

                c_batch = x.reshape(x.size(0), -1)   # [B, c_dim]
                x_batch = label                   # [B, x_dim]

                y_samps = generator.sample(100, c_batch)     # (S, B, D)
                y_np = y_samps.detach().cpu().numpy()
                x_np = x_batch.detach().cpu().numpy()

                all_crps.extend(crps_batch_per_marginal(y_np, x_np))

        all_crps = np.array(all_crps)
        all_crps = all_crps.mean()
        trial.report(all_crps, epoch)

        if all_crps < best_crps:
            best_crps = all_crps

        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch} \
                                     with crps={all_crps}")
    return best_crps
