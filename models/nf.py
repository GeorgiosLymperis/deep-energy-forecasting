"""
Normalizing Flow models (NAF, UNAF, NSF) for conditional probabilistic forecasting.

Defines:
    - MyFlow base class
    - UNAF
    - NAF
    - NSF

Each flow implements:
    - log_prob(x, context)
    - sample(n, context) -> (n, B, D)
    - quantiles(context, q, n)
    - fit(train_loader, validation_loader, ...)

The flows use zuko.flows.* under the hood.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import zuko
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna

from evaluation.metrics import crps_batch_per_marginal


class MyFlow(nn.Module):
    """Base class for conditional normalizing flows.

    Subclasses must implement `_build_flow()` and (optionally) `get_config()`.

    This wrapper exposes a unified interface for:
      - log probability computation
      - sampling
      - quantile estimation
      - model fitting (MLE training)

    Attributes:
        flow (Callable): Zuko flow constructor created by `_build_flow()`.
    """

    def __init__(self) -> None:
        super(MyFlow, self).__init__()
        self.flow = self._build_flow()

    def _build_flow(self) -> Any:
        """Build and return a Zuko flow instance.

        This method must be implemented by subclasses.

        Returns:
            Any: A Zuko flow object.
        """
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute log probability of samples under the flow.

        Args:
            x (torch.Tensor): Input samples of shape (B, D) or similar.
            context (Optional[torch.Tensor]): Conditional context of shape (B, C).

        Returns:
            torch.Tensor: Log probability values of shape (B,).
        """
        if context is None:
            return self.flow().log_prob(x)
        return self.flow(context).log_prob(x)

    def sample(self, n: int, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Draw samples from the flow.

        Args:
            n (int): Number of samples to generate per context.
            c (Optional[torch.Tensor]): Context. If None, unconditional sampling.

        Returns:
            torch.Tensor: Samples of shape (n, B, D).
        """
        if c is None:
            return self.flow().sample((n,))
        return self.flow(c).sample((n,))

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Alias for log_prob, enabling `flow(x, context)` usage.

        Args:
            x (torch.Tensor): Input samples.
            context (Optional[torch.Tensor]): Conditional context.

        Returns:
            torch.Tensor: Log probability values.
        """
        if context is None:
            return self.flow().log_prob(x)
        return self.flow(context).log_prob(x)

    def quantiles(
        self,
        context: Optional[torch.Tensor] = None,
        q: Iterable[float] = (0.1, 0.5, 0.9),
        n: int = 100
    ) -> np.ndarray:
        """Estimate quantiles by Monte Carlo sampling.

        Args:
            context (Optional[torch.Tensor]): Conditional context.
            q (Iterable[float]): Quantiles to compute.
            n (int): Number of Monte Carlo samples.

        Returns:
            np.ndarray: Array of shape (len(q), B, D) with quantiles.
        """
        with torch.no_grad():
            y = self.sample(n, context)  # (n, B, D)
            y = y.detach().cpu().numpy()
            return np.quantile(y, q, axis=0)

    def fit(
        self,
        trainloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None,
        epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 20,
        device: Optional[torch.device] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Train the flow using maximum likelihood estimation (MLE).

        Args:
            trainloader (DataLoader): Training dataloader yielding (x, y).
            validation_dataloader (Optional[DataLoader]): Validation dataloader.
            epochs (int): Maximum number of epochs.
            lr (float): Learning rate.
            patience (int): Early stopping patience.
            device (Optional[torch.device]): Target device. If None, auto-detect.
            save_path (Optional[str]): If provided, save the best model.

        Returns:
            Dict[str, List[float]]: Training and validation error history.
        """
        device = device or (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
        self.to(device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        best_val = float('inf')
        best_state: Optional[Dict[str, torch.Tensor]] = None
        bad_epochs = 0

        history = {'train_error': [], 'val_error': []}

        for epoch in range(1, epochs + 1):
            # --------------------------
            # Training
            # --------------------------
            self.train()
            train_losses: List[float] = []
            pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)

            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                # flatten context
                c = x.reshape(x.size(0), -1)

                optimizer.zero_grad(set_to_none=True)
                loss = -self.log_prob(y, c).mean()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.detach().cpu().item())
                pbar.set_postfix(train=f"{np.mean(train_losses):.4f}")

            train_error = float(np.mean(train_losses))
            history['train_error'].append(train_error)

            # --------------------------
            # Validation
            # --------------------------
            if validation_dataloader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for x_val, y_val in validation_dataloader:
                        x_val = x_val.to(device, non_blocking=True)
                        y_val = y_val.to(device, non_blocking=True)
                        c_val = x_val.reshape(x_val.size(0), -1)
                        val_loss = -self.log_prob(y_val, c_val).mean()
                        val_losses.append(val_loss.detach().cpu().item())

                val_error = float(np.mean(val_losses))
                history['val_error'].append(val_error)
                tqdm.write(f"Epoch {epoch}/{epochs} [train: {train_error:.4f} | val: {val_error:.4f}]")

                sched.step(val_error)

                if val_error < best_val - 1e-4:
                    best_val = val_error
                    best_state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(f"Early stopping at epoch {epoch} (best val error = {best_val:.4f}).")
                        break
            else:
                tqdm.write(f"Epoch {epoch}/{epochs} [train: {train_error:.4f}]")

        if best_state is not None:
            self.load_state_dict(best_state)

        if save_path is not None:
            self.save(save_path)

        return history

    def get_config(self) -> Dict[str, Any]:
        """Return constructor arguments for recreation.

        Subclasses should override this.

        Returns:
            Dict[str, Any]: Empty dict by default.
        """
        return {}

    def save(self, path: str) -> None:
        """Save model state and configuration.

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
    def load_from(cls, path: str, map_location: Optional[torch.device] = None) -> "MyFlow":
        """Load model from disk.

        Args:
            path (str): Path to the saved checkpoint.
            map_location (Optional[torch.device]): Device mapping.

        Returns:
            MyFlow: Loaded model instance.
        """
        payload = torch.load(path, map_location=map_location)
        cfg = payload.get("config", {})
        model = cls(**cfg) if cfg else cls()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


class UNAF(MyFlow):
    """Unconstrained Neural Autoregressive Flow (UNAF).

    Wrapper around `zuko.flows.UNAF` supporting conditional features.
    """

    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        hidden_features: int = 64,
        transforms: int = 3,
        randperm: bool = False,
        signal: int = 16
    ) -> None:
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.hidden_features = hidden_features
        self.transforms = transforms
        self.randperm = randperm
        self.signal = signal
        super(UNAF, self).__init__()

    def _build_flow(self) -> Any:
        return zuko.flows.UNAF(
            features=self.x_dim,
            context=self.c_dim,
            transforms=self.transforms,
            randperm=self.randperm,
            signal=self.signal,
            network={
                "hidden_features": self.hidden_features,
                "activation": nn.ReLU,
            }
        )

    def get_config(self) -> Dict[str, Any]:
        return dict(
            x_dim=self.x_dim,
            c_dim=self.c_dim,
            hidden_features=self.hidden_features,
            transforms=self.transforms,
            randperm=self.randperm,
            signal=self.signal,
        )


class NAF(MyFlow):
    """Neural Autoregressive Flow (NAF)."""

    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        hidden_features: int = 64,
        transforms: int = 3,
        randperm: bool = False,
        signal: int = 16
    ) -> None:
        
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.hidden_features = hidden_features
        self.transforms = transforms
        self.randperm = randperm
        self.signal = signal
        super(NAF, self).__init__()

    def _build_flow(self) -> Any:
        return zuko.flows.NAF(
            features=self.x_dim,
            context=self.c_dim,
            transforms=self.transforms,
            randperm=self.randperm,
            signal=self.signal,
            network={
                "hidden_features": self.hidden_features,
                "activation": nn.ReLU,
            }
        )

    def get_config(self) -> Dict[str, Any]:
        return dict(
            x_dim=self.x_dim,
            c_dim=self.c_dim,
            hidden_features=self.hidden_features,
            transforms=self.transforms,
            randperm=self.randperm,
            signal=self.signal,
        )


class NSF(MyFlow):
    """Neural Spline Flow (NSF)."""

    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        transforms: int,
        hidden_features: int
    ) -> None:
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.transforms = transforms
        self.hidden_features = hidden_features
        super(NSF, self).__init__()

    def _build_flow(self) -> Any:
        return zuko.flows.NSF(
            features=self.x_dim,
            context=self.c_dim,
            transforms=self.transforms,
            hidden_features=self.hidden_features,
        )

    def get_config(self) -> Dict[str, Any]:
        return dict(
            x_dim=self.x_dim,
            c_dim=self.c_dim,
            transforms=self.transforms,
            hidden_features=self.hidden_features,
        )


def plot_nf_training(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Learning curve"
) -> Tuple[Any, Any]:
    """Plot training and validation curve for flows.

    Args:
        history (Dict[str, List[float]]): Dictionary with 'train_error' and 'val_error'.
        save_path (Optional[str]): If provided, save the figure.
        title (str): Plot title.

    Returns:
        Tuple[Figure, Axes]: Matplotlib figure and axis.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['train_error'], label='train')
    ax.plot(history['val_error'], label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path)

    return fig, ax


def flow_losses(flow: MyFlow, dataloader: DataLoader) -> np.ndarray:
    """Compute losses per batch for Diebold–Mariano tests.

    Args:
        flow (MyFlow): Trained flow.
        dataloader (DataLoader): Dataloader yielding (x, y).

    Returns:
        np.ndarray: Array of batch losses.
    """
    device = next(flow.parameters()).device
    flow.eval()
    losses = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = x.reshape(x.size(0), -1)
            loss = -flow.log_prob(y, c).mean()
            losses.append(loss)

    return np.array(losses)


def naf_objective(
    trial: optuna.Trial,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    c_dim: int,
    x_dim: int,
    device: torch.device,
    obj_epochs: int = 15
) -> float:
    """Optuna objective for NAF hyperparameter search.

    Args:
        trial (optuna.Trial): Optuna trial.
        train_dataloader (DataLoader): Training data.
        validation_dataloader (DataLoader): Validation data.
        c_dim (int): Context dimension.
        x_dim (int): Output dimension.
        device (torch.device): Device.
        obj_epochs (int): Objective evaluation epochs.

    Returns:
        float: Best CRPS metric.
    """
    n_layers = trial.suggest_categorical("n_layers", [2, 3])
    base_exp = trial.suggest_categorical("width_exp", [3, 4, 5])
    hidden_features = [2 ** base_exp] * n_layers
    transforms = trial.suggest_categorical("transforms", [1, 2])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    naf = NAF(x_dim, c_dim, hidden_features=hidden_features, transforms=transforms, randperm=False, signal=8)

    best_crps = np.inf
    for epoch in range(obj_epochs):
        naf.fit(train_dataloader, validation_dataloader=None, epochs=1, lr=lr, device=device)

        naf.eval()
        all_crps = []

        with torch.no_grad():
            for x, label in validation_dataloader:
                x = x.to(device)
                label = label.to(device)

                c_batch = x.reshape(x.size(0), -1)
                x_batch = label

                y_samps = naf.sample(25, c_batch)
                y_np = y_samps.detach().cpu().numpy()
                x_np = x_batch.detach().cpu().numpy()

                all_crps.extend(crps_batch_per_marginal(y_np, x_np))

        all_crps = np.mean(all_crps)
        trial.report(all_crps, epoch)

        if all_crps < best_crps:
            best_crps = all_crps

        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch} with crps={all_crps}")

    return best_crps


def nsf_objective(
    trial: optuna.Trial,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    c_dim: int,
    x_dim: int,
    device: torch.device,
    obj_epochs: int = 15
) -> float:
    """Optuna objective for NSF hyperparameter tuning.

    Args:
        trial (optuna.Trial): Optuna trial.
        train_dataloader (DataLoader): Training data.
        validation_dataloader (DataLoader): Validation data.
        c_dim (int): Context dimension.
        x_dim (int): Output dimension.
        device (torch.device): Device.
        obj_epochs (int): Number of epochs per trial.

    Returns:
        float: Best CRPS achieved.
    """
    n_layers = trial.suggest_categorical("n_layers", [2, 3])
    base_exp = trial.suggest_categorical("width_exp", [3, 4, 5])
    hidden_features = [2 ** base_exp] * n_layers
    transforms = trial.suggest_categorical("transforms", [1, 3, 5])

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    nsf = NSF(x_dim, c_dim, transforms=transforms, hidden_features=hidden_features)

    best_crps = np.inf
    for epoch in range(obj_epochs):
        nsf.fit(train_dataloader, validation_dataloader=None, epochs=1, lr=lr, device=device)

        nsf.eval()
        all_crps = []

        with torch.no_grad():
            for x, label in validation_dataloader:
                x = x.to(device)
                label = label.to(device)

                c_batch = x.reshape(x.size(0), -1)
                x_batch = label

                y_samps = nsf.sample(25, c_batch)
                y_np = y_samps.detach().cpu().numpy()
                x_np = x_batch.detach().cpu().numpy()

                all_crps.extend(crps_batch_per_marginal(y_np, x_np))

        all_crps = np.mean(all_crps)
        trial.report(all_crps, epoch)

        if all_crps < best_crps:
            best_crps = all_crps

        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch} with crps={all_crps}")

    return best_crps
