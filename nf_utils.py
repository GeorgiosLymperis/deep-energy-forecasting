import torch
import torch.nn as nn
import zuko
import numpy as np
from evaluation.metrics import crps_batch_per_marginal, energy_score_per_batch, variogram_score_per_batch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class MyFlow(nn.Module):
    def __init__(self):
        super(MyFlow, self).__init__()
        self.flow = self._build_flow()

    def _build_flow(self):
        raise NotImplementedError
    
    def log_prob(self, x, context=None):
        if context is None:
            return self.flow().log_prob(x)
        return self.flow(context).log_prob(x)
    
    def sample(self, n, context=None):
        if context is None:
            return self.flow().sample((n,))
        return self.flow(context).sample((n,))

    def forward(self, x, context=None):
        if context is None:
            return self.flow().log_prob(x)
        return self.flow(context).log_prob(x)
    
    # def save(self, path):
    #     torch.save(self.state_dict(), path)

    # def load(self, path):
    #     self.load_state_dict(torch.load(path))

    def quantiles(self, context=None, q=100, n=100):
        with torch.no_grad():
            y = self.sample(n, context)              # (n, B, D)
            y = y.detach().cpu().numpy()
            qv = np.quantile(y, q, axis=0)          # (len(q), B, D)
        return qv

    def fit(self, trainloader, validation_dataloader=None, epochs=50, lr=1e-3, patience=20, device=None, save_path=None):
        device = device or (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
        self.to(device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val = float('inf')
        best_state = None
        bad_epochs = 0

        history = {
            'train_error': [],
            'val_error': []
        }

        for epoch in range(1, epochs + 1):
            # --- Train ---
            self.train()
            train_losses = []
            pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                c = x.reshape(x.size(0), -1)
                optimizer.zero_grad(set_to_none=True)
                loss = -self.log_prob(y, c).mean()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                train_losses.append(loss.detach().cpu().item())

                pbar.set_postfix(train=f"{np.mean(train_losses):.4f}")
                
            train_error = float(np.mean(train_losses))
            history['train_error'].append(train_error)

            # --- Validation ---
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
                tqdm.write(f"Epoch {epoch}/{epochs} [train: {train_error:.4f} | val: {val_error:.4f}]")

                history['val_error'].append(val_error)

                # scheduler & early stopping
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
    
    def get_config(self):
        """Override in subclasses to return init kwargs."""
        return {}

    def save(self, path: str):
        payload = {
            "state_dict": self.state_dict(),
            "config": self.get_config(),
            "class": self.__class__.__name__,
        }
        torch.save(payload, path)

    @classmethod
    def load_from(cls, path: str, map_location=None):
        payload = torch.load(path, map_location=map_location)
        # Recreate the instance with the saved config
        cfg = payload.get("config", {})
        model = cls(**cfg) if cfg else cls()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

class UNAF(MyFlow):
    """
    Unconstrained Neural Autoregressive Flow (UNAF)
    """  
    def __init__(self, x_dim, c_dim, hidden_features=64, transforms=3, randperm=False, signal=16):
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.hidden_features = hidden_features
        self.transforms = transforms
        self.randperm = randperm
        self.signal = signal
        super(UNAF, self).__init__()

    def _build_flow(self):

        return zuko.flows.UNAF(
                features= self.x_dim,
                context=self.c_dim,
                transforms=self.transforms,
                randperm=self.randperm,
                signal=self.signal,
                network={
                    "hidden_features": self.hidden_features,
                    "activation": nn.ReLU
                }
            )
    
    def get_config(self):
        return dict(
            x_dim=self.x_dim,
            c_dim=self.c_dim,
            hidden_features=self.hidden_features,
            transforms=self.transforms,
            randperm=self.randperm,
            signal=self.signal,
        )
    
class NAF(MyFlow):
    """
    Neural Autoregressive Flow (NAF)
    """ 
    def __init__(self, x_dim, c_dim, hidden_features=64, transforms=3, randperm=False, signal=16):
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.hidden_features = hidden_features
        self.transforms = transforms
        self.randperm = randperm
        self.signal = signal
        super(NAF, self).__init__()

    def _build_flow(self):

        return zuko.flows.NAF(
                features= self.x_dim,
                context=self.c_dim,
                transforms=self.transforms,
                randperm=self.randperm,
                signal=self.signal,
                network={
                    "hidden_features": self.hidden_features,
                    "activation": nn.ReLU
                }
            )
    
    def get_config(self):
        return dict(
            x_dim=self.x_dim,
            c_dim=self.c_dim,
            hidden_features=self.hidden_features,
            transforms=self.transforms,
            randperm=self.randperm,
            signal=self.signal,
        )

class NSF(MyFlow):
    """
    Neural Spline Flow (NSF)
    """
    def __init__(self, x_dim, c_dim, transforms, hidden_features):
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.transforms = transforms
        self.hidden_features = hidden_features
        super(NSF, self).__init__()

    def _build_flow(self):
        return zuko.flows.NSF(
            features=self.x_dim,
            context=self.c_dim,
            transforms=self.transforms, 
            hidden_features=self.hidden_features)
    
    def get_config(self):
        return dict(
            x_dim=self.x_dim,
            c_dim=self.c_dim,
            transforms=self.transforms,
            hidden_features=self.hidden_features,
        )

def evaluate_flow(flow, test_dataloader, model_label, save_path=None, **kwargs):
    n_samples = kwargs.get('samples', 20)
    device = kwargs.get('device', next(flow.parameters()).device)

    flow.eval()
    all_crps, all_energy, all_vario = [], [], []

    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc=f"Evaluating", leave=False)
        for x, label in pbar:
            x = x.to(device)
            label = label.to(device)

            c_batch = x.reshape(x.size(0), -1)   # [B, c_dim]
            x_batch = label                   # [B, x_dim]

            y_samps = flow.sample(n_samples, c_batch)     # (S, B, D)
            y_np = y_samps.detach().cpu().numpy()
            x_np = x_batch.detach().cpu().numpy()

            all_crps.append(crps_batch_per_marginal(y_np, x_np))
            all_energy.append(energy_score_per_batch(y_np, x_np))
            all_vario.append(variogram_score_per_batch(y_np, x_np))

    results = {
        'label': model_label,
        'crps': float(np.mean(all_crps)),
        'energy': float(np.mean(all_energy)),
        'variogram': float(np.mean(all_vario)),
    }

    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(results, f)
    return results

def plot_training(history, save_path=None, title='Learning curve'):
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

def make_24h_forecast_with_bands(flow, context, samples=100):
    """
    Makes a forecast for the next 24 hours

    flow: Trained flow
    context: tensor with context the weather conditions of the day [B, c_dim]
             Context must be scaled
    samples: the size of the samples generated by flow

    Returns: (Q1, median, Q3) numpy arrays
    """
    flow.eval()
    device = next(flow.parameters()).device

    Q1, median, Q3 = [], [], []
    context = context.to(device)

    prediction_quantiles = flow.quantiles(context, [0.25, 0.50, 0.75], n=samples)
    Q1 = prediction_quantiles[0,:,:]
    median = prediction_quantiles[1,:,:]
    Q3 = prediction_quantiles[2,:,:]

    return Q1, median, Q3

def flow_losses(flow, dataloader):
    """
    Generate losses for Diebold-Mariano test.
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
