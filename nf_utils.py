import torch
import torch.nn as nn
import zuko
import numpy as np
from evaluation.metrics import crps_batch_per_marginal, energy_score_per_batch, variogram_score_per_batch, quantile_score_averaged_fast

class MyFlow(nn.Module):
    def __init__(self):
        super(MyFlow, self).__init__()
        self.flow = self._build_flow()
         # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.to(self.device)

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
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def quantiles(self, context=None, q=100, n=100):
        with torch.no_grad():
            y = self.sample(n, context)              # (n, B, D)
            y = y.detach().cpu().numpy()
            qv = np.quantile(y, q, axis=0)          # (len(q), B, D)
        return qv

    def fit(self, trainloader, validation_dataloader=None, epochs=50, lr=1e-3, patience=20):
        # self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val = float('inf')
        best_state = None
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            # --- Train ---
            self.train()
            train_losses = []
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                c = x.view(x.size(0), -1)
                optimizer.zero_grad()
                loss = -self.log_prob(y, c).mean()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                train_losses.append(loss.detach().cpu().item())
            train_error = float(np.mean(train_losses))

            # --- Validation ---
            if validation_dataloader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for x_val, y_val in validation_dataloader:
                        x_val = x_val.to(self.device)
                        y_val = y_val.to(self.device)
                        c_val = x_val.view(x_val.size(0), -1)
                        val_loss = -self.log_prob(y_val, c_val).mean()
                        val_losses.append(val_loss.detach().cpu().item())
                val_error = float(np.mean(val_losses))
                print(f"Epoch {epoch}: train error = {train_error:.4f} | val error = {val_error:.4f}")

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
                print(f"Epoch {epoch}: train error = {train_error:.4f}")

        if best_state is not None:
            self.load_state_dict(best_state)
        return self

class NormalizingFlowUNAF(MyFlow):
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
        super(NormalizingFlowUNAF, self).__init__()

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
    
class NormalizingFlowNAF(MyFlow):
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
        super(NormalizingFlowNAF, self).__init__()

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

class NormalizingFlowNSF(MyFlow):
    """
    Neural Spline Flow (NSF)
    """
    def __init__(self, features_dim, transforms, hidden_features):
        self.features_dim = features_dim
        self.transforms = transforms
        self.hidden_features = hidden_features
        super(NormalizingFlowNSF, self).__init__()

    def _build_flow(self):
        return zuko.flows.NSF(features=self.features_dim, transforms=self.transforms, hidden_features=self.hidden_features)

    def fit(self, trainloader, validation_dataloader=None, epochs=50, lr=1e-3, patience=20):
            # self.to(self.device)
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

            best_val = float('inf')
            best_state = None
            bad_epochs = 0

            for epoch in range(1, epochs + 1):
                # --- Train ---
                self.train()
                train_losses = []
                for _, y in trainloader:
                    # x = x.to(self.device)
                    y = y.to(self.device)
                    optimizer.zero_grad()
                    loss = -self.log_prob(y).mean()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                    optimizer.step()
                    train_losses.append(loss.detach().cpu().item())
                train_error = float(np.mean(train_losses))

                # --- Validation ---
                if validation_dataloader is not None:
                    self.eval()
                    val_losses = []
                    with torch.no_grad():
                        for x_val, y_val in validation_dataloader:
                            x_val = x_val.to(self.device)
                            y_val = y_val.to(self.device)
                            c_val = x_val.view(x_val.size(0), -1)
                            val_loss = -self.log_prob(y_val, c_val).mean()
                            val_losses.append(val_loss.detach().cpu().item())
                    val_error = float(np.mean(val_losses))
                    print(f"Epoch {epoch}: train error = {train_error:.4f} | val error = {val_error:.4f}")

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
                    print(f"Epoch {epoch}: train error = {train_error:.4f}")

            if best_state is not None:
                self.load_state_dict(best_state)
            return self


def evaluate_flow(flow, test_dataloader, model_label, **kwargs):
    n_samples = kwargs.get('samples', 20)
    device = kwargs.get('device', next(flow.parameters()).device)

    flow.eval()
    all_crps, all_energy, all_vario, all_qs = [], [], [], []

    with torch.no_grad():
        for x, label in test_dataloader:
            x = x.to(device)
            label = label.to(device)

            c_batch = x.view(x.size(0), -1)   # [B, c_dim]
            x_batch = label                   # [B, x_dim]

            y_samps = flow.sample(n_samples, c_batch)     # (S, B, D)
            y_np = y_samps.detach().cpu().numpy()
            x_np = x_batch.detach().cpu().numpy()

            # quantiles: shape (Q, B, D)
            qs = [0.01 * i for i in range(1, 100)]
            q_arr = flow.quantiles(c_batch, qs, n=max(n_samples, 100))

            all_crps.append(crps_batch_per_marginal(y_np, x_np))
            all_energy.append(energy_score_per_batch(y_np, x_np))
            all_vario.append(variogram_score_per_batch(y_np, x_np))
            all_qs.append(quantile_score_averaged_fast(q_arr, x_np))

    results = {
        'label': model_label,
        'crps': float(np.mean(all_crps)),
        'energy': float(np.mean(all_energy)),
        'variogram': float(np.mean(all_vario)),
        'quantile_score': float(np.mean(all_qs))
    }
    return results


    


if __name__ == '__main__':
    import torch
    import torch.optim as optim

    # Create your flow model
    flow = NormalizingFlowUNAF(x_dim=2, c_dim=2, hidden_features=[16, 16])

    
    x_data = torch.randn(1000, 2)
    c_data = torch.randn(1000, 2)

    print(flow.quantiles(c_data, [0.1, 0.5]))
