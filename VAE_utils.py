import zuko
import torch.nn as nn
import torch
from torch.distributions import Independent, Normal
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from evaluation.metrics import crps_batch_per_marginal, energy_score_per_batch, variogram_score_per_batch
import json
from tqdm import tqdm

def ensure_batch_context(x, c):
    """
    Επιστρέφει c σε σχήμα (batch, c_dim), όπου batch = x.size(0).
    Δέχεται:
      - c: (batch,)    -> (batch,1)
      - c: (batch,1)   -> (batch,1)
      - c: (1,c_dim)   -> (batch,c_dim) (expand)
      - c: (batch,c_dim)-> (batch,c_dim)
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

class ELBO(nn.Module):
    def __init__(self, encoder, decoder, prior,
                 warmup_steps=50_000,  # set roughly: epochs * steps_per_epoch
                 beta_start=0.0, beta_end=1.0, mode='linear'):
        super(ELBO, self).__init__()
        self.encoder = encoder    # (x,c)-> q(z|x,c)
        self.decoder = decoder    # (z,c)-> p(x|z,c)
        self.prior   = prior      # c   -> p(z|c)  (use standard Normal if unconditional)
        self.beta_sched = BetaScheduler(
            warmup_steps=warmup_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            mode=mode
        )
    
    @property
    def beta(self):
        return self.beta_sched.beta()

    def step_beta(self):
        self.beta_sched.update()

    def reset_beta(self):
        self.beta_sched.reset()

    def forward(self, x, c):
        q = self.encoder(x, c)                 # q(z|x,c)
        z = q.rsample()
        log_px = self.decoder(z, c).log_prob(x)
        log_pz = self.prior(c).log_prob(z)     # or self.prior().log_prob(z) if unconditional
        log_qz = q.log_prob(z)
        
        beta = self.beta
        elbo = log_px + beta * (log_pz - log_qz)
        loss = -elbo.mean()
        return loss, {
            "beta":  beta,
            "elbo":  elbo.mean().item(),
            "recon": log_px.mean().item(),
            "kl":    (log_qz - log_pz).mean().item()
        }
    
class Encoder(zuko.lazy.LazyDistribution):
    """
    Gaussian encoder
    """
    def __init__(self, z_dim, c_dim, x_dim, hidden_dim=128, device=None):
        """
        z_dim: dimension of latent space
        c_dim: dimension of context
        x_dim: dimension of input, the random variable to model
        """
        super(Encoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and device is None else "cpu")
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + c_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim * 2)
        )

    def forward(self, x, c):
        c = ensure_batch_context(x, c)
        h = self.net(torch.cat([x, c], dim=1))
        mu, raw = h.chunk(2, dim=-1)
        # Stable positive scale: softplus + eps
        std = F.softplus(raw) + 1e-6
        return Independent(Normal(mu, std), 1)
    
class Decoder(zuko.lazy.LazyDistribution):
    def __init__(
        self,
        z_dim, c_dim, x_dim,
        hidden_dim=128,
        fixed_logvar=None,
        var_mode="tanh",          # "tanh" | "softplus"
        sigma_min=1e-3,           # min std
        sigma_max=1.0,            # only used for var_mode="tanh"
        device=None
    ):
        super().__init__()
        self.fixed_logvar = fixed_logvar
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.var_mode = var_mode
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max) if sigma_max is not None else None

        self.net = nn.Sequential(
            nn.Linear(z_dim + c_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*x_dim if fixed_logvar is None else x_dim)
        )

        if fixed_logvar is True:
            # learnable global log-variance per dimension
            self.logvar_param = nn.Parameter(torch.zeros(x_dim))
        else:
            self.logvar_param = None

    def _pos_std(self, raw):
        """Map unconstrained raw -> strictly positive std with a safe floor (and optional ceiling)."""
        if self.var_mode == "softplus":
            std = F.softplus(raw) + self.sigma_min
            if self.sigma_max is not None:
                std = torch.clamp(std, max=self.sigma_max)
            return std
        elif self.var_mode == "tanh":
            # Bound log-std between log(sigma_min) and log(sigma_max)
            # This avoids extreme tiny/huge variances cleanly.
            assert self.sigma_max is not None and self.sigma_max > self.sigma_min
            lo = math.log(self.sigma_min)
            hi = math.log(self.sigma_max)
            # map raw in R -> (0,1) via tanh, then to [lo,hi]
            s = 0.5 * (torch.tanh(raw) + 1.0)  # in (0,1)
            log_std = lo + s * (hi - lo)
            return torch.exp(log_std)
        else:
            raise ValueError(f"Unknown var_mode: {self.var_mode}")

    def forward(self, z, c):
        c = ensure_batch_context(z, c)
        h = self.net(torch.cat([z, c], dim=1))

        if self.fixed_logvar is None:
            mu, raw = h.chunk(2, dim=-1)
            std = self._pos_std(raw)
        elif self.fixed_logvar is True:
            mu = h
            std = torch.exp(0.5 * self.logvar_param).unsqueeze(0)
            std = torch.clamp(std, min=self.sigma_min)  # guard even the learned one
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
    def __init__(self, c_dim, z_dim, hidden_dim=128, device=None):
        super().__init__()
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() and device is None else "cpu")
        self.net = nn.Sequential(
            nn.Linear(c_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*z_dim)
            )

    def forward(self, c):
        # Use the same robust context handling
        # Build a dummy x to reuse ensure_batch_context’s shape logic
        if not isinstance(c, torch.Tensor):
            c = torch.as_tensor(c)
        dummy_x = torch.zeros(c.size(0) if c.dim() > 0 else 1, 1, device=c.device, dtype=c.dtype)
        c_batched = ensure_batch_context(dummy_x, c)
        h = self.net(c_batched)
        mu, raw = h.split(self.z_dim, dim=1)
        std = F.softplus(raw) + 1e-6
        return Independent(Normal(mu, std), 1)
    
class BetaScheduler:
    """
    β schedule for KL weight.
    Modes:
      - 'linear'  : 0 -> 1 over warmup_steps
      - 'cosine'  : slow start & end
      - 'sigmoid' : very gentle warmup, then sharp rise
      - 'constant': always beta_end
    """
    def __init__(self, warmup_steps, beta_start=0.0, beta_end=1.0, mode='linear'):
        self.warmup_steps = max(int(warmup_steps), 1)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.mode = mode
        self._step = 0

    @property
    def step(self):
        return self._step

    def reset(self):
        self._step = 0

    def update(self):
        self._step += 1

    def beta(self):
        if self.mode == 'constant':
            return self.beta_end
        t = min(self._step / self.warmup_steps, 1.0)  # 0..1
        if self.mode == 'linear':
            s = t
        elif self.mode == 'cosine':
            s = 0.5 - 0.5 * math.cos(math.pi * t)
        elif self.mode == 'sigmoid':
            # centered sigmoid; k controls steepness
            k = 10.0
            s = 1.0 / (1.0 + math.exp(-k*(t - 0.5)))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return (1 - s) * self.beta_start + s * self.beta_end
    
def train_vae(elbo, trainloader, lr=1e-4, weight_decay=0.0, epochs=10, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elbo.to(device)
    elbo.train()

    # Optionally exclude scale/logvar params from weight decay
    decay, no_decay = [], []
    for n, p in elbo.named_parameters():
        if any(k in n for k in ["logvar", "raw", "logvar_param"]):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.Adam(param_groups, lr=lr)

    for epoch in range(1, epochs + 1):
        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "elbo": 0.0, "beta": 0.0}
        n_batches = 0
        for c, x in trainloader:
            x = x.to(device)
            c = c.to(device)
            loss, metrics = elbo(x, c)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            elbo.step_beta()

            running["loss"] += loss.item()
            running["recon"] += metrics["recon"]
            running["kl"]    += metrics["kl"]
            running["elbo"]  += metrics["elbo"]
            running["beta"]  += metrics["beta"]
            n_batches += 1

        for k in running:
            running[k] /= max(n_batches, 1)

        print(f'Epoch {epoch}/{epochs} '
              f'- Loss: {running["loss"]:.4f} '
              f'- Recon: {running["recon"]:.4f} '
              f'- KL: {running["kl"]:.4f} '
              f'- Beta: {running["beta"]:.3f}')


    return elbo


class VAElinear(nn.Module):
    """
    Conditional VAE (MLP) with heteroscedastic decoder:
      q_phi(z|x,y)  -> outputs mu_z, log_var_z
      p_theta(x|z,y)-> outputs mu_x, log_var_x   (variance learned)

    x: R^{in_size},  y: R^{cond_in},  z: R^{latent_s}
    """

    def __init__(self, **kwargs):
        super().__init__()
        # Required
        self.latent_s = kwargs['latent_s']
        self.in_size  = kwargs['in_size']
        self.cond_in  = kwargs['cond_in']

        # Optional
        enc_w = kwargs.get('enc_w', 128)
        enc_l = kwargs.get('enc_l', 2)
        dec_w = kwargs.get('dec_w', 128)
        dec_l = kwargs.get('dec_l', 2)
        gpu   = kwargs.get('gpu', True)

        # Device
        self.device = torch.device("cuda:0" if (gpu and torch.cuda.is_available()) else "cpu")

        # Encoder: [x||y] -> [mu_z || log_var_z]
        enc_sizes = [self.in_size + self.cond_in] + [enc_w] * enc_l + [2 * self.latent_s]
        enc_layers = []
        for a, b in zip(enc_sizes[:-1], enc_sizes[1:]):
            enc_layers += [nn.Linear(a, b), nn.ReLU()]
        enc_layers.pop()                       # drop last ReLU
        self.enc = nn.Sequential(*enc_layers)

        # Decoder: [z||y] -> [mu_x || log_var_x]
        dec_sizes = [self.latent_s + self.cond_in] + [dec_w] * dec_l + [2 * self.in_size]
        dec_layers = []
        for a, b in zip(dec_sizes[:-1], dec_sizes[1:]):
            dec_layers += [nn.Linear(a, b), nn.ReLU()]
        dec_layers.pop()                       # drop last ReLU
        self.dec = nn.Sequential(*dec_layers)

        self.to(self.device)

    # -------- helpers --------
    def _ensure_2d(self, t):
        if t.dim() > 2:
            return t.view(t.size(0), -1)
        return t

    def encode(self, x, y):
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        h = torch.cat([x, y], dim=1)
        mu_z, log_var_z = torch.split(self.enc(h), self.latent_s, dim=1)
        return mu_z, log_var_z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        out = self.dec(zy)
        mu_x, log_var_x = torch.split(out, self.in_size, dim=1)
        # Clamp log-variance for numerical stability
        log_var_x = torch.clamp(log_var_x, min=-6.0, max=4.0)
        return mu_x, log_var_x

    # -------- core API --------
    def loss(self, x0, cond_in=None, beta: float = 1.0):
        """
        ELBO = E_q[ -log p(x|z,y) ] + beta * KL(q||p)
        -log p(x|z,y) for Gaussian = 0.5 * [ log_var_x + (x-mu_x)^2 / exp(log_var_x) ] (const dropped)
        """
        x0 = self._ensure_2d(x0).to(self.device)
        bs = x0.size(0)

        if cond_in is None:
            cond_in = torch.zeros(bs, self.cond_in, device=self.device, dtype=x0.dtype)
        else:
            cond_in = self._ensure_2d(cond_in).to(dtype=x0.dtype, device=self.device)

        # q(z|x,y)
        mu_z, log_var_z = self.encode(x0, cond_in)
        z = self.reparameterize(mu_z, log_var_z)

        # p(x|z,y)
        mu_x, log_var_x = self.decode(z, cond_in)

        # Gaussian NLL (per-sample sum over dims)
        inv_var_x = torch.exp(-log_var_x)
        recon_nll = 0.5 * (log_var_x + (x0 - mu_x).pow(2) * inv_var_x)
        recon = recon_nll.sum(dim=1).mean()

        # KL(q(z|x,y) || N(0,I))
        kl = 0.5 * (torch.exp(log_var_z) + mu_z.pow(2) - 1.0 - log_var_z)
        kl = kl.sum(dim=1).mean()

        total = recon + beta * kl
        return total, {'recon': recon.detach(), 'kl': kl.detach()}

    @torch.no_grad()
    def forward(self, x0, cond_in=None, return_latent=False):
        self.eval()
        x0 = self._ensure_2d(x0).to(self.device)
        bs = x0.size(0)
        if cond_in is None:
            cond_in = torch.zeros(bs, self.cond_in, device=self.device, dtype=x0.dtype)
        else:
            cond_in = self._ensure_2d(cond_in).to(dtype=x0.dtype, device=self.device)

        mu_z, log_var_z = self.encode(x0, cond_in)
        z = self.reparameterize(mu_z, log_var_z)
        mu_x, log_var_x = self.decode(z, cond_in)
        if return_latent:
            return (mu_x, log_var_x), (mu_z, log_var_z, z)
        return mu_x

    def to(self, device):
        ret = super().to(device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        return ret

    @torch.no_grad()
    def sample(self, n: int, c, sample_mean: bool = False, var_scale: float = 1.0):
        """
        Vectorized sampler with the same logic as Generator.sample:
          - c: (B, cond_in) or (cond_in,) or (B,)  -> normalized to (B, cond_in)
          - returns: (S=n, B, in_size)

        Args:
            n           : number of samples per context (S)
            c           : context(s)
            sample_mean : if True, return decoder mean instead of stochastic sample
            var_scale   : multiply the predictive variance by this factor during sampling
        """
        self.eval()
        device = self.device

        # --- normalize context to (B, cond_in) ---
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float32, device=device)
        else:
            c = c.to(device=device, dtype=torch.float32)

        if c.dim() == 1:
            # either (cond_in,) or (B,) scalar context; if scalar-per-sample, expand to (B,1)
            if c.numel() == self.cond_in:
                c = c.view(1, -1)  # single context -> (1, cond_in)
            else:
                c = c.view(-1, 1)  # per-sample scalar -> (B, 1)

        elif c.dim() == 2:
            pass  # already (B, cond_in)
        else:
            # flatten any higher dims: (B, *, cond_in) -> (B, cond_in)
            c = c.view(c.size(0), -1)

        B = c.size(0)
        assert c.size(1) == self.cond_in, f"Context dim mismatch: got {c.size(1)}, expected {self.cond_in}"

        # --- draw latent z for each (S,B) pair ---
        z = torch.randn(n * B, self.latent_s, device=device)

        # --- repeat context for each of the S samples ---
        c_rep = c.repeat(n, 1)  # (S*B, cond_in)

        # --- decode to (S*B, in_size) params ---
        mu_x, log_var_x = self.decode(z, c_rep)

        if sample_mean:
            x = mu_x
        else:
            std = torch.exp(0.5 * log_var_x) * (var_scale ** 0.5)
            eps = torch.randn_like(std)
            x = mu_x + std * eps

        # --- reshape to (S, B, in_size) ---
        x = x.view(n, B, self.in_size)
        return x

    @torch.no_grad()
    def quantiles(self, c, q=(0.25, 0.5, 0.75), n: int = 100, sample_mean: bool = False, var_scale: float = 1.0):
        """
        Match Generator.quantiles:
          - c: flexible shapes as in .sample
          - returns np.array with shape (len(q), B, in_size)
        """
        y = self.sample(n=n, c=c, sample_mean=sample_mean, var_scale=var_scale)  # (S, B, D)
        y = y.detach().cpu().numpy()
        qv = np.quantile(y, q, axis=0)  # (len(q), B, D)
        return qv


def plot_training(history, save_path=None, title='Learning curve'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['train_loss'], label='loss')
    ax.plot(history['train_recon'], label='recon')
    ax.plot(history['train_kl'], label='kl')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path)

    return fig, ax

def make_24h_forecast_with_bands(vae, context, samples=100):
    vae.eval()
    device = next(vae.parameters()).device

    Q1, median, Q3 = [], [], []
    context = context.to(device)

    prediction_quantiles = vae.quantiles(context, q=[0.25, 0.50, 0.75], n=samples)
    Q1 = prediction_quantiles[0,:,:]
    median = prediction_quantiles[1,:,:]
    Q3 = prediction_quantiles[2,:,:]

    return Q1, median, Q3

def vae_losses(vae, dataloader):
    device = next(vae.parameters()).device
    vae.eval()
    losses = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = x.reshape(x.size(0), -1)
            loss = -vae.log_prob(y, c).mean()
            losses.append(loss)

    return np.array(losses)

def evaluate_vae(vae, test_dataloader, model_label, save_path=None, **kwargs):
    n_samples = kwargs.get('samples', 20)
    device = kwargs.get('device', next(vae.parameters()).device)

    vae.eval()
    all_crps, all_energy, all_vario = [], [], []

    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc=f"Evaluating", leave=False)
        for x, label in pbar:
            x = x.to(device)
            label = label.to(device)

            c_batch = x.reshape(x.size(0), -1)   # [B, c_dim]
            x_batch = label                   # [B, x_dim]

            y_samps = vae.sample(n_samples, c_batch)     # (S, B, D)
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