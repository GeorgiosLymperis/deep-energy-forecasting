import torch
import torch.nn as nn
import numpy as np
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

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, y_dim, hidden_dim=128, device=None):
        """
        z_dim: dimension of latent space
        c_dim: dimension of context
        y_dim: dimension of output
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

    def forward(self, z, c):
        c = ensure_batch_context(z, c)
        x = torch.cat([z, c], dim=1)
        return self.net(x)
    
    def _weights_initialize(self, mean: float, std: float):
        """
        Initialize self model parameters following a normal distribution based on mean and std
        :param mean: mean of the standard distribution
        :param std : standard deviation of the normal distribution
        :return: None
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=mean, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # @torch.no_grad()
    # def sample(self, n, c):          # n scenarios for context c
    #     device = next(self.parameters()).device
    #     S = []
    #     for _ in range(n):
    #         z = torch.randn(1, self.z_dim, device=device)
    #         S.append(self.forward(z, c).squeeze(0))
    #     return torch.stack(S)
    
    @torch.no_grad()
    def sample(self, n, c):
        """
        Returns S samples for each context in c.
        c: (B, c_dim) or (c_dim,) or (B,)
        -> returns (S, B, y_dim)
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


    # @torch.no_grad()
    # def quantiles(self, c, q=100, n=100):
    #     y = self.sample(n, c)
    #     y = y.detach().cpu().numpy()
    #     qv = np.quantile(y, q, axis=0)
    #     return qv
    
    @torch.no_grad()
    def quantiles(self, c, q=(0.25, 0.5, 0.75), n=100):
        """
        Returns np.array with shape (len(q), B, y_dim).
        """
        y = self.sample(n, c)                      # (S, B, D)
        y = y.detach().cpu().numpy()
        qv = np.quantile(y, q, axis=0)            # (len(q), B, D)
        return qv
    
    def get_config(self):
        return {
            "z_dim": self.z_dim,
            "c_dim": self.c_dim,
            "y_dim": self.y_dim,
            "hidden_dim": self.hidden_dim,
            "device": str(self.device),
        }
    
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
    
class Discriminator(nn.Module):
    def __init__(self, c_dim, y_dim, hidden_dim=128, alpha=0.01, device=None):
        """
        c_dim: dimension of context
        y_dim: dimension of Generator output
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

    def forward(self, z, c):
        c = ensure_batch_context(z, c)
        x = torch.cat([z, c], dim=1)
        return self.net(x)
    
    def _weights_initialize(self, mean: float, std: float):
        """
        Initialize self model parameters following a normal distribution based on mean and std
        :param mean: mean of the standard distribution
        :param std : standard deviation of the normal distribution
        :return: None
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=mean, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_config(self):
        return {
            "c_dim": self.c_dim,
            "y_dim": self.y_dim,
            "hidden_dim": self.hidden_dim,
            "alpha": self.alpha,
            "device": str(self.device),
        }
    
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

    
def gradient_penalty(discriminator, real_data, fake_data, c):
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
    )[0] # (batch, y_dim)

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgan_gp(
        generator, discriminator, trainloader, validationloader=None, g_lr=1e-4, d_lr=1e-4, 
        gp_lambda=10, n_critic=5, epochs=20, save_path=None, patience=20
        ):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.0, 0.9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.0, 0.9))

    g_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, mode='min', factor=0.5, patience=3)
    d_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, mode='min', factor=0.5, patience=3)

    device = next(generator.parameters()).device
    z_dim = generator.z_dim

    best_val = float('inf')
    best_state = None
    bad_epochs = 0

    history = {
        "d_loss": [],
        "g_loss": [],
        "val_loss": []
    }
    for epoch in range(1, epochs+1):
        # samples_0 = generator.sample(10000, torch.tensor([0.0]))
        # samples_1 = generator.sample(10000, torch.tensor([1.0]))
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # axs[0].hist2d(*samples_0.T, bins=64, range=((-2, 2), (-2, 2)))
        # axs[0].set_title('cond=0')
        # axs[1].hist2d(*samples_1.T, bins=64, range=((-2, 2), (-2, 2)))
        # axs[1].set_title('cond=1')
        # plt.show()
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for c_batch, y_real in pbar:
            c_batch = c_batch.to(device)
            y_real = y_real.to(device)
            batch_size = y_real.size(0)
            d_losses = []
            # Train discriminator
            for _ in range(n_critic):
                discriminator.train()
                z = torch.randn(batch_size, z_dim, device=device)
                y_fake = generator.forward(z, c_batch).detach()

                d_real = discriminator(y_real, c_batch)
                d_fake = discriminator(y_fake, c_batch)

                gp = gradient_penalty(discriminator, y_real, y_fake, c_batch)
                d_loss = -(d_real.mean() - d_fake.mean()) + gp_lambda * gp

                d_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                d_optimizer.step()

                d_losses.append(d_loss.detach().cpu().item())
            d_loss = np.mean(d_losses)
            history['d_loss'].append(d_loss)
            # d_sched.step(d_loss)


            # Train generator
            generator.train()
            z = torch.randn(batch_size, z_dim, device=device)
            y_fake = generator(z, c_batch)
            d_fake = discriminator(y_fake, c_batch)
            g_loss = -d_fake.mean()

            history['g_loss'].append(g_loss.detach().cpu().item())

            g_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            g_optimizer.step()
            # g_sched.step(g_loss)

            if validationloader is not None:
                with torch.no_grad():
                    generator.eval()
                    discriminator.eval()
                    val_losses = []
                    for val_c, val_y in validationloader:
                        val_c = val_c.to(device)
                        val_y = val_y.to(device)
                        val_batch_size = val_y.size(0)
                        val_z = torch.randn(val_batch_size, z_dim, device=device)
                        val_y_fake = generator(val_z, val_c)
                        val_d_fake = discriminator(val_y_fake, val_c)
                        val_losses.append(-val_d_fake.mean().detach().cpu().item())
                    val_loss = np.mean(val_losses)
                    history['val_loss'].append(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = (generator.state_dict(), discriminator.state_dict())
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        generator.load_state_dict(best_state[0])
                        discriminator.load_state_dict(best_state[1])
                        break
        tqdm.write(f"Epoch {epoch}/{epochs} [train] Disc. Loss: {np.mean(d_losses):.4f}, Gen. Loss: {g_loss:.4f}"
                    f"Val. Loss: {np.mean(val_loss):.4f}" if validationloader is not None else 
                    f"Epoch {epoch}/{epochs} [train] Disc. Loss: {np.mean(d_losses):.4f}, Gen. Loss: {g_loss:.4f}")
                
    if save_path is not None:
        torch.save(generator.state_dict(), "gen_" + save_path)
        torch.save(discriminator.state_dict(), "disc_" +  save_path)
    return discriminator, generator, history

def train_wgan_gp_(
    generator, discriminator, trainloader, validationloader=None,
    g_lr=1e-4, d_lr=1e-4, gp_lambda=10, n_critic=5, epochs=20,
    save_path=None, patience=20
):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.0, 0.9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.0, 0.9))

    # g_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, mode='min', factor=0.5, patience=3)
    # d_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, mode='min', factor=0.5, patience=3)

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

        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
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
                d_loss = -(d_real.mean() - d_fake.mean()) + gp_lambda * gp + epsilon_drift * (d_real**2).mean()

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
            # g_sched.step(val_loss)
            # d_sched.step(d_epoch)

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                best_state = (generator.state_dict(), discriminator.state_dict())
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
            # g_sched.step(g_epoch)
            # d_sched.step(d_epoch)
            pass

        history["d_loss"].append(d_epoch)
        history["g_loss"].append(g_epoch)

        tqdm.write(
            f"Epoch {epoch}/{epochs} | D: {d_epoch:.4f} | G: {g_epoch:.4f}" +
            (f" | Val: {history['val_loss'][-1]:.4f}" if validationloader is not None else "")
        )

    if save_path is not None:
        torch.save(generator.state_dict(), f"gen_{save_path}")
        torch.save(discriminator.state_dict(), f"disc_{save_path}")

    return discriminator, generator, history

def plot_training(history, save_path=None, title='Learning losses'):
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

def evaluate_gan(generator, testloader, model_label, save_path=None, **kwargs):
    n_samples = kwargs.get('samples', 20)
    device = kwargs.get('device', next(generator.parameters()).device)

    generator.eval()
    all_crps, all_energy, all_vario = [], [], []

    with torch.no_grad():
        pbar = tqdm(testloader, desc=f"Evaluating", leave=False)
        for c, label in pbar:
            c = c.to(device)
            label = label.to(device)

            c_batch = c.reshape(c.size(0), -1)   # [B, c_dim]
            x_batch = label                   # [B, x_dim]

            y_samps = generator.sample(n_samples, c_batch)     # (S, B, D)
            y_np = y_samps.detach().cpu().numpy()
            x_np = x_batch.detach().cpu().numpy()

            all_crps.append(crps_batch_per_marginal(y_np, x_np))
            all_energy.append(energy_score_per_batch(y_np, x_np))
            all_vario.append(variogram_score_per_batch(y_np, x_np))

    results = {
        'label': model_label,
        'crps': float(np.mean(all_crps)),
        'energy': float(np.mean(all_energy)),
        'variogram': float(np.mean(all_vario))
    }

    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(results, f)
    return results

def make_24h_forecast_with_bands(generator, context, samples=100):
    """
    Makes a forecast for the next 24 hours

    generator: Trained flow
    context: tensor with context the weather conditions of the day [B, c_dim]
             Context must be scaled
    samples: the size of the samples generated by flow

    Returns: (Q1, median, Q3) numpy arrays
    """
    generator.eval()
    device = next(generator.parameters()).device

    Q1, median, Q3 = [], [], []
    context = context.to(device)

    prediction_quantiles = generator.quantiles(context, [0.25, 0.50, 0.75], n=samples)
    Q1 = prediction_quantiles[0,:,:]
    median = prediction_quantiles[1,:,:]
    Q3 = prediction_quantiles[2,:,:]

    return Q1, median, Q3

def generator_losses(generator, dataloader):
    """
    Generate losses for Diebold-Mariano test.
    """
    device = next(generator.parameters()).device
    generator.eval()
    losses = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = x.reshape(x.size(0), -1)
            loss = -generator.log_prob(y, c).mean()
            losses.append(loss)

    return np.array(losses)