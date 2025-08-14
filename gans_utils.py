import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
        self.device = torch.device("cuda" if torch.cuda.is_available() and device is None else "cpu")
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

    @torch.no_grad()
    def sample(self, n, c):          # n scenarios for context c
        device = next(self.parameters()).device
        S = []
        for _ in range(n):
            z = torch.randn(1, self.z_dim, device=device)
            S.append(self.forward(z, c).squeeze(0))
        return torch.stack(S)


    @torch.no_grad()
    def quantiles(self, c, q=100, n=100):
        y = self.sample(n, c)
        y = y.detach().cpu().numpy()
        qv = np.quantile(y, q, axis=0)
        return qv
    
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
        self.device = torch.device("cuda" if torch.cuda.is_available() and device is None else "cpu")
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

def train_wgan_gp(generator, discriminator, train_loader, g_lr=1e-4, d_lr=1e-4, gp_lambda=10, n_critic=5, epochs=20):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.0, 0.9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.0, 0.9))

    device = next(generator.parameters()).device
    z_dim = generator.z_dim

    discriminator.train()
    generator.train()

    for epoch in range(1, epochs+1):
        # samples_0 = generator.sample(10000, torch.tensor([0.0]))
        # samples_1 = generator.sample(10000, torch.tensor([1.0]))
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # axs[0].hist2d(*samples_0.T, bins=64, range=((-2, 2), (-2, 2)))
        # axs[0].set_title('cond=0')
        # axs[1].hist2d(*samples_1.T, bins=64, range=((-2, 2), (-2, 2)))
        # axs[1].set_title('cond=1')
        # plt.show()
        for i, (c_batch, y_real) in enumerate(train_loader):
            c_batch = c_batch.to(device)
            y_real = y_real.to(device)
            batch_size = y_real.size(0)
            
            # Train discriminator
            for _ in range(n_critic):
                z = torch.randn(batch_size, z_dim, device=device)
                y_fake = generator.forward(z, c_batch).detach()

                d_real = discriminator(y_real, c_batch)
                d_fake = discriminator(y_fake, c_batch)

                gp = gradient_penalty(discriminator, y_real, y_fake, c_batch)
                d_loss = -(d_real.mean() - d_fake.mean()) + gp_lambda * gp

                d_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                d_optimizer.step()

            # Train generator
            z = torch.randn(batch_size, z_dim, device=device)
            y_fake = generator(z, c_batch)
            d_fake = discriminator(y_fake, c_batch)
            g_loss = -d_fake.mean()

            g_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            g_optimizer.step()

            if i % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Batch [{i}/{len(train_loader)}], '
                        f'Disc. Loss: {d_loss.item():.4f}, Gen. Loss: {g_loss.item():.4f}')
                
    
    return discriminator, generator