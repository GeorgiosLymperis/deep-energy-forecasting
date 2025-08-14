from gans_utils import Generator, Discriminator, train_wgan_gp
from utils_data import GEFcomLoadLoader

dataset = GEFcomLoadLoader()
dataset.split(test_size=0.05, validation_size=0.05)

train_dataloader, validation_dataloader, test_dataloader = dataset.get_dataloaders(batch_size=64)

x_dim = 24
c_dim = 24 * len(dataset.features)

generator = Generator(z_dim=256, c_dim=c_dim, y_dim=x_dim, hidden_dim=256 * 4)
discriminator = Discriminator(c_dim=c_dim, y_dim=x_dim,hidden_dim=256 * 4)

train_wgan_gp(generator, discriminator, train_dataloader, d_lr=5e-5, g_lr=5e-5, epochs=200)
