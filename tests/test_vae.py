import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import pytest

from torch.utils.data import DataLoader, TensorDataset

from models import vae as vae_mod


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(0)
    np.random.seed(0)
    yield


def test_ensure_batch_context_vae_shapes():
    x = torch.zeros(3, 5)
    # 1D per-sample
    c = [1, 2, 3]
    out = vae_mod.ensure_batch_context(x, c)
    assert out.shape == (3, 1)

    c2 = torch.randn(1, 2)
    out2 = vae_mod.ensure_batch_context(x, c2)
    assert out2.shape == (3, 2)

    c3 = torch.randn(3, 2)
    out3 = vae_mod.ensure_batch_context(x, c3)
    assert out3.shape == (3, 2)


def test_beta_scheduler_modes_and_reset():
    # linear
    s = vae_mod.BetaScheduler(warmup_steps=4, beta_start=0.0, beta_end=1.0, mode='linear')
    assert math.isclose(s.beta(), 0.0)
    s.update()
    assert math.isclose(s.beta(), 0.25)
    s.reset()
    assert s.step == 0

    # cosine
    s2 = vae_mod.BetaScheduler(4, 0.0, 1.0, mode='cosine')
    s2.update(); s2.update()
    b2 = s2.beta()
    assert 0.0 <= b2 <= 1.0

    # sigmoid
    s3 = vae_mod.BetaScheduler(4, 0.0, 1.0, mode='sigmoid')
    s3.update(); s3.update()
    assert 0.0 <= s3.beta() <= 1.0

    # constant
    s4 = vae_mod.BetaScheduler(4, 0.2, 0.8, mode='constant')
    assert math.isclose(s4.beta(), 0.8)

    # unknown mode raises
    s5 = vae_mod.BetaScheduler(4, 0.0, 1.0, mode='linear')
    s5.mode = 'unknown'
    with pytest.raises(ValueError):
        _ = s5.beta()


def test_encoder_decoder_prior_shapes_and_variance_modes():
    z_dim, c_dim, x_dim = 2, 3, 4
    B = 5
    enc = vae_mod.Encoder(z_dim=z_dim, c_dim=c_dim, x_dim=x_dim, hidden_dim=8)
    dec = vae_mod.Decoder(z_dim=z_dim, c_dim=c_dim, x_dim=x_dim, hidden_dim=8, var_mode='softplus')
    prior = vae_mod.Prior(c_dim=c_dim, z_dim=z_dim)

    x = torch.randn(B, x_dim)
    c = torch.randn(B, c_dim)

    q = enc(x, c)
    z = q.rsample()
    assert z.shape == (B, z_dim)

    pz = prior(c)
    zp = pz.rsample()
    assert zp.shape == (B, z_dim)

    px = dec(z, c)
    xs = px.rsample()
    assert xs.shape == (B, x_dim)

    # test tanh mode requires sigma_max provided
    dec2 = vae_mod.Decoder(z_dim=z_dim, c_dim=c_dim, x_dim=x_dim, hidden_dim=8, var_mode='tanh', sigma_min=1e-3, sigma_max=1.0)
    px2 = dec2(z, c)
    assert px2.rsample().shape == (B, x_dim)

    # fixed_logvar True
    dec3 = vae_mod.Decoder(z_dim=z_dim, c_dim=c_dim, x_dim=x_dim, fixed_logvar=True)
    p3 = dec3(torch.randn(B, z_dim), c)
    assert p3.rsample().shape == (B, x_dim)


def test_vaelinear_basic_operations_and_sampling(tmp_path):
    latent_s = 2
    in_size = 4
    cond_in = 3
    vae = vae_mod.VAElinear(latent_s=latent_s, in_size=in_size, cond_in=cond_in, enc_w=16, enc_l=1, dec_w=16, dec_l=1, gpu=False)

    B = 6
    x = torch.randn(B, in_size)
    c = torch.randn(B, cond_in)

    mu_z, log_var_z = vae.encode(x, c)
    assert mu_z.shape == (B, latent_s)

    z = vae.reparameterize(mu_z, log_var_z)
    assert z.shape == (B, latent_s)

    mu_x, log_var_x = vae.decode(z, c)
    assert mu_x.shape == (B, in_size)

    loss, logs = vae.loss(x, c, beta=0.5)
    assert torch.is_tensor(loss) and loss.dim() == 0
    assert 'recon' in logs and 'kl' in logs

    # forward and return_latent
    out = vae.forward(x, c)
    assert out.shape == (B, in_size)

    (mu_x2, lv2), (mu_z2, lvz2, z2) = vae.forward(x, c, return_latent=True)
    assert mu_x2.shape == (B, in_size)

    samples = vae.sample(3, c)
    assert samples.shape == (3, B, in_size)

    qs = vae.quantiles(c, q=(0.25, 0.5), n=10)
    assert isinstance(qs, np.ndarray)


def test_plot_vae_training_and_train_loop_short(tmp_path):
    # plot
    history = {'train_loss': [1.0, 0.8], 'train_recon': [0.9, 0.7], 'train_kl': [0.1, 0.1]}
    out = tmp_path / 'vae_plot.png'
    fig, ax = vae_mod.plot_vae_training(history, save_path=str(out), title='Test VAE')
    assert out.exists()

    # short train loop with tiny dataset
    latent_s = 2
    in_size = 4
    cond_in = 3
    vae = vae_mod.VAElinear(latent_s=latent_s, in_size=in_size, cond_in=cond_in, enc_w=16, enc_l=1, dec_w=16, dec_l=1, gpu=False)

    B = 16
    x = torch.randn(B, in_size)
    c = torch.randn(B, cond_in)

    ds = TensorDataset(c, x)
    dl = DataLoader(ds, batch_size=8)

    # run 2 epochs only
    trained, h = vae_mod.train_vae_linear(vae, dl, validation_dataloader=dl, lr=1e-3, epochs=2)
    assert 'train_loss' in h
    assert len(h['train_loss']) == 2
