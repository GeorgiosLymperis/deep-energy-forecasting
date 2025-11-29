import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import pytest

from models.wgan import (
    ensure_batch_context,
    Generator,
    Discriminator,
    gradient_penalty,
    plot_gan_training,
)


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Make tests deterministic where randomness is used."""
    torch.manual_seed(0)
    np.random.seed(0)
    yield


def test_ensure_batch_context_various_shapes():
    x = torch.zeros(4, 3)

    # 1D list/array per-sample
    c1 = [1, 2, 3, 4]
    out1 = ensure_batch_context(x, c1)
    assert out1.shape == (4, 1)

    # (1, c_dim) should broadcast to (B, c_dim)
    c2 = torch.randn(1, 2)
    out2 = ensure_batch_context(x, c2)
    assert out2.shape == (4, 2)

    # (B, c_dim) unchanged
    c3 = torch.randn(4, 2)
    out3 = ensure_batch_context(x, c3)
    assert out3.shape == (4, 2)

    # (B, *, *) flattened to (B, -1)
    c4 = torch.randn(4, 2, 1)
    out4 = ensure_batch_context(x, c4)
    assert out4.shape == (4, 2)

    # mismatched 1D length should raise
    with pytest.raises(ValueError):
        ensure_batch_context(x, [1, 2, 3])


def test_generator_forward_sample_quantiles_and_save_load(tmp_path):
    z_dim, c_dim, y_dim = 2, 3, 4
    gen = Generator(z_dim=z_dim, c_dim=c_dim, y_dim=y_dim, hidden_dim=8, device='cpu')

    B = 5
    z = torch.randn(B, z_dim)
    c = torch.randn(B, c_dim)

    out = gen(z, c)
    assert out.shape == (B, y_dim)

    samples = gen.sample(3, c)
    assert samples.shape == (3, B, y_dim)

    q = gen.quantiles(c, q=(0.5,), n=10)
    assert isinstance(q, np.ndarray)
    assert q.shape == (1, B, y_dim)

    # save and load
    p = tmp_path / "gen_test.pth"
    gen.save(str(p))
    loaded = Generator.load_from(str(p), map_location='cpu')

    # config should match keys and shapes
    cfg = gen.get_config()
    lcfg = loaded.get_config()
    for k in ['z_dim', 'c_dim', 'y_dim']:
        assert cfg[k] == lcfg[k]

    # loaded model should produce outputs of correct shape
    z2 = torch.randn(2, z_dim)
    c2 = torch.randn(2, c_dim)
    out2 = loaded(z2, c2)
    assert out2.shape == (2, y_dim)


def test_discriminator_forward_and_gradient_penalty():
    c_dim, y_dim = 3, 4
    disc = Discriminator(c_dim=c_dim, y_dim=y_dim, hidden_dim=8, device='cpu')

    B = 4
    y_real = torch.randn(B, y_dim, requires_grad=True)
    y_fake = torch.randn(B, y_dim)
    c = torch.randn(B, c_dim)

    out = disc(y_real, c)
    assert out.shape == (B, 1)

    gp = gradient_penalty(disc, y_real.detach(), y_fake.detach(), c)
    assert torch.is_tensor(gp)
    assert gp.dim() == 0
    assert gp.item() >= 0


def test_plot_gan_training_saves_file(tmp_path):
    history = {'d_loss': [1.0, 0.5, 0.2], 'g_loss': [1.2, 0.6, 0.3]}
    out_path = tmp_path / 'train_plot.png'
    fig, ax = plot_gan_training(history, save_path=str(out_path), title='Test')

    assert fig is not None
    assert ax is not None
    assert out_path.exists()
