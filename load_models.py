from gans_utils import (
    Generator, Discriminator, train_wgan_gp_, plot_gan_training)
from utils_data import GEFcomLoadLoader, create_load_dataset
import torch
import numpy as np
from tqdm import tqdm
from evaluation.evaluation_utils import (
                            evaluate_model, plot_load_forecasts, plot_qs, plot_crps, 
                            plot_correlations, plot_dm, plot_roc_many_scenarios,
                            collect_fake_features, collect_real_features
                            )
from VAE_utils import VAElinear, plot_vae_training
from nf_utils import NAF, NSF, plot_nf_training

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.init()
    _ = torch.empty(1, device='cuda')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = create_load_dataset()
train_dataset = dataset[dataset["TIMESTAMP"] < "2011-12-08 00:00:00"]
prediction_dataset = dataset[dataset["TIMESTAMP"] >= "2011-12-08 00:00:00"]
print(len(prediction_dataset))
train_loader = GEFcomLoadLoader(train_dataset.copy())

train_dataloader, validation_dataloader, test_dataloader = train_loader.get_dataloaders(
                                                                batch_size=32, shuffle=True, use_gpu=True, test_size=0.1, validation_size=0.5
                                                                )

prediction_loader = GEFcomLoadLoader(prediction_dataset.copy(), x_scaler=train_loader.x_scaler, y_scaler=train_loader.y_scaler)

prediction_dataloader, _, _ = prediction_loader.get_dataloaders(
                                            batch_size=32, use_gpu=True, shuffle=False, 
                                            test_size=0.0, validation_size=0.0
                                            )
prediction_dataset = prediction_loader.create_dataset(shuffle=False)
c_dim = train_loader.context_dim
x_dim = 24

# ---------- Train GAN ----------

EPOCHS = 1
D_LR = 2e-4
GP_LAMBDA = 5
N_CRITIC = 5

generator_path = None
discriminator_path = None

if generator_path is not None:
    generator = Generator(z_dim=256, c_dim=c_dim, y_dim=x_dim, hidden_dim=256 * 4, device=device)
    generator.load_from(generator_path)
    discriminator = Discriminator(c_dim=c_dim, y_dim=x_dim,hidden_dim=256 * 4, device=device)
    discriminator.load_from(discriminator_path)
else:
    generator = Generator(z_dim=256, c_dim=c_dim, y_dim=x_dim, hidden_dim=256 * 4, device=device)
    discriminator = Discriminator(c_dim=c_dim, y_dim=x_dim,hidden_dim=256 * 4, device=device)

    discriminator, generator, history = train_wgan_gp_(generator, discriminator, train_dataloader, validationloader=validation_dataloader, d_lr=D_LR, epochs=EPOCHS, gp_lambda=GP_LAMBDA, n_critic=N_CRITIC)
    discriminator.save('models/GAN_DISCRIMINATOR_LOAD.pth')
    generator.save('models/GAN_GENERATOR_LOAD.pth')
    plot_gan_training(history, title='Learning curve (GAN)', save_path='plots/GAN_LEARNING_CURVE_LOAD.png')

results = evaluate_model(generator, test_dataloader, "GAN_LOAD", save_path='evaluations/GAN_EVALUATION.pkl', device=device)
plot_load_forecasts(generator, prediction_loader, save_path='plots/GAN_FORECASTS_LOAD.jpg')

gan_crps = results['crps'] # (days, T)
gan_energy = results['energy'] # (days,)
gan_vario = results['variogram'] # (days,)
gan_quantiles = results['quantile'] # (99, days, T)
gan_qs = np.mean(gan_quantiles, axis=0) # (days, T)

c_pred = prediction_dataset.sample(1).drop(columns=['LOAD' + str(h) for h in range(1, 25)])
plot_correlations(generator, c_pred, train_loader.x_scaler, save_path='plots/GAN_CORRELATIONS_LOAD.jpg', title='GAN correlations (Solar)')

# ---------- Train VAE ----------

EPOCHS = 1
vae = VAElinear(latent_s=8, cond_in=c_dim, in_size=x_dim,
                gpu=True, enc_w=128, enc_l=3, dec_w=128, dec_l=3)
opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

history = {
    'train_loss': [], 'train_recon': [], 'train_kl': [],
    'val_loss':   [], 'val_recon':   [], 'val_kl':   [],
    'beta':       [],
}

best = float('inf')
patience = 20
bad_epochs = 0  # <-- initialize

for epoch in range(EPOCHS):
    beta = min(1.0, (epoch + 1) / 50.0)

    # ---- train ----
    vae.train()
    t_loss = t_recon = t_kl = 0.0
    train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [train] β={beta:.2f}", leave=False)
    for i, (c_batch, x) in enumerate(train_bar, start=1):
        c_batch = c_batch.to(vae.device, dtype=torch.float32)
        x = x.to(vae.device, dtype=torch.float32)

        loss, logs = vae.loss(x, c_batch, beta=beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 5.0)
        opt.step()

        t_loss  += loss.item()
        t_recon += logs['recon'].item()
        t_kl    += (beta * logs['kl']).item()

        train_bar.set_postfix(loss=t_loss/i, recon=t_recon/i, kl=t_kl/i)

    history['train_loss'].append(t_loss / i)
    history['train_recon'].append(t_recon / i)
    history['train_kl'].append(t_kl / i)

    # ---- validate ----
    vae.eval()
    v_loss = v_recon = v_kl = 0.0
    with torch.no_grad():
        val_bar = tqdm(validation_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", leave=False)
        for j, (c_batch, x) in enumerate(val_bar, start=1):
            c_batch = c_batch.to(vae.device, dtype=torch.float32)
            x = x.to(vae.device, dtype=torch.float32)

            loss, logs = vae.loss(x, c_batch, beta=beta)
            v_loss  += loss.item()
            v_recon += logs['recon'].item()
            v_kl    += (beta * logs['kl']).item()

            val_bar.set_postfix(loss=v_loss/j, recon=v_recon/j, kl=v_kl/j)

    history['val_loss'].append(v_loss / j)
    history['val_recon'].append(v_recon / j)
    history['val_kl'].append(v_kl / j)
    history['beta'].append(beta)

    # epoch summary
    tqdm.write(
        f"Epoch {epoch+1:03d} | "
        f"train {history['train_loss'][-1]:.4f} "
        f"(recon={history['train_recon'][-1]:.4f}, kl={history['train_kl'][-1]:.4f}) | "
        f"val {history['val_loss'][-1]:.4f} "
        f"(recon={history['val_recon'][-1]:.4f}, kl={history['val_kl'][-1]:.4f}) | "
        f"β={beta:.2f}"
    )

    # ---- early stopping ----
    if history['val_loss'][-1] < best - 1e-6:
        best = history['val_loss'][-1]
        bad_epochs = 0
        torch.save(vae.state_dict(), 'models/vae_LOAD.pth')
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            tqdm.write(f"Early stopping at epoch {epoch+1}. Best val_loss={best:.4f}")
            break

vae.load_state_dict(torch.load('models/vae_LOAD.pth', map_location=vae.device))
plot_vae_training(history, title='Learning curve (VAE)', save_path='plots/VAE_LOAD.jpg')
plot_load_forecasts(vae, prediction_loader, save_path='plots/VAE_FORECASTS_LOAD.jpg')
plot_correlations(vae, c_pred, train_loader.x_scaler, save_path='plots/VAE_CORRELATIONS_LOAD.jpg', title='VAE correlations (Solar)')

results = evaluate_model(vae, test_dataloader, "VAE_LOAD", save_path='evaluations/VAE_EVALUATION.pkl', device=device)
vae_crps = results['crps'] # (days, T)
vae_energy = results['energy'] # (days,)
vae_vario = results['variogram'] # (days,)
vae_quantiles = results['quantile'] # (99, days, T)
vae_qs = np.mean(vae_quantiles, axis=0) # (days, T)

#______________ Training NAF ________________
EPOCHS = 1
lr = 1e-3

flow = NAF(x_dim=x_dim, c_dim=c_dim, hidden_features=[16, 16], signal=8)
history = flow.fit(train_dataloader, validation_dataloader=validation_dataloader, epochs=EPOCHS, lr=lr, patience=20, device=device, save_path='models/NAF_MODEL_LOAD.pt')
plot_nf_training(history, title='Learning curve (NAF)', save_path='plots/NAF_LEARNING_CURVE_LOAD.png')
plot_load_forecasts(flow, prediction_loader, save_path='plots/NAF_FORECASTS_LOAD.jpg')
plot_correlations(flow, c_pred, train_loader.x_scaler, save_path='plots/NAF_CORRELATIONS_LOAD.jpg', title='NAF correlations (Solar)')

results = evaluate_model(flow, test_dataloader, "NAF_LOAD", save_path='evaluations/NAF_EVALUATION.pkl', device=device)

naf_crps = results['crps'] # (days, T)
naf_energy = results['energy'] # (days,)
naf_vario = results['variogram'] # (days,)
naf_quantiles = results['quantile'] # (99, days, T)
naf_qs = np.mean(naf_quantiles, axis=0) # (days, T)

#______________ Training NSF ________________
EPOCHS = 1
lr = 1e-3

flow = NSF(x_dim=x_dim, c_dim=c_dim, hidden_features=[16, 16], transforms=3)
history = flow.fit(train_dataloader, validation_dataloader=validation_dataloader, epochs=EPOCHS, lr=lr, patience=20, device=device, save_path='models/NSF_MODEL_LOAD.pt')
plot_nf_training(history, title='Learning curve (NSF)', save_path='plots/NSF_LEARNING_CURVE_LOAD.png')
plot_load_forecasts(flow, prediction_loader, save_path='plots/NSF_FORECASTS_LOAD.jpg')
plot_correlations(flow, c_pred, train_loader.x_scaler, save_path='plots/NSF_CORRELATIONS_LOAD.jpg', title='NSF correlations (Solar)')

results = evaluate_model(flow, test_dataloader, "NSF_LOAD", save_path='evaluations/NSF_EVALUATION.pkl', device=device)

nsf_crps = results['crps'] # (days, T)
nsf_energy = results['energy'] # (days,)
nsf_vario = results['variogram'] # (days,)
nsf_quantiles = results['quantile'] # (99, days, T)
nsf_qs = np.mean(nsf_quantiles, axis=0) # (days, T)

plot_crps([gan_crps, vae_crps, naf_crps, nsf_crps], ['GAN', 'VAE', 'NAF', 'NSF'], save_path='plots/CRPS_LOAD.png', title='CRPS (Solar)')
plot_qs([gan_quantiles, vae_quantiles, naf_quantiles, nsf_quantiles], ['GAN', 'VAE', 'NAF', 'NSF'], save_path='plots/QS_LOAD.png', title='Quantile score (LOAD)')

labels = ['GAN', 'VAE', 'NAF', 'NSF']
losses = {
    "CRPS": {"NAF": naf_crps, "VAE": vae_crps, "GAN": gan_crps, "NSF": nsf_crps},
    "QS"  : {"NAF": naf_qs,   "VAE": vae_qs,   "GAN": gan_qs, "NSF": nsf_qs},
    "ES"  : {"NAF": naf_energy,   "VAE": vae_energy,   "GAN": gan_energy, "NSF": nsf_energy},
    "VS"  : {"NAF": naf_vario,   "VAE": vae_vario,   "GAN": gan_vario, "NSF": nsf_vario}
}

plot_dm(losses, labels, h=1, save_path='plots/DM_TEST_LOAD.png', title='Diebold-Mariano Test (Solar)')

X_real = collect_real_features(prediction_dataloader)

fake_by_model = {
    "GAN": collect_fake_features(generator, prediction_dataloader, device, n_runs=50),
    "VAE": collect_fake_features(vae, prediction_dataloader, vae.device, n_runs=50),
    "NAF": collect_fake_features(flow, prediction_dataloader, device, n_runs=50),
    "NSF": collect_fake_features(flow, prediction_dataloader, device, n_runs=50)
}

plot_roc_many_scenarios(X_real, fake_by_model, n_runs=50,
                        title="LOAD track classifier-based metrics", save_path='plots/ROC_LOAD.png')