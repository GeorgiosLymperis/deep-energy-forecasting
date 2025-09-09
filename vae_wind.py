from VAE_utils import VAElinear, make_24h_forecast_with_bands, plot_training, evaluate_vae
from utils_data import GEFcomWindLoader, create_wind_dataset
import torch
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.init()
    _ = torch.empty(1, device='cuda')  # pre-warm context

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def plot_forecasts(vae, prediction_loader, save_path=None, **kwargs):
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    dataset = prediction_loader.create_dataset(shuffle=False)
    target_names = ['TARGETVAR' + str(h) for h in range(1, 25)]
    zones = ['ZONE_' + str(i) for i in range(1, 10 + 1)]
    samples = kwargs.get('samples', 100)
    
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12,14), sharex=True, sharey=True)
    axes = axes.flatten()

    pbar = tqdm(zones, desc="Predict Zones", leave=False)
    for axe, zone in zip(axes, pbar):
        df_zone = dataset[dataset[zone] == 1]
        targets = df_zone[target_names].values.reshape(-1)
        contexts = df_zone.drop(columns=target_names)
        contexts_scaled = prediction_loader.x_scaler.transform(contexts)
        c = torch.from_numpy(contexts_scaled).to(torch.float)
        Q1, median, Q3 = make_24h_forecast_with_bands(vae, c, samples=samples)
        Q1 = prediction_loader.y_scaler.inverse_transform(Q1).reshape(-1)
        median = prediction_loader.y_scaler.inverse_transform(median).reshape(-1)
        Q3 = prediction_loader.y_scaler.inverse_transform(Q3).reshape(-1)

        mae = mean_absolute_error(targets, median)
        axe.fill_between(np.arange(len(targets))/24, Q1, Q3, alpha=0.2)
        axe.plot(np.arange(len(targets))/24, median, label='median')
        axe.plot(np.arange(len(targets))/24, targets, label='targets')
        axe.set_title(f"{zone} (MAE={mae:.2f})")
        axe.legend()
        axe.set_ylabel('Power')

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)

dataset = create_wind_dataset()
train_dataset = dataset[dataset["TIMESTAMP"] < "2013-11-01 01:00:00"]
prediction_dataset = dataset[dataset["TIMESTAMP"] >= "2013-11-01 01:00:00"]

train_loader = GEFcomWindLoader(train_dataset.copy())

train_dataloader, validation_dataloader, test_dataloader = train_loader.get_dataloaders(
                                                                batch_size=32, shuffle=True, use_gpu=True, test_size=0.1, validation_size=0.5
                                                                )

prediction_loader = GEFcomWindLoader(prediction_dataset.copy(), x_scaler=train_loader.x_scaler, y_scaler=train_loader.y_scaler)

prediction_dataloader, _, _ = prediction_loader.get_dataloaders(
                                            batch_size=32, use_gpu=True, shuffle=False, 
                                            test_size=0.0, validation_size=0.0
                                            )
c_dim = train_loader.context_dim
epochs = 1


#______________ Training VAE ________________

vae = VAElinear(latent_s=8, cond_in=c_dim, in_size=24,
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

for epoch in range(epochs):
    beta = min(1.0, (epoch + 1) / 50.0)

    # ---- train ----
    vae.train()
    t_loss = t_recon = t_kl = 0.0
    train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [train] β={beta:.2f}", leave=False)
    for i, (c, x) in enumerate(train_bar, start=1):
        c = c.to(vae.device, dtype=torch.float32)
        x = x.to(vae.device, dtype=torch.float32)

        loss, logs = vae.loss(x, c, beta=beta)
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
        val_bar = tqdm(validation_dataloader, desc=f"Epoch {epoch+1}/{epochs} [val]", leave=False)
        for j, (c, x) in enumerate(val_bar, start=1):
            c = c.to(vae.device, dtype=torch.float32)
            x = x.to(vae.device, dtype=torch.float32)

            loss, logs = vae.loss(x, c, beta=beta)
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
        torch.save(vae.state_dict(), 'models/vae_wind.pth')
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            tqdm.write(f"Early stopping at epoch {epoch+1}. Best val_loss={best:.4f}")
            break

vae.load_state_dict(torch.load('models/vae_wind.pth', map_location=vae.device))
evaluate_vae(vae, test_dataloader, "VAE_WIND", save_path='evaluations/VAE_EVALUATION.json', device=device)
plot_training(history, title='Learning curve (VAE)', save_path='plots/VAE_WIND.jpg')
plot_forecasts(vae, prediction_loader, save_path='plots/VAE_FORECASTS_WIND.jpg')