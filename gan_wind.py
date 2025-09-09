from gans_utils import (
    Generator, Discriminator, train_wgan_gp, train_wgan_gp_, evaluate_gan,
    make_24h_forecast_with_bands, plot_training)
from utils_data import GEFcomWindLoader, create_wind_dataset
import torch
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.init()
    _ = torch.empty(1, device='cuda')  # pre-warm context

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def plot_forecasts(generator, prediction_dataloader, save_path=None, **kwargs):
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    dataset = prediction_dataloader.create_dataset(shuffle=False)
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
        contexts_scaled = prediction_dataloader.x_scaler.transform(contexts)
        c = torch.from_numpy(contexts_scaled).to(torch.float)
        Q1, median, Q3 = make_24h_forecast_with_bands(generator, c, samples=samples)
        Q1 = prediction_dataloader.y_scaler.inverse_transform(Q1).reshape(-1)
        median = prediction_dataloader.y_scaler.inverse_transform(median).reshape(-1)
        Q3 = prediction_dataloader.y_scaler.inverse_transform(Q3).reshape(-1)

        mae = mean_absolute_error(targets, median)
        axe.fill_between(np.arange(len(targets))/24, Q1, Q3, alpha=0.2)
        axe.plot(np.arange(len(targets))/24, median, label='median')
        axe.plot(np.arange(len(targets))/24, targets, label='targets')
        axe.set_title(f"{zone} (MAE={mae:.2f})")
        axe.legend()
        axe.set_ylabel('Power')

    if save_path is not None:
        plt.savefig(save_path)

    return fig, axes

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

x_dim = 24
c_dim = train_loader.context_dim

generator = Generator(z_dim=256, c_dim=c_dim, y_dim=x_dim, hidden_dim=256 * 4, device=device)
discriminator = Discriminator(c_dim=c_dim, y_dim=x_dim,hidden_dim=256 * 4, device=device)

discriminator, generator, history = train_wgan_gp_(generator, discriminator, train_dataloader, validationloader=validation_dataloader, d_lr=2e-4, epochs=2, gp_lambda=5, n_critic=5)
discriminator.save('models/GAN_DISCRIMINATOR_WIND.pth')
generator.save('models/GAN_GENERATOR_WIND.pth')
evaluate_gan(generator, test_dataloader, "GAN_WIND", save_path='evaluations/GAN_EVALUATION.json', device=device)
plot_training(history, title='Learning curve (GAN)', save_path='plots/GAN_LEARNING_CURVE_WIND.png')
plot_forecasts(generator, prediction_loader, save_path='plots/GAN_FORECASTS_WIND.jpg')