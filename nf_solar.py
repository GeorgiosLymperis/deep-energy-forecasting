from nf_utils import (
    UNAF, NAF, NSF,
      plot_training, make_24h_forecast_with_bands, evaluate_flow
            )
from utils_data import GEFcomSolarLoader, create_solar_dataset
import torch
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.init()
    _ = torch.empty(1, device='cuda')  # pre-warm context

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def plot_forecasts(flow, prediction_loader, save_path=None, **kwargs):
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    dataset = prediction_loader.create_dataset(shuffle=False)
    active_hours = prediction_loader.active_hours
    target_names = ['POWER' + str(h) for h in active_hours]
    zones = ['ZONE_' + str(i) for i in range(1, 3 + 1)]
    samples = kwargs.get('samples', 100)
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12,14), sharex=True, sharey=True)
    axes = axes.flatten()

    pbar = tqdm(zones, desc="Predict Zones", leave=False)
    for axe, zone in zip(axes, pbar):
        df_zone = dataset[dataset[zone] == 1]
        targets = df_zone[target_names].values.reshape(-1)
        contexts = df_zone.drop(columns=target_names)
        contexts_scaled = prediction_loader.x_scaler.transform(contexts)
        c = torch.from_numpy(contexts_scaled).to(torch.float)
        Q1, median, Q3 = make_24h_forecast_with_bands(flow, c, samples=samples)
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

dataset = create_solar_dataset()
train_dataset = dataset[dataset["TIMESTAMP"] < "2013-11-01 01:00:00"]
prediction_dataset = dataset[dataset["TIMESTAMP"] >= "2013-11-01 01:00:00"]

train_loader = GEFcomSolarLoader(train_dataset.copy())

train_dataloader, validation_dataloader, test_dataloader = train_loader.get_dataloaders(
                                                                batch_size=32, shuffle=True, use_gpu=True, test_size=0.1, validation_size=0.5
                                                                )

prediction_loader = GEFcomSolarLoader(prediction_dataset.copy(), x_scaler=train_loader.x_scaler, y_scaler=train_loader.y_scaler)

prediction_dataloader, _, _ = prediction_loader.get_dataloaders(
                                            batch_size=32, use_gpu=True, shuffle=False, 
                                            test_size=0.0, validation_size=0.0
                                            )
c_dim = train_loader.context_dim
x_dim = len(train_loader.active_hours)
epochs = 2
lr = 1e-3

#______________ Training UNAF ________________

flow = UNAF(x_dim=x_dim, c_dim=c_dim, hidden_features=[16, 16], signal=8)
history = flow.fit(train_dataloader, validation_dataloader=validation_dataloader, epochs=epochs, lr=lr, patience=20, device=device, save_path='models/UNAF_MODEL_SOLAR.pt')
plot_training(history, title='Learning curve (UNAF)', save_path='plots/UNAF_LEARNING_CURVE_SOLAR.png')
evaluate_flow(flow, test_dataloader, 'UNAF_SOLAR', save_path='evaluations/NF_EVALUATIONS.json')
plot_forecasts(flow, prediction_loader, save_path='plots/UNAF_FORECASTS_SOLAR.jpg')

#______________ Training NAF ________________

flow = NAF(x_dim=x_dim, c_dim=c_dim, hidden_features=[16, 16], signal=8)
history = flow.fit(train_dataloader, validation_dataloader=validation_dataloader, epochs=epochs, lr=lr, patience=20, device=device, save_path='models/NAF_MODEL_SOLAR.pt')
plot_training(history, title='Learning curve (NAF)', save_path='plots/NAF_LEARNING_CURVE_SOLAR.png')
evaluate_flow(flow, test_dataloader, 'NAF_SOLAR', save_path='evaluations/NF_EVALUATIONS.json')
plot_forecasts(flow, prediction_loader, save_path='plots/NAF_FORECASTS_SOLAR.jpg')

#______________ Training NSF ________________

flow = NSF(x_dim=x_dim, c_dim=c_dim, hidden_features=[16, 16], transforms=3)
history = flow.fit(train_dataloader, validation_dataloader=validation_dataloader, epochs=epochs, lr=lr, patience=20, device=device, save_path='models/NSF_MODEL_SOLAR.pt')
plot_training(history, title='Learning curve (NSF)', save_path='plots/NSF_LEARNING_CURVE_SOLAR.png')
evaluate_flow(flow, test_dataloader, 'NSF_SOLAR', save_path='evaluations/NF_EVALUATIONS.json')
plot_forecasts(flow, prediction_loader, save_path='plots/NSF_FORECASTS_SOLAR.jpg')