from nf_utils import (
    UNAF, NAF, NSF,
      plot_training, make_24h_forecast_with_bands, evaluate_flow
            )
from utils_data import GEFcomLoadLoader, create_load_dataset
import torch
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.init()
    _ = torch.empty(1, device='cuda')  # pre-warm context

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def plot_forecasts(flow, prediction_dataloader, save_path=None, **kwargs):
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error
    dataset = prediction_dataloader.create_dataset(shuffle=False)
    target_names = ['LOAD' + str(h) for h in range(1, 25)]
    samples = kwargs.get('samples', 100)
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,14))
    targets = dataset[target_names].values.reshape(-1)
    contexts = dataset.drop(columns=target_names)
    contexts_scaled = prediction_dataloader.x_scaler.transform(contexts)
    c = torch.from_numpy(contexts_scaled).to(torch.float)
    Q1, median, Q3 = make_24h_forecast_with_bands(flow, c, samples=samples)
    Q1 = prediction_dataloader.y_scaler.inverse_transform(Q1).reshape(-1)
    median = prediction_dataloader.y_scaler.inverse_transform(median).reshape(-1)
    Q3 = prediction_dataloader.y_scaler.inverse_transform(Q3).reshape(-1)

    mae = mean_absolute_error(targets, median)
    axes.fill_between(np.arange(len(targets))/24, Q1, Q3, alpha=0.2)
    axes.plot(np.arange(len(targets))/24, median, label='median')
    axes.plot(np.arange(len(targets))/24, targets, label='targets')
    axes.set_title(f"(MAE={mae:.2f})")
    axes.legend()
    axes.set_ylabel('Power')

    if save_path is not None:
        plt.savefig(save_path)

    return fig, axes

dataset = create_load_dataset()
train_dataset = dataset[dataset["TIMESTAMP"] < "2011-12-08 00:00:00"]
prediction_dataset = dataset[dataset["TIMESTAMP"] >= "2011-12-08 00:00:00"]

train_loader = GEFcomLoadLoader(train_dataset.copy())

train_dataloader, validation_dataloader, test_dataloader = train_loader.get_dataloaders(
                                                                batch_size=32, shuffle=True, use_gpu=True, test_size=0.1, validation_size=0.5
                                                                )

prediction_loader = GEFcomLoadLoader(prediction_dataset.copy(), x_scaler=train_loader.x_scaler, y_scaler=train_loader.y_scaler)

prediction_dataloader, _, _ = prediction_loader.get_dataloaders(
                                            batch_size=32, use_gpu=True, shuffle=False, 
                                            test_size=0.0, validation_size=0.0
                                            )
c_dim = train_loader.context_dim
epochs = 2
lr = 1e-3

#______________ Training UNAF ________________

flow = UNAF(x_dim=24, c_dim=c_dim, hidden_features=[16, 16], signal=8)
history = flow.fit(train_dataloader, validation_dataloader=validation_dataloader, epochs=epochs, lr=lr, patience=20, device=device, save_path='models/UNAF_MODEL_LOAD.pt')
plot_training(history, title='Learning curve (UNAF)', save_path='plots/UNAF_LEARNING_CURVE_LOAD.png')
# evaluate_flow(flow, test_dataloader, 'UNAF_LOAD', save_path='evaluations/NF_EVALUATION.json')
plot_forecasts(flow, prediction_loader, save_path='plots/UNAF_FORECASTS_LOAD.jpg')

#______________ Training NAF ________________

flow = NAF(x_dim=24, c_dim=c_dim, hidden_features=[16, 16], signal=8)
history = flow.fit(train_dataloader, validation_dataloader=validation_dataloader, epochs=epochs, lr=lr, patience=20, device=device, save_path='models/NAF_MODEL_LOAD.pt')
plot_training(history, title='Learning curve (NAF)', save_path='plots/NAF_LEARNING_CURVE_LOAD.png')
evaluate_flow(flow, test_dataloader, 'NAF_LOAD', save_path='evaluations/NF_EVALUATION.json')
plot_forecasts(flow, prediction_loader, save_path='plots/NAF_FORECASTS_LOAD.jpg')

#______________ Training NSF ________________

flow = NSF(x_dim=24, c_dim=c_dim, hidden_features=[16, 16], transforms=3)
history = flow.fit(train_dataloader, validation_dataloader=validation_dataloader, epochs=epochs, lr=lr, patience=20, device=device, save_path='models/NSF_MODEL_LOAD.pt')
plot_training(history, title='Learning curve (NSF)', save_path='plots/NSF_LEARNING_CURVE_LOAD.png')
evaluate_flow(flow, test_dataloader, 'NSF_LOAD', save_path='evaluations/NF_EVALUATION.json')
plot_forecasts(flow, prediction_loader, save_path='plots/NSF_FORECASTS_LOAD.jpg')