from gans_utils import (
    Generator, Discriminator, train_wgan_gp_, plot_gan_training)
from utils_data import GEFcomSolarLoader, create_solar_dataset
import torch
import numpy as np
from evaluation.evaluation_utils import (
                            evaluate_model, plot_solar_forecasts, plot_qs, plot_crps, 
                            plot_correlations, plot_dm, plot_roc_many_scenarios,
                            collect_fake_features, collect_real_features,
                            compare_models_on_solar_forecasts
                            )
from VAE_utils import VAElinear, plot_vae_training, train_vae_linear
from nf_utils import NAF, NSF, plot_nf_training
import optuna

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.init()
    _ = torch.empty(1, device='cuda')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = create_solar_dataset()
train_dataset = dataset[dataset["TIMESTAMP"] < "2014-06-01 01:00:00"]
prediction_dataset = dataset[dataset["TIMESTAMP"] >= "2014-06-01 01:00:00"]

train_loader = GEFcomSolarLoader(train_dataset.copy())

# filter_hours = []
filter_hours = ['2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24']

train_dataloader, validation_dataloader, test_dataloader = train_loader.get_dataloaders(
                                                                batch_size=32, shuffle=True, use_gpu=True, 
                                                                test_size=0.1, validation_size=0.2,
                                                                filter_hours=filter_hours
                                                                )

prediction_loader = GEFcomSolarLoader(prediction_dataset.copy(), x_scaler=train_loader.x_scaler, y_scaler=train_loader.y_scaler)

prediction_dataloader, _, _ = prediction_loader.get_dataloaders(
                                            batch_size=32, use_gpu=True, shuffle=False, 
                                            test_size=0.0, validation_size=0.0,
                                            filter_hours=filter_hours
                                            )

prediction_dataset = prediction_loader.create_dataset(shuffle=False, filter_hours=filter_hours)
c_dim = train_loader.context_dim
x_dim = len(train_loader.active_hours)
# ---------- Train GAN ----------
gan_study = optuna.load_study(study_name="solar", storage="sqlite:///gan_study.db")
best_params = gan_study.best_params

EPOCHS = 100
D_LR = best_params['d_lr']
GP_LAMBDA = best_params['gp_lambda']
N_CRITIC = best_params['n_critic']
HIDDEN_DIM = best_params['hidden_dim']
Z_DIM = best_params['z_dim']

generator = Generator(z_dim=Z_DIM, c_dim=c_dim, y_dim=x_dim, hidden_dim=HIDDEN_DIM, device=device)
discriminator = Discriminator(c_dim=c_dim, y_dim=x_dim, hidden_dim=HIDDEN_DIM, device=device)

discriminator, generator, gan_history = train_wgan_gp_(generator, discriminator, train_dataloader, validationloader=validation_dataloader, d_lr=D_LR, epochs=EPOCHS, gp_lambda=GP_LAMBDA, n_critic=N_CRITIC)
discriminator.save('models/GAN_DISCRIMINATOR_SOLAR.pth')
generator.save('models/GAN_GENERATOR_SOLAR.pth')
plot_gan_training(gan_history, title='Learning curve (GAN)', save_path='plots/GAN_LEARNING_CURVE_SOLAR.png')

gan_results = evaluate_model(generator, test_dataloader, "GAN_SOLAR", save_path='evaluations/GAN_EVALUATION.pkl', device=device)
plot_solar_forecasts(generator, prediction_loader, save_path='plots/GAN_FORECASTS_SOLAR.jpg', filter_hours=filter_hours)

gan_crps = gan_results['crps'] # (days, T)
gan_energy = gan_results['energy'] # (days,)
gan_vario = gan_results['variogram'] # (days,)
gan_quantiles = gan_results['quantile'] # (99, days, T)
gan_qs = np.mean(gan_quantiles, axis=0) # (days, T)

c_pred = prediction_dataset.sample(1).drop(columns=['POWER' + str(h) for h in train_loader.active_hours])
plot_correlations(generator, c_pred, train_loader.x_scaler, save_path='plots/GAN_CORRELATIONS_SOLAR.jpg', title='GAN correlations (Solar)')

# ---------- Train VAE ----------
vae_study = optuna.load_study(study_name="load", storage="sqlite:///vae_study.db")
best_params = vae_study.best_params

EPOCHS = 100
LATENT_S = best_params['latent_s']
ENC_W = best_params['enc_w']
ENC_L = best_params['enc_l']
DEC_W = best_params['dec_w']
DEC_L = best_params['dec_l']
LR = best_params['lr']

vae = VAElinear(latent_s=LATENT_S, cond_in=c_dim, in_size=x_dim,
                gpu=True, enc_w=ENC_W, enc_l=ENC_L, dec_w=DEC_W, dec_l=DEC_L)
vae, vae_history = train_vae_linear(vae, train_dataloader, validation_dataloader, epochs=EPOCHS, save_path='models/vae_SOLAR.pth', lr=LR)

plot_vae_training(vae_history, title='Learning curve (VAE)', save_path='plots/VAE_SOLAR.jpg')
plot_solar_forecasts(vae, prediction_loader, save_path='plots/VAE_FORECASTS_SOLAR.jpg', filter_hours=filter_hours)
plot_correlations(vae, c_pred, train_loader.x_scaler, save_path='plots/VAE_CORRELATIONS_SOLAR.jpg', title='VAE correlations (Solar)')

vae_results = evaluate_model(vae, test_dataloader, "VAE_SOLAR", save_path='evaluations/VAE_EVALUATION.pkl', device=device)
vae_crps = vae_results['crps'] # (days, T)
vae_energy = vae_results['energy'] # (days,)
vae_vario = vae_results['variogram'] # (days,)
vae_quantiles = vae_results['quantile'] # (99, days, T)
vae_qs = np.mean(vae_quantiles, axis=0) # (days, T)

#______________ Training NAF ________________
naf_study = optuna.load_study(study_name="solar", storage="sqlite:///naf_study.db")
best_params = naf_study.best_params

EPOCHS = 100
LR = best_params['lr']
TRANSFORMS = best_params['transforms']
n_layers = best_params['n_layers']
width_exp = best_params['width_exp']

naf = NAF(x_dim=x_dim, c_dim=c_dim, hidden_features=[2**width_exp]*n_layers, signal=8, transforms=TRANSFORMS)
naf_history = naf.fit(train_dataloader, validation_dataloader=validation_dataloader, epochs=EPOCHS, lr=LR, patience=20, device=device, save_path='models/NAF_MODEL_LOAD.pt')
plot_nf_training(naf_history, title='Learning curve (NAF)', save_path='plots/NAF_LEARNING_CURVE_SOLAR.png')
plot_solar_forecasts(naf, prediction_loader, save_path='plots/NAF_FORECASTS_SOLAR.jpg', filter_hours=filter_hours)
plot_correlations(naf, c_pred, train_loader.x_scaler, save_path='plots/NAF_CORRELATIONS_SOLAR.jpg', title='NAF correlations (Solar)')

naf_results = evaluate_model(naf, test_dataloader, "NAF_SOLAR", save_path='evaluations/NAF_EVALUATION.pkl', device=device)

naf_crps = naf_results['crps'] # (days, T)
naf_energy = naf_results['energy'] # (days,)
naf_vario = naf_results['variogram'] # (days,)
naf_quantiles = naf_results['quantile'] # (99, days, T)
naf_qs = np.mean(naf_quantiles, axis=0) # (days, T)

#______________ Training NSF ________________
nsf_study = optuna.load_study(study_name="solar", storage="sqlite:///nsf_study.db")
best_params = nsf_study.best_params

EPOCHS = 100
LR = best_params['lr']
TRANSFORMS = best_params['transforms']
n_layers = best_params['n_layers']
width_exp = best_params['width_exp']

nsf = NSF(x_dim=x_dim, c_dim=c_dim, hidden_features=[2**width_exp]*n_layers, transforms=TRANSFORMS)
nsf_history = nsf.fit(train_dataloader, validation_dataloader=validation_dataloader, epochs=EPOCHS, lr=LR, patience=20, device=device, save_path='models/NSF_MODEL_LOAD.pt')
plot_nf_training(nsf_history, title='Learning curve (NSF)', save_path='plots/NSF_LEARNING_CURVE_SOLAR.png')
plot_solar_forecasts(nsf, prediction_loader, save_path='plots/NSF_FORECASTS_SOLAR.jpg', filter_hours=filter_hours)
plot_correlations(nsf, c_pred, train_loader.x_scaler, save_path='plots/NSF_CORRELATIONS_SOLAR.jpg', title='NSF correlations (Solar)')

nsf_results = evaluate_model(nsf, test_dataloader, "NSF_SOLAR", save_path='evaluations/NSF_EVALUATION.pkl', device=device)

nsf_crps = nsf_results['crps'] # (days, T)
nsf_energy = nsf_results['energy'] # (days,)
nsf_vario = nsf_results['variogram'] # (days,)
nsf_quantiles = nsf_results['quantile'] # (99, days, T)
nsf_qs = np.mean(nsf_quantiles, axis=0) # (days, T)

plot_crps([gan_crps, vae_crps, naf_crps, nsf_crps], ['GAN', 'VAE', 'NAF', 'NSF'], save_path='plots/CRPS_SOLAR.png', title='CRPS (Solar)')
plot_qs([gan_quantiles, vae_quantiles, naf_quantiles, nsf_quantiles], ['GAN', 'VAE', 'NAF', 'NSF'], save_path='plots/QS_SOLAR.png', title='Quantile score (SOLAR)')

compare_models_on_solar_forecasts(prediction_loader, [generator, vae, naf, nsf], save_path='plots/COMPARE_FORECASTS_SOLAR.png', filter_hours=filter_hours)

labels = ['GAN', 'VAE', 'NAF', 'NSF']
losses = {
    "CRPS": {"NAF": naf_crps, "VAE": vae_crps, "GAN": gan_crps, "NSF": nsf_crps},
    "QS"  : {"NAF": naf_qs,   "VAE": vae_qs,   "GAN": gan_qs, "NSF": nsf_qs},
    "ES"  : {"NAF": naf_energy,   "VAE": vae_energy,   "GAN": gan_energy, "NSF": nsf_energy},
    "VS"  : {"NAF": naf_vario,   "VAE": vae_vario,   "GAN": gan_vario, "NSF": nsf_vario}
}

plot_dm(losses, labels, h=x_dim-1, save_path='plots/DM_TEST_SOLAR.png', title='Diebold-Mariano Test (Solar)')

X_real = collect_real_features(prediction_dataloader)

fake_by_model = {
    "GAN": collect_fake_features(generator, prediction_dataloader, device, n_runs=50),
    "VAE": collect_fake_features(vae, prediction_dataloader, vae.device, n_runs=50),
    "NAF": collect_fake_features(naf, prediction_dataloader, device, n_runs=50),
    "NSF": collect_fake_features(nsf, prediction_dataloader, device, n_runs=50)
}

plot_roc_many_scenarios(X_real, fake_by_model, n_runs=50,
                        title="SOLAR track classifier-based metrics", save_path='plots/ROC_SOLAR.png')

for loss in losses:
    print("\nLoss:", loss)
    print("  NAF:", losses[loss]["NAF"].mean().mean())
    print("  VAE:", losses[loss]["VAE"].mean())
    print("  GAN:", losses[loss]["GAN"].mean())
    print("  NSF:", losses[loss]["NSF"].mean().mean())