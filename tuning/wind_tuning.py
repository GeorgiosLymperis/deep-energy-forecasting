from functools import partial

import torch
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

from ..gans_utils import gan_objective
from ..VAE_utils import vae_objective
from ..nf_utils import naf_objective, nsf_objective
from ..utils_data import GEFcomWindLoader, create_wind_dataset

import warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.init()
    _ = torch.empty(1, device='cuda')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = create_wind_dataset()
train_dataset = dataset[dataset["TIMESTAMP"] < "2013-01-01 01:00:00"] # one year

train_loader = GEFcomWindLoader(train_dataset.copy())

train_dataloader, validation_dataloader, test_dataloader = train_loader.get_dataloaders(
                                                                batch_size=32, use_gpu=True, test_size=0.1, validation_size=0.2,
                                                                shuffle=True
                                                                )

c_dim = train_loader.context_dim
x_dim = 24

OBJ_EPOCHS = 15
N_TRIALS = 25

pruner = SuccessiveHalvingPruner(min_resource=5, reduction_factor=3)

# ---- Helper functions ----
SEED = 1
def new_sampler():
    return TPESampler(multivariate=True, group=True, seed=SEED)

def make_study(db_name, study_name, sampler):
    return optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{db_name}",
        load_if_exists=True,
        study_name=study_name,
    )

def log_callback(study, trial):
    print(f"[{study.study_name}] trial#{trial.number} value={trial.value} params={trial.params}")

#  ---- Objectives ----
gan_objective_ = partial(
    gan_objective,
    train_dataloader=train_dataloader,
    validation_dataloader=validation_dataloader,
    c_dim=c_dim, x_dim=x_dim, device=device, obj_epochs=OBJ_EPOCHS
)

vae_objective_ = partial(
    vae_objective,
    train_dataloader=train_dataloader,
    validation_dataloader=validation_dataloader,
    c_dim=c_dim, x_dim=x_dim, device=device, obj_epochs=OBJ_EPOCHS
)

# NAF and NSF are slower
OBJ_EPOCHS = 10
naf_objective_ = partial(
    naf_objective,
    train_dataloader=train_dataloader,
    validation_dataloader=validation_dataloader,
    c_dim=c_dim, x_dim=x_dim, device=device, obj_epochs=OBJ_EPOCHS
)

nsf_objective_ = partial(
    nsf_objective,
    train_dataloader=train_dataloader,
    validation_dataloader=validation_dataloader,
    c_dim=c_dim, x_dim=x_dim, device=device, obj_epochs=OBJ_EPOCHS
)

# ---- Studies ----
gan_study = make_study("gan_study.db", "wind", new_sampler())
gan_study.optimize(gan_objective_, n_trials=N_TRIALS, callbacks=[log_callback])

vae_study = make_study("vae_study.db", "wind", new_sampler())
vae_study.optimize(vae_objective_, n_trials=N_TRIALS, callbacks=[log_callback])

# NAF and NSF are slower
N_TRIALS = 10
naf_study = make_study("naf_study.db", "wind", new_sampler())
naf_study.optimize(naf_objective_, n_trials=N_TRIALS, callbacks=[log_callback])

nsf_study = make_study("nsf_study.db", "wind", new_sampler())
nsf_study.optimize(nsf_objective_, n_trials=N_TRIALS, callbacks=[log_callback])
