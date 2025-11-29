from functools import partial

import torch
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler, RandomSampler

from models.wgan import gan_objective
from models.vae import vae_objective
from models.nf import naf_objective, nsf_objective
from data.utils_data import GEFcomLoadLoader, create_load_dataset

import warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.init()
    _ = torch.empty(1, device='cuda')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = create_load_dataset()
train_dataset = dataset[dataset["TIMESTAMP"] < "2006-01-01 00:00:00"] # one year

train_loader = GEFcomLoadLoader(train_dataset.copy())

train_dataloader, validation_dataloader, test_dataloader = train_loader.get_dataloaders(
                                                                batch_size=32, use_gpu=True, test_size=0.1, validation_size=0.2,
                                                                shuffle=True
                                                                )

c_dim = train_loader.context_dim
x_dim = 24

OBJ_EPOCHS = 10
N_TRIALS = 25

pruner = SuccessiveHalvingPruner(min_resource=11, reduction_factor=3)

# ---- Helper functions ----
SEED = 1
def new_sampler(sampler='random'):
    ''''
    sampler: 'random'(default) or 'tpe'
    returns sampler
    raise NotImplementedError
    '''
    if sampler == 'random':
        return RandomSampler(seed=SEED)
    elif sampler == 'tpe':
        return TPESampler(multivariate=True, group=True, seed=SEED)
    else:
        raise NotImplementedError(f"Sampler {sampler} not implemented")


def make_study(db_name, study_name, sampler):
    return optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///tuning/{db_name}",
        load_if_exists=True,
        study_name=study_name,
    )

def make_and_run_study(db_name, study_name, objective, random_n=25, tpe_n=25):
    study = make_study(db_name, study_name, new_sampler("random"))
    study.optimize(objective, n_trials=random_n)
    study.sampler = new_sampler("tpe")
    study.optimize(objective, n_trials=tpe_n)
    

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
make_and_run_study("gan_study.db", "load", gan_objective_, random_n=25, tpe_n=25)

make_and_run_study("vae_study.db", "load", vae_objective_, random_n=25, tpe_n=25)

make_and_run_study("naf_study.db", "load", naf_objective_, random_n=10, tpe_n=10)

make_and_run_study("nsf_study.db", "load", nsf_objective_, random_n=10, tpe_n=15)
