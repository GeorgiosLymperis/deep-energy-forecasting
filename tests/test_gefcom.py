import pytest
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..utils_data import GEFcomWindLoader

@pytest.fixture
def fake_data(tmp_path):
    # Δημιουργία toy csv αρχείου
    path = tmp_path / "wind"
    path.mkdir()
    df = pd.DataFrame({
        'TIMESTAMP': pd.date_range(start='2021-01-01', periods=24, freq='h'),
        'U10': np.random.rand(24),
        'V10': np.random.rand(24),
        'U100': np.random.rand(24),
        'V100': np.random.rand(24),
        'TARGETVAR': np.random.rand(24),
        'ZONEID': [1]*24
    })
    file = path / "toy.csv"
    df.to_csv(file, index=False)
    return path

def test_create_dataset(fake_data):
    loader = GEFcomWindLoader(folder=str(fake_data))
    assert not loader.dataframe.empty
    assert 'U10' in loader.dataframe.columns
    assert any(col.startswith('ZONE_') for col in loader.dataframe.columns)

def test_build_features(fake_data):
    loader = GEFcomWindLoader(folder=str(fake_data))
    loader.build_features()
    for feat in ['ws10', 'ws100', 'we10', 'we100', 'wd10', 'wd100']:
        assert feat in loader.dataframe.columns

def test_split(fake_data):
    loader = GEFcomWindLoader(folder=str(fake_data))
    loader.build_features()
    train, val, test = loader.split()
    total = len(train) + len(val) + len(test)
    assert total == len(loader.dataframe)

def test_dataloader(fake_data):
    loader = GEFcomWindLoader(folder=str(fake_data))
    loader.build_features()
    loader.split()
    train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=4)
    x, y = next(iter(train_loader))
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape[0] <= 4
    assert x.shape[1] == len(loader.features)
