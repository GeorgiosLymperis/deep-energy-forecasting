import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler

def get_device(use_gpu=False):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class GEFcom2014:
    def __init__(self):
        pass

    def build_features(self):
        raise NotImplementedError
    
    def split(self):
        raise NotImplementedError
    
    def load_data(self):
        raise NotImplementedError
    
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    
class GEFcomWindLoader(GEFcom2014):
    def __init__(self, folder='data/wind'):
        self.folder = folder
        self.create_dataset()
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def create_dataset(self, save=False):
        files = os.listdir(self.folder)
        df_wind = pd.DataFrame()
        for file in files:
            df = pd.read_csv(os.path.join(self.folder, file), parse_dates=['TIMESTAMP'], index_col='TIMESTAMP')
            df_wind = pd.concat([df_wind, df], axis=0)
        df_wind = pd.get_dummies(df_wind, prefix='ZONE', prefix_sep='_', columns=['ZONEID'], dtype='int')
        if save:
            df_wind.to_csv('data/wind_data.csv')
        
        self.dataframe = df_wind.copy()
    
    def build_features(self, density=1):
        '''
        The predictors included wind forecasts at two heights, 10 and 100 m above ground level, obtained from the European Centre for Medium-range Weather Forecasts (ECMWF).
        These forecasts were for the zonal and meridional wind components (denoted u and v), i.e., projections of the wind vector on the west-east and south-north axes, respectively.

        U10 zonal wind component at 10 m
        V10 meridional wind component at 10 m
        U100 zonal wind component at 100 m
        V100 meridional wind component at 100 m

        Outputs Description
        Wind speed (ws), wind energy (we), and wind direction (wd) were as follows,
        where u and v are the wind components provided and d is the density, for which we used a constant 1.0
        ws = sqrt[u**2  + v**2]
        we = 0.5 × d × ws**3
        wd = 180/π × arctan(u, v)
        '''
        u10 = self.dataframe['U10']
        u100 = self.dataframe['U100']
        v10 = self.dataframe['V10']
        v100 = self.dataframe['V100']

        self.dataframe['ws10'] = np.sqrt(u10**2 + v10**2)
        self.dataframe['ws100'] = np.sqrt(u100**2 + v100**2)
        self.dataframe['we10'] = 0.5 * density * self.dataframe['ws10']**3
        self.dataframe['we100'] = 0.5 * density * self.dataframe['ws100']**3
        self.dataframe['wd10'] = np.arctan2(v10, u10) * 180 / np.pi
        self.dataframe['wd100'] = np.arctan2(v100, u100) * 180 / np.pi
        self.dataframe = self.dataframe.bfill()

        self.features = ['U10', 'V10', 'U100', 'V100', 'ws10', 'ws100', 'we10', 'we100', 'wd10', 'wd100']
        self.zones = [i for i in range(1, 10 + 1)]

    def split(self, random_state=42, test_size=0.2, validation_size=0.2):
        X = self.dataframe.drop(columns=['TARGETVAR'])
        y = self.dataframe['TARGETVAR']

        self.training_days = int((X.shape[0] * (1 - test_size)) // 24)
        self.validation_days = int((self.training_days * validation_size) // 24)
        self.test_days = int((X.shape[0] * test_size) // 24)

        X_train = X.iloc[:(self.training_days * 24)]
        X_validation = X.iloc[(self.training_days * 24):((self.training_days + self.validation_days) * 24)]
        X_test = X.iloc[((self.training_days + self.validation_days) * 24):]

        y_train = y.iloc[:(self.training_days * 24)]
        y_validation = y.iloc[(self.training_days * 24):((self.training_days + self.validation_days) * 24)]
        y_test = y.iloc[((self.training_days + self.validation_days) * 24):]

        # Fit scaler on X
        X_train_scaled = self.x_scaler.fit_transform(X_train)
        

        # Scale validation and test
        X_validation_scaled = self.x_scaler.transform(X_validation)
        X_test_scaled = self.x_scaler.transform(X_test)
        # Fit scaler on y
        self.y_scaler = StandardScaler()
        y_train_scaled = self.y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_validation_scaled = self.y_scaler.transform(y_validation.values.reshape(-1, 1)).flatten()
        y_test_scaled = self.y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

        # Make tensors
        self.train_dataset = MyDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32).view(-1, 24, X.shape[1]),
            torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 24)
        )

        self.validation_dataset = MyDataset(
            torch.tensor(X_validation_scaled, dtype=torch.float32).view(-1, 24, X.shape[1]),
            torch.tensor(y_validation_scaled, dtype=torch.float32).view(-1, 24)
        )

        self.test_dataset = MyDataset(
            torch.tensor(X_test_scaled, dtype=torch.float32).view(-1, 24, X.shape[1]),
            torch.tensor(y_test_scaled, dtype=torch.float32).view(-1, 24)
        )


        return self.train_dataset, self.validation_dataset, self.test_dataset

    def get_dataloaders(self, batch_size=32, shuffle=False, num_workers=4, use_gpu=False):

        device = get_device(use_gpu)
        pin = True if device.type == 'cuda' and use_gpu else False

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )
        self.validation_dataloader = DataLoader(
            self.validation_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )

        return self.train_dataloader, self.validation_dataloader, self.test_dataloader

class GEFcomSolarLoader(GEFcom2014):
    def __init__(self, folder='data/solar'):
        self.folder = folder
        # from 9:00 to 20:00 we have zero power
        self.inactive_hours = [i for i in range(9, 20 + 1)]
        self.active_hours = [i for i in range(0, 24) if i not in self.inactive_hours]
        self.create_dataset()
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def create_dataset(self, save=False):
        files = os.listdir(self.folder)
        df_solar = pd.DataFrame()
        for file in files:
            df = pd.read_csv(os.path.join(self.folder, file), parse_dates=['TIMESTAMP'], index_col='TIMESTAMP')
            df_solar = pd.concat([df_solar, df], axis=0)
        
        df_solar['time'] = df_solar.index.hour
        df_solar = df_solar[df_solar['time'].isin(self.active_hours)].drop(columns=['time'])
        df_solar = pd.get_dummies(df_solar, prefix='ZONE', prefix_sep='_', columns=['ZONEID'], dtype='int')
        if save:
            df_solar.to_csv('data/solar_data.csv')
        self.dataframe = df_solar.copy()

    def build_features(self):
        '''
        The target variable is solar power.  There are 12 independent variables from the ECMWF NWP output to be used as below.

        078.128 Total column liquid water (tclw) - unit (kg m**-2) - Vertical integral of cloud liquid water content
        079.128 Total column ice water (tciw) - unit (kg m**-2) - Vertical integral of cloud ice water content
        134.128 surface pressure (SP) - Unit: Pa
        157.128 Relative humidity at 1000 mbar (r) -unit (%)-   Relative humidity is defined with respect to saturation of the mixed phase, i.e. with respect to saturation over ice below -23C and with respect to saturation over water above 0C. In the regime in between a quadratic interpolation is applied.
        164.128 total cloud cover (TCC) - Unit: (0-1) - Total cloud cover derived from model levels using the model's overlap assumption
        165.128 10 metre U wind component (10u) - unit  (m s**-1)
        166.128 10 metre V wind component (10V) - unit  (m s**-1)
        167.128 2 metre temperature (2T) - Unit: K
        169.128 surface solar rad down (SSRD) - Unit: J m-2 - Accumulated field
        175.128 surface thermal rad  down (STRD) - Unit: J m-2 - Accumulated field
        178.128 top net solar rad (TSR)   Unit: J m-2 - Net solar radiation at the top of the atmosphere. Accumulated field
        228.128 total precipitation (TP) - Unit: m - Convective precipitation + stratiform precipitation (CP +LSP). Accumulated field.
        '''
        self.dataframe = self.dataframe.bfill()
        self.features = [
            'VAR78', 'VAR79', 'VAR134', 'VAR157', 'VAR164', 'VAR165',
            'VAR166', 'VAR167', 'VAR169', 'VAR175', 'VAR178', 'VAR228'
        ]
        self.zones = [i for i in range(1, 3 + 1)]

    def split(self, test_size=0.2, validation_size=0.2):
        X = self.dataframe.drop(columns=['POWER'])
        y = self.dataframe['POWER']

        hours = len(self.active_hours)
        self.training_days = int((X.shape[0] * (1 - test_size)) // hours)
        self.validation_days = int(self.training_days * validation_size)
        self.test_days = int(self.training_days * test_size)

        X_train = X.iloc[:(self.training_days * hours)]
        X_validation = X.iloc[(self.training_days * hours):((self.training_days + self.validation_days) * hours)]
        X_test = X.iloc[((self.training_days + self.validation_days) * hours):]

        y_train = y.iloc[:(self.training_days * hours)]
        y_validation = y.iloc[(self.training_days * hours):((self.training_days + self.validation_days) * hours)]
        y_test = y.iloc[((self.training_days + self.validation_days) * hours):]

        # Fit scaler on X
        X_train_scaled = self.x_scaler.fit_transform(X_train)
        X_validation_scaled = self.x_scaler.transform(X_validation)
        X_test_scaled = self.x_scaler.transform(X_test)

        # Fit scaler on y
        y_train_scaled = self.y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_validation_scaled = self.y_scaler.transform(y_validation.values.reshape(-1, 1)).flatten()
        y_test_scaled = self.y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

        # Make tensors
        self.train_dataset = MyDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32).view(-1, hours, X.shape[1]),
            torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, hours)
        )

        self.validation_dataset = MyDataset(
            torch.tensor(X_validation_scaled, dtype=torch.float32).view(-1, hours, X.shape[1]),
            torch.tensor(y_validation_scaled, dtype=torch.float32).view(-1, hours)
        )

        self.test_dataset = MyDataset(
            torch.tensor(X_test_scaled, dtype=torch.float32).view(-1, hours, X.shape[1]),
            torch.tensor(y_test_scaled, dtype=torch.float32).view(-1, hours)
        )


        return self.train_dataset, self.validation_dataset, self.test_dataset

    def get_dataloaders(self, batch_size=32, shuffle=False, num_workers=4, use_gpu=False):

        device = get_device(use_gpu)
        pin = True if device.type == 'cuda' and use_gpu else False

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )
        self.validation_dataloader = DataLoader(
            self.validation_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )

        return self.train_dataloader, self.validation_dataloader, self.test_dataloader
    
class GEFcomLoadLoader():
    def __init__(self, folder='data/load'):
        self.folder = folder
        self.create_dataset()
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()


    def create_dataset(self, save=False):
        files = os.listdir(self.folder)
        df_load = pd.DataFrame()
        for file in files:
            df = pd.read_csv(os.path.join(self.folder, file))
            df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%m%d%Y %H:%M')
            df = df.set_index('TIMESTAMP')
            df_load = pd.concat([df_load, df], axis=0)
        df_load = df_load.drop(columns=['ZONEID'])
        df_load = df_load.drop_duplicates()
        if save:
            df_load.to_csv('data/load_data.csv')
        self.dataframe = df_load.iloc[23:].copy() # remove first day because is 23 hours

    def build_features(self):
        self.dataframe = self.dataframe.bfill()
        self.features = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10',
                'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18', 'w19', 'w20',
                'w21', 'w22', 'w23', 'w24', 'w25']
        
    def split(self, test_size=0.2, validation_size=0.2):
        X = self.dataframe.drop(columns=['LOAD'])
        y = self.dataframe['LOAD']

        self.training_days = int((X.shape[0] * (1 - test_size)) // 24)
        self.validation_days = int((self.training_days * validation_size) // 24)
        self.test_days = int((X.shape[0] * test_size) // 24)

        X_train = X.iloc[:(self.training_days * 24)]
        X_validation = X.iloc[(self.training_days * 24):((self.training_days + self.validation_days) * 24)]
        X_test = X.iloc[((self.training_days + self.validation_days) * 24):]

        y_train = y.iloc[:(self.training_days * 24)]
        y_validation = y.iloc[(self.training_days * 24):((self.training_days + self.validation_days) * 24)]
        y_test = y.iloc[((self.training_days + self.validation_days) * 24):]

        # Fit scaler on X
        X_train_scaled = self.x_scaler.fit_transform(X_train)
        

        # Scale validation and test
        X_validation_scaled = self.x_scaler.transform(X_validation)
        X_test_scaled = self.x_scaler.transform(X_test)
        # Fit scaler on y
        self.y_scaler = StandardScaler()
        y_train_scaled = self.y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_validation_scaled = self.y_scaler.transform(y_validation.values.reshape(-1, 1)).flatten()
        y_test_scaled = self.y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

        # Make tensors
        self.train_dataset = MyDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32).view(-1, 24, X.shape[1]),
            torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 24)
        )

        self.validation_dataset = MyDataset(
            torch.tensor(X_validation_scaled, dtype=torch.float32).view(-1, 24, X.shape[1]),
            torch.tensor(y_validation_scaled, dtype=torch.float32).view(-1, 24)
        )

        self.test_dataset = MyDataset(
            torch.tensor(X_test_scaled, dtype=torch.float32).view(-1, 24, X.shape[1]),
            torch.tensor(y_test_scaled, dtype=torch.float32).view(-1, 24)
        )


        return self.train_dataset, self.validation_dataset, self.test_dataset

    def get_dataloaders(self, batch_size=32, shuffle=False, num_workers=4, use_gpu=False):

        device = get_device(use_gpu)
        pin = True if device.type == 'cuda' and use_gpu else False

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )
        self.validation_dataloader = DataLoader(
            self.validation_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )

        return self.train_dataloader, self.validation_dataloader, self.test_dataloader
        

if __name__ == '__main__':
    dataset = GEFcomLoadLoader()
    dataset.create_dataset()
    dataset.build_features()
    dataset.split()
    dataset.get_dataloaders()