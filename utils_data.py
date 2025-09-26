import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_device(use_gpu=False):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def create_wind_dataset(folder='data/wind',save=False):
        files = os.listdir(folder)
        df_wind = pd.DataFrame()
        for file in files:
            df = pd.read_csv(os.path.join(folder, file), parse_dates=['TIMESTAMP'])
            df_wind = pd.concat([df_wind, df], axis=0)
        df_wind = pd.get_dummies(df_wind, prefix='ZONE', prefix_sep='_', columns=['ZONEID'], dtype='int')
        if save:
            df_wind.to_csv('data/wind_data.csv')
        
        return df_wind

def create_solar_dataset(folder='data/solar', save=False):
    inactive_hours = [i for i in range(9, 20 + 1)]
    active_hours = [i for i in range(0, 24) if i not in inactive_hours]
    files = os.listdir(folder)
    df_solar = pd.DataFrame()
    for file in files:
        df = pd.read_csv(os.path.join(folder, file), parse_dates=['TIMESTAMP'], index_col='TIMESTAMP')
        df_solar = pd.concat([df_solar, df], axis=0)
    
    df_solar['time'] = df_solar.index.hour
    df_solar = df_solar[df_solar['time'].isin(active_hours)].drop(columns=['time'])
    df_solar = pd.get_dummies(df_solar, prefix='ZONE', prefix_sep='_', columns=['ZONEID'], dtype='int')
    if save:
        df_solar.to_csv('data/solar_data.csv')

    return df_solar.reset_index()

def create_load_dataset(folder='data/load', save=False):
    files = os.listdir(folder)
    df_load = pd.DataFrame()
    for file in files:
        df = pd.read_csv(os.path.join(folder, file))
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%m%d%Y %H:%M')
        df_load = pd.concat([df_load, df], axis=0)
    df_load = df_load.drop(columns=['ZONEID']) # It has only one zone, so drop it
    df_load = df_load.drop_duplicates(subset=['TIMESTAMP'])
    df_load = df_load.dropna()
    df_load = df_load.sort_values(by='TIMESTAMP').iloc[23:]
    if save:
        df_load.to_csv('data/load_data.csv')
    return df_load
    
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    
class GEFcomWindLoader():
    def __init__(self, dataframe, x_scaler=None, y_scaler=None):
        self.dataframe = dataframe
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def __build_features(self, density=1, save=False):
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

        if save:
            self.dataframe.to_csv('data/wind_data_features.csv')

    def create_dataset(self, shuffle=True):
        self.__build_features()
        zones = ['ZONE_' + str(i) for i in range(1, 10 + 1)]
        col_names = []
        target_names = ['TARGETVAR' + str(h) for h in range(1, 25)]
        for col in self.features:
            for h in range(1, 25):
                col_names.append(col + '_' + str(h))
        df = self.dataframe.copy()
        nb_days = int(len(df) / 24)
        index_1d = df['TIMESTAMP'].values.reshape(nb_days, 24)[:,0]
        index = pd.DatetimeIndex(index_1d)
        x = [df[col].values.reshape(nb_days, 24) for col in self.features]
        x.extend([df[zone].values.reshape(nb_days, 24)[:, 0].reshape(nb_days, 1) for zone in zones]) # ZONEID is the same for all hours
        col_names.extend(zones)
        y = df['TARGETVAR'].values.reshape(nb_days, 24)
        df_per_day = pd.DataFrame(
            np.concatenate([*x, y], axis=1), 
            columns=col_names + target_names,
            index=index
            )

        df_per_day['month'] = df_per_day.index.month
        df_per_day['day_of_week'] = df_per_day.index.dayofweek
        for zone in zones:
            zone_indices = df_per_day[zone] == 1
            df_per_day.loc[zone_indices, target_names] = df_per_day.loc[zone_indices, target_names].shift(-1)

        df_per_day.dropna(inplace=True)
        if shuffle:
            df_per_day = df_per_day.sample(frac=1)

        return df_per_day

    def split(self, random_state=42, test_size=0.2, validation_size=0.2, shuffle=True):
        df_per_day = self.create_dataset(shuffle=shuffle)
        X = df_per_day.drop(columns=['TARGETVAR' + str(h) for h in range(1, 25)])
        self.context_dim = len(X.columns)
        y = df_per_day[['TARGETVAR' + str(h) for h in range(1, 25)]]

        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
        else:
            X_train, X_test, y_train, y_test = X, [], y, []

        if validation_size > 0 and len(X_test) > 0: 
            X_test, X_val, y_test, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=random_state, shuffle=shuffle)
        else:
            X_test, X_val, y_test, y_val = X_test, [], y_test, []


        if self.x_scaler is None:
            self.x_scaler = StandardScaler()
            X_train_scaled = self.x_scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.x_scaler.transform(X_train)

        if self.y_scaler is None:
            self.y_scaler = StandardScaler()
            y_train_scaled = self.y_scaler.fit_transform(y_train)
        else:
            y_train_scaled = self.y_scaler.transform(y_train)
        
        # Scale test
        self.test_dataset = None
        if len(X_test) > 0:
            X_test_scaled = self.x_scaler.transform(X_test)
            y_test_scaled = self.y_scaler.transform(y_test)
            self.test_dataset = MyDataset(
                    torch.tensor(X_test_scaled, dtype=torch.float32),
                    torch.tensor(y_test_scaled, dtype=torch.float32)
                )

        # Scale validation
        self.validation_dataset = None
        if len(X_val) > 0:
            X_validation_scaled = self.x_scaler.transform(X_val)        
            y_validation_scaled = self.y_scaler.transform(y_val)
            self.validation_dataset = MyDataset(
                    torch.tensor(X_validation_scaled, dtype=torch.float32),
                    torch.tensor(y_validation_scaled, dtype=torch.float32)
                )
        

        # Make tensors
        self.train_dataset = MyDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train_scaled, dtype=torch.float32)
        )

        return self.train_dataset, self.validation_dataset, self.test_dataset

    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=4, use_gpu=False, **kwargs):
        self.split(shuffle=shuffle, **kwargs)

        device = get_device(use_gpu)
        pin = True if device.type == 'cuda' and use_gpu else False

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )
        if self.validation_dataset is not None:
            self.validation_dataloader = DataLoader(
                self.validation_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
            )
        else:
            self.validation_dataloader = None

        if self.test_dataset is not None:
            self.test_dataloader = DataLoader(
                 self.test_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )
        else:
            self.test_dataloader = None

        return self.train_dataloader, self.validation_dataloader, self.test_dataloader

class GEFcomSolarLoader():
    def __init__(self, dataframe, x_scaler=None, y_scaler=None):
        self.dataframe = dataframe
        # from 9:00 to 20:00 we have zero power
        self.inactive_hours = [i for i in range(9, 20 + 1)]
        self.active_hours = [i for i in range(0, 24) if i not in self.inactive_hours]
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def __build_features(self):
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

    def create_dataset(self, shuffle=False):
        self.__build_features()
        zones = ['ZONE_' + str(i) for i in range(1, 3 + 1)]
        col_names = []
        target_names = ['POWER' + str(h) for h in self.active_hours]
        for col in self.features:
            for h in self.active_hours:
                col_names.append(col + '_' + str(h))
        df = self.dataframe.copy()
        hours_per_day = len(self.active_hours)
        nb_days = int(len(df) / hours_per_day)
        index_1d = df['TIMESTAMP'].values.reshape(nb_days, hours_per_day)[:,0]
        index = pd.DatetimeIndex(index_1d)
        x = [df[col].values.reshape(nb_days, hours_per_day) for col in self.features]
        x.extend([df[zone].values.reshape(nb_days, hours_per_day)[:, 0].reshape(nb_days, 1) for zone in zones]) # ZONEID is the same for all hours
        col_names.extend(zones)
        y = df['POWER'].values.reshape(nb_days, hours_per_day)
        df_per_day = pd.DataFrame(
            np.concatenate([*x, y], axis=1), 
            columns=col_names + target_names,
            index=index
            )

        df_per_day['month'] = df_per_day.index.month
        df_per_day['day_of_week'] = df_per_day.index.dayofweek
        for zone in zones:
            zone_indices = df_per_day[zone] == 1
            df_per_day.loc[zone_indices, target_names] = df_per_day.loc[zone_indices, target_names].shift(-1)

        df_per_day.dropna(inplace=True)
        
        if shuffle:
            self.dataframe = df_per_day.sample(frac=1)
        return df_per_day

    def split(self, random_state=42, test_size=0.2, validation_size=0.2, shuffle=True):
        df_per_day = self.create_dataset(shuffle=shuffle)
        X = df_per_day.drop(columns=['POWER' + str(h) for h in self.active_hours])
        self.context_dim = len(X.columns)
        y = df_per_day[['POWER' + str(h) for h in self.active_hours]]

        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
        else:
            X_train, X_test, y_train, y_test = X, [], y, []
        
        if validation_size > 0 and len(X_test) > 0: 
            X_test, X_val, y_test, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=random_state, shuffle=shuffle)
        else:
            X_test, X_val, y_test, y_val = X_test, [], y_test, []


        if self.x_scaler is None:
            self.x_scaler = StandardScaler()
            X_train_scaled = self.x_scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.x_scaler.transform(X_train)

        if self.y_scaler is None:
            self.y_scaler = StandardScaler()
            y_train_scaled = self.y_scaler.fit_transform(y_train)
        else:
            y_train_scaled = self.y_scaler.transform(y_train)

        # Scale test
        self.test_dataset = None
        if len(X_test) > 0:
            X_test_scaled = self.x_scaler.transform(X_test)
            y_test_scaled = self.y_scaler.transform(y_test)
            self.test_dataset = MyDataset(
                    torch.tensor(X_test_scaled, dtype=torch.float32),
                    torch.tensor(y_test_scaled, dtype=torch.float32)
                )

        # Scale validation
        self.validation_dataset = None
        if len(X_val) > 0:
            X_validation_scaled = self.x_scaler.transform(X_val)        
            y_validation_scaled = self.y_scaler.transform(y_val)
            self.validation_dataset = MyDataset(
                    torch.tensor(X_validation_scaled, dtype=torch.float32),
                    torch.tensor(y_validation_scaled, dtype=torch.float32)
                )
        

        # Make tensors
        self.train_dataset = MyDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train_scaled, dtype=torch.float32)
        )

        return self.train_dataset, self.validation_dataset, self.test_dataset

    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=4, use_gpu=False, **kwargs):
        self.split(shuffle=shuffle, **kwargs)

        device = get_device(use_gpu)
        pin = True if device.type == 'cuda' and use_gpu else False

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )
        if self.validation_dataset is not None:
            self.validation_dataloader = DataLoader(
                self.validation_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
            )
        else:
            self.validation_dataloader = None

        if self.test_dataset is not None:
            self.test_dataloader = DataLoader(
                 self.test_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )
        else:
            self.test_dataloader = None

        return self.train_dataloader, self.validation_dataloader, self.test_dataloader
    
class GEFcomLoadLoader():
    def __init__(self, dataframe, x_scaler=None, y_scaler=None):
        self.dataframe = dataframe
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def __build_features(self):
        self.dataframe = self.dataframe.dropna()
        self.features = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10',
                'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18', 'w19', 'w20',
                'w21', 'w22', 'w23', 'w24', 'w25']
        
    def create_dataset(self, shuffle=True):
        self.__build_features()
        col_names = []
        target_names = ['LOAD' + str(h) for h in range(1, 25)]
        for col in self.features:
            for h in range(1, 25):
                col_names.append(col + '_' + str(h))
        df = self.dataframe.copy()
        nb_days = int(len(df) / 24)
        index_1d = df['TIMESTAMP'].values.reshape(nb_days, 24)[:,0]
        index = pd.DatetimeIndex(index_1d)
        x = [df[col].values.reshape(nb_days, 24) for col in self.features]
        y = df['LOAD'].values.reshape(nb_days, 24)
        df_per_day = pd.DataFrame(
            np.concatenate([*x, y], axis=1), 
            columns=col_names + target_names,
            index=index
            )

        df_per_day['month'] = df_per_day.index.month
        df_per_day['day_of_week'] = df_per_day.index.dayofweek

        df_per_day.dropna(inplace=True)
        if shuffle:
            df_per_day = df_per_day.sample(frac=1)

        return df_per_day
    
    def split(self, random_state=42, test_size=0.2, validation_size=0.2, shuffle=True):
        df_per_day = self.create_dataset(shuffle=shuffle)
        X = df_per_day.drop(columns=['LOAD' + str(h) for h in range(1, 25)])
        self.context_dim = len(X.columns)
        y = df_per_day[['LOAD' + str(h) for h in range(1, 25)]]

        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
        else:
            X_train, X_test, y_train, y_test = X, [], y, []

        if validation_size > 0 and len(X_test) > 0: 
            X_test, X_val, y_test, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=random_state, shuffle=shuffle)
        else:
            X_test, X_val, y_test, y_val = X_test, [], y_test, []


        if self.x_scaler is None:
            self.x_scaler = StandardScaler()
            X_train_scaled = self.x_scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.x_scaler.transform(X_train)

        if self.y_scaler is None:
            self.y_scaler = StandardScaler()
            y_train_scaled = self.y_scaler.fit_transform(y_train)
        else:
            y_train_scaled = self.y_scaler.transform(y_train)
        
        # Scale test
        self.test_dataset = None
        if len(X_test) > 0:
            X_test_scaled = self.x_scaler.transform(X_test)
            y_test_scaled = self.y_scaler.transform(y_test)
            self.test_dataset = MyDataset(
                    torch.tensor(X_test_scaled, dtype=torch.float32),
                    torch.tensor(y_test_scaled, dtype=torch.float32)
                )

        # Scale validation
        self.validation_dataset = None
        if len(X_val) > 0:
            X_validation_scaled = self.x_scaler.transform(X_val)        
            y_validation_scaled = self.y_scaler.transform(y_val)
            self.validation_dataset = MyDataset(
                    torch.tensor(X_validation_scaled, dtype=torch.float32),
                    torch.tensor(y_validation_scaled, dtype=torch.float32)
                )
        

        # Make tensors
        self.train_dataset = MyDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train_scaled, dtype=torch.float32)
        )

        return self.train_dataset, self.validation_dataset, self.test_dataset

    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=4, use_gpu=False, **kwargs):
        self.split(shuffle=shuffle, **kwargs)

        device = get_device(use_gpu)
        pin = True if device.type == 'cuda' and use_gpu else False

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )
        if self.validation_dataset is not None:
            self.validation_dataloader = DataLoader(
                self.validation_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
            )
        else:
            self.validation_dataloader = None

        if self.test_dataset is not None:
            self.test_dataloader = DataLoader(
                 self.test_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers, pin_memory=pin
        )
        else:
            self.test_dataloader = None

        return self.train_dataloader, self.validation_dataloader, self.test_dataloader
