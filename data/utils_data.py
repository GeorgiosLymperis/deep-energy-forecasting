import os
from typing import List

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
import torch

def get_device(use_gpu: bool=False)->torch.device:
    """
    Returns the device to be used for training.
    If use_gpu is True and a GPU is available, the GPU will be used.
    Otherwise, the CPU will be used.
    """
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_wind_dataset(folder: str = 'wind', save: bool = False, savepath: str = 'wind_data.csv') -> pd.DataFrame:
    """
    Create the wind dataset.

    Args:
        folder (str): The folder containing the wind data.
        save (bool): Whether to save the dataset.
        savepath (str): The path to save the dataset.

    Returns:
        pd.DataFrame: The wind dataset.
    """
    files = os.listdir(folder)
    df_wind = pd.DataFrame()
    for file in files:
        df = pd.read_csv(os.path.join(folder, file), parse_dates=['TIMESTAMP'])
        df_wind = pd.concat([df_wind, df], axis=0)
    df_wind = pd.get_dummies(df_wind, prefix='ZONE', prefix_sep='_', columns=['ZONEID'], dtype='int')
    if save:
        df_wind.to_csv(savepath)
    
    return df_wind


def create_solar_dataset(folder: str = 'solar', save: bool = False, savepath: str = 'solar_data.csv') -> pd.DataFrame:
    """
    Create the solar dataset.

    Args:
        folder (str): The folder containing the solar data.
        save (bool): Whether to save the dataset.
        savepath (str): The path to save the dataset.

    Returns:
        pd.DataFrame: The solar dataset.
    """
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
        df_solar.to_csv(savepath)

    return df_solar.reset_index()

def create_load_dataset(folder: str='load', save: bool=False, savepath: str='load_data.csv')->pd.DataFrame:
    """
    Create the load dataset.

    Args:
        folder (str): The folder containing the load data.
        save (bool): Whether to save the dataset.
        savepath (str): The path to save the dataset.

    Returns:
        pd.DataFrame: The load dataset.
    """
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
        df_load.to_csv(savepath)
    return df_load
    
class MyDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]

    
class GEFcomWindLoader():
    """
    Loader for the GEFcom wind dataset.
    If the scalers are None, they will be created from the data.
    If the loader is used for scenario generation, the scalers must be provided from the training data.

    Args:
        dataframe (pd.DataFrame): The wind dataset.
        x_scaler (StandardScaler): The scaler for the input features.
        y_scaler (StandardScaler): The scaler for the output features.

    Attributes:
        dataframe (pd.DataFrame): The wind dataset.
        x_scaler (StandardScaler): The scaler for the input features.
        y_scaler (StandardScaler): The scaler for the output features.
        features (list): A list of the features.
        context_dim (int): The dimension of the context vector.
        train_dataset (MyDataset): The training dataset.
        validation_dataset (MyDataset): The validation dataset.
        test_dataset (MyDataset): The test dataset.
        train_dataloader (DataLoader): The training dataloader.
        validation_dataloader (DataLoader): The validation dataloader.
        test_dataloader (DataLoader): The test dataloader.
    """
    def __init__(self, dataframe: pd.DataFrame, x_scaler: StandardScaler=None, y_scaler: StandardScaler=None):
        self.dataframe = dataframe
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def __build_features(self, density: int=1, save: bool=False):
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

    def create_dataset(self, shuffle: bool=True, filter_hours: List[str]=[])->pd.DataFrame:
        """
        Creates the dataset in the format expected by the model.
        
        Args:
            shuffle (bool): Whether to shuffle the dataset.
            filter_hours (List[str]): List of hours to filter out.

        Returns:
            pd.DataFrame: The dataset in the format expected by the model.
        """
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
        numerical_features = col_names.copy() # is for use later for filtering
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

        if filter_hours:
            filter_columns = [col for col in numerical_features if col.split('_')[1] in filter_hours]
            df_per_day = df_per_day.drop(columns=filter_columns)

        if shuffle: 
            df_per_day = df_per_day.sample(frac=1)

        return df_per_day

    def split(self, random_state: int=42, test_size: float=0.5, validation_size: float=0.2, shuffle: bool=True, filter_hours: List[str]=[])->tuple:
        """
        Splits the dataset into train, validation and test sets.

        Args:
            random_state (int): The random state to use for shuffling the dataset.
            validation_size (float): The percentage of the dataset to use for the validation set.
            test_size (float): The percentage of the validation dataset to use for the test set.
            shuffle (bool): Whether to shuffle the dataset.
            filter_hours (List[str]): List of hours to filter out.

        Returns:
            tuple: A tuple containing the train, validation and test sets.
        """
        df_per_day = self.create_dataset(shuffle=shuffle, filter_hours=filter_hours)
        X = df_per_day.drop(columns=['TARGETVAR' + str(h) for h in range(1, 25)])
        self.context_dim = len(X.columns)
        y = df_per_day[['TARGETVAR' + str(h) for h in range(1, 25)]]

        if validation_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, random_state=random_state, shuffle=shuffle)
        else:
            X_train, X_val, y_train, y_val = X, [], y, []

        if validation_size > 0 and len(X_val) > 0: 
            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size, random_state=random_state, shuffle=shuffle)
        else:
            X_val, X_test, y_val, y_test = X_val, [], y_val, []


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

    def get_dataloaders(self, batch_size: int=32, shuffle: bool=True, num_workers: int=4, use_gpu: bool=False, filter_hours: List[str]=[], **kwargs)->tuple:
        """
        Create the dataloaders for the train, validation and test sets.

        Args:
            batch_size (int): The batch size to use for the dataloaders.
            shuffle (bool): Whether to shuffle the dataset.
            num_workers (int): The number of workers to use for the dataloaders.
            use_gpu (bool): Whether to use the GPU.
            filter_hours (List[str]): List of hours to filter out.

        Returns:
            tuple: A tuple containing the train, validation and test dataloaders.
        """
        self.split(shuffle=shuffle, filter_hours=filter_hours, **kwargs)

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
    """
    Loader for the GEFcom solar dataset.
    If the scalers are None, they will be created from the data.
    If the loader is used for scenario generation, the scalers must be provided from the training data.

    Args:
        dataframe (pd.DataFrame): The solar dataset.
        x_scaler (StandardScaler): The scaler for the input features.
        y_scaler (StandardScaler): The scaler for the output features.

    Attributes:
        dataframe (pd.DataFrame): The solar dataset.
        x_scaler (StandardScaler): The scaler for the input features.
        y_scaler (StandardScaler): The scaler for the output features.
        active_hours: A list of the active hours.
        inactive_hours: A list of the inactive hours.
        features (list): A list of the features.
        context_dim (int): The dimension of the context vector.
        train_dataset (MyDataset): The training dataset.
        validation_dataset (MyDataset): The validation dataset.
        test_dataset (MyDataset): The test dataset.
        train_dataloader (DataLoader): The training dataloader.
        validation_dataloader (DataLoader): The validation dataloader.
        test_dataloader (DataLoader): The test dataloader.
    """
    def __init__(self, dataframe: pd.DataFrame, x_scaler: StandardScaler=None, y_scaler: StandardScaler=None):
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

    def create_dataset(self, shuffle: bool=True, filter_hours: List[str]=[])->pd.DataFrame:
        """
        Creates the dataset in the format expected by the model.
        
        Args:
            shuffle (bool): Whether to shuffle the dataset.
            filter_hours (List[str]): List of hours to filter out.

        Returns:
            pd.DataFrame: The dataset in the format expected by the model.
        """
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
        numerical_features = col_names.copy() # is for use later for filtering
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

        if filter_hours:
            filter_columns = [col for col in numerical_features if col.split('_')[1] in filter_hours]
            df_per_day = df_per_day.drop(columns=filter_columns)
        
        
        if shuffle:
            df_per_day = df_per_day.sample(frac=1)
        return df_per_day

    def split(self, random_state: int=42, test_size: float=0.5, validation_size: float=0.2, shuffle: bool=True, filter_hours: List[str]=[])->tuple:
        """
        Splits the dataset into train, validation and test sets.

        Args:
            random_state (int): The random state to use for shuffling the dataset.
            validation_size (float): The percentage of the dataset to use for the validation set.
            test_size (float): The percentage of the validation dataset to use for the test set.
            shuffle (bool): Whether to shuffle the dataset.
            filter_hours (List[str]): List of hours to filter out.

        Returns:
            tuple: A tuple containing the train, validation and test sets.
        """
        df_per_day = self.create_dataset(shuffle=shuffle, filter_hours=filter_hours)
        X = df_per_day.drop(columns=['POWER' + str(h) for h in self.active_hours])
        self.context_dim = len(X.columns)
        y = df_per_day[['POWER' + str(h) for h in self.active_hours]]

        if validation_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, random_state=random_state, shuffle=shuffle)
        else:
            X_train, X_val, y_train, y_val = X, [], y, []

        if validation_size > 0 and len(X_val) > 0: 
            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size, random_state=random_state, shuffle=shuffle)
        else:
            X_val, X_test, y_val, y_test = X_val, [], y_val, []


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

    def get_dataloaders(self, batch_size: int=32, shuffle: bool=True, num_workers: int=4, use_gpu: bool=False, filter_hours: List[str]=[], **kwargs)->tuple:
        """
        Create the dataloaders for the train, validation and test sets.

        Args:
            batch_size (int): The batch size to use for the dataloaders.
            shuffle (bool): Whether to shuffle the dataset.
            num_workers (int): The number of workers to use for the dataloaders.
            use_gpu (bool): Whether to use the GPU.
            filter_hours (List[str]): List of hours to filter out.

        Returns:
            tuple: A tuple containing the train, validation and test dataloaders.
        """
        self.split(shuffle=shuffle, filter_hours=filter_hours, **kwargs)

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
    """
    Loader for the GEFcom load dataset.
    If the scalers are None, they will be created from the data.
    If the loader is used for scenario generation, the scalers must be provided from the training data.

    Args:
        dataframe (pd.DataFrame): The load dataset.
        x_scaler (StandardScaler): The scaler for the input features.
        y_scaler (StandardScaler): The scaler for the output features.

    Attributes:
        dataframe (pd.DataFrame): The load dataset.
        x_scaler (StandardScaler): The scaler for the input features.
        y_scaler (StandardScaler): The scaler for the output features.
        features (list): A list of the features.
        context_dim (int): The dimension of the context vector.
        train_dataset (MyDataset): The training dataset.
        validation_dataset (MyDataset): The validation dataset.
        test_dataset (MyDataset): The test dataset.
        train_dataloader (DataLoader): The training dataloader.
        validation_dataloader (DataLoader): The validation dataloader.
        test_dataloader (DataLoader): The test dataloader.
    """
    def __init__(self, dataframe: pd.DataFrame, x_scaler: StandardScaler=None, y_scaler: StandardScaler=None):
        self.dataframe = dataframe
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def __build_features(self):
        self.dataframe = self.dataframe.dropna()
        self.features = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10',
                'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18', 'w19', 'w20',
                'w21', 'w22', 'w23', 'w24', 'w25']
        
    def create_dataset(self, shuffle: bool=True, filter_hours: List[str]=[])->pd.DataFrame:
        """
        Creates the dataset in the format expected by the model.
        
        Args:
            shuffle (bool): Whether to shuffle the dataset.
            filter_hours (List[str]): List of hours to filter out.

        Returns:
            pd.DataFrame: The dataset in the format expected by the model.
        """
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

        if filter_hours:
            filter_columns = [col for col in col_names if col.split('_')[1] in filter_hours]
            df_per_day = df_per_day.drop(columns=filter_columns)

        if shuffle:
            df_per_day = df_per_day.sample(frac=1)

        return df_per_day
    
    def split(self, random_state: int=42, test_size: float=0.5, validation_size: float=0.2, shuffle: bool=True, filter_hours: List[str]=[])->tuple:
        """
        Splits the dataset into train, validation and test sets.

        Args:
            random_state (int): The random state to use for shuffling the dataset.
            validation_size (float): The percentage of the dataset to use for the validation set.
            test_size (float): The percentage of the validation dataset to use for the test set.
            shuffle (bool): Whether to shuffle the dataset.
            filter_hours (List[str]): List of hours to filter out.

        Returns:
            tuple: A tuple containing the train, validation and test sets.
        """
        df_per_day = self.create_dataset(shuffle=shuffle, filter_hours=filter_hours)
        X = df_per_day.drop(columns=['LOAD' + str(h) for h in range(1, 25)])
        self.context_dim = len(X.columns)
        y = df_per_day[['LOAD' + str(h) for h in range(1, 25)]]

        if validation_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, random_state=random_state, shuffle=shuffle)
        else:
            X_train, X_val, y_train, y_val = X, [], y, []

        if validation_size > 0 and len(X_val) > 0: 
            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size, random_state=random_state, shuffle=shuffle)
        else:
            X_val, X_test, y_val, y_test = X_val, [], y_val, []


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

    def get_dataloaders(self, batch_size: int=32, shuffle: bool=True, num_workers: int=4, use_gpu: bool=False, filter_hours: List[str]=[], **kwargs)->tuple:
        """
        Create the dataloaders for the train, validation and test sets.

        Args:
            batch_size (int): The batch size to use for the dataloaders.
            shuffle (bool): Whether to shuffle the dataset.
            num_workers (int): The number of workers to use for the dataloaders.
            use_gpu (bool): Whether to use the GPU.
            filter_hours (List[str]): List of hours to filter out.

        Returns:
            tuple: A tuple containing the train, validation and test dataloaders.
        """
        self.split(shuffle=shuffle, filter_hours=filter_hours, **kwargs)

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
