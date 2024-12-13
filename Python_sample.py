import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from prophet import Prophet
import os
import warnings
import sys
import random
from scipy.stats import loguniform
import holidays

warnings.filterwarnings("ignore")

# def
class LSTM(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, -1])
    return np.array(X), np.array(y)

# MAPE 
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    y_true_non_zero = y_true[non_zero_indices]
    y_pred_non_zero = y_pred[non_zero_indices]
    return np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100

# RMSE 
def Root_Mean_Square_Error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    y_true_non_zero = y_true[non_zero_indices]
    y_pred_non_zero = y_pred[non_zero_indices]
    rmse = np.sqrt(sum((y_true_non_zero - y_pred_non_zero)**2)/len(y_true_non_zero))
    return rmse

# define holidays for Prophet model
temp = holidays.US(years=[2023,2024])
chosen_holidays = []
ds = []
for ptr in temp.items():
    ds.append(ptr[0])
    chosen_holidays.append(ptr[1])
    
chosen_holidays = pd.DataFrame({"holiday":chosen_holidays, "ds":ds}).sort_values("ds")
chosen_holidays = chosen_holidays.drop([11,17],axis=0)
chosen_holidays.index = list(range(len(chosen_holidays)))
lower_window = [-2,0,0,-2,0,-3,-2,0,0,-1,-2,
                -1,0,0,-2,0,-2,-2,0,0,-1,-3]
upper_window = [0,0,0,1,0,1,1,0,0,4,5,
                0,0,0,1,0,1,0,0,0,4,5]
chosen_holidays["lower_window"] = lower_window
chosen_holidays["upper_window"] = upper_window

# File paths
current_directory = os.path.dirname(__file__)
#paths = [os.path.join(current_directory, 'fhvhv_final', f'fhvhv_final_{str(i).zfill(2)}.csv') for i in range(1, 14)]
#test_path = os.path.join(current_directory, 'fhvhv_final', 'fhvhv_final_14.csv')

months = pd.date_range(start="2023-01", end="2024-02", freq="M").strftime("%Y-%m")
paths = [os.path.join(current_directory, 'fhvhv_final', f'fhvhv_final_{month}.csv') for month in months]
test_path = os.path.join(current_directory, 'fhvhv_final', 'fhvhv_final_2024-02.csv')
zone_path = os.path.join(current_directory, 'taxi_zone_lookup.csv')
log_file = os.path.join(current_directory, 'output_log.txt')
# Load and merge data
dfs = [pd.read_csv(path) for path in paths]
df = pd.concat(dfs)
print(df.tail())
test = pd.read_csv(test_path)
zone = pd.read_csv(zone_path)
df = df.merge(zone, on='zoneID', how='left')
test = test.merge(zone, on='zoneID', how='left')
# Filter data (yearly count >= 1 million)
'''a = df[['zoneID', 'count']].groupby('zoneID').sum('count')
a = a.sort_values['count']
target_zones = a[a['count'] >= 1000000].reset_index()['zoneID']
'''
a = df.groupby('zoneID', as_index=False)['count'].sum()
a = a.sort_values(by='count', ascending=False)
target_zones = a[a['count'] >= 1_000_000]['zoneID'].reset_index(drop=True)

'''
# Filter data from Manhattan
df = df[df['Borough'] == 'Manhattan']
test = test[test['Borough'] == 'Manhattan']
ids = zone[zone['Borough'] == 'Manhattan']['zoneID']
'''

# save
results_train_all = []
results_valid_all = []
results_test_all = []
evaluation_test_all =[]

with open(log_file, 'w') as f:
    # Redirect stdout to the log file
    sys.stdout = f
# area for
    for id in target_zones:
        if id !=79:
            continue
        print(f"zone {id}/n")
        df_zone = df[df['zoneID'] == id].copy()
        test_zone = test[test['zoneID'] == id].copy()

        df_zone.sort_index(inplace=True) 
        df_prophet = df_zone.reset_index()[["time_period", "count"]].copy()
        df_prophet['datetime'] = pd.to_datetime(df_prophet['time_period'])
        df_prophet = df_prophet.rename(columns = {'datetime':'ds','count':'y'})
        # df_prophet = df_prophet.groupby('ds')['y'].sum().reset_index()‘
        test_zone.sort_index(inplace=True)
        test_prophet = test_zone.reset_index()[["time_period", "count"]].copy()
        test_prophet['datetime'] = pd.to_datetime(test_prophet['time_period'])
        test_prophet = test_prophet.rename(columns = {'datetime':'ds','count':'y'})

        # Initialize Prophet model with custom seasonalities
        model_prophet = Prophet(growth = 'linear', 
                                yearly_seasonality=False,
                                weekly_seasonality=True,
                                daily_seasonality=False,
                                holidays = chosen_holidays,
                                changepoint_prior_scale=0.5) 
        
        
        model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=6)
        # model_prophet.add_country_holidays(country_name="US")
        model_prophet.add_seasonality(name='weekday', period=1, fourier_order=5, condition_name="weekday")
        model_prophet.add_seasonality(name='weekend', period=1, fourier_order=5, condition_name="weekend")

        # Prepare the training dataset with custom conditions
        df_prophet["weekday"] = df_prophet["ds"].apply(lambda x: x.weekday() < 5)
        df_prophet["weekend"] = df_prophet["ds"].apply(lambda x: x.weekday() >= 5)

        # Fit the model on the training data
        model_prophet.fit(df_prophet)

        # Make future dataframe for training predictions and add custom conditions
        future_train = model_prophet.make_future_dataframe(periods=0, freq='30T')
        future_train["weekday"] = future_train["ds"].apply(lambda x: x.weekday() < 5)
        future_train["weekend"] = future_train["ds"].apply(lambda x: x.weekday() >= 5)

        # Predict on training data
        forecast_train = model_prophet.predict(future_train)

        # Prepare test data with custom conditions
        test_prophet["weekday"] = test_prophet["ds"].apply(lambda x: x.weekday() < 5)
        test_prophet["weekend"] = test_prophet["ds"].apply(lambda x: x.weekday() >= 5)
        test_prophet = test_prophet.set_index('ds')

        # Make future dataframe for test predictions
        future = model_prophet.make_future_dataframe(periods=len(test_prophet), freq='30T')
        future["weekday"] = future["ds"].apply(lambda x: x.weekday() < 5)
        future["weekend"] = future["ds"].apply(lambda x: x.weekday() >= 5)

        # Predict on test data
        forecast_test = model_prophet.predict(future)
        forecast_test = forecast_test[-len(test_prophet):]  # Select only the test period rows
        print("forecast_test.columns")
        print(forecast_test.columns)
        # part of prophet predictions used in final predictions
        forecast_train['addedseasonality'] = forecast_train['weekly'] + forecast_train['weekday'] + forecast_train['weekend']+ forecast_train['monthly'] + forecast_train['holidays']
        forecast_test['addedseasonality'] = forecast_test['weekly'] + forecast_test['weekday'] + forecast_test['weekend']+ forecast_test['monthly'] + forecast_test['holidays']

        print("forecast_test.head")
        print(forecast_test.head())

        # print('forecast_train', forecast_train.shape)
        df_zone['ds'] = pd.to_datetime(df_zone['time_period']).copy()
        df_zone = df_zone.merge(forecast_train, on='ds', how='left')
        residuals_train = df_zone['count']-df_zone['yhat']
        df_zone['residual_trend'] = residuals_train + df_zone['trend']
        df_zone = df_zone.dropna()
        df_zone.set_index('time_period', inplace=True)######################

        valid_zone = df_zone[df_zone['datetime'].dt.year == 2024]
        train_zone = df_zone[df_zone['datetime'].dt.year == 2023]

        #test_zone.loc[:, 'ds'] = pd.to_datetime(test_zone['time_period'])
        test_zone['ds'] = pd.to_datetime(test_zone['time_period']).copy()
        # print("check1")
        # print(test_zone.head())
        test_zone = test_zone.merge(forecast_test, on='ds', how='left')
        # print("check2")
        # print(test_zone.head())
        # print("check3")
        # print(test_zone.head())
        test_zone['residual_trend'] = test_zone['count'] - test_zone['addedseasonality']
        # print("check4")
        # print(test_zone.head())
        test_zone = test_zone.dropna()
        # print("check5")
        # print(test_zone.head())
        test_zone.set_index('time_period', inplace=True)
        # print("check6")
        # print(test_zone.head())

        # lstm, data processing
        forecast_train = forecast_train.reset_index()[['ds','yhat','trend']].copy()
        forecast_test = forecast_test.reset_index()[['ds','yhat','trend']].copy()
        
        print("forecast_test.head")
        print(forecast_test.head())

        # standardize the train set
        resid_scaler = MinMaxScaler(feature_range=(-1, 1))
        residual_scaled_train = resid_scaler.fit_transform(df_zone['residual_trend'].values.reshape(-1, 1))
        # use the parameters of train set to standardize the test set
        residual_scaled_test = resid_scaler.transform(test_zone['residual_trend'].values.reshape(-1, 1))
        columns = ['temperature_2m', 'apparent_temperature','relative_humidity_2m',
                   'rain','snowfall','snow_depth','cloud_cover_low','shortwave_radiation',
                   'wind_speed_10m','wind_direction_10m','felony','misdemeanor','residual_trend']
        features = df_zone[columns]
        features_test = test_zone[columns]

        # standardize all features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_features_train = scaler.fit_transform(features) #ndarrays
        scaled_features_test = scaler.transform(features_test)
        
        time_step=4
        train = scaled_features_train[df_zone['ds'].dt.year == 2023]
        valid = scaled_features_train[df_zone['ds'].dt.year == 2024]
        # Create datasets
        X_trainandvalid, y_trainandvalid = create_dataset(scaled_features_train, time_step)
        X_train, y_train = create_dataset(train, time_step)
        X_valid, y_valid = create_dataset(valid, time_step)
        X_test, y_test = create_dataset(scaled_features_test, time_step)

        # tensor
        X_trainandvalid, y_trainandvalid = torch.tensor(X_trainandvalid, dtype=torch.float32), torch.tensor(y_trainandvalid, dtype=torch.float32)
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        X_valid, y_valid = torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32)
        X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

        
        param_grid = {
            'batch_size' : [8,16,32,64],
            'hidden_size': [32, 64],
            'num_layers': [1, 2],
            'lr': loguniform.rvs(1e-4,1e-1,size=100)
        }
        grid = list(ParameterGrid(param_grid))
        best_mape = float('inf')
        best_rmse = float('inf')
        best_params = None
        
        # trial
        for _ in range(10):
            params = random.choice(grid)
            print(f'{params}')
            model = LSTM(batch_size = params['batch_size'], input_size=X_train.shape[2], hidden_size=params['hidden_size'], num_layers=params['num_layers'])
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

            num_epochs = 50
            for epoch in range(num_epochs):
                model.train()
                outputs = model(X_train)
                loss = criterion(outputs.flatten(), y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

            model.eval()
            with torch.no_grad():
                predictions = model(X_valid).numpy().flatten()
                y_valid_true = df_zone['count'].iloc[-len(y_valid):].values
                y_valid_pred = resid_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()+df_zone['addedseasonality'].tail(len(y_valid))
                mape = mean_absolute_percentage_error(y_valid_true, y_valid_pred)
                rmse = Root_Mean_Square_Error(y_valid_true, y_valid_pred)
                if rmse < best_rmse:
                    best_mape = mape
                    best_rmse = rmse
                    best_params = params
                    best_model = model

        print(f"zone {id} best paras: {best_params}, test MAPE: {best_mape:.2f}%")

        best_model.eval()
        with torch.no_grad():
            predictions_train = best_model(X_train).numpy().flatten()
            predictions_valid = best_model(X_valid).numpy().flatten()
            predictions_test = best_model(X_test).numpy().flatten()
        
        predictions_train_rescaled = resid_scaler.inverse_transform(predictions_train.reshape(-1, 1)).flatten() 
        predictions_valid_rescaled = resid_scaler.inverse_transform(predictions_valid.reshape(-1, 1)).flatten() 
        predictions_test_rescaled = resid_scaler.inverse_transform(predictions_test.reshape(-1, 1)).flatten() 

        # final predictions: lstm pred(trend+residual) + prophet pred(season+holidays)
        predictions_train_rescaled = predictions_train_rescaled + train_zone['addedseasonality'][time_step:-1]
        predictions_valid_rescaled = predictions_valid_rescaled + valid_zone['addedseasonality'][time_step:-1]
        predictions_test_rescaled = predictions_test_rescaled + test_zone['addedseasonality'][time_step:-1]
        
        y_true_test = test_zone['count'][time_step:-1]
        test_mape = mean_absolute_percentage_error(y_true_test, predictions_test_rescaled)
        test_rmse = Root_Mean_Square_Error(y_true_test, predictions_test_rescaled)
        # print(f'MAPE of training set of zone {int(id)}: {mape_train:.2f}%')
        # print(f'MAPE of test set of zone {int(id)}: {mape_test:.2f}%')
        # print(f'RMSE of training set of zone {int(id)}: {rmse_train:.2f}')
        # print(f'RMSE of test set of zone {int(id)}: {rmse_test:.2f}')

        results_train = pd.DataFrame({
            'time_period': df_zone['ds'][time_step:-1].values,
            'zoneID': [id] * len(predictions_train),
            'count': df_zone['count'][time_step:-1].values,
            'predicted_train': predictions_train_rescaled
        })
        results_valid = pd.DataFrame({
            'time_period': valid_zone['ds'][time_step:-1].values,
            'zoneID': [id] * len(predictions_valid),
            'count': valid_zone['count'][time_step:-1].values,
            'predicted_valid': predictions_valid_rescaled
        })
        results_test = pd.DataFrame({
            'time_period': test_zone['ds'][time_step:-1].values,
            'zoneID': [id] * len(predictions_test),
            'count': test_zone['count'][time_step:-1].values,
            'predicted_test': predictions_test_rescaled
        })

        evaluation_test = pd.DataFrame({
        'zoneID': [id],
        'best_para_batchsize': [best_params['batch_size']],
        'best_para_hiddensize': [best_params['hidden_size']],
        'best_para_numlayer': [best_params['num_layers']],
        'best_para_lr': [best_params['lr']],
        'rmse_valid': [best_rmse],
        'mape_valid': [best_mape],
        'rmse_test':[test_rmse],
        'rmse_test':[test_mape]
        })
    
        results_train_all.append(results_train)
        results_valid_all.append(results_valid)
        results_test_all.append(results_test)
        evaluation_test_all.append(evaluation_test)

    # combine
    final_train_results = pd.concat(results_train_all)
    final_valid_results = pd.concat(results_valid_all)
    final_test_results = pd.concat(results_test_all)
    final_evaluation_test_results = pd.concat(evaluation_test_all)

    final_train_results.to_csv(os.path.join(current_directory, 'Prophet_LSTM_results_train.csv'), index=False)
    final_valid_results.to_csv(os.path.join(current_directory, 'Prophet_LSTM_results_valid.csv'), index=False)
    final_test_results.to_csv(os.path.join(current_directory, 'Prophet_LSTM_results_test.csv'), index=False)
    final_evaluation_test_results.to_csv(os.path.join(current_directory, 'Prophet_LSTM_evaluation_results_test.csv'), index=False)
    sys.stdout = sys.__stdout__
