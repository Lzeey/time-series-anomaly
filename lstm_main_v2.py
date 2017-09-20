# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 2017

@author: lzeey
This script contains the experiment of seq2seq RNN autoencoders
"""
import sys
import pandas as pd
import numpy as np
import numpy.matlib
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.layers import LSTM, Activation, Dense, Input, Flatten
from keras.optimizers import RMSprop
from keras.models import Model
from statsmodels.tsa.stattools import adfuller

import plotly as py
import cufflinks as cf


def plot_graph_wrapper(df, title='', xTitle='', yTitle='',
                       filename='output.html'):
    """Cufflink offline plot for easy viewing"""
    cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
    cf_output = df.iplot(kind='scatter', title=title, xTitle=xTitle,
                         online=False, asFigure=True)
    py.offline.plot(cf_output, filename=filename)


def lstm_anomaly_model(input_dtype='float32',
                       input_dim=1, look_back=20, h_dim=64, out_dim=10, verbose=True):
    """Create a stateful LSTM time-series anomaly detection model
    input_dtype: Either 'int32' or 'float32'. Defined the numeric type for input
    input_dim: Input time series dimension (number of input PER time bin)
    h_dim: Dimension of hidden state in LSTM
    out_dim: Number of look-ahead prediction step"""
    # We take in only one timestep at a time
    in_layer = Input(batch_shape=(look_back, look_back, input_dim), dtype=input_dtype, name='input')
    # Use stateful to remember state from previous batch
    x = LSTM(h_dim, stateful=True, recurrent_dropout=0.2, name='LSTM', return_sequences=True)(in_layer)
    x1 = LSTM(h_dim, stateful=True, recurrent_dropout=0.2, name='LSTM2')(x)
    y = Dense(out_dim * input_dim)(x1)
    model = Model(inputs=in_layer, outputs=y)
    opt = RMSprop()
    model.compile(loss='mse', optimizer=opt)
    if verbose:
        print(model.summary())
    return model

def seasonal_decomposition(x, period):
    """Extracts the seasonal components of the signal x, according to period"""
    num_period = len(x) // period
    assert(num_period > 0)
    
    x_trunc = x[:num_period*period].reshape((num_period, period))
    x_season = np.mean(x_trunc, axis=0)
    x_season = np.concatenate((np.tile(x_season, num_period), x_season[:len(x) % period]))
    return x_season
    
def remove_seasonality(x):
    """Detects the seasonality and removes it"""
    x_trun


def time_series_anomaly(x_raw, look_ahead=5, look_back=36, epochs=2):
    """Perform anomaly detection on time-series x
    x: np.array of dimension (, n_dim), 0-th index is time,
    and 1st index is for multi_dimension input"""
    single_dim = False
    if len(x_raw.shape) == 1:  # 1 dimensional time-series
        input_dim = 1
        x_raw = x_raw.reshape(-1, 1)
        single_dim = True
    elif len(x_raw.shape) == 2:  # Multi-dim time-series
        input_dim = x_raw.shape[1]
    else:
        raise(Exception("Unexpected input shape. Expected 1 or 2 dimension. Received: {}".format(len(x_raw.shape))))
    t_len = x_raw.shape[0]
    x_raw = x_raw.astype(np.float32)
    
    #Perform normalization
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    x = scaler.fit_transform(x_raw)
    model = lstm_anomaly_model(input_dim=input_dim,
                                   look_back=look_back,
                                    h_dim=64, out_dim=look_ahead)
    pred = np.zeros((t_len - look_back, look_ahead, input_dim), dtype=np.float32)
    loss = None
    num_iters = (t_len - look_ahead - look_back) // look_back
    for ep in range(epochs + 1):
        model.reset_states()
        print("Starting epoch {}".format(ep))
        for i in range(num_iters - 1):  # Iterate through time-series
            if i % 1 == 0:
                print("Epoch: {} ({}/{}): Loss = {}"
                      .format(ep, i, num_iters-1, loss), end='\r', flush=True)
                sys.stdout.flush()
            #Chop up data
            in_x = np.zeros((look_back, look_back, input_dim))
            out_x = np.zeros((look_back, look_ahead, input_dim))
            for idx in range(look_back): #Within batch data-prep
                in_x[idx, :, :] = x[i*look_back + idx : (i+1)*look_back + idx].reshape((1, look_back, input_dim))
                out_x[idx, :, :] = x[(i+1)*look_back + idx  : (i+1)*look_back + idx + look_ahead].reshape((1, look_ahead, input_dim))
            
            #  We perform prediction THEN training.
            #  No way to extract prediction during forward-pass of training
            if ep == epochs:
                pred_x = model.predict(in_x)
                pred[i*look_back:(i+1)*look_back, :, :] = pred_x.reshape((look_back, look_ahead, input_dim))
            else:
                loss = model.train_on_batch(in_x, np.squeeze(out_x))

    pred = scaler.inverse_transform(pred)
    
    pred_full = np.full((t_len, look_ahead, input_dim), np.nan, dtype=np.float32)
    for i in range(look_ahead):  # Shift the time-frames forward
        pred_full[look_back+i:t_len, i, :] = pred[i:, i, :]

    error = pred_full - x_raw.reshape((-1, 1, input_dim))
    pred_mean = np.nanmean(pred_full, axis=1)
    pred_dev = np.std(pred_full, axis=1)
    if single_dim:
        pred_full = pred_full.squeeze()
        error = error.squeeze()
        pred_mean = pred_mean.squeeze()
        
    output = {'pred_full': pred_full,
              'pred': pred_mean,
              'error': error,
              'conf': pred_dev,
              }
    return output


if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)
    df = pd.read_csv("power_data.txt")
    df.columns = ['power']
    date_range = pd.date_range('1/1/1997', periods=35039, freq='15Min')
    
    # Set date to df
    df = pd.DataFrame(np.array(df['power']),index=date_range)
    df.columns = ['power']
    df = df.resample('2H').mean()
    
    #Transform into a delta problem
    #delta = df['power'][1:].values - df['power'][:-1].values
    df['season'] = seasonal_decomposition(df['power'].values, period=74)
    df['deseason'] = df['power'] - df['season']
               
    
    output = time_series_anomaly(df.deseason.values, look_ahead=5, look_back=31, epochs=1)
    #results = pd.DataFrame({'delta_pred':output['pred'], 'delta':delta})
    df['pred'] = output['pred'] + df['season']
    plot_graph_wrapper(df, title="Prediction")

    #FFT
    f_x = np.fft.rfft(df.power.values - np.mean(df.power.values))
    f_x = np.real(f_x * f_x.conj())
    plt.plot(f_x)
    max_powers = f_x.argsort()[-6:][::-1]
    
    period = len(df) / max_powers
    print(period)
    #Stationarity test
    p_val = adfuller(df.power.values)
    
    # Split timeseries per week
#    weeks = [g for n, g in df.groupby(pd.TimeGrouper('W'))]
#    red_weeks = weeks[1:52]
    
    """
    red_weeks[0:10,13:15,18,20:37,39] is normal
    red_weeks[11:12,16:17,19,38,50] is anomaly
    """
