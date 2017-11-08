# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 2017

@author: lzeey
This script contains the experiment of seq2seq RNN autoencoders
"""
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.layers import LSTM, Activation, Dense, Input, Flatten
from keras.optimizers import RMSprop
from keras.models import Model
from statsmodels.tsa.stattools import adfuller

import plotly as py
import plotly.graph_objs as go
import cufflinks as cf


def plot_graph_wrapper(df, title='', xTitle='', yTitle='',
                       filename='LSTM_output.html'):
    """Cufflink offline plot for easy viewing"""
    cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
    cf_output = df.iplot(kind='scatter', title=title, xTitle=xTitle,
                         online=False, asFigure=True)
    
    py.offline.plot(cf_output, filename=filename)

def plot_timeseries_with_bound(df, val_col, bound_col, true_col, xTitle='', yTitle='',
                       title='', filename='LSTM_output_bounds.html'):
    cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
#    cf_output = df.iplot(kind='scatter', title=title, xTitle=xTitle,
#                         online=False, asFigure=True)
    
    upper_bound = go.Scatter(
        name='Upper Bound',
        #x=df['Time'],
        y=df[val_col]+df[bound_col],
        mode='lines',
        marker=dict(color="444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty' )

    trace = go.Scatter(
        name='Measurement',
        #x=df['Time'],
        y=df[val_col],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty' )
    
    lower_bound = go.Scatter(
        name='Lower Bound',
        #x=df['Time'],
        y=df[val_col]-df[bound_col],
        marker=dict(color="444"),
        line=dict(width=0),
        mode='lines' )
    
    true_line = go.Scatter(
        name='Actual',
        #x=df['Time'],
        y=df[true_col],
        mode='lines',
        line=dict(color='rgb(180, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='lines' )
    
    # Trace order can be important
    # with continuous error bars
    data = [lower_bound, trace, upper_bound, true_line]
    
    layout = go.Layout(
        yaxis=dict(title=yTitle),
        title=title,
        showlegend = True)
    fig = go.Figure(data=data, layout=layout)
    
    py.offline.plot(fig, filename=filename)    
    


def lstm_anomaly_model(input_dtype='float32',
                       input_dim=1, h_dim=128, out_dim=10, layers=16, verbose=True):
    """Create a stateful LSTM time-series anomaly detection model
    input_dtype: Either 'int32' or 'float32'. Defined the numeric type for input
    input_dim: Input time series dimension (number of input PER time bin)
    h_dim: Dimension of hidden state in LSTM
    out_dim: Number of look-ahead prediction step"""
    # We take in only one timestep at a time
    in_layer = Input(batch_shape=(1, 1, input_dim), dtype=input_dtype, name='input')
    # Use stateful to remember state from previous batch
    x = LSTM(h_dim, stateful=True, recurrent_dropout=0.2, name='LSTM')(in_layer)
#    if layers == 1:
#        x = LSTM(h_dim, stateful=True, recurrent_dropout=0.2, name='LSTM')(in_layer)
#    else:
#        x = LSTM(h_dim, stateful=True, recurrent_dropout=0.2, name='LSTM', return_sequences=True)(in_layer)
#        for layer in range(1, layers-1):
#            x = LSTM(h_dim, stateful=True, recurrent_dropout=0.2, name='LSTM' + str(layer), return_sequences=True)(x)
#        x = LSTM(h_dim, stateful=True, recurrent_dropout=0.2, name='LSTM' + str(layers))(x)
#    
    y = Dense(out_dim * input_dim)(x)
    model = Model(inputs=in_layer, outputs=y)
    opt = RMSprop()
    model.compile(loss='mse', optimizer=opt)
    if verbose:
        print(model.summary())
    return model


def time_series_anomaly(x_raw, look_ahead=16, epochs=2, layers=16, hidden_size=16):
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
                                    h_dim=hidden_size, out_dim=look_ahead, layers=layers)
    pred = np.zeros((t_len-look_ahead, input_dim, look_ahead), dtype=np.float32)
    loss = None
    for ep in range(epochs):
        model.reset_states()
        for i in range(t_len - look_ahead):  # Iterate through time-series
            if i % 300 == 0:
                print("It {} ({}/{}): Loss = {}"
                      .format(ep, i, t_len-look_ahead, loss))
            in_x = x[i].reshape((1, 1, input_dim))
            out_x = x[i+1:i+1+look_ahead].reshape((1, -1))
            #  We perform prediction THEN training.
            #  No way to extract prediction during forward-pass of training            
            #Save prediction
            if ep == epochs - 1:
                pred_x = model.predict(in_x)
                pred[i, :, :] = pred_x
            else:
                loss = model.train_on_batch(in_x, out_x)
    pred = scaler.inverse_transform(pred)
    
    pred_full = np.full((t_len, input_dim, look_ahead), np.nan, dtype=np.float32)
    for i in range(look_ahead):  # Shift the time-frames forward
        pred_full[i+1:i+1+t_len-look_ahead, :, i] = pred[:, :, i]
    
    error = pred_full - x_raw.reshape((-1, input_dim, 1))
    pred_mean = np.nanmean(pred_full, axis=2)
    pred_dev = np.std(pred_full, axis=2)
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

def seasonal_decomposition(x, period):
    """Extracts the seasonal components of the signal x, according to period"""
    num_period = len(x) // period
    assert(num_period > 0)
    
    x_trunc = x[:num_period*period].reshape((num_period, period))
    x_season = np.mean(x_trunc, axis=0)
    x_season = np.concatenate((np.tile(x_season, num_period), x_season[:len(x) % period]))
    return x_season

def automatic_seasonality_remover(x, k_components=10, verbose=True):
    """Extracts the most likely seasonal component via FFT"""
    f_x = np.fft.rfft(x - np.mean(x))
    f_x = np.real(f_x * f_x.conj())
    periods = len(x) / f_x.argsort()[-k_components:][::-1]
    periods = np.rint(periods).astype(int)
    min_error = None
    best_period = 0
    best_season = np.zeros(len(x))
    
    for period in periods:
        if period == len(x): continue
        type(period)
        x_season = seasonal_decomposition(x, period)
        error = np.average((x - x_season)**2)
        if min_error is None or error < min_error:
            min_error = error
            best_season = x_season
            best_period = period
    
    if verbose:
        print("Best fit period: {}".format(best_period))
    return best_season
    
    
if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)
    sample_period = '1H' # '30Min'
    
    df = pd.read_csv("power_data.txt")
    df.columns = ['power']
    date_range = pd.date_range('1/1/1997', periods=35039, freq='15Min')
    
    # Set date to df
    df = pd.DataFrame(np.array(df['power']),index=date_range)
    df.columns = ['power']
    df = df.resample(sample_period).mean()
    
    #Transform into a delta problem
    df['season'] = automatic_seasonality_remover(df['power'].values)
    df['deseason'] = df['power'] - df['season']
               
    output = time_series_anomaly(df.deseason.values, 
                                 look_ahead=16, 
                                 epochs=2, 
                                 layers=2,
                                 hidden_size=16)
    
    df['pred'] = output['pred'] + df['season']
    df['conf'] = output['conf']
    df['error'] = df['power'] - df['pred']
    plot_graph_wrapper(df, title="Prediction")
    plot_timeseries_with_bound(df, 'pred', 'conf', 'power', title="Prediction")
    #Stationarity test
    p_val = adfuller(df.power.values)
    
    # Split timeseries per week
#    weeks = [g for n, g in df.groupby(pd.TimeGrouper('W'))]
#    red_weeks = weeks[1:52]
    
    """
    red_weeks[0:10,13:15,18,20:37,39] is normal
    red_weeks[11:12,16:17,19,38,50] is anomaly
    """
