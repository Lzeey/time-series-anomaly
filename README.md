# Time Series Anomaly Detection with Deep Learning
Detecting Anomaly

## Intro
The goal of this project is to design an assumption-free anomaly detection tool that can take in any time series data (one-dimensional for now) and flag out datapoints that are significantly higher/lower than usual.

Problem: Given a time series $X = [x_0, x_1, … , x_n] \in \R^n$, return a series of time indices $i$ which are anomalous.

## Description

### LSTMs
Currently, there are two predictors in this repo. The first is a stateful LSTM (see `lstm_main.py`). The LSTM works fine for time-series with shorter seasonality, but will have issues with long-term seasonalities.

The stateful LSTM takes one input at the time and predict k time-steps ahead. The state is retained for the next input (hence stateful).

Currently, the model works quite well on the power dataset, with 1 hour sampling (week seasonality of $7*24$ time-steps)

![LSTM anomalies](/figures/lstm_1hr.png)

but falls short when the seasonality gets too long (e.g. 30 Mins sampling with $7*2*24$)

![LSTM anomlies 30mins](/figures/lstm_30mins.png)

### 1D Causal Convolution
We use a wave-net like structure to capture patterns from very long timeseries.

This is extremely fast (per epoch time of ~10s with GPU), but output can be easily thrown off with large anomalies. May need to rethink training strategy (cross-validation of some kind). Also, the anomalies does not feature as strongly as LSTM for now.

![Wavenet Anomalies](/figures/wavenet_1hr_more_epochs.png)

Check out the .html plotly plots to see full results.
## Other similar detectors
See STL, or for a more sophisticated [Seaonsal Hybrid ESD](https://github.com/twitter/AnomalyDetection) (Twitter's).

While there has been attempts to use other deep learning techniques, e.g. autoencoder.

## Examples
To be completed. In the meanwhile, refer to `atrous_conv_exp_03.ipynb`.

## TODO:
1. Differentiate between point anomalies and structural anomalies (e.g. one whole day missing vs sudden spike). Possible idea is to use a CUMSUM algorithm for the structural ones (since prediction error fits normal distribution well), then use a higher threshold for point anomalies.

## Dependencies
TODO

## Known errors with jupyter notebook
If you encounter a long error message for the plotly diagrams, upgrade nbformat.
`pip install --upgrade nbformat`.  

**If you are still getting an error**
Try updating jupyter notebook. The current 
For newer version of jupyter, you might get an error.

``IOPub data rate exceeded.
The notebook server will temporarily stop sending output
to the client in order to avoid crashing it.
To change this limit, set the config variable
`--NotebookApp.iopub_data_rate_limit`.``

To fix this, run with 
``jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000.``

## History
1. The conventional STL techniques work brilliant on this dataset.
2. LSTMs have issues capturing long term dependencies