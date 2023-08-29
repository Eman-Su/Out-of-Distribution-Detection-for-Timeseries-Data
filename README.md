# Comparison of Deep Learning Techniques for Stock Price Prediction.

AI702 Deep Learning Course Project submitted by Eman Al Suradi, Kevin Toner and Shahina Kunhimon.

## Overview

This project is a comparative study of different deep learning techniques and statistical ARIMA model for predicting
the future stock prices of the Alphabet company.  Univariate probabilistic model ARIMA is used as the baseline. The deep learning
models used are:

1. Long Short-Term Memory (LSTM) [1]
2. Gated Recurrent Unit (GRU) [2]
3. N-BEATS [3]
4. Temporal Fusion Transformer (TFT) [4]


## Contents
### Dataset folder
Contains four datasets and the notebook used to generate them: <br />
|-> _Data_Preprocessing.ipynb_ <br />
|-> _normal_test_data.csv_ <br />
|-> _normal_train_data.csv_ <br />
|-> _small_dataset.csv_ <br />
|-> _big_dataset.csv_ 


### Models folder
Contains a notebook/python file for each of the models: <br />
|-> _ARIMA.ipynb_ <br />
|-> _NBEATS.py_ <br />
|-> _LSTM.ipynb_ <br />
|-> _GRU.ipynb_ <br />
|-> _TFT.ipynb_<br />
**|-> Demos folder** <br />
&emsp;Contains a demo instructions file, and demo notebooks for some of the models: <br />
&emsp;|-> ARIMA_Demo.ipynb <br />
&emsp;|-> Demo Instructions.md <br />
&emsp;|-> LSTM_and_GRU_Demo.ipynb <br />
&emsp;|-> NBEATS_Demo.ipynb <br />


## References
1. Hochreiter, S. and Schmidhuber, J., 1997. Long short-term memory. Neural computation, 9(8), pp.1735-1780.http://www.bioinf.jku.at/publications/older/2604.pdf
2. Cho, K., Van Merriënboer, B., Bahdanau, D. and Bengio, Y., 2014. On the properties of neural machine translation: Encoder-decoder approaches https://arxiv.org/abs/1409.1259
3.  Oreshkin, B.N., Carpov, D., Chapados, N. and Bengio, Y., 2019. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.
https://arxiv.org/abs/1905.10437
4. Lim, B., Arik, S.O., Loeff, N. and Pfister, T., 2019. Temporal fusion transformers for interpretable multi-horizon time series forecasting. https://arxiv.org/abs/1912.09363
