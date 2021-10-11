import sys
import keras
import numpy as np
from scipy.stats import randint
import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from keras.layers import Bidirectional
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def load():

    print('Load finished!')
    return x_train, y_train, x_val, y_val


def classify(x_train, y_train, x_val, y_val, num):

    #
    lstm_num_units = num[0]
    lstm_epochs = num[1]
    lstm_dropout = num[2]/10
    lstm_batch_size = num[3]


    model = Sequential()
    model.add(Bidirectional(LSTM(lstm_num_units, activation='relu', return_sequences=False)))
    model.add(Dropout(lstm_dropout))
    model.add(Dense(4))
    model.compile(loss='mae', optimizer='adam')

    history = model.fit(x_train, y_train, epochs=lstm_epochs, batch_size= lstm_batch_size, verbose=2,  shuffle=False)


    print('LSTM finished!')
    y_pred = model.predict(x_val)

    mae = mean_absolute_error(y_val, y_pred)
    print('Test mae: ', mae)

    return 1 / mae