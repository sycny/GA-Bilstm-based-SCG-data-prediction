#!/usr/bin/env python
# coding: utf-8



import sys 
import keras
import numpy as np # linear algebra
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

# feature extraction
# read data
df = pd.read_csv('/Users/ycs/Desktop/Task/data/sensor-27minutes.csv', usecols=[2, 1])
df.describe()

# %%

# take out outliers
df_clear = df.drop(df[(df['Value'] - df['Value'].mean()).abs() > 3 * df['Value'].std()].index)
print(df_clear.shape)



# These two boundaries are picked up manually from the timestamp on the excel,where the sensor data begin and end.
dataindex = pd.read_excel('/Users/ycs/Desktop/Task/data/label-27minutes.xlsx')
index = dataindex.loc[
    (dataindex['TimeStamp (mS)'] > 495369) & (dataindex['TimeStamp (mS)'] < 2114095), ['TimeStamp (mS)']]


# feature calculation
i = 0
feature = np.zeros((10, 1622))
for i in range(len(index) - 1):
    lower = index.iloc[i]['TimeStamp (mS)']
    upper = index.iloc[i + 1]['TimeStamp (mS)']
    lower_requried = df_clear['time'].map(lambda x: x >= lower)
    upper_requried = df_clear['time'].map(lambda x: x <= upper)
    required = df_clear[lower_requried & upper_requried]
    feature[0, i] = np.var(required.Value)
    feature[1, i] = np.mean(required.Value)
    feature[2, i] = np.ptp(required.Value)
    feature[3, i] = np.sum(required.Value)
    feature[4, i] = np.sum(required.Value * required.Value)
    feature[5, i] = np.median(required.Value)
    feature[6, i] = np.max(required.Value)
    feature[7, i] = np.min(required.Value)
    feature[8, i] = np.std(required.Value)
    feature[9, i] = np.size(required.Value)

# %%

# save feature data
featuredata = pd.DataFrame(feature)
#featuredata.to_csv('featuredata10_clear.csv', index=False)'''







#read input and output
dataindex=pd.read_excel('/Users/ycs/Desktop/Task/data/label-27minutes.xlsx')
Label=dataindex.loc[ (dataindex['TimeStamp (mS)']>495369) & (dataindex['TimeStamp (mS)']< 2114095) , ['TimeStamp (mS)','S','D','H','R']]
Feature=pd.read_csv('/Users/ycs/Desktop/Task/data/featuredata10_clear.csv',header=None)



feature = Feature.values
label = Label.values
#data normalization
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaled1 = scaler1.fit_transform(feature)
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaled2 = scaler2.fit_transform(label)


#train
x_train=scaled1[0:983,[1,2,3,4,5,6,7,8,9,10]]
y_train=scaled2[0:983,[1,2,3,4]]#four different labels

#validation
x_val=scaled1[984:1214,[1,2,3,4,5,6,7,8,9,10]]
y_val=scaled2[984:1214,[1,2,3,4]]#four different labels

#test
x_test=scaled1[1215:1621,[1,2,3,4,5,6,7,8,9,10]]
y_test=scaled2[1215:1621,[1,2,3,4]]#four different labels

print(x_train.shape,y_train.shape,x_val.shape,y_val.shape,x_test.shape,y_test.shape)


#reshape for the BiLSTM
train_X = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
val_X = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
test_X = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(train_X.shape, val_X.shape, test_X.shape)



# model for 4 labels
model = Sequential()
model.add(Bidirectional(LSTM(9, activation='relu',return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(4))
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_X, y_train, epochs=35, batch_size=4, validation_data=(val_X, y_val), verbose=2, shuffle=False)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

model.save('m.h5')



