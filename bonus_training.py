# %%


import sys
import numpy as np  # linear algebra
from scipy.stats import randint
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt  # this is used for the plot the graph
from sklearn.model_selection import train_test_split  # to split the data into two parts
from sklearn.model_selection import KFold  # use for cross validation
from sklearn.preprocessing import StandardScaler  # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline  # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics  # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error, r2_score


df1 = pd.read_csv('/Users/ycs/Desktop/UGA/task/UGA/sensor1104-6e22.csv', usecols=[1, 2])
df2 = pd.read_csv('/Users/ycs/Desktop/UGA/task/UGA/sensor1107-6e22.csv', usecols=[1, 2])
df3 = pd.read_csv('/Users/ycs/Desktop/UGA/task/UGA/sensor1111-1ccf.csv', usecols=[1, 2])
print(df1.shape, df2.shape, df3.shape)


df1_clear = df1.drop(df1[(df1['Value'] - df1['Value'].mean()).abs() > 3 * df1['Value'].std()].index)
df2_clear = df2.drop(df2[(df2['Value'] - df2['Value'].mean()).abs() > 3 * df2['Value'].std()].index)
df3_clear = df3.drop(df3[(df3['Value'] - df3['Value'].mean()).abs() > 3 * df3['Value'].std()].index)
print(df1_clear.shape, df2_clear.shape, df3_clear.shape)


dataindex1 = pd.read_excel('/Users/ycs/Desktop/UGA/task/UGA/label1104-6e22.xlsx')
index1 = dataindex1.loc[
    (dataindex1['TimeStamp (mS)'] > 447339) & (dataindex1['TimeStamp (mS)'] < 2091339), ['TimeStamp (mS)']]
dataindex2 = pd.read_excel('/Users/ycs/Desktop/UGA/task/UGA/label1107-6e22.xlsx')
index2 = dataindex2.loc[
    (dataindex2['TimeStamp (mS)'] > 440388) & (dataindex2['TimeStamp (mS)'] < 2025378), ['TimeStamp (mS)']]
dataindex3 = pd.read_excel('/Users/ycs/Desktop/UGA/task/UGA/label1111-1ccf.xlsx')
index3 = dataindex3.loc[
    (dataindex3['TimeStamp (mS)'] > 2891478) & (dataindex3['TimeStamp (mS)'] < 6447498), ['TimeStamp (mS)']]


print(index1.shape, index2.shape, index3.shape)


# 1104 feature
i = 0
feature1 = np.zeros((10, 1732))
for i in range(1731):
    lower = index1.iloc[i]['TimeStamp (mS)']
    upper = index1.iloc[i + 1]['TimeStamp (mS)']
    lower_requried = df1_clear['timestamp'].map(lambda x: x >= lower)
    upper_requried = df1_clear['timestamp'].map(lambda x: x <= upper)
    required = df1_clear[lower_requried & upper_requried]
    feature1[0, i] = np.var(required.Value)
    feature1[1, i] = np.median(required.Value)
    feature1[2, i] = np.ptp(required.Value)
    feature1[3, i] = np.sum(required.Value)
    feature1[4, i] = np.sum(required.Value * required.Value)
    feature1[5, i] = np.mean(required.Value)
    feature1[6, i] = np.max(required.Value)
    feature1[7, i] = np.min(required.Value)
    feature1[8, i] = np.std(required.Value)
    feature1[9, i] = np.size(required.Value)


# %%featuredata = pd.DataFrame(feature1)
# featuredata.to_csv('featuredata10_1104_clear.csv', index=False)



# %%

# 1107 feature
i = 0
feature2 = np.zeros((10, 1803))
for i in range(len(index2) - 1):
    lower = index2.iloc[i]['TimeStamp (mS)']
    upper = index2.iloc[i + 1]['TimeStamp (mS)']
    lower_requried = df2_clear['timestamp'].map(lambda x: x >= lower)
    upper_requried = df2_clear['timestamp'].map(lambda x: x <= upper)
    required = df2_clear[lower_requried & upper_requried]
    feature2[0, i] = np.var(required.Value)
    feature2[1, i] = np.mean(required.Value)
    feature2[2, i] = np.ptp(required.Value)
    feature2[3, i] = np.sum(required.Value)
    feature2[4, i] = np.sum(required.Value * required.Value)
    feature2[5, i] = np.median(required.Value)
    feature2[6, i] = np.max(required.Value)
    feature2[7, i] = np.min(required.Value)
    feature2[8, i] = np.std(required.Value)
    feature2[9, i] = np.size(required.Value)


#featuredata = pd.DataFrame(feature2)
#featuredata.to_csv('featuredata10_1107_clear.csv', index=False)


# 1107 feature
i = 0
feature3 = np.zeros((10, 3395))
for i in range(len(index3) - 1):
    lower = index3.iloc[i]['TimeStamp (mS)']
    upper = index3.iloc[i + 1]['TimeStamp (mS)']
    lower_requried = df3_clear['timestamp'].map(lambda x: x >= lower)
    upper_requried = df3_clear['timestamp'].map(lambda x: x <= upper)
    required = df3_clear[lower_requried & upper_requried]
    feature3[0, i] = np.var(required.Value)
    feature3[1, i] = np.mean(required.Value)
    feature3[2, i] = np.ptp(required.Value)
    feature3[3, i] = np.sum(required.Value)
    feature3[4, i] = np.sum(required.Value * required.Value)
    feature3[5, i] = np.median(required.Value)
    feature3[6, i] = np.max(required.Value)
    feature3[7, i] = np.min(required.Value)
    feature3[8, i] = np.std(required.Value)
    feature3[9, i] = np.size(required.Value)



# %%

#featuredata = pd.DataFrame(feature3)
#featuredata.to_csv('featuredata10_1111_clear.csv', index=False)

# %%

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
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#%%

dataindex04=pd.read_excel('/Users/ycs/Desktop/UGA/task/UGA/label1104-6e22.xlsx')
Label04=dataindex04.loc[ (dataindex04['TimeStamp (mS)']>447339) & (dataindex04['TimeStamp (mS)']<2091339) , ['TimeStamp (mS)','S','D','H','R']]
Feature04=pd.read_csv('/Users/ycs/Desktop/UGA/task/UGA/featuredata10_1104_clear.csv',header=None)

dataindex07=pd.read_excel('/Users/ycs/Desktop/UGA/task/UGA/label1107-6e22.xlsx')
Label07=dataindex07.loc[ (dataindex07['TimeStamp (mS)']>440388) & (dataindex07['TimeStamp (mS)']<2025378) , ['TimeStamp (mS)','S','D','H','R']]
Feature07=pd.read_csv('/Users/ycs/Desktop/UGA/task/UGA/featuredata10_1107_clear.csv',header=None)

dataindex11=pd.read_excel('/Users/ycs/Desktop/UGA/task/UGA/label1111-1ccf.xlsx')
Label11=dataindex11.loc[ (dataindex11['TimeStamp (mS)']>2891478) & (dataindex11['TimeStamp (mS)']<6447498) , ['TimeStamp (mS)','S','D','H','R']]
Feature11=pd.read_csv('/Users/ycs/Desktop/UGA/task/UGA/featuredata10_1111_clear.csv',header=None)



#%%


feature04 = Feature04.values
label04 = Label04.values
#Normalized data
scaler104 = MinMaxScaler(feature_range=(0, 1))
scaled104 = scaler104.fit_transform(feature04)
scaler204 = MinMaxScaler(feature_range=(0, 1))
scaled204 = scaler204.fit_transform(label04)
#scaled2

feature07 = Feature07.values
label07 = Label07.values
#Normalized data
scaler107 = MinMaxScaler(feature_range=(0, 1))
scaled107 = scaler107.fit_transform(feature07)
scaler207 = MinMaxScaler(feature_range=(0, 1))
scaled207 = scaler207.fit_transform(label07)

feature11 = Feature11.values
label11 = Label11.values
#Normalized data
scaler111 = MinMaxScaler(feature_range=(0, 1))
scaled111 = scaler111.fit_transform(feature11)
scaler211 = MinMaxScaler(feature_range=(0, 1))
scaled211 = scaler211.fit_transform(label11)

#%%

#train
x_train04=scaled104[0:1218,[1,2,3,4,5,6,7,8,9,10]]
y_train04=scaled204[0:1218,[1,2,3,4]]#four different labels

#validation
x_val04=scaled104[1218:1418,[1,2,3,4,5,6,7,8,9,10]]
y_val04=scaled204[1218:1418,[1,2,3,4]]#four different labels

#test
x_test04=scaled104[1419:1732,[1,2,3,4,5,6,7,8,9,10]]
y_test04=scaled204[1419:1732,[1,2,3,4]]#four different labels



x_train07=scaled107[0:1199,[1,2,3,4,5,6,7,8,9,10]]
y_train07=scaled207[0:1199,[1,2,3,4]]#four different labels

#validation
x_val07=scaled107[1200:1439,[1,2,3,4,5,6,7,8,9,10]]
y_val07=scaled207[1200:1439,[1,2,3,4]]#four different labels

#test
x_test07=scaled107[1440:1802,[1,2,3,4,5,6,7,8,9,10]]
y_test07=scaled207[1440:1802,[1,2,3,4]]#four different labels




x_train11=scaled111[0:2399,[1,2,3,4,5,6,7,8,9,10]]
y_train11=scaled211[0:2399,[1,2,3,4]]#four different labels

#validation
x_val11=scaled111[2400:2715,[1,2,3,4,5,6,7,8,9,10]]
y_val11=scaled211[2400:2715,[1,2,3,4]]#four different labels

#test
x_test11=scaled111[2716:3394,[1,2,3,4,5,6,7,8,9,10]]
y_test11=scaled211[2716:3394,[1,2,3,4]]#four different labels

print(x_train04.shape,y_train04.shape,x_val04.shape,y_val04.shape,x_test04.shape,y_test04.shape,
      x_train07.shape,y_train07.shape,x_val07.shape,y_val07.shape,x_test07.shape,y_test07.shape,
    x_train11.shape,y_train11.shape,x_val11.shape,y_val11.shape,x_test11.shape,y_test11.shape)

#%%

#the training data and val data need to be combined
x_train=np.vstack((x_train04,x_train07,x_train11))
x_val=np.vstack((x_val04,x_val07,x_val11))

y_train=np.vstack((y_train04,y_train07,y_train11))


y_val=np.vstack((y_val04,y_val07,y_val11))

print(x_train.shape,y_train.shape,x_val.shape,y_val.shape)

#%%

train_X = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
val_X = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
#test_X = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(train_X.shape, val_X.shape)

#%%

# model for 4 labels
model = Sequential()
model.add(Bidirectional(LSTM(9, activation='relu',return_sequences=False)))
model.add(Dense(4))
model.compile(loss='mae', optimizer='adam')
# 拟合神经网络模型
history = model.fit(train_X, y_train, epochs=35, batch_size=4, validation_data=(val_X, y_val), verbose=2, shuffle=False)
# 绘制历史数据
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

model.save('mb.h5')
