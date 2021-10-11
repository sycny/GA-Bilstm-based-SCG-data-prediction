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
import webbrowser
from datetime import datetime
from util import write_influx



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


#test
x_test04=scaled104[1419:1732,[1,2,3,4,5,6,7,8,9,10]]
y_test04=scaled204[1419:1732,[1,2,3,4]]#four different labels




x_test07=scaled107[1440:1802,[1,2,3,4,5,6,7,8,9,10]]
y_test07=scaled207[1440:1802,[1,2,3,4]]#four different labels


#test
x_test11=scaled111[2716:3394,[1,2,3,4,5,6,7,8,9,10]]
y_test11=scaled211[2716:3394,[1,2,3,4]]#four different labels



from keras.models import load_model

model= load_model('mb.h5')


test_X04 = x_test04.reshape((x_test04.shape[0], 1, x_test04.shape[1]))
yhat04 = model.predict(test_X04)
inv_yhat04 = concatenate((scaled204[1419:1732,[0]], yhat04), axis=1)
inv_yhat04 = scaler204.inverse_transform(inv_yhat04)
results04=inv_yhat04[:,1]
y_tests04= label04[1419:1732,1]
resultd04=inv_yhat04[:,2]
y_testd04= label04[1419:1732,2]
resulth04=inv_yhat04[:,3]
y_testh04= label04[1419:1732,3]
resultr04=inv_yhat04[:,4]
y_testr04= label04[1419:1732,4]
mae_s04 = mean_absolute_error(results04, y_tests04)
mae_d04 = mean_absolute_error(resultd04, y_testd04)
mae_h04 = mean_absolute_error(resulth04, y_testh04)
mae_r04 = mean_absolute_error(resultr04, y_testr04)
print(mae_s04,mae_d04,mae_h04,mae_r04)

#%%

test_X07 = x_test07.reshape((x_test07.shape[0], 1, x_test07.shape[1]))
yhat07 = model.predict(test_X07)
inv_yhat07 = concatenate((scaled207[1440:1802,[0]], yhat07), axis=1)#维数凑齐
inv_yhat07 = scaler207.inverse_transform(inv_yhat07)#只转换预测结果
results07=inv_yhat07[:,1]
y_tests07= label07[1440:1802,1]
resultd07=inv_yhat07[:,2]
y_testd07= label07[1440:1802,2]
resulth07=inv_yhat07[:,3]
y_testh07= label07[1440:1802,3]
resultr07=inv_yhat07[:,4]
y_testr07= label07[1440:1802,4]
mae_s07 = mean_absolute_error(results07, y_tests07)
mae_d07 = mean_absolute_error(resultd07, y_testd07)
mae_h07 = mean_absolute_error(resulth07, y_testh07)
mae_r07 = mean_absolute_error(resultr07, y_testr07)
print(mae_s07,mae_d07,mae_h07,mae_r07)

#%%

test_X11 = x_test11.reshape((x_test11.shape[0], 1, x_test11.shape[1]))
yhat11 = model.predict(test_X11)
inv_yhat11 = concatenate((scaled211[2716:3394,[0]], yhat11), axis=1)#维数凑齐
inv_yhat11 = scaler211.inverse_transform(inv_yhat11)#只转换预测结果
results11=inv_yhat11[:,1]
y_tests11= label11[2716:3394,1]
resultd11=inv_yhat11[:,2]
y_testd11= label11[2716:3394,2]
resulth11=inv_yhat11[:,3]
y_testh11= label11[2716:3394,3]
resultr11=inv_yhat11[:,4]
y_testr11= label11[2716:3394,4]
mae_s11 = mean_absolute_error(results11, y_tests11)
mae_d11 = mean_absolute_error(resultd11, y_testd11)
mae_h11 = mean_absolute_error(resulth11, y_testh11)
mae_r11 = mean_absolute_error(resultr11, y_testr11)
print(mae_s11,mae_d11,mae_h11,mae_r11)

#%%

print('Test Score1: %.3f' % (100-(mae_s04+mae_d04+mae_r04+mae_h04)))

#%%

print('Test Score2: %.3f' % (100-(mae_s07+mae_d07+mae_r07+mae_h07)))

#%%

print('Test Score3: %.3f' % (100-(mae_s11+mae_d11+mae_r11+mae_h11)))


#write online

verbose = False

if len(sys.argv) >= 6:
    ip = sys.argv[1]
    db = sys.argv[2]
    user = sys.argv[3]
    passw = sys.argv[4]
    unit = sys.argv[5]
    #start = local_time_epoch(sys.argv[6], "America/New_York")
    #end = local_time_epoch(sys.argv[7], "America/New_York")
else:

    sys.exit()

dest = {'ip': ip, 'db': db, 'user': user, 'passw': passw}

start = datetime.now().timestamp()  # uncomment this line to use the current time as the start timestamp of the plot

timestamp = start

# for example, fs =1 and n = 60 means 60 data point and the data interval is 1 second
fs = 1  # 1 Hz
n = 2000  # n is the data length

url = ip + ":3000/d/5JyA06aMk/vital-signs?orgId=1" + unit
url = url + "&from=" + str(int(start * 1))  # + "000"
# url = url + "&to=" + str(int(end*1000)) #+ "000"

print("Click here to see the results in Grafana (user/password:viewer/guest):\n" + url)
#  input("Press any key to continue")
webbrowser.open(url, new=2)

# user your first name as the unit name or location tag if you are in a class and want to avoid overwriting with each other
write_influx(dest, unit, 'labeled', 'S1', y_tests04, timestamp, fs)
write_influx(dest, unit, 'labeled', 'D1', y_testd04, timestamp, fs)
write_influx(dest, unit, 'labeled', 'H1', y_testh04, timestamp, fs)
write_influx(dest, unit, 'labeled', 'R1', y_testr04, timestamp, fs)
write_influx(dest, unit, 'labeled', 'S2', y_tests07, timestamp, fs)
write_influx(dest, unit, 'labeled', 'D2', y_testd07, timestamp, fs)
write_influx(dest, unit, 'labeled', 'H2', y_testh07, timestamp, fs)
write_influx(dest, unit, 'labeled', 'R2', y_testr07, timestamp, fs)
write_influx(dest, unit, 'labeled', 'S3', y_tests11, timestamp, fs)
write_influx(dest, unit, 'labeled', 'D3', y_testd11, timestamp, fs)
write_influx(dest, unit, 'labeled', 'H3', y_testh11, timestamp, fs)
write_influx(dest, unit, 'labeled', 'R3', y_testr11, timestamp, fs)

# user your first name as the unit name or location tag if you are in a class and want to avoid overwriting with each other
write_influx(dest, unit, 'predicted', 'S1', results04, timestamp, fs)
write_influx(dest, unit, 'predicted', 'D1', resultd04, timestamp, fs)
write_influx(dest, unit, 'predicted', 'H1', resulth04, timestamp, fs)
write_influx(dest, unit, 'predicted', 'R1', resultr04, timestamp, fs)
write_influx(dest, unit, 'predicted', 'S2', results07, timestamp, fs)
write_influx(dest, unit, 'predicted', 'D2', resultd07, timestamp, fs)
write_influx(dest, unit, 'predicted', 'H2', resulth07, timestamp, fs)
write_influx(dest, unit, 'predicted', 'R2', resultr07, timestamp, fs)
write_influx(dest, unit, 'predicted', 'S3', results11, timestamp, fs)
write_influx(dest, unit, 'predicted', 'D3', resultd11, timestamp, fs)
write_influx(dest, unit, 'predicted', 'H3', resulth11, timestamp, fs)
write_influx(dest, unit, 'predicted', 'R3', resultr11, timestamp, fs)


# time.sleep(n) # sleep n seconds, which can be removed
end = timestamp + n / fs  # add n seconds

print("start:", start, (datetime.fromtimestamp(start).strftime('%Y-%m-%dT%H:%M:%S.%f')), "end:", end,
      (datetime.fromtimestamp(end).strftime('%Y-%m-%dT%H:%M:%S.%f')))

print("Click here to see the results in Grafana (user/password:viewer/guest):\n" + url)




results04=pd.DataFrame(results04)
resultd04=pd.DataFrame(resultd04)
resultr04=pd.DataFrame(resultr04)
resulth04=pd.DataFrame(resulth04)
results04.to_csv('results04_dataset3.csv',index=False)
resultd04.to_csv('resultd04_dataset3.csv',index=False)
resultr04.to_csv('resultr04_dataset3.csv',index=False)
resulth04.to_csv('resulth04_dataset3.csv',index=False)

#%%

results07=pd.DataFrame(results07)
resultd07=pd.DataFrame(resultd07)
resultr07=pd.DataFrame(resultr07)
resulth07=pd.DataFrame(resulth07)
results07.to_csv('results07_dataset3.csv',index=False)
resultd07.to_csv('resultd07_dataset3.csv',index=False)
resultr07.to_csv('resultr07_dataset3.csv',index=False)
resulth07.to_csv('resulth07_dataset3.csv',index=False)

#%%

results11=pd.DataFrame(results11)
resultd11=pd.DataFrame(resultd11)
resultr11=pd.DataFrame(resultr11)
resulth11=pd.DataFrame(resulth11)
results11.to_csv('results11_dataset3.csv',index=False)
resultd11.to_csv('resultd11_dataset3.csv',index=False)
resultr11.to_csv('resultr11_dataset3.csv',index=False)
resulth11.to_csv('resulth11_dataset3.csv',index=False)
