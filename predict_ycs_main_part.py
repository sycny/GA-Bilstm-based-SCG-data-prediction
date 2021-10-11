#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
import numpy as np
from numpy import concatenate
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import webbrowser
from datetime import datetime
from util import write_influx


# read input and output

dataindex=pd.read_excel('/Users/ycs/Desktop/Task/data/label-27minutes.xlsx')
Label=dataindex.loc[ (dataindex['TimeStamp (mS)']>495369) & (dataindex['TimeStamp (mS)']< 2114095) , ['TimeStamp (mS)','S','D','H','R']]
Feature=pd.read_csv('/Users/ycs/Desktop/Task/data/featuredata10_clear.csv',header=None)




feature = Feature.values
label = Label.values
# data normalization
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaled1 = scaler1.fit_transform(feature)
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaled2 = scaler2.fit_transform(label)


# test
x_test = scaled1[1215:1621, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
ys_test = scaled2[1215:1621, [1]]  # four different labels
yd_test = scaled2[1215:1621, [2]]
yh_test = scaled2[1215:1621, [3]]
yr_test = scaled2[1215:1621, [4]]


# reshape for the BiLSTM
test_X = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(test_X.shape)


# predict
from keras.models import load_model

model= load_model('m.h5')

yhat4label = model.predict(test_X)
yhats = yhat4label[:,[0]]
yhatd = yhat4label[:,[1]]
yhath = yhat4label[:,[2]]
yhatr = yhat4label[:,[3]]

inv_yhats = concatenate((scaled2[1215:1621, [0]], yhats, scaled2[1215:1621, [2, 3, 4]]), axis=1)  # 维数凑齐
inv_yhats = scaler2.inverse_transform(inv_yhats)
results = inv_yhats[:,1]
y_tests = label[1215:1621,1]
mae_s = mean_absolute_error(results, y_tests)
print('Test MAE_S: %.3f' % mae_s)

inv_yhatd = concatenate((scaled2[1215:1621, [0, 1]], yhatd, scaled2[1215:1621, [3, 4]]), axis=1)
inv_yhatd = scaler2.inverse_transform(inv_yhatd)  # denormalization
resultd = inv_yhatd[:,2]
y_testd = label[1215:1621,2]
mae_d = mean_absolute_error(resultd, y_testd)
print('Test MAE_D: %.3f' % mae_d)

inv_yhath = concatenate((scaled2[1215:1621, [0, 1, 2]], yhath, scaled2[1215:1621, [4]]), axis=1)
inv_yhath = scaler2.inverse_transform(inv_yhath)
resulth = inv_yhath[:,3]
y_testh = label[1215:1621,3]
mae_h = mean_absolute_error(resulth, y_testh)
print('Test MAE_H: %.3f' % mae_h)


inv_yhatr = concatenate((scaled2[1215:1621, [0, 1, 2, 3]], yhatr), axis=1)
inv_yhatr = scaler2.inverse_transform(inv_yhatr)  # denormalization
resultr = inv_yhatr[:,4]
y_testr = label[1215:1621,4]
mae_r = mean_absolute_error(resultr, y_testr)
print('Test MAE_R: %.3f' % mae_r)

print('Test Score: %.3f' % (100 - (mae_s + mae_d + mae_r + mae_h)))

resultspd = pd.DataFrame(results)
resultdpd = pd.DataFrame(resultd)
resultrpd = pd.DataFrame(resultr)
resulthpd = pd.DataFrame(resulth)
resultspd.to_csv('results.csv', index=False)
resultdpd.to_csv('resultd.csv', index=False)
resultrpd.to_csv('resultr.csv', index=False)
resulthpd.to_csv('resulth.csv', index=False)


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
    print("Example: " + sys.argv[
        0] + " https://sensorweb.us testdb test sensorweb yu:ch:en:gs:hi:mp")  # 2020-08-13T02:03:00.200")
    print("yu:ch:en:gs:hi:mp")
    sys.exit()

dest = {'ip': ip, 'db': db, 'user': user, 'passw': passw}

start = datetime.now().timestamp()  # uncomment this line to use the current time as the start timestamp of the plot

timestamp = start

# for example, fs =1 and n = 60 means 60 data point and the data interval is 1 second
fs = 1  # 1 Hz
n = 406 # n is the data length

url = ip + ":3000/d/Vv7164WMz/vital-signs?orgId=1&refresh=1s&var-unit=" + unit
url = url + "&from=" + str(int(start * 1))  # + "000"
# url = url + "&to=" + str(int(end*1000)) #+ "000"

print("Click here to see the results in Grafana (user/password:viewer/guest):\n" + url)
#  input("Press any key to continue")
webbrowser.open(url, new=2)


# user your first name as the unit name or location tag if you are in a class and want to avoid overwriting with each other
write_influx(dest, unit, 'labeled', 'S', y_tests, timestamp, fs)
write_influx(dest, unit, 'labeled', 'D', y_testd, timestamp, fs)
write_influx(dest, unit, 'labeled', 'H', y_testh, timestamp, fs)
write_influx(dest, unit, 'labeled', 'R', y_testr, timestamp, fs)



# user your first name as the unit name or location tag if you are in a class and want to avoid overwriting with each other
write_influx(dest, unit, 'predicted', 'S', results, timestamp, fs)
write_influx(dest, unit, 'predicted', 'D', resultd, timestamp, fs)
write_influx(dest, unit, 'predicted', 'H', resulth, timestamp, fs)
write_influx(dest, unit, 'predicted', 'R', resultr, timestamp, fs)


# time.sleep(n) # sleep n seconds, which can be removed
end = timestamp + n / fs  # add n seconds

print("start:", start, (datetime.fromtimestamp(start).strftime('%Y-%m-%dT%H:%M:%S.%f')), "end:", end,
      (datetime.fromtimestamp(end).strftime('%Y-%m-%dT%H:%M:%S.%f')))

print("Click here to see the results in Grafana (user/password:viewer/guest):\n" + url)

