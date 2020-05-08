
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer, Dense, Dropout, LSTM
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.compat.v1.keras.layers import CuDNNLSTM



df = pd.read_csv('Dataset_7MAIO.csv', delimiter = ',',
                 encoding = 'ISO-8859-1')

# Ordenar por mês, dia e hora.
df.sort_values(['Month (number)', 'Day of month', 'Hour'],
               ascending = [True, True, True], inplace = True)

# Separação por ruas.
df_1 = df[df['road_num'] == 1]
df_2 = df[df['road_num'] == 2]
df_3 = df[df['road_num'] == 3]
df_4 = df[df['road_num'] == 4]

SPLIT_INDEX = len(df_1) - 8*24
df_1.drop('road_num', axis = 1, inplace = True)




# Vamos tratar da rua 1.

#dataset = df_1_train.dropna(subset=["speed_diff"])

#dataset=dataset.reset_index(drop=True)

df_1=df_1.reset_index(drop=True)

df_1_train = df_1[:SPLIT_INDEX]
df_1_test  = df_1[SPLIT_INDEX:]

'''training_set = df_1.iloc[:,4:5].values   # Só contem valores do speed_diff
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)'''

'''
We will create a training set such that for every 7 days (7*24 hours) we will provide the next 24 hours
speed_diff as output. In other words, input for our RNN would be 7 days temperature data and
 the output would be 1 day forecast of speed_diff
'''

x_train = []
y_train = []


n_future = 24 # next 4 days temperature forecast
n_past = 24*7 # Past 30 days

label = df_1_train['speed_diff']


for i in range(0,len(df_1_train)-n_past-n_future+1):
  dias = df_1_train.iloc[i : i + n_past+24]
  mes = dias.iloc[0]['Month (number)']
  dia_1 = dias.iloc[0]['Day of month']
  dia_168 = dias.iloc[168]['Day of month']
  if (mes == 4 or mes == 6 or mes == 9 or mes == 11) and (dia_168 - dia_1 == 7 or dia_168 - dia_1 == -25):
    x_train.append(df_1_train.iloc[i : i + n_past])
    y_train.append(label.iloc[i + n_past : i + n_past + n_future ])
  elif (mes == 1 or mes == 3 or mes == 5 or mes == 7 or mes == 8 or mes == 10 or mes == 12) and (dia_168 - dia_1 == 7 or dia_168 - dia_1 == -24):
    x_train.append(df_1_train.iloc[i : i + n_past])
    y_train.append(label.iloc[i + n_past : i + n_past + n_future ])
  elif mes == 2 and (dia_168 - dia_1 == 7 or dia_168 - dia_1 == -22):
    x_train.append(df_1_train.iloc[i : i + n_past])
    y_train.append(label.iloc[i + n_past : i + n_past + n_future ])
'''x_train1=[]
y_train1=[]
for i in range(len(x_train)-1):
  x_train1.append(x_train[i]+x_train[i+1])
  y_train1.append(y_train[i]+y_train[i+1])
print(x_train1[0].size)
print(len(x_train1))
x_train=x_train1
y_train=y_train1'''


for i in range(len(x_train)):
    x_train[i]=np.array(x_train[i])
for i in range(len(y_train)):
    y_train[i]=np.array(y_train[i])
for i in range(len(x_train)):
    x_train[i]=np.array(x_train[i])
for i in range(len(y_train)):
    y_train[i]=np.array(y_train[i])
for i in range(len(x_train)):
    x_train[i] = np.reshape(x_train[i], (x_train[0].shape[0],x_train[0].shape[1]) )

for i in range(len(y_train)):
    y_train[i] = np.reshape(y_train[i], (y_train[0].shape[0]))

x_train=np.array(x_train)
y_train=np.array(y_train)


scalers=[]
for i in range(17):

  sc = MinMaxScaler(feature_range=(0,1))
  x_train[:,i] = sc.fit_transform(x_train[:,i])
  scalers.append(sc)

print('x_train',len(x_train),len(x_train[0]))
sc1 = MinMaxScaler(feature_range=(0,1))
y_train = sc1.fit_transform(y_train)


regressor = Sequential()
regressor.add(CuDNNLSTM(units=24*7, return_sequences=True, input_shape = (168,17) ) )
regressor.add(Dropout(0.2))
regressor.add(CuDNNLSTM(24*7 , return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(CuDNNLSTM(24*7, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(CuDNNLSTM(24*7))
regressor.add(Dropout(0.2))
regressor.add(Dense(24,activation='sigmoid'))
regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])
regressor.fit(x_train, y_train, epochs=10 )


##############################
# read test dataset
testdataset = df_1_test
#get only the temperature column
testdataset = testdataset.iloc[:24*7]



x_test = []

label1 = testdataset['speed_diff']
x_test.append(testdataset.iloc[0 : n_past])
for i in range(len(x_test)):
    x_test[i]=np.array(x_test[i])
for i in range(len(x_test)):

    x_test[i] = np.reshape(x_test[i], (x_test[0].shape[0] , x_test[0].shape[1]) )

print(x_test)

x_test=np.array(x_test)


real_temperature = df_1_test
real_temperature = real_temperature.iloc[24*7:]
print(real_temperature)
y_real = []

label2 = real_temperature['speed_diff']

y_real.append(label2.iloc[:n_future ])

for i in range(len(y_real)):
    y_real[i]=np.array(y_real[i])

for i in range(len(y_real)):
    y_real[i] = np.reshape(y_real[i], (y_real[0].shape[0]))


for i in range(17):
  x_test[:,i] = scalers[i].fit_transform(x_test[:,i])



print('############################')

#testing = sc.transform(testdataset)
#testing = np.reshape(x_test,(testing.shape[1],testing.shape[0]))



predicted_temperature = regressor.predict(x_test)
predicted_temperature = sc1.inverse_transform(predicted_temperature)
predicted_temperature = np.reshape(predicted_temperature,(predicted_temperature.shape[1],predicted_temperature.shape[0]))


for i in range(len(predicted_temperature)):
  print('Valor real na hora: ',i ,y_real[0][i],'Valor previsto: ',predicted_temperature[i][0])
