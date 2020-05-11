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
import numpy as np
import random as rd
import tensorflow as tf
import pandas as pd
import io
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras import optimizers


df = pd.read_csv('Dataset_11Maio.csv', delimiter = ',',
                 encoding = 'ISO-8859-1')

# Ordenar por mês, dia e hora.
df.sort_values(['Month (number)', 'Day of month', 'Hour'],
               ascending = [True, True, True], inplace = True)

# Separação por ruas.
df_1 = df[df['road_num'] == 2]
df_2 = df[df['road_num'] == 2]
df_3 = df[df['road_num'] == 3]
df_4 = df[df['road_num'] == 4]

df_1.drop('road_num', axis = 1, inplace = True)
'''
df_1.drop('Month (number)', axis = 1, inplace = True)
df_1.drop('Day of month', axis = 1, inplace = True)
df_1.drop('Hour', axis = 1, inplace = True)
df_1.drop('Day of week (name)', axis = 1, inplace = True)
df_1.drop('Distance', axis = 1, inplace = True)
df_1.drop('incident_category_desc', axis = 1, inplace = True)

'''


# Vamos tratar da rua 1.

#dataset = df_1_train.dropna(subset=["speed_diff"])

#dataset=dataset.reset_index(drop=True)

df_1=df_1.reset_index(drop=True)



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
n_past = 24*3 # Past 30 days

label = df_1['speed_diff']


for i in range(0,len(df_1)-n_past-n_future+1):
  dias = df_1.iloc[i : i + n_past+24]
  mes = dias.iloc[0]['Month (number)']
  dia_1 = dias.iloc[0]['Day of month']
  dia_168 = dias.iloc[24*3+1]['Day of month']
  if i == 1020 or i==1900 :
    print('Mes',dias.iloc[24*3+1]['Month (number)'])
    print('DIA',dias.iloc[24*3+1]['Day of month'])
  if (mes == 4 or mes == 6 or mes == 9 or mes == 11) and (dia_168 - dia_1 == 3 or dia_168 - dia_1 == -29):
    a=df_1.iloc[i : i + n_past]
    a.drop('Month (number)', axis = 1, inplace = True)
    a.drop('Day of month', axis = 1, inplace = True)
    a.drop('Hour', axis = 1, inplace = True)
    a.drop('Day of week (name)', axis = 1, inplace = True)
    a.drop('Distance', axis = 1, inplace = True)
    a.drop('incident_category_desc', axis = 1, inplace = True)

    x_train.append(a)
    y_train.append(label.iloc[i + n_past : i + n_past + n_future ])
  elif (mes == 1 or mes == 3 or mes == 5 or mes == 7 or mes == 8 or mes == 10 or mes == 12) and (dia_168 - dia_1 == 3 or dia_168 - dia_1 == -28):
    a=df_1.iloc[i : i + n_past]
    a.drop('Month (number)', axis = 1, inplace = True)
    a.drop('Day of month', axis = 1, inplace = True)
    a.drop('Hour', axis = 1, inplace = True)
    a.drop('Day of week (name)', axis = 1, inplace = True)
    a.drop('Distance', axis = 1, inplace = True)
    a.drop('incident_category_desc', axis = 1, inplace = True)
    x_train.append(a)
    y_train.append(label.iloc[i + n_past : i + n_past + n_future ])
  elif mes == 2 and (dia_168 - dia_1 == 3 or dia_168 - dia_1 == -26):
    a=df_1.iloc[i : i + n_past]
    a.drop('Month (number)', axis = 1, inplace = True)
    a.drop('Day of month', axis = 1, inplace = True)
    a.drop('Hour', axis = 1, inplace = True)
    a.drop('Day of week (name)', axis = 1, inplace = True)
    a.drop('Distance', axis = 1, inplace = True)
    a.drop('incident_category_desc', axis = 1, inplace = True)
    x_train.append(a)
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
DADOS_TREINO=[]
DADOS_TESTE=[]
LABELS_TREINO=[]
LABELS_TESTE=[]
for i in range(len(x_train)):
  if i >len(x_train)-8:
    DADOS_TESTE.append(x_train[i])
    LABELS_TESTE.append(y_train[i])
  else:
    DADOS_TREINO.append(x_train[i])
    LABELS_TREINO.append(y_train[i])

x_train=DADOS_TREINO
y_train=LABELS_TREINO
x_test=DADOS_TESTE
y_test=LABELS_TESTE

x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)

print(x_train.shape)

scalers=[]
for i in range(11):

  sc = MinMaxScaler(feature_range=(0,1))
  x_train[:,i] = sc.fit_transform(x_train[:,i])
  x_test[:,i] = sc.fit_transform(x_test[:,i])
  scalers.append(sc)

sc1 = MinMaxScaler(feature_range=(0,1))
y_train = sc1.fit_transform(y_train)

def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

regressor = Sequential()
regressor.add(CuDNNLSTM(units=24*3, return_sequences=True, input_shape = (24*3,11) ) )
regressor.add(Dropout(0.2))
regressor.add(CuDNNLSTM(24*3 , return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(CuDNNLSTM(24*3, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(CuDNNLSTM(24*2))
regressor.add(Dropout(0.2))
regressor.add(Dense(24,activation='sigmoid'))
regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])
regressor.fit(x_train, y_train, epochs=1000 )







print('############################')


predicted_temperature = regressor.predict(x_test)
previstos=[]
for i in range(5):
  k=predicted_temperature[i].reshape(1,24)
  a = sc1.inverse_transform(k)
  a = np.reshape(a,(a.shape[1],a.shape[0]))
  previstos.append(a)

for i in range(5):
  print('PREVISÃO DO DIA %d' %i)
  for j in range(24):

      print('Valor real na hora: ',j ,y_test[i][j],'Valor previsto: ',previstos[i][j][0])
  print('-------------------------------------------------')
  print('-------------------------------------------------')
  print('-------------------------------------------------')
