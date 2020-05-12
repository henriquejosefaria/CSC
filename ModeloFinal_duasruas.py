import numpy as np
import random
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
df_1 = df[df['road_num'] == 1]
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
nome_dia=[]


n_future = 24 # next 4 days temperature forecast
n_past = 24*3 # Past 30 days

label = df_1['speed_diff']


for i in range(0,len(df_1)-n_past-n_future+1):
  dias = df_1.iloc[i : i + n_past+24]
  mes = dias.iloc[0]['Month (number)']
  dia_1 = dias.iloc[0]['Day of month']
  dia_168 = dias.iloc[24*3+1]['Day of month']
  d=dias.iloc[24*3+1]['Day of week (name)']
  if (mes == 4 or mes == 6 or mes == 9 or mes == 11) and (dia_168 - dia_1 == 3 or dia_168 - dia_1 == -29):
    a=df_1.iloc[i : i + n_past]
    nome_dia.append(d)
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
    nome_dia.append(d)
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
    nome_dia.append(d)
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
l=[]


indices=[5172, 2728, 3728, 2069, 3439, 2078, 3032, 4545, 2117, 2638, 2731, 2819, 4882, 3695, 1773, 2335, 3090, 3839, 2684, 2864, 3242, 2479, 3803, 1910, 2832, 4956, 3014, 3314, 5075, 4222, 3645, 4180, 4430, 3457, 1809, 4303, 4311, 3484, 2627, 4845, 3509, 3375, 4650, 4480, 2366, 2594, 3748, 2125, 2347, 1844, 5244, 3153, 3515, 4939, 2086, 2666, 3579, 4169, 4257, 1706, 3787, 2643, 1858, 3110, 2434, 1185, 4714, 4446, 1153, 4607, 3220, 4941, 1394, 3745, 3755, 3874, 4877, 1061, 1534, 1229, 4662, 2496, 3741, 4902, 3178, 2604, 2081, 4237, 2143, 2780, 1044, 4689, 2618, 2852, 3356, 4042, 1692, 1002, 2511, 3347]

'''
indices=[]
i=0
while i<100:
  r=random.randint(1000,len(x_train)-1)
  if r not in indices:
    indices.append(r)
    i+=1
'''
for i in range(len(x_train)):
  if i in indices:
    DADOS_TESTE.append(x_train[i])
    LABELS_TESTE.append(y_train[i])
    l.append(nome_dia[i])
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


#----------------------
df_1 = df[df['road_num'] == 2]
df_1.drop('road_num', axis = 1, inplace = True)
df_1=df_1.reset_index(drop=True)



'''training_set = df_1.iloc[:,4:5].values   # Só contem valores do speed_diff
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)'''

'''
We will create a training set such that for every 7 days (7*24 hours) we will provide the next 24 hours
speed_diff as output. In other words, input for our RNN would be 7 days temperature data and
 the output would be 1 day forecast of speed_diff
'''

x_train1=[]
y_train1=[]
nome_dia1=[]

n_future = 24 # next 4 days temperature forecast
n_past = 24*3 # Past 30 days

label = df_1['speed_diff']

for i in range(0,len(df_1)-n_past-n_future+1):
  dias = df_1.iloc[i : i + n_past+24]
  mes = dias.iloc[0]['Month (number)']
  dia_1 = dias.iloc[0]['Day of month']
  dia_168 = dias.iloc[24*3+1]['Day of month']
  d=dias.iloc[24*3+1]['Day of week (name)']
  if (mes == 4 or mes == 6 or mes == 9 or mes == 11) and (dia_168 - dia_1 == 3 or dia_168 - dia_1 == -29):
    a=df_1.iloc[i : i + n_past]
    nome_dia1.append(d)
    a.drop('Month (number)', axis = 1, inplace = True)
    a.drop('Day of month', axis = 1, inplace = True)
    a.drop('Hour', axis = 1, inplace = True)
    a.drop('Day of week (name)', axis = 1, inplace = True)
    a.drop('Distance', axis = 1, inplace = True)
    a.drop('incident_category_desc', axis = 1, inplace = True)

    x_train1.append(a)
    y_train1.append(label.iloc[i + n_past : i + n_past + n_future ])
  elif (mes == 1 or mes == 3 or mes == 5 or mes == 7 or mes == 8 or mes == 10 or mes == 12) and (dia_168 - dia_1 == 3 or dia_168 - dia_1 == -28):
    a=df_1.iloc[i : i + n_past]
    nome_dia1.append(d)
    a.drop('Month (number)', axis = 1, inplace = True)
    a.drop('Day of month', axis = 1, inplace = True)
    a.drop('Hour', axis = 1, inplace = True)
    a.drop('Day of week (name)', axis = 1, inplace = True)
    a.drop('Distance', axis = 1, inplace = True)
    a.drop('incident_category_desc', axis = 1, inplace = True)
    x_train1.append(a)
    y_train1.append(label.iloc[i + n_past : i + n_past + n_future ])
  elif mes == 2 and (dia_168 - dia_1 == 3 or dia_168 - dia_1 == -26):
    a=df_1.iloc[i : i + n_past]
    nome_dia1.append(d)
    a.drop('Month (number)', axis = 1, inplace = True)
    a.drop('Day of month', axis = 1, inplace = True)
    a.drop('Hour', axis = 1, inplace = True)
    a.drop('Day of week (name)', axis = 1, inplace = True)
    a.drop('Distance', axis = 1, inplace = True)
    a.drop('incident_category_desc', axis = 1, inplace = True)
    x_train1.append(a)
    y_train1.append(label.iloc[i + n_past : i + n_past + n_future ])
'''x_train1=[]
y_train1=[]
for i in range(len(x_train)-1):
  x_train1.append(x_train[i]+x_train[i+1])
  y_train1.append(y_train[i]+y_train[i+1])
print(x_train1[0].size)
print(len(x_train1))
x_train=x_train1
y_train=y_train1'''


for i in range(len(x_train1)):
    x_train1[i]=np.array(x_train1[i])
for i in range(len(y_train1)):
    y_train1[i]=np.array(y_train1[i])
for i in range(len(x_train1)):
    x_train1[i]=np.array(x_train1[i])
for i in range(len(y_train1)):
    y_train1[i]=np.array(y_train1[i])
for i in range(len(x_train1)):
    x_train1[i] = np.reshape(x_train1[i], (x_train1[0].shape[0],x_train1[0].shape[1]) )

for i in range(len(y_train1)):
    y_train1[i] = np.reshape(y_train1[i], (y_train1[0].shape[0]))


x_train1=np.array(x_train1)
y_train1=np.array(y_train1)
DADOS_TREINO1=[]
DADOS_TESTE1=[]
LABELS_TREINO1=[]
LABELS_TESTE1=[]
l1=[]


i=0
while i<100:
  r=random.randint(1000,len(x_train1)-1)
  if r not in indices:
    indices.append(r)
    i+=1

for i in range(len(x_train1)):
  if i in indices:
    DADOS_TESTE1.append(x_train1[i])
    LABELS_TESTE1.append(y_train1[i])
    l1.append(nome_dia1[i])
  else:
    DADOS_TREINO1.append(x_train1[i])
    LABELS_TREINO1.append(y_train1[i])



x_train1=DADOS_TREINO1
y_train1=LABELS_TREINO1
x_test1=DADOS_TESTE1
y_test1=LABELS_TESTE1

x_train1=np.array(x_train1)
y_train1=np.array(y_train1)
x_test1=np.array(x_test1)
y_test1=np.array(y_test1)

x_train2=np.concatenate((x_train, x_train1), axis=0)
y_train2=np.concatenate((y_train, y_train1), axis=0)
x_test2=np.concatenate((x_test, x_test1), axis=0)
y_test2=np.concatenate((y_test, y_test1), axis=0)

print('x_train:',x_train2.shape)
print('y_train:',y_train2.shape)
print('x_test:',x_test2.shape)
print('y_test:',y_test2.shape)

x_train=x_train2
y_train=y_train2
x_test=x_test2
y_test=y_test2
#------------------------------

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
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
history=regressor.fit(x_train, y_train, epochs=1000, callbacks=[callback] )

print('############################')

predicted_temperature = regressor.predict(x_test)
previstos=[]
for i in range(100):
  k=predicted_temperature[i].reshape(1,24)
  a = sc1.inverse_transform(k)
  a = np.reshape(a,(a.shape[1],a.shape[0]))
  previstos.append(a)


erros=[]
for i in range(100):
  soma=0
  #print('PREVISÃO DO DIA %d' %i)
  for j in range(24):
    soma += abs(y_test[i][j]-previstos[i][j][0])
  erros.append(soma/24)
    #print('Hora: ',j,'. Diferença entre o valor real e previsto: ',abs(y_test[i][j]-previstos[i][j][0]))
  #print('-------------------------------------------------')
  #print("Média dos erros: ",soma/24)
  #print('-------------------------------------------------')
  #print('-------------------------------------------------')
  #print('-------------------------------------------------')

for i in range(5):
  for j in range(24):
    print('Hora: ',j,'. Real: ',y_test[i][j], 'Previsto: ',previstos[i][j][0])

print('Número de ocurrências de cada dia da semana')
n = [0,0,0,0,0,0,0]
m = [0,0,0,0,0,0,0]
for i in range(len(l)):
  n[int(l[i])]+=1
  m[int(l[i])]+=erros[i]
for i in range(7):
  print(n[i])
  print(m[i]/n[i])

print('-------------------------------------------------')
print('-------------------------------------------------')
print('-------------------------------------------------')