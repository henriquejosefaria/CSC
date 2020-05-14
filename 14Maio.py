import numpy as np
import pandas as pd
import tensorflow as tf
import random

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

import matplotlib.pyplot as plt

df_1 = pd.read_csv('Dataset_Rua1.csv', delimiter = ',',
                 encoding = 'ISO-8859-1')

# Ordenar por mês, dia e hora.
#df.sort_values(['Month (number)', 'Day of month', 'Hour'],ascending = [True, True, True], inplace = True)

# Separação por ruas.
#df_1 = df[df['road_num'] == 1]
#df_2 = df[df['road_num'] == 2]
#df_3 = df[df['road_num'] == 3]
#df_4 = df[df['road_num'] == 4]

# Vamos tratar da rua 1.
df_1.drop('road_num', axis = 1, inplace = True)
df_1=df_1.reset_index(drop=True)

#=======================================================================================================================
#
#                                               Supervised Problem
#
#=======================================================================================================================

x_train = []
y_train = []
nome_dia=[]

n_future = 24 # next 24 hours speed diff forecast
n_past = 24*3 # Past 3 days

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

print(x_train.shape)

print('=====================================================')

DADOS_TREINO=[]
DADOS_TESTE=[]
LABELS_TREINO=[]
LABELS_TESTE=[]
l=[]

#=======================================================================================================================
#
#                                               Dados de Validação
#
#=======================================================================================================================

indices=[]
i=0
while i<200:
  r=random.randint(1000,len(x_train)-1)
  if r not in indices:
    indices.append(r)
    i+=1

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
print(x_test.shape)

#=======================================================================================================================
#
#                                               Normalização
#
#=======================================================================================================================

# Normalização das features
scalers=[]
for i in range(11):
  sc = MinMaxScaler(feature_range=(0,1))
  x_train[:,i] = sc.fit_transform(x_train[:,i])
  x_test[:,i] = sc.fit_transform(x_test[:,i])
  scalers.append(sc)

# Normalização da label
sc1 = MinMaxScaler(feature_range=(0,1))
y_train = sc1.fit_transform(y_train)
y_test_n = sc1.fit_transform(y_test)

#=======================================================================================================================
#
#                                               Modelo
#
#=======================================================================================================================

def norm_inf(y_true, y_pred):
    return tf.keras.backend.max(tf.keras.backend.abs(y_pred - y_true))

def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

model = Sequential()
model.add(CuDNNLSTM(units=24*3, return_sequences=True, input_shape = (24*3,11) ) )
model.add(Dropout(0.2))
model.add(CuDNNLSTM(24*3 , return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(24*3, return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(24*2))
model.add(Dropout(0.2))
model.add(Dense(24,activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error',metrics=rmse)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
history=model.fit(x_train, y_train, validation_data=(x_test, y_test_n), epochs=1000, callbacks=[callback] )

#=======================================================================================================================
#
#                                                  Previsão
#
#=======================================================================================================================

predicted_temperature = model.predict(x_test)
previstos=[]
for i in range(200):
  k=predicted_temperature[i].reshape(1,24)
  a = sc1.inverse_transform(k)
  a = np.reshape(a,(a.shape[1],a.shape[0]))
  previstos.append(a)

#=======================================================================================================================
#
#                                                  Resultados
#
#=======================================================================================================================

erros=[]
for i in range(200):
  soma=0
  for j in range(24):
    soma += abs(y_test[i][j]-previstos[i][j][0])
  erros.append(soma/24)

for i in range(5):
  for j in range(24):
    print('Hora: ',j,'. Real: ',y_test[i][j], 'Previsto: ',previstos[i][j][0])
  print('=================================================================')

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

erros1=[]
for i in range(200):
  soma=[]
  for j in range(24):
    soma.append(abs(y_test[i][j]-previstos[i][j][0]))
  erros1.append(max(soma))

print('Número de ocurrências de cada dia da semana')
n1 = [0,0,0,0,0,0,0]
m1 = [0,0,0,0,0,0,0]
for i in range(len(l)):
  n1[int(l[i])]+=1
  m1[int(l[i])]+=erros1[i]
for i in range(7):
  print(n1[i])
  print(m1[i]/n1[i])

print('-------------------------------------------------')
print('-------------------------------------------------')
print('-------------------------------------------------')

#=======================================================================================================================
#
#                                                  Gráficos
#
#=======================================================================================================================

def print_history_loss(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

print_history_loss(history)

real=[]
prev=[]
for i in range(len(previstos)):
  for j in range(24):
    real.append(y_test[i][j])
    prev.append(previstos[i][j][0])

def plot_resultados(real, prev):
  plt.plot(range(len(real)), real, 'o', color='red', label="Valor Real")
  plt.plot(range(len(prev)), prev, 'o', color='black', label="Valor Previsto")
  plt.legend(loc="upper right")
  plt.xlabel('Hora')
  plt.ylabel('Speed Diff')
  plt.show()

plot_resultados(real[0:24], prev[0:24])