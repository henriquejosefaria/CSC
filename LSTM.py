import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
"""
Created with love on Fri Mar 20 14:33:22 2020

@author: Henrique Faria
"""
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer, Dense

df=pd.read_csv('Dataset_Finalissimo.csv', delimiter=',',encoding = 'ISO-8859-1')
df_1 = df[df['road_num'] == 1]
y = df_1.speed_diff
df_1.head(10)


'''
df_2 = df[df['road_num'] == 2]
df_3 = df[df['road_num'] == 3]
df_4 = df[df['road_num'] == 4]'''
#cols = cols[-1:] + cols[:-1]




LABEL = 'speed_diff'

X = df_1.drop(LABEL, axis = 'columns')
y = df_1[LABEL]


x_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.10, random_state = 42
)



print(x_train.shape)
print(y_train.shape)


def normalizer(data):
    scalor = MinMaxScaler(feature_range=(-1, 1))
    data[['Cases']] = scalor.fit_transform(data[['Cases']])
    return scalor




def build_model(timesteps, features, neurons=64):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(neurons,return_sequences=True, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.LSTM(neurons*2, return_sequences=False, input_shape=(timesteps, features)))
    model.add(tf.keras.layers.Dense(neurons, activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae'])
    return model


def forecast(model, dados_limpos, timesteps, multisteps):
    prev = dados_limpos[-timesteps:].values
    predictions = []
    for step in range(1, multisteps + 1):
        previsao = model.predict(prev)
        #previsao_Desnormalizada = scaler.inverse_transform(previsao)
        #predictions.append(previsao_Desnormalizada[0][0])
        predictions.append(previsao[0][0])
        prev = np.append(prev[0], previsao)
        prev = prev[-timesteps:]
    return predictions


def predict_first():
    #X, y = to_supervised(timesteps)
    model = build_model(timesteps, univariate)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
    print('oi')
    predictions = forecast(model, df_1, timesteps, multisteps, scaler)
    for i in predictions:
        print("predicted => ", i)
    plt.plot(predictions, label='id %s' % 0)
    plt.show()


# Executar


timesteps = 7
univariate = 17
multisteps = 1
batch_size = 6
epochs = 110


predict_first()
