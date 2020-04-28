import pandas as pd
from math import radians, sin, cos, atan2, sqrt

df = pd.read_csv('PauPau.csv', delimiter = ',', error_bad_lines = False,
                 encoding = 'ISO-8859-1')

def distance(p1, n):
  R = 6371.0
  if n == 1:
    lat2 = radians(41.548331)
    lon2 = radians(-8.421298)
  elif n == 2:
    lat2 = radians(41.551356)
    lon2 = radians(-8.420001)
  elif n == 3:
    lat2 = radians(41.546639)
    lon2 = radians(-8.433517)
  else:
    lat2 = radians(41.508849)
    lon2 = radians(-8.462299)
  lat1, lon1 = radians(p1[0]), radians(p1[1])
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  c = 2 * atan2(sqrt(a), sqrt(1 - a))
  distance = R * c
  return distance

df['Distance'] = df.apply(lambda row: distance((row['latitude'],row['longitude']), row['road_num']), axis=1)


'''print('max:', df['Distance'].max())
print('min:', df['Distance'].min())
print('mean:', df['Distance'].mean())
print('standard deviation:', df['Distance'].std())'''


# row ids para distancia GRANDE
#df.index[df['Distance'] == 6313.251265773197].tolist()

#df.iloc[17807]

# o professor pos coordenadas dos USA?
# agora vamos ter que procurar por dados falsos!!! meh

# ver rows tq distancia >= threshold
# remover essas
print(len(df))
print('oi')
thresh = 1.5 # braga é pequena
ind_dados_errados = df.index[df['Distance'] > thresh].tolist()
print(len(ind_dados_errados))
print('oi')


df.drop(ind_dados_errados , inplace=True)

print(len(df))

df.to_csv('Dataset1.5.csv', index = False)


'''print('max:', df['Distance'].max())
print('min:', df['Distance'].min())
print('mean:', df['Distance'].mean())
print('standard deviation:', df['Distance'].std())'''
