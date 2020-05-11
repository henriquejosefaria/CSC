import pandas as pd

df=pd.read_csv('Dataset_9Maio.csv', delimiter=',',encoding = 'ISO-8859-1')

df.head(10)

print(len(df))
i=0
for i in range(1,13):
  for j in range(1,32):
    L=df[(df['Month (number)']==i)&(df['Day of month']==j)].dropna()

    L1=L[['Month (number)','Day of month','Hour','road_num']]
    L1 = L1.drop_duplicates()
    indexNames = df[(df['Month (number)']==i)&(df['Day of month']==j) ].index
    if len(L1)>96:
      print('ola')

    if (len(L1)>=92 and len(L1)<96):
      i=i+1
    if len(L1)<96:
      try:
        df.drop(indexNames, inplace=True)
      except:
        pass

print('Numero de dias que apenas falta uma hora',i)
print(len(df))
df.to_csv('Dataset_11Maio.csv', index = False)
