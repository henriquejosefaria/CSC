import pandas as pd

df=pd.read_csv('Dataset0.5.csv', delimiter=',')

df.head(10)

for i in range(1,13):
  for j in range(1,32):
    L=df[(df['Month (number)']==i)&(df['Day of month']==j)].dropna()
    for k in range(1,5):
      for l in range(24):
        L1=L[(L['road_num']==k)&(L['Hour']==l)].dropna()
        if len(L1)==0:
          try:
            df.drop(L)
          except:
            pass
