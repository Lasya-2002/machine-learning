import pandas as pd
df=pd.read_csv('auto-mpg.csv')
print(df)
print(df.dtypes=='object')
print(df.isnull().sum())
print('\n')
df=df.replace(to_replace='?',value=0)
d1={'horsepower':int}
df=df.astype(d1)
print(df.dtypes)
print(df.isnull().sum())
print(df)