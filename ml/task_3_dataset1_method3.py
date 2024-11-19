import pandas as pd
df=pd.read_csv('employees.csv')
print(df)
print(df.isnull().sum())
print('\n')
df=df.dropna()
print(df.isnull().sum())
print(df)