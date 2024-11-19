import numpy as np
import pandas as pd
df=pd.read_csv('employees.csv')
print(df)
print(df.isnull().sum())
print('\n')
df=df.replace(to_replace=np.nan,value=0)
print(df.isnull().sum())