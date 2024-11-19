import pandas as pd
import random
df=pd.read_csv('auto-mpg.csv')
print(df)
lst1 = random.sample(range(500), len(df))
lst2= random.sample(range(399),len(df))
df.insert(5,'random_value',lst1,True)
print(df)
print('\n')
df=df.assign(new_col=lst2)
print(df)
print('\n')

