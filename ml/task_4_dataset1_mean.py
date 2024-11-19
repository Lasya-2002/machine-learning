import pandas as pd
import numpy as np
from scipy import stats
import random
df=pd.read_csv('iris.csv')
print(df)
print(df.dtypes=='object')
print(df.isnull().sum())
print('Mean of sepal_length is:', np.average(df.sepal_length))
print('Mean of sepal_width is:',np.average(df.sepal_width))
print('Mean of petal_length is:',np.average(df.petal_length))
print('Mean of petal_width is:', np.average(df.petal_width))
print('\n')
print('Median of sepal_length is:', np.median(df.sepal_length))
print('Median of sepal_width is:',np.median(df.sepal_width))
print('Median of petal_length is:',np.median(df.petal_length))
print('Median of petal_width is:', np.median(df.petal_width))
print('\n')
print('trimmed Mean of sepal_length is:', stats.trim_mean(df.sepal_length,0.25))
print('Trimmed Mean of sepal_width is:',stats.trim_mean(df.sepal_width,0.25))
print('Trimmed Mean of petal_length is:',stats.trim_mean(df.petal_length,0.25))
print('Trimmed Mean of petal_width is:', stats.trim_mean(df.petal_width,0.25))
lst1=random.randint(1,10)
df.insert(4,'Weights',lst1,True)
print('\n')
print('Weighted mean of sepal_length is:',np.average(df.sepal_length,weights=df.Weights))
print('Weighted mean of sepal_width is:',np.average(df.sepal_width,weights=df.Weights))
print('Weighted mean of petal_length is:',np.average(df.petal_length,weights=df.Weights))
print('Weighted mean of petal_width is:',np.average(df.petal_width,weights=df.Weights))