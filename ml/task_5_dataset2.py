import pandas as pd
df=pd.read_csv('iris.csv')
print(df)
print(df.dtypes=='object','\n\n')
print(df.isnull().sum(),'\n\n')
#using loc[]
x=df.loc[:,['sepal_length','sepal_width','petal_length','petal_width']]
y=df.loc[:,'species']
print(x)
print(y)
#using iloc[]
print('\n')
x1=df.iloc[:,:-1]
y1=df.iloc[:,-1]
print(x1)
print(y1)
