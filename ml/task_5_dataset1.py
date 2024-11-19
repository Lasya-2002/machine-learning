import pandas as pd
df=pd.read_csv('nba-2.csv',index_col='Name')
print(df)
print(df.dtypes=='object','\n\n')
print(df.isnull().sum(),'\n\n')
#indexing using dataframe index operator
print(df[['Age','College','Salary']],'\n\n\n')
#indexing using loc[]
first=df.loc['Avery Bradley']
second=df.loc['R.J. Hunter']
print(first,'\n\n\n',second,'\n')
print(df.loc[['Avery Bradley','R.J. Hunter']])
print(df.loc[['Avery Bradley','R.J. Hunter'],['Team','Number','Position']])
print(df.loc[:,['Team','Number','Position']])
#indexing using iloc[]
row2=df.iloc[3]
print(row2,'\n\n')
print(df.iloc[[3,5,7]],'\n\n\n')
print(df.iloc[[3,4],[1,2]],'\n\n\n')
print(df.iloc[:,[1,2]],'\n\n')
#indexing using ix[]
#first=df.ix['Avery Bradley']
#second=df.ix[1]
#print(first,'\n\n',second)
print(df.head())
print(df.tail())
