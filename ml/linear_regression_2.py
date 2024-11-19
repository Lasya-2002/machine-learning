#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

#importing the dataset
df=pd.read_csv('USA_housing.csv')
print(df.head())

print(df.isnull().sum(),'\n')
print(df.shape)
print(df.info(),'\n')
print(df.describe())

df=df.drop('Address',axis=1)

#EDA using plots
sns.pairplot(data=df,hue='Price')
plt.show()
#Correlation heatmap
sns.heatmap(data=df.corr(),cmap='Pastel2',annot=True,fmt='.2f')
plt.show()

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df_scaled=sc.fit_transform(df)

#Extracting the target and features from the data frame
x=df.drop('Price',axis=1)
y=df['Price']

#training and testing data splitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
print('x_train shape is: ',x_train.shape,'\n y_train shape is:',y_train.shape,'\n x_test shape is: ',x_test.shape,'\n y_test shape is: ',y_test.shape)

# Model selection
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error
print('The R squared value is :', r2_score(y_test,y_pred))
print('The mean squared value is: ',mean_squared_error(y_test,y_pred))





