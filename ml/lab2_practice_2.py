#Neural Network
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

df=pd.read_csv('Churn_Modelling.csv')
print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.info())
print(df.dtypes=='object')

df=df.drop(['RowNumber','CustomerId','Surname'],axis=1)
print(df.dtypes)
print(df.shape)

print(df['Geography'].value_counts)
print(df['Gender'].value_counts)

obj_list=['Geography','Gender']
from sklearn.preprocessing import LabelEncoder
for col in obj_list:
    en=LabelEncoder()
    en.fit(df[col])
    df[col]=en.transform(df[col])
print(df.dtypes)
print(df.head())    

x=df.iloc[:,:9].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=10)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(units=4,activation='relu',kernel_initializer='uniform',input_dim=9))
model.add(Dense(units=4,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

model.fit(x_train,y_train,batch_size=2,epochs=10)
y_pred=model.predict(x_test)

print(y_pred[10:20])
print(y_test[10:20])

