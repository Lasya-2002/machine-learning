#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing and loading the dataset
df=pd.read_csv("C:/Users/mvsla/OneDrive/Desktop/ml/iris.csv")

#Exploratory Data analysis
print(df.head())
print(df.dtypes=='object')
print(df.info())
print(df.describe())

#EDA using visualization
#univariate analysis
count=df['species'].value_counts()
plt.bar(count.index,count,color='blue')
plt.xlabel('target classes')
plt.ylabel('Count')
plt.title('Count plot of the target feature')
plt.show()
#there is no class imbalance

#Bivariate analysis
plt.figure(figsize=(10,8))
sns.set_style('darkgrid')
sns.set_palette('pastel')
sns.boxplot(data=df,orient='v',width=0.5)
plt.show()

#pairplot
plt.figure(figsize=(10,8))
sns.pairplot(data=df,hue='species')
plt.show()

#selecting the necessary features from the dataframe
from sklearn.preprocessing import OneHotEncoder
enc= OneHotEncoder()

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
y_df=pd.DataFrame(y)
y_enc=pd.get_dummies(y_df,prefix=[''],dtype='float')

print(y_enc)

#training and testing data split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y_enc,test_size=0.20,random_state=42)

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(units=64, activation='relu', input_dim=4,kernel_initializer='uniform'))
model.add(Dense(units=64, activation='relu',kernel_initializer='uniform'))
model.add(Dense(units=3,activation='softmax',kernel_initializer='uniform'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

model.fit(x_train,y_train,epochs=10,batch_size=2)
acc_tr= model.evaluate(x_train)
acc_tst=model.evaluate(x_test)
