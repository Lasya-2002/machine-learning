#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

#importing the dataset
df=pd.read_csv("C:/Users/mvsla/OneDrive/Documents/Linear_regression_on_datasets[1]/Week-2- Packages, EDA, Simple Linear regression/Salary_dataset.csv")
print(df.head())

df=df.drop(df.columns[[0]],axis=1)

#EDA and plots
print(df.isnull().sum(),'\n')
print(df.describe,'\n')
print(df.shape,'\n')
print(df.info(),'\n')
print(df.duplicated().sum(),'\n')

#plot to visualize the relationship between the x and y
plt.figure(figsize=(10,8))
sns.scatterplot(data=df,x='YearsExperience',y='Salary')
plt.show()

#x=df['YearsExperience']
#y=df['Salary']

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
print('Training data X shape:',x_train.shape)
print('Training data Y shape:',y_train.shape)
print('Testing data X shape:',x_test.shape)
print('Testing data Y shape:',y_test.shape)

#x_train=x_train[:,np.newaxis]
#x_test=x_test[:,np.newaxis]

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print('Mean squared error is: ',mean_squared_error(y_test,y_pred))
print('Mean absolute error is: ',mean_absolute_error(y_test,y_pred))
print('R sqaured error is: ',r2_score(y_test,y_pred))

# Intecept and coeff of the line
print('Intercept of the model:',lr.intercept_)
print('Coefficient of the line:',lr.coef_)
x1= [[1.3]] 
y_pred1=lr.predict(x1)
print(y_pred1)

y1 = lr.coef_ * 1.3 + lr.intercept_
print(y1)

#visualizing the test set results
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,lr.predict(x_train),color='blue')
plt.title('Salary vs Experience test set results')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()