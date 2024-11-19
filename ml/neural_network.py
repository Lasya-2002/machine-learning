#Neural Network
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing label encoder for encoding, category data
from sklearn.preprocessing import LabelEncoder
#read the dataset
df = pd.read_csv("Churn_Modelling.csv")
print(df.head())
print(df.dtypes)
print(df.columns)
print(df.shape)
#drop un-necessary columns
df=df.drop(['RowNumber','CustomerId','Surname'], axis=1)
print(df.shape)
print(df.dtypes)
# object dtype means categorical data, see classes of objects
print(df.Gender.value_counts())
print(df.Geography.value_counts())
#object is categorical data, so convert to numeric by encoding
obj_list= ['Geography','Gender']

for col in obj_list:
    encoder = LabelEncoder()
    encoder.fit(df[col])
    df[col] = encoder.transform(df[col])
#see all dtypes are numeric
print(df.dtypes)
print(df.head())
#separate X and target
X = df.iloc[:,:9].values
y = df.iloc[:,-1].values
# apply train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
# Feature Scaling, observe som efeatures of X are with too large values, so scale down to small values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#build your keras ANN model
from keras.models import Sequential
from keras.layers import Dense
# Defining the ANN model
model = Sequential()
# Adding the first hidden layer, by passing X with 8 features
model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))

# Adding the second hidden layer
model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 2, epochs = 10)
y_pred = model.predict(X_test)
print(y_pred[10:20])
print(y_test[10:20])