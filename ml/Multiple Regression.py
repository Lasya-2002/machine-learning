import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

housing_data = pd.read_csv("USA_Housing.csv")

housing_data = pd.get_dummies(housing_data, columns=['Avg. Area Income', 'Avg. Area House Age', 
                                                     'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 
                                                     'Area Population', 'Price', 'Address' ])
print(housing_data.head())
print(housing_data.isnull().sum())
X, y = housing_data.drop('Price', axis=1), housing_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

metrics = {'Mean Squared Error': mean_squared_error, 'R^2 Score': r2_score}
[print(f"{name}: {metric(y_test, y_pred)}") for name, metric in metrics.items()]

coefficients = pd.DataFrame({'feature': X.columns, 'coefficient': model.coef_})
print(coefficients)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()