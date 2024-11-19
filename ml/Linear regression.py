import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and explore data
df = pd.read_csv("Salary_dataset.csv")
df.drop('Unnamed: 0', axis=1, inplace=True)

# Data visualization
sns.scatterplot(x=df['YearsExperience'], y=df['Salary'])
plt.xlabel('Years')
plt.ylabel('Salary')
plt.title('Salary Prediction')
plt.show()

# Model training and evaluation
X, Y = df['YearsExperience'].values.reshape(-1, 1), df['Salary']
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

lr = LinearRegression().fit(x_train, y_train)
y_pred = lr.predict(x_test)

metrics = {'Mean Squared Error': mean_squared_error,
           'Mean Absolute Error': mean_absolute_error,
           'R2 Score': r2_score}
for name, metric in metrics.items():
    print(f'{name}: {metric(y_test, y_pred)}')

print(f'Intercept of the model: {lr.intercept_}')
print(f'Coefficient of the line: {lr.coef_}')

# Validation with new data
x_new = np.array([[1.3]])  # New test value as a list of list
print(lr.predict(x_new))

# Calculate manually to verify
manual_calculation = 9339.08172382 * 1.3 + 24985.53016251169
print(manual_calculation)