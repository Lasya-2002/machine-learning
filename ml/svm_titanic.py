import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
# Load the dataset
data = pd.read_csv('titanic_train.csv')
# Drop rows with NaN values
data = data.dropna()
# Split features and target
X = data[['Age', 'Fare']]  # Assuming 'Age' and 'Fare' are features
y = data['Survived']  # Assuming 'Survived' is the target variable
# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)
# Build the model with class weights
svm = SVC(kernel="linear", gamma=0.5, C=1.0, class_weight='balanced')
# Train the model
svm.fit(x_train, y_train)
# Scatter plot
plt.scatter(X['Age'], X['Fare'], c=y, s=20, edgecolors="k")
plt.xlabel('Age')
plt.ylabel('Fare')
# Plot Decision Boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# Create a meshgrid of points
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
# Show the plot
plt.show()
# Calculate accuracy score on test set
accuracy = svm.score(x_test, y_test)
print("Accuracy:", accuracy)
# Make predictions
pred = svm.predict(x_test)
# Generate confusion matrix
cm = confusion_matrix(y_test, pred)
# Visualize confusion matrix
ax = sns.heatmap(cm, annot=True, cmap='Greens')
ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values')
plt.show()
# Display classification report without zero division warnings
print(classification_report(y_test, pred, zero_division=0))