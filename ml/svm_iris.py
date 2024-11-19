# Load the important packages
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

# Load the datasets
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

#Build the model
svm = SVC(kernel="linear", gamma=0.5, C=1.0)
# Trained the model
svm.fit(X, y)

# Plot Decision Boundary
DecisionBoundaryDisplay.from_estimator(
		svm,
		X,
		response_method="predict",
		cmap=plt.cm.Spectral,
		alpha=0.8,
		xlabel=iris.feature_names[0],
		ylabel=iris.feature_names[1],
        )

# Scatter plot
plt.scatter(X[:, 0], X[:, 1], 
			c=y, 
			s=20, edgecolors="k")
plt.show()
