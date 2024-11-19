from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

# Load the dataset
cancer = load_breast_cancer()
X, y = cancer.data[:, :2], cancer.target

# Build and train the model
svm = SVC(kernel="linear", gamma=0.5, C=1.0).fit(X, y)

# Plot Decision Boundary and scatter plot
[DecisionBoundaryDisplay.from_estimator(
    svm, X, response_method="predict", cmap=plt.cm.Spectral, alpha=0.8,
    xlabel=cancer.feature_names[0], ylabel=cancer.feature_names[1]) or
 plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors="k") for _ in range(2)]

plt.show()