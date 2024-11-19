import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Importing and splitting the data
def import_and_split():
    balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
                               sep=',', header=None)
    X = balance_data.iloc[:, 1:]
    y = balance_data.iloc[:, 0]
    return train_test_split(X, y, test_size=0.3, random_state=100)

# Training the decision tree and plotting
def train_and_plot_decision_tree(criterion):
    X_train, X_test, y_train, y_test = import_and_split()
    clf = DecisionTreeClassifier(criterion=criterion, random_state=100, max_depth=3, min_samples_leaf=5)
    clf.fit(X_train, y_train)
    plt.figure(figsize=(15, 10))
    plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=clf.classes_, rounded=True)
    plt.show()
    return clf, X_test, y_test

# Predictions and accuracy calculation
def predict_and_calculate_accuracy(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Report:\n{classification_report(y_test, y_pred)}")

# Main code
if_name_== "_main_":
    for criterion in ["gini", "entropy"]:
        clf, X_test, y_test = train_and_plot_decision_tree(criterion)
        predict_and_calculate_accuracy(clf, X_test, y_test)