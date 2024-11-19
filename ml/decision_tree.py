import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')
df=pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-' +
        'databases/balance-scale/balance-scale.data',
        sep=',', header=None)
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())

sns.set_palette('bright')
plots=plt.figure(figsize=(10,6))

sns.set_style('darkgrid')
num_col=df.select_dtypes(include=['int64','float64']).columns
plt.figure(figsize=(14,len(num_col)*3))
for idx, feature in enumerate(num_col,1):
    plt.subplot(len(num_col),2,idx)
    sns.histplot(df[feature],kde=True)
    plt.title(f"{feature}| skewness:{round(df[feature].skew(),2)}")

plt.tight_layout()
plt.show()
    
sns.pairplot(df,hue=0)
plt.show()

x=df.iloc[:,1:5].values
y=df.iloc[:,0]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=20)


from sklearn.tree import DecisionTreeClassifier
gini=DecisionTreeClassifier(criterion='gini',max_depth=3,
                            random_state=100,min_samples_leaf=5)
gini.fit(x_train,y_train)
y_pred_gini=gini.predict(x_test)

entropy=DecisionTreeClassifier(criterion='entropy',random_state=100,
                               max_depth=3,min_samples_leaf=5)
entropy.fit(x_train,y_train)
y_pred_entropy=entropy.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cm1=confusion_matrix(y_test,y_pred_gini)
acc1=accuracy_score(y_test,y_pred_gini)*100
rep1=classification_report(y_test,y_pred_gini)

cm2=confusion_matrix(y_test,y_pred_entropy)
acc2=accuracy_score(y_test,y_pred_entropy)*100
rep2=classification_report(y_test,y_pred_entropy)

print('The confusion matrix in case of using decision criteria as GINI INDEX is :\n',cm1)
print('The accuracy in case of using decision criteria as GINI INDEX is :\n',acc1)
print('The classification report in case of using decision criteria as GINI INDEX is :\n',rep1)

print('The confusion matrix in case of using decision criteria as ENTROPY is : \n',cm2)
print('The accuracy in case of using decision criteria as ENTROPY is : \n',acc2)
print('The classification report in case of using decision criteria as ENTROPY is : \n',rep2)

  
# Function to plot the decision tree
from sklearn import tree
def plot_decision_tree(clf_object, feature_names, class_names):
    plt.figure(figsize=(15, 10))
    plt.title('Decision Tree')
    tree.plot_tree(clf_object, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    plt.show()
    
plot_decision_tree(gini,['X1','X2','X3','X4'],['B','L','R'])
plot_decision_tree(entropy,['X1','X2','X3','X4'],['B','L','R'])
