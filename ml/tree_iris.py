import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
print(sns.get_dataset_names())

df=pd.DataFrame(sns.load_dataset('iris'))
print(df.head())

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=None)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy',random_state=20)
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

plt.figure(figsize=(10,8))
ax=sns.heatmap(cm,annot=True,cmap='Blues')
ax.set_title('Confusion Matrix with labels \n\n')
ax.set_xlabel('\nPredicted labels')
ax.set_ylabel('\nActual values')
plt.show()