import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df=pd.read_csv('tips.csv')
print(df.head(10))

#MATPLOTLIB
#scatter plot
plt.figure(figsize=(10,8))
plt.scatter(df['day'],df['tip'])
plt.title('scatter plot using Matplotlib')
plt.xlabel('days')
plt.ylabel('tips')
plt.show()

#scatter plot with colour bar
plt.figure(figsize=(10,8))
plt.scatter(df['day'],df['tip'],c=df['size'],s=df['total_bill'])
plt.title('scatter plot with colour bar using Matplotlib')
plt.xlabel('days')
plt.ylabel('tips')
plt.colorbar()
plt.show()

#barplot
plt.bar(df['day'],df['tip'],color='maroon')
plt.title('Bar chart using Matplotlib')
plt.xlabel('days')
plt.ylabel('tips')
plt.show()

#histogram
plt.hist(df['total_bill'])
plt.title('Histogram using matplotlib')
plt.xlabel('total bill')
plt.ylabel('frequency')
plt.show()

#SEABORN
#line plot
sns.lineplot(x='sex',y='total_bill',data=df)
plt.title('Line plot using Seaborn')
plt.show()
sns.lineplot(data=df.drop(['total_bill'],axis=1))
plt.title('Lineplot with seaborn dropping the total bill')
plt.show()
#scatter plot
sns.scatterplot(x='day',y='tip',data=df)
plt.title('Scatter plot using seaborn')
plt.show()
#scatter plot with colour bar
sns.scatterplot(x='day',y='tip',data=df,hue='sex')
plt.title('Scatter plot in sns with colours')
plt.show()
#barplot
sns.barplot(x='day',y='tip', data=df,hue='sex')
plt.title('Barplot showing days vs tips')
plt.show()
#histogram
sns.histplot(x='total_bill', data=df, kde=True, hue='sex')
plt.show()

