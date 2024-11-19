#data visualizations
import pandas as pd
data=pd.read_csv('tips.csv')
#USING MATPLOTLIB
import matplotlib.pyplot as plt
plt.scatter(data['day'],data['tip'])
plt.show()
#with colour bar
plt.scatter(data['day'],data['tip'],c=data['size'],s=data['total_bill'])
plt.colorbar()
plt.show()
#linechart
plt.plot(data['tip'])
plt.plot(data['size'])
plt.show()
#barplot
plt.bar(data['day'],data['tip'])
plt.show()
#histogram
plt.hist(data['total_bill'])
plt.show()

#USING SEABORN 
import seaborn as sns
sns.lineplot(x='sex',y='total_bill',data=data)
plt.show()

sns.scatterplot(x='day',y='tip',data=data)
plt.show()
#using bokeh
from bokeh.plotting import figure,show
from bokeh.palettes import magma
graph=figure(title='Bokeh Scatterplot')
color=magma(256)
graph.scatter(data['total_bill'],data['tip'],color=color)
show(graph)

df=data['tip'].value_counts()
graph.line(df,data['tip'])
show(graph)

graph=figure(title='Bokeh Bar Chart')
graph.vbar(data['total_bill'],top=data['tip'])
show(graph)
