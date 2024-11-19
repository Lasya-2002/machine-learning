import sys
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

full_health_data=pd.read_csv('data.csv',header=0,delimiter=',')
print(full_health_data.head())

x=full_health_data['Average_Pulse']
y=full_health_data['Calorie_Burnage']

slope,intercept,r,p,std_err=stats.linregress(x,y)

def myFunc(l):
    return slope*x+intercept

mymodel=list(map(myFunc,x))
plt.figure(figsize=(10,8))
plt.scatter(x,y)
plt.plot(x,mymodel)
plt.xlim(xmin=0,xmax=200)
plt.ylim(ymin=0,ymax=200)
plt.xlabel('Average Pulse')
plt.ylabel('Calorie Burnage')
plt.title('Linear regression')
plt.show()

plt.savefig('p1.jpeg')
import numpy as np
print(np.std(full_health_data))
