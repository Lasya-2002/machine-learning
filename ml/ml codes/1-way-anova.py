#basic scipy packages
import pandas as pd

#stats packages
from scipy import stats

#graphing packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
cvd_dataset=pd.read_csv('CVD_Dataset_EDA_ANOVA.csv')

sns.boxplot(x='Power',y='GrowthRate',data=cvd_dataset)
plt.xlabel(r'Power(w)')
plt.ylabel(r'Growth Rate ($\AA$/min')

cvd_pivot=cvd_dataset.pivot(index='RunID',
                            columns='Power',
                            values='GrowthRate')
print(cvd_pivot)

def degrees_of_freedom(df_pivot):
    #between the samples is across the row
    #within the server is down the row
    n=df_pivot.count(axis=1)
    a=df_pivot.count()
    n=n.iloc[0]
    a=a.iloc[0]
    TotalN=n*a
    dft=a-1
    dfe=TotalN-a
    dftotal=TotalN-1
    return dft,dfe,dftotal,n,a
dft,dfe,dftotal,n,a=degrees_of_freedom(cvd_pivot)
print('degrees of freedom between treatments: {:d}'.format(dft))
print("Degrees of freedom within treatments: {:d}".format(dfe))
print("Total degrees of freedom: {:d}".format(dftotal))

def compute_sums(df_pivot):
    yi_sum=df_pivot.sum()
    yi_avg=df_pivot.mean()
    y_sum=yi_sum.sum()
    y_avg=yi_avg.mean()
    return yi_sum,yi_avg,y_sum,y_avg
yi_sum,yi_avg,y_sum,y_avg=compute_sums(cvd_pivot)
print(r"The sum under each Power (W) treatment is: {}".format(yi_sum))
print(r"The average under each Power (W) treatment is: {}".format(yi_avg))
print(r"The grand sum is: {:.2f}".format(y_sum))
print(r"The grand average is: {:.2f}".format(y_avg))

def sum_of_squares(yi_avg,y_avg,df_pivot):
    sstr=(yi_avg-y_avg)**2
    sstr=sstr.sum()*n
    sse=df_pivot.sub(yi_avg.values)**2
    sse=sse.sum().sum()
    sst=sstr+sse
    return sstr,sse,sst
sstr,sse,sst=sum_of_squares(yi_avg,y_avg,cvd_pivot)
print("Sum of Squares between treatments is: {:.2f}".format(sstr))
print("Sum of Squares within treatments is: {:.2f}".format(sse))
print("Total Sum of Squares is: {:.2f}".format(sse))

def mean_squares(sstr,sse,dft,dfe):
    mstr=sstr/dft
    mse=sse/dfe
    return mstr,mse
mstr,mse=mean_squares(sstr,sse,dft,dfe)
print("Mean squares between treatments is: {:.2f}".format(mstr))
print("Mean squares within treatments is: {:.2f}".format(mse))

f=mstr/mse
p=stats.f.sf(f,dft,dfe)
print("F-value is: {:.2f}".format(f))
print("p-value is: {:.3f}".format(p))

#built in method for one way anova
f_val, p_val = stats.f_oneway(cvd_pivot[8], 
                cvd_pivot[10], cvd_pivot[12], cvd_pivot[14])  
print("F-value is: {:.2f}".format(f_val))
print("p-value is: {:.3f}".format(p_val))