#importing the libraries
import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

#importing the dataset
df=pd.read_csv('winequality-red.csv')
print(df.head(),'\n')
print(df.shape,'\n')
print(df.dtypes=='object','\n\n')
print(df.info(),'\n')
print(df.describe(),'\n')
print(df.nunique())
print(df.isnull().sum(),'\n')
print(df.columns.tolist(),'\n')

#EXPLORATORY DATA ANALYSIS
#Univariate analysis

#creating a count plot
quality_counts=df['quality'].value_counts()
plt.figure(figsize=(8,6))
plt.bar(quality_counts.index,quality_counts,color='green')
plt.title('Count plot of quality')
plt.xlabel('count')
plt.ylabel('quality')
plt.show()

#creating the kernel density plots
sns.set_style('darkgrid')
numerical_columns=df.select_dtypes(include=['int64','float64']).columns
plt.figure(figsize=(14,len(numerical_columns)*3))
for idx, feature in enumerate(numerical_columns,1):
    plt.subplot(len(numerical_columns),2,idx)
    sns.histplot(df[feature],kde=True)
    plt.title(f"{feature}| skewness:{round(df[feature].skew(),2)}")

plt.tight_layout()
plt.show()

#creating a swarm plot
plt.figure(figsize=(10,8))
sns.swarmplot(x='quality',y='alcohol',data=df,palette='viridis')
plt.title('Swarm plot for quality and Alcohol')
plt.xlabel('Quality')
plt.ylabel('Alcohol')
plt.show()

#Bivariate analysis
#pair plots
sns.set_palette('pastel')
plt.figure(figsize=(10,6))
sns.pairplot(df)
plt.suptitle('Pair Plot Of dataframe')
plt.show()

#creating viloin plots
df['quality']=df['quality'].astype(str)
plt.figure(figsize=(10,8))
sns.violinplot(x="quality", y="alcohol", data=df, palette={
               '3': 'lightcoral', '4': 'lightblue', '5': 'lightgreen', '6': 'gold', 
               '7':'lightskyblue', '8': 'lightpink'}, alpha=0.7)
plt.title('violin plot for quality and alcohol')
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()

#creating a box plot
sns.boxplot(x='quality',y='alcohol',data=df)

#Multivariate analysis
#Heatmap using correlation matrix
# Assuming 'df' is your DataFrame
plt.figure(figsize=(15, 10))
# Using Seaborn to create a heatmap
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Pastel2', linewidths=2)
plt.title('Correlation Heatmap')
plt.show()
