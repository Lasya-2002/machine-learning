import pandas as pd
import csv
df1=pd.read_csv('CardioGoodFitness.csv')
print(df1)

df2=pd.read_table('CardioGoodFitness.csv')
print(df2)

with open('CardioGoodFitness.csv') as csv_file:
    csv_reader= csv.reader(csv_file)
    df3=pd.DataFrame([csv_reader],index=None)
print(df3)
