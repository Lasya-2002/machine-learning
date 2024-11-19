import pandas as pd
import csv
df1=pd.read_csv('housing.csv')
print(df1)

df2=pd.read_table('housing.csv')
print(df2)

with open('housing.csv') as csv_file:
    csv_reader=csv.reader(csv_file)
    df3=pd.DataFrame([csv_file],index=None)
print(df3)