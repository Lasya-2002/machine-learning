import pandas as pd
import csv
df1=pd.read_csv('bank.csv')
print(df1)

df2=pd.read_table('bank.csv')
print(df2)

with open('bank.csv') as csv_file:
    csv_reader=csv.reader(csv_file)
    df3=pd.DataFrame([csv_reader],index=None)
print(df3)    