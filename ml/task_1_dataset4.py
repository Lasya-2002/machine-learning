import pandas as pd
import csv
df1=pd.read_csv('auto-mpg.csv')
print(df1)

df2=pd.read_table('auto-mpg.csv')
print(df2)

with open('auto-mpg.csv') as csv_file:
    csv_reader=csv.reader(csv_file)
    df3=pd.DataFrame([csv_file],index=None)
print(df3)