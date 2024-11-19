import pandas as pd
import csv
df1=pd.read_csv('train.csv')
print(df1)

df2=pd.read_table('train.csv')
print(df2)

with open('train.csv') as csv_file:
    csv_reader= csv.reader(csv_file)
    df3=pd.DataFrame([csv_reader],index=None)
csv_file.close()
print(df3)