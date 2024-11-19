import pandas as pd
df=pd.DataFrame([['a','b'],['c','d']],
                index=['row1','row2'],
                columns=['col1','col2'])
print(df.to_json(orient='split'))

print(df.to_json(orient='index'))

#data=pd.read_json('http://api.population.io/1.0/population/India/today-and-tomorrow/?format=json')
#print(data)

import json
from pandas.io.json import json_normalize
Wwith open('https://github.com/a9k00r/python-test/blob/master/raw_nyc_phil.json') as f:
    d=json.load(f)
nycphil=json_normalize(d['programs'])
nycphil.head(3)

works_data=json_normalize(data=d['programs'],
                          record_path='works',
                          meta=['id','orchestra','programID','season'])
works_data.head(3)

soloist_data=json_normalize(data=d['programs'],
                             record_path=['works','soloists'],
                             meta=['id'])
soloist_data.head(3)'''