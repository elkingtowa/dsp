import datetime as dt
import pylab

fp = open('LehmanBrothers_data.txt')
columns = ['Close/Last','Volume']
data = empty((0,len(columns)))
dates = []
for line in fp.readlines()[1:]:
    fields = line.split('\t')
    dates.append(dt.datetime.strptime(fields[0], "%Y/%m/%d"))
    data = vstack([data, array([[float(fields[1].replace(',','')), float(fields[2].replace(',',''))]])])
fp.close()

import pandas as pd
df = pd.DataFrame(data, index=dates, columns=columns)
df.head()