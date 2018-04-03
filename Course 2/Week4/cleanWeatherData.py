# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 20:47:12 2018

@author: manma
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv('annArborWeatherData.txt',
                 delim_whitespace=True,
                 header=None)

# Re index columns
df.columns = df.columns + 1


df[1] = df[1].str.replace('USC00200230', '')
df = df[(df[1].str.endswith('TMAX')) |
        (df[1].str.endswith('TMIN')) |
        (df[1].str.endswith('SNOW')) |
        (df[1].str.endswith('SNWD'))]

# TMAX is df.loc['yearmonth'].iloc[0]
df[1] = df[1].str.replace('TMAX', '')
# TMIN is df.loc['yearmonth'].iloc[1]
df[1] = df[1].str.replace('TMIN', '')
# SNOW is df.loc['yearmonth'].iloc[2]
df[1] = df[1].str.replace('SNOW', '')
# SNWD is df.loc['yearmonth'].iloc[3]
df[1] = df[1].str.replace('SNWD', '')

df = df.set_index(df[1])
df = df.drop(columns=1)

# only even columns are days of the month
cols = [col for col in df.columns if col % 2 == 0]
df = df[cols]

df.columns = list(range(1,32))

# %%
rows = []
for yearmonth in tqdm(df.index.unique()):
    year = int(yearmonth[:4])
    month = int(yearmonth[4:6])
    for day in df.columns:
        day = day
        try:
            tmax = df.loc[yearmonth].iloc[0].iloc[day]
        except:
            tmax = np.nan
        try:
            tmin = df.loc[yearmonth].iloc[1].iloc[day]
        except:
            tmin = np.nan
        try:
            snow = df.loc[yearmonth].iloc[2].iloc[day]
        except:
            snow = np.nan
        try:
            snwd = df.loc[yearmonth].iloc[3].iloc[day]
        except:
            snwd = np.nan
        rows.append([year, month, day, tmin, tmax, snow, snwd])

# %%

data = pd.DataFrame(rows)

data['date'] = (data[0].astype(str) + '/' +
                data[1].astype(str) + '/' +
                data[2].astype(str))

data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.set_index('date')
data = data.drop(columns=[0,1,2])
data.columns = ['TMIN', 'TMAX', 'SNOW', 'SNWD']
data = data.replace('-9999', np.nan)
data['doy'] = data.index.dayofyear

# %%
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data[(data.index.year >= 2004) & (data.index.year <= 2016)]

# %%
data.to_pickle('weatherData.pickle')