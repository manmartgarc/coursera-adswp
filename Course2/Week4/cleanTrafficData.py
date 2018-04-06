# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:23:28 2018

@author: manma
"""

import pandas as pd

# %% import
df = pd.read_csv('Data Query Tool Table.csv',
                 index_col=0)

# %% clean variable names
df.columns = df.columns.str.lower().str.replace(' ', '_')

month_dict = {'January':1,
              'February':2,
              'March':3,
              'April':4,
              'May':5,
              'June':6,
              'July':7,
              'August':8,
              'September':9,
              'October':10,
              'November':11,
              'December':12}

df['crash_month'] = df['crash_month'].replace(month_dict)

# %% slice data
# snow related accidents

df.to_csv('incidents.csv')
#snow_crashes_nowet = df[['crash_month', 'snow', 'ice', 'slush']].copy()
#snow_crashes_wet = df[['crash_month', 'wet', 'snow']].copy()
#
#snow_crashes_nowet['total'] = snow_crashes_nowet.sum(axis=1,
#                                                     numeric_only=True)
#
#snow_crashes_wet['total'] = snow_crashes_wet.sum(axis=1,
#                                                numeric_only=True)

#snow_crashes_nowet.to_csv('snow_crashes_nowet.csv')
#snow_crashes_wet.to_csv('snow_crashes_wet.csv')