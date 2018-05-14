# -*- coding: utf-8 -*-
"""
Created on Sun May  6 21:31:43 2018

@author: manma
"""

import pandas as pd
import numpy as np

doc = []
with open('dates.txt', 'r') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)        

# %%
# 04/20/2009; 04/20/09; 4/20/09; 4/3/09
fmt1_a = df.str.extractall(r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})')
#fmt1_b = df.str.extractall(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})')
fmt1 = pd.concat([fmt1_a])
fmt1.reset_index(inplace=True)
fmt1_index = fmt1['level_0']

# %%
# Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
fmt2 = df.str.extractall(
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[.]* )'
        '(\d{1,2})[?:, -]*(\d{4})')

#fmt2_b = df.str.extractall(
#        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(\d{1,2}),\s'
#        '(\d{2,4})')
#
#fmt2_c = df.str.extractall(
#        r'(January|February|March|April|May|June|July|August|'
#        'September|October|November|December)\s(\d{1,2}),\s(\d{2,4})')
#
#fmt2_d = df.str.extractall(
#        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.\s(\d{1,2}),\s'
#        '(\d{2,4})')
#
#fmt2_e = df.str.extractall(
#        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(\d{1,2})\s'
#        '(\d{2,4})')

#fmt2 = pd.concat([fmt2_a, fmt2_b, fmt2_c, fmt2_d, fmt2_e])

fmt2.reset_index(inplace=True)
fmt2_index = fmt2['level_0']

# %%
# 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009

fmt3 = df.str.extractall(r'((?:\d{1,2} ))?'
           '((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
           '[a-z]*[?:, -]* )(\d{4})')

fmt3.rename(columns={0:1, 1:0}, inplace=True)
fmt3.reset_index(inplace=True)
fmt3_index = fmt3['level_0']

# %%
# Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
#fmt4 = df.str.extractall(
#        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(\d{1,2})'
#        '(st|nd|rd|th),\s(\d{2,4})')
#fmt4.index.rename('level_0', inplace=True)
#fmt4.reset_index(inplace=True)
#fmt4_index = fmt4['level_0']


# %%
# Feb 2009; Sep 2009; Oct 2010
#fmt5 = df.str.extractall(
#        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(\d{2,4})')
#
#fmt5 = fmt5.rename(columns={1:2})
#fmt5.reset_index(inplace=True)
#fmt5_index = fmt5['level_0']

# %%
# 6/2008; 12/2009
fmt6 = df.str.extractall(r'(\d{1,2})/(\d{4})')

fmt6 = fmt6.rename(columns={1:2})
fmt6.reset_index(inplace=True)
fmt6_index = fmt6['level_0']

# %%
# 2009; 2010
fmt7_a = df.str.extractall(r'.*(\d{4})')
fmt7_b = df.str.extractall(r'^(\d{4})[^0-9]')

fmt7 = pd.concat([fmt7_a, fmt7_b])

fmt7 = fmt7.rename(columns={0:2})
fmt7.reset_index(inplace=True)
fmt7_index = fmt7['level_0']

# %%
df2 = pd.concat([fmt1, fmt2, fmt3, fmt6, fmt7])
df2['level_0'] = df2['level_0'].astype(int)
    
for col in df2.select_dtypes(include='object'):
    df2[col] = df2[col].str.strip()
    
df2.rename(columns={0:'month', 1:'day', 2:'year'}, inplace=True)
df2 = df2[df2['match'] == 0]
df2.drop(columns='match', inplace=True)

months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

months_num = dict(enumerate(months, start=1))
months_dict = {v:k for k,v in months_num.items()}
months_dict3 = {v[:3]:k for k,v in months_num.items()}

df2['month'] = df2['month'].str.replace(r'[\,\.]', '')
df2['month'] = df2['month'].replace({'Decemeber':'December',
                                     'Janaury':'January'})
df2['month'].replace(months_dict, inplace=True)
df2['month'].replace(months_dict3, inplace=True)
df2['month'].fillna(1, inplace=True)
df2['day'].fillna(1, inplace=True)
df2['month'] = df2['month'].astype(int)
df2['month'] = np.where(df2['month'] > 12, df2['day'], df2['month'])
df2['month'] = np.where(df2['month'] > 12, 1, df2['month'])

df2['day'] = df2['day'].astype(int)
df2['year'] = df2['year'].astype(int)
df2['year'] = np.where(df2['year'].astype(str).str.len() == 2,
                       '19' + df2['year'].astype(str),
                       df2['year'])

df2 = df2.drop_duplicates(subset='level_0', keep='first')

df2['date'] = (
        pd.to_datetime(df2['month'].astype(str)
        + '/'
        + df2['day'].astype(str)
        + '/'
        + df2['year'].astype(str)))

df2 = df2.sort_values(by='date')
ordered = list(enumerate(df2['level_0']))

chrono = pd.Series(data=[i[1] for i in ordered],
                   index=[i[0] for i in ordered])
