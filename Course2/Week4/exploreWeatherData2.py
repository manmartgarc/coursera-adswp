# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:14:14 2018

@author: manma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


weather = pd.read_pickle('weatherData.pickle')
weather = weather.fillna(method='ffill')

incidents = pd.read_csv('incidents.csv', index_col=0)

# %%
# more than 1 inch of snow depth
weather.loc[weather['SNWD'] >= 25.4, 'snwd_inch'] = 1
#weather['SNWD'] = weather['SNWD'] / 25.4

# more than 1 inch of snowfall
weather.loc[weather['SNOW'] >= 25.4, 'snow_inch'] = 1
#weather['SNOW'] = weather['SNOW'] / 25.4

# fraction of snow crashes
incidents['frac_snow'] = incidents['snow'] / incidents['total']

# %%

avg_temps = weather[['TMAX', 'TMIN']].mean(axis=1)
avg_temps = avg_temps.groupby([avg_temps.index.year,
                               avg_temps.index.month]).mean()

avg_snowf= weather.groupby([weather.index.year,
                            weather.index.month])['SNOW'].mean()

avg_snowd = weather.groupby([weather.index.year,
                             weather.index.month])['SNWD'].mean()

n_snowdays = weather.groupby([weather.index.year,
                                weather.index.month])['snow_inch'].sum()

totcrash = incidents.groupby([incidents.index,
                              'crash_month'])['total'].mean()

totsnowcrash = incidents.groupby([incidents.index,
                               'crash_month'])['snow'].mean()

frac_snow = incidents.groupby([incidents.index,
                               'crash_month'])['frac_snow'].mean()

df = pd.DataFrame({'Average Snow Fall':avg_snowf,
                   'Average Snow Depth':avg_snowd,
                   'Number of Snow Days':n_snowdays,
                   'Total Crashes':totcrash,
                   'Total Snow Crashes':totsnowcrash,
                   'Fraction of Snow Crashes':frac_snow,
                   'Average Temp':avg_temps / 10},
                     index=avg_snowf.index)
# %%
plt.style.use('seaborn-colorblind')

fig, ax = plt.subplots(1,1)
ax1b = ax.twinx()
ax1b.get_yaxis().set_ticks([])

rolling = df[['Total Crashes',
              'Average Temp']].rolling(window=12,
                                       center=False)
ma = rolling.mean()
ma.plot(secondary_y='Total Crashes', ax=ax)

#ax.set_xticks(len(ma.index.get_level_values(level=0).unique()))
#ticks = np.arange(min(ma.index.get_level_values(0).unique()),
#                       max(ma.index.get_level_values(0).unique()))

ax.set_xticks(list(range(int(ax.get_xlim()[0]),
                         int(ax.get_xlim()[1]+10), 10)))

labels = []
for i in range(len(ma.index.get_values())):
    if i % 10 == 0:
        labels.append(ma.index.get_values()[i])

month_dict = {1:'Feb', 2:'Jan', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
              7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}



ax.set_xticklabels(labels,
                   rotation=45)

ax.set_xlabel('Year')
ax.set_ylabel('Temp ($\circ$ C - MA)')
ax1b.set_ylabel('Total Traffic Incidents (MA)', rotation=270,
                labelpad=50)

fig.suptitle('12-month Moving Average for\n'
             'Temperature and Traffic Accidents in Ann Arbor, MI')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
fig.savefig('tempTraffic.jpeg', dpi=300)
plt.close(fig)
