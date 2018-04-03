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

fig, (ax1, ax2) = plt.subplots(2,1)

pd.rolling_mean(df[['Average Temp', 'Total Crashes']], window=12).plot(secondary_y='Total Crashes')

#sns.jointplot(df['Average Temp'], df['Total Crashes'],
#              kind='reg')

#sns.jointplot(df['avg_snowf'],
#              df['avg_totcrash'],
#              kind='reg')

yearly = df.groupby(level=0)['Average Temp', 'Total Crashes'].agg(
                    {'Average Temp':'mean', 'Total Crashes':'sum'})
            
monthly = df.groupby(level=1)['Average Temp', 'Total Crashes'].mean()

l1, = ax1.plot(yearly.index, yearly['Average Temp'])
ax1b = ax1.twinx()
l2, = ax1b.plot(yearly.index, yearly['Total Crashes'], 'g')

ax2.plot(monthly.index, monthly['Average Temp'])
ax2b = ax2.twinx()
ax2b.plot(monthly.index, monthly['Total Crashes'], 'g')

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

ax1.set_title('Yearly Summary', loc='left',
              fontdict={'fontsize':8})
ax1.set_xticks(yearly.index)
ax1.set_xticklabels(yearly.index,
                    rotation=45)
ax1.set_xlabel('Year')
ax1.set_ylabel('Average Temp ($\circ$C)')
ax1b.set_ylabel('Total Crashes', rotation=270, labelpad=10)

ax2.set_title('Monthly Summary', loc='left',
              fontdict={'fontsize':8})
ax2.set_xticks(monthly.index)
ax2.set_xticklabels(months, rotation=45)
ax2.set_xlabel('Month')
ax2.set_ylabel('Average Temp ($\circ$C)')
ax2b.set_ylabel('Total Crashes', rotation=270, labelpad=10)


# force ax1 to show xticklabels for some reason.
#plt.setp(ax1.get_xticklabels(), visible=True)
#plt.setp(ax1.get_xlabel(), visible=True)
#fig.set_label('test')

for ax in [ax1, ax2, ax1b, ax2b]:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(5)

fig.legend([l1, l2], ['Average Temp', 'Total Crashes'], fontsize=5,
           bbox_to_anchor=(0.56, 0.92))
fig.suptitle('Average Temperature and Traffic Accidents in Ann Arbor, MI')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
fig.savefig('tempTraffic.jpeg', dpi=300)
plt.close(fig)
