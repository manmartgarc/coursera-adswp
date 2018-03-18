
# coding: utf-8

# # Assignment 2
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# An NOAA dataset has been stored in the file `data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv`. The data for this assignment comes from a subset of The National Centers for Environmental Information (NCEI) [Daily Global Historical Climatology Network](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt) (GHCN-Daily). The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.
# 
# Each row in the assignment datafile corresponds to a single observation.
# 
# The following variables are provided to you:
# 
# * **id** : station identification code
# * **date** : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
# * **element** : indicator of element type
#     * TMAX : Maximum temperature (tenths of degrees C)
#     * TMAX : MAXimum temperature (tenths of degrees C)
# * **value** : data value for element (tenths of degrees C)
# 
# For this assignment, you must:
# 
# 1. Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high and record low temperatures for each day should be shaded.
# 2. Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
# 3. Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
# 4. Make the visual nice! Leverage principles from the first module in this course when developing your solution. Consider issues such as legends, labels, and chart junk.
# 
# The data you have been given is near **Ann Arbor, Michigan, United States**, and the stations the data comes from are shown on the map below.

# In[9]:

import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)
    return mplleaflet.display()

leaflet_plot_stations(400,'fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89')


# In[3]:

df = pd.read_csv('data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv')
df.head()


# In[4]:

df['Date'] = pd.to_datetime(df['Date'])

mins = df[df['Element'] == 'TMIN']
maxs = df[df['Element'] == 'TMAX']

mins = mins.groupby(['Date', 'Element'])['Data_Value'].min().unstack()
maxs = maxs.groupby(['Date', 'Element'])['Data_Value'].max().unstack()


# In[160]:

by_day = mins.merge(maxs, left_index=True, right_index=True)
by_day['year'] = by_day.index.year

# Convert to celcius
by_day['TMIN'] = by_day['TMIN'] / 10
by_day['TMAX'] = by_day['TMAX'] / 10


# In[161]:

by_day_15 = by_day[by_day['year'] == 2015]
by_year = by_day.groupby('year').agg({'TMIN':'min', 'TMAX':'max'})

by_year_05to14 = by_year.loc[by_year.index < 2015]

record_min = by_day.loc[by_day['year'] < 2015, 'TMIN'].min()
record_max = by_day.loc[by_day['year'] < 2015, 'TMAX'].max()
by_day_15 = by_day.loc[by_day['year'] == 2015]

below_min_15 = by_day_15[by_day_15['TMIN'] < record_min]
above_max_15 = by_day_15[by_day_15['TMAX'] > record_max]


# In[300]:

import numpy as np
np.unique(by_day.index.year)


# In[312]:

get_ipython().magic('matplotlib notebook')
import matplotlib.dates as dates

fig = plt.figure(figsize=(12,6), dpi=300)

plt.plot(by_day.index, by_day['TMIN'],
         c='#53868b', lw=0.8, zorder=2)

plt.plot(by_day.index, by_day['TMAX'],
         c='#ee7621', lw=0.8, zorder=2)

# days below 2005-2014 record min
plt.scatter(below_min_15.index, below_min_15['TMIN'],
            c='#ee7621', marker='v', s=15, zorder=3)

# days above 2005-2014 record max
plt.scatter(above_max_15.index, above_max_15['TMAX'],
            c='#53868b', marker='^', s=15, zorder=3)

plt.title('Historical Temperatures in Ann Arbor, MI')
plt.legend(['Daily Min', 'Daily Max', 'Record Low Broken in 2015', 'Record High Broken in 2015'],
           frameon=False, loc=0)

# Bar at 0 celcius for reference
plt.plot(by_day.index, [0 for i in range(len(by_day))],
         c='black', lw=0.5, zorder=1, alpha=0.5)
plt.plot(by_day.index, [record_min for i in range(len(by_day))],
         c='black', lw=0.5, zorder=1, alpha=0.5)
plt.plot(by_day.index, [record_max for i in range(len(by_day))],
         c='black', lw=0.5, zorder=1, alpha=0.5)

plt.ylim([-50, 50])
plt.xlabel('Day of the Year')
plt.ylabel('Temperature - C$^\circ$')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


plt.fill_between(by_day.index,
                 by_day['TMIN'], by_day['TMAX'],
                 facecolor='y',
                 alpha=0.5)

# Fixes from discussion board.
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)

