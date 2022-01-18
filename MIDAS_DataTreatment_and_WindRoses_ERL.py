# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:13:41 2021

@author: Oscar Brousse

INFO:
This script has to be run once the MIDAS wind data has been downloaded and standardized
It works for one station only and would have to be updated for working with multiple stations
The script can get rid of hours when the wind is changing its orientation and keep only hours
when the wind orientation hasn't changed for 3 hours'
"""

import pandas as pd
import numpy as np
# import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import windrose
from datetime import datetime

### Change below as necessary

### Chose the name of your city of interest
city = 'London'
years = [2015, 2016, 2017, 2018, 2019, 2020]
datadir_MIDAS = ''  ### Directory where the MIDAS data has been downloaded
aws_name = 'Heathrow'
savedir = datadir_MIDAS + 'Figures/'
filt_quad = True ### Use to filter out hours when winds are not oriented in the same upwind quadrant for at least 3 hours

df = pd.read_csv(datadir_MIDAS + aws_name + '_' + str(years[0]) + '.csv', index_col=0)
for yr in years[1::]:
    df_tmp = pd.read_csv(datadir_MIDAS + aws_name + '_' + str(yr) + '.csv',  index_col=0)
    df = df.append(df_tmp)
    del df_tmp
df = df.set_index(pd.DatetimeIndex(df.index))

wind_dir = [0,90,180,270] ### Wind orientation thresholds (0 = North, 90 = East, 180 = South and 270 = West; see https://www.metoffice.gov.uk/binaries/content/assets/metofficegovuk/pdf/research/library-and-archive/library/publications/factsheets/factsheet_17-observations.pdf)
wind_dir_name = ['NE', 'SE', 'SW', 'NW'] ### Wind orientation classes

### Normalize wind speed in m/s

df['normalized_wind_speed'] = df.wind_speed
df['normalized_wind_speed'][df.wind_speed_unit_id == 4] = df.wind_speed*0.51444444

### Filter hours when wind direction is not in the same quadrant for at least 3 hours

if filt_quad == True:

    df['Wind_Quadrant'] = np.nan
    for wd_i in range(len(wind_dir)):
        if (wd_i == len(wind_dir) - 1):
            df.Wind_Quadrant = df.Wind_Quadrant.mask(df.wind_direction >= wind_dir[wd_i], wind_dir_name[wd_i])
        elif (wd_i != len(wind_dir) - 1):
            df.Wind_Quadrant = df.Wind_Quadrant.mask(((df.wind_direction >= wind_dir[wd_i]) & 
               (df.wind_direction < wind_dir[wd_i + 1])), wind_dir_name[wd_i])
    
    for t in range(2,len(df),1):
        if (df.Wind_Quadrant[t] != df.Wind_Quadrant[t-1]) | (df.Wind_Quadrant[t] != df.Wind_Quadrant[t-2]):
            df.normalized_wind_speed[t] = np.nan
            df.wind_direction[t] = np.nan

df[['normalized_wind_speed', 'wind_direction']] = df[['normalized_wind_speed', 'wind_direction']].mask(
                                                                                        df.normalized_wind_speed == 0)
### Print how many observations above 9 m/s
df.normalized_wind_speed[df.normalized_wind_speed > 9.0].count() / df.normalized_wind_speed.count() * 100

### Print how many observations in each threshold
for i in range(0,12,3):
    if i < 9:
        print(np.around(df.normalized_wind_speed[(df.normalized_wind_speed > i) & (df.normalized_wind_speed < i+3)].count() /
        df.normalized_wind_speed.count() * 100, 
        decimals = 2))
    else:
        print(np.around(df.normalized_wind_speed[df.normalized_wind_speed > i].count() / 
                        df.normalized_wind_speed.count() * 100, 
              decimals = 2))

############
### Plot wind roses
############

### Fig 1 : Hourly data

col = 3 
row = 4
fig, ax = plt.subplots(row, col, figsize = (12,4.5*row), subplot_kw=dict(projection="windrose"))
ax = ax.flatten()
rad_ticks = np.linspace(0,25,5,endpoint=True)
min_val, max_val = (0.2,1)
n = 4
orig_cmap = plt.cm.Greys
colors = orig_cmap(np.linspace(min_val, max_val, n))
cmap = clrs.LinearSegmentedColormap.from_list("mycmap", colors)
bins = np.arange(0, 12, 3)

for mon in list(dict.fromkeys(df.index.month)):
    df_mon = df[df.index.month == mon]
    mon_name = datetime.strptime(str(mon), "%m").strftime("%B")
    im = ax[mon-1].bar(df_mon.wind_direction, df_mon.normalized_wind_speed, bins=np.arange(0, 12, 3), nsector = 16,
            normed=True, opening=1, edgecolor='white', cmap = cmap)
    ax[mon-1].set_xticklabels(['N', 'NE',  'E', 'SE', 'S', 'SW','W', 'NW'])
    ax[mon-1].set_theta_zero_location('N')
    ax[mon-1].set_theta_direction(-1)
    ax[mon-1].set_rticks(rad_ticks)
    ax[mon-1].set_title(mon_name, color = 'dimgrey', fontsize = 12, y=1.1)
    del df_mon
    
for ax_i in ax:
    ax_i.tick_params(axis='y',which='both',left=False,right=False,labelright=False)
    ax_i.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)

ax[-3].set_yticklabels(map(str, rad_ticks))
ax[-3].set_ylabel('[%]', rotation=0)
ax[-3].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)
ax[-3].set_rlabel_position(247.5)
ax[-3].yaxis.set_label_coords(-0.05, 0.30)

ax[-2].legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5,-0.3))

if filt_quad == True:
    fig.savefig(savedir + 'WindRose_Heathrow_Hourly_2015-2020_Filt.png', dpi=600)
    fig.savefig(savedir + 'WindRose_Heathrow_Hourly_2015-2020_Filt.pdf')
else:
    fig.savefig(savedir + 'WindRose_Heathrow_Hourly_2015-2020.png', dpi=600)
    fig.savefig(savedir + 'WindRose_Heathrow_Hourly_2015-2020.pdf')



