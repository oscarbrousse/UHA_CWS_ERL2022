# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 14:16:52 2021

@author: Oscar Brousse

INFO:
This script is used to test how often prevailing winds are coming from the same quadrant in multiple AWS locations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs


#########################
###    ATTRIBUTES     ###
#########################

### City of interest

city = 'London'

bbox_llat = 51.1
bbox_ulat = 51.9
bbox_llon = -0.72
bbox_ulon = 0.48

quadrants = {'NE': [(bbox_llon + bbox_ulon)/2, (bbox_llat + bbox_ulat)/2, bbox_ulon, bbox_ulat],
             'SE': [(bbox_llon + bbox_ulon)/2, bbox_llat, bbox_ulon, (bbox_llat + bbox_ulat)/2],
             'SW': [bbox_llon, bbox_llat, (bbox_llon + bbox_ulon)/2, (bbox_llat + bbox_ulat)/2],
             'NW': [bbox_llon, (bbox_llat + bbox_ulat)/2, (bbox_llon + bbox_ulon)/2, bbox_ulat]}

### Wind classes and plotting information

breeze_classes = [0,3,6,9]  ### In meters per second
breeze_y_classes = [0.3,0.1,-0.1,-0.3] ### Y axis coordinates
breezes = ['Calm or Light Breeze', 'Gentle to Moderate Breeze', 'Moderate to Fresh Breeze', 'Strong Breeze'] ### Classes
wind_dir = [0,90,180,270] ### Wind orientation thresholds (0 = North, 90 = East, 180 = South and 270 = West; see https://www.metoffice.gov.uk/binaries/content/assets/metofficegovuk/pdf/research/library-and-archive/library/publications/factsheets/factsheet_17-observations.pdf)
wind_dir_name = ['NE', 'SE', 'SW', 'NW'] ### Wind orientation classes

## Modulable years and month for Had-UK calculations
startyear = '2015'
startmon = '01'
startday = '01'
endyear = '2021'
endmon = '01'
endday = '01'

## Exact dates for time slicing in Xarray
startdate = startyear + '-' + startmon + '-' + startday
enddate = endyear + '-' + endmon + '-' + endday

## List of covered dates for plotting labels
dates_list = [d.strftime('%Y-%m-%d') for d in pd.date_range(startdate, enddate, freq='1d').to_list()]
years = [2015, 2016, 2017, 2018, 2019, 2020]


#########################
### AWS OBSERVATIONS  ###
#########################

datadir_MIDAS = '' + city + '/Filtered/'  ### Directory where MIDAS standardized data is located
### Adapt the list of stations name depending on the location
MIDAS_wnd_aws = ['Heathrow', 'Kenley-airfield', 'Kew-gardens', 'Northolt']

for aws in MIDAS_wnd_aws:
    tmp_df = pd.read_csv(datadir_MIDAS + aws + '_' + str(years[0]) + '.csv', index_col=0)
    for yr in years[1::]:
        tmp_df_y = pd.read_csv(datadir_MIDAS + aws + '_' + str(yr) + '.csv',  index_col=0)
        tmp_df = tmp_df.append(tmp_df_y)
        del tmp_df_y
    tmp_df = tmp_df.set_index(pd.DatetimeIndex(tmp_df.index))
    if aws == MIDAS_wnd_aws[0]:
        df_wind_comp = pd.DataFrame(index = tmp_df.index)
    ### Normalize wind speed in m/s
    
    tmp_df['normalized_wind_speed'] = tmp_df.wind_speed
    tmp_df['normalized_wind_speed'][tmp_df.wind_speed_unit_id == 4] = tmp_df.wind_speed*0.51444444
    
    ### Filter hours where wind direction is not in the same quadrant for at least 3 hours
    
    tmp_df['Wind_Quadrant'] = np.nan
    for wd_i in range(len(wind_dir)):
        if (wd_i == len(wind_dir) - 1):
            tmp_df.Wind_Quadrant = tmp_df.Wind_Quadrant.mask(tmp_df.wind_direction >= wind_dir[wd_i], wind_dir_name[wd_i])
        elif (wd_i != len(wind_dir) - 1):
            tmp_df.Wind_Quadrant = tmp_df.Wind_Quadrant.mask(((tmp_df.wind_direction >= wind_dir[wd_i]) & 
               (tmp_df.wind_direction < wind_dir[wd_i + 1])), wind_dir_name[wd_i])
    
    for t in range(2,len(tmp_df),1):
        if (tmp_df.Wind_Quadrant[t] != tmp_df.Wind_Quadrant[t-1]) | (tmp_df.Wind_Quadrant[t] != tmp_df.Wind_Quadrant[t-2]):
            tmp_df.normalized_wind_speed[t] = np.nan
            tmp_df.wind_direction[t] = np.nan
    
    tmp_df[['normalized_wind_speed', 'wind_direction']] = tmp_df[['normalized_wind_speed', 'wind_direction']].mask(
                                                                                            tmp_df.normalized_wind_speed == 0)
    df_wind_comp[[aws + '_nws', aws + '_wdir', aws + '_quad']] = tmp_df[['normalized_wind_speed', 'wind_direction', 'Wind_Quadrant']]
    del tmp_df
### Count how many times the wind is blowing from the same quadrant across all MIDAS stations in the domain

sub_eq_df = df_wind_comp.filter(regex='_quad$',axis=1).copy()
### Test within quadrant (could test within certain wind direction or wind speeds ranges)
test_eq_df = sub_eq_df.eq(sub_eq_df.iloc[:, 0], axis=0).all(1)
print(test_eq_df[test_eq_df == True].count() / test_eq_df.count())
