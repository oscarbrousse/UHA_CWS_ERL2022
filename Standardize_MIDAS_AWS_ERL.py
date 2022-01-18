# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:18:37 2021

@author: Oscar Brousse

INFO:
This script is run to extract variables of interest in a MIDAS AWS data file
It creates a standardized data structure used in the UHA analysis and in the script MIDAS_DataTreatment_and_WindRoses_ERL.py
"""

import pandas as pd
import os.path

city = 'London'
years = [2015, 2016, 2017, 2018, 2019, 2020]
datadir = '' + city + '/'  ### Directory where MIDAS stations has been downloaded
savedir = '' + city + '/Filtered/' ### Directory where the newly organized data will be stored



def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

list_of_weather_stations_f = find_csv_filenames(datadir)
f_info_name_out = 'MIDAS_MetaData_' + city + '.csv'
info_length = 279
var_names_loc = info_length + 1 
var_interest = ['id', 'air_temperature', 'wind_direction', 'wind_speed', 'wind_speed_unit_id', 'rltv_hum']

for yr in years:
    list_of_weather_stations_f_year = list(filter(lambda x: x.endswith(str(yr) + '.csv'), 
                                                  list_of_weather_stations_f))
    period_interest = pd.date_range(start='01-01-' + str(yr), end='01-01-' + str(yr + 1), 
                           freq='1H', closed='left')
    
    for aws in list_of_weather_stations_f_year:
        info_aws = pd.read_csv(datadir + aws, skiprows=0, nrows=info_length, index_col=False, header=None)
        data_aws = pd.read_csv(datadir + aws, skiprows=var_names_loc, index_col=0, nrows=len(period_interest))
        if data_aws.index[-1] == 'end data':
            data_aws = data_aws.drop('end data')
        data_aws = data_aws.set_index(pd.DatetimeIndex(data_aws.index))
        
        # We just subsample the variables of interest and aggregate the data over a standardized year-hourly time scale
        data_aws_new = pd.DataFrame(index=period_interest)
        data_aws_new = pd.concat([data_aws_new,data_aws[var_interest]], axis=1)
        data_aws_new['height'] = info_aws.iloc[14,2]
        data_aws_new['MIDAS_id'] = info_aws.iloc[12,2]
        data_aws_new['name'] = info_aws.iloc[10,2].capitalize()
        data_aws_new['lon'] = info_aws.iloc[13,3]
        data_aws_new['lat'] = info_aws.iloc[13,2]
        
        data_aws_new.to_csv(savedir + data_aws_new['name'][0] + '_' + str(yr) + '.csv')

