# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 11:09:19 2021

@author: Oscar Brousse

INFO:
This script has to be run once all the Netatmo data has been filtered
It will prepare all the data suitable for the UHA calculation and analyses done in NetAtmo_UHI_per_MIDAS_Wind_ERL.py
In particular it filters out data that hasn't passed the M4 quality check and prepares a file where all spatial info per CWS is stored'
"""

import pandas as pd
import numpy as np
import xarray as xr
import os.path
from osgeo import gdal

######################
## DEFINE CONSTANTS ##
######################

### Change according to your requirements

### Chose the name of your city of interest
city = 'London'
year = [2015, 2016, 2017, 2018, 2019, 2020]
datadir_MIDAS = '' ### Directory of the MIDAS weather station used for wind speed and orientation measurements
datadir_NA = '' ### Directory of the freshly downloaded Netatmo data
savedir = datadir_NA + 'Figures/'

NaN_tolerance = 0.9  ### Get rid of Netatmo data with less than 10% of measurements for the whole period.

### Chose the upper and lower pixels of your domain 
### (better to have it slightly bigger than the domain used for Netatmo data collection)
    
ulon = 1.3
llon = -1.9
ulat = 52.4
llat = 50.5

date_range = pd.date_range(start='01-01-2015 00:30:00+00:00', end='01-01-2021 00:30:00+00:00', 
                           freq='1H', closed='left')

f_list_stat_csv = 'List_Netatmo_stations_' + city + '_2015-2020.csv'
list_of_cws = pd.read_csv(datadir_NA + f_list_stat_csv)

def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

list_of_aws = find_csv_filenames(datadir_MIDAS)

### LCZ Map

### Directory of the LCZ map downloaded on the WUDAPT website
datadir_EU_LCZ = ''

if ((os.path.isfile(datadir_NA + 'WUDAPT_EU_LCZ_' + city + '.tif')) & 
    (os.path.isfile(datadir_NA + 'WUDAPT_EU_LCZ_' + city + '_LAEA.tif'))):
    print ("Clipped TIFF files already exist")
else:
    ### Clip the EU map to the desired domain extension if not already existing
    TranslateOptions_Clip = gdal.TranslateOptions(
        gdal.ParseCommandLine("-projwin " + str(llon) + " " + 
                              str(ulat) + " " + str(ulon) + " " + str(llat)))    
    LCZ_EU_WGS84 = gdal.Open(datadir_EU_LCZ + 'EU_LCZ_map_WGS84.tif')
    gdal.Translate(datadir_NA + 'WUDAPT_EU_LCZ_' + city + '.tif', 
                   LCZ_EU_WGS84,
                   options = TranslateOptions_Clip)
    cropped_LCZ_city_WGS84 = gdal.Open(datadir_NA + 'WUDAPT_EU_LCZ_' + city + '.tif')
    WarpOptions_LAEA = gdal.WarpOptions(gdal.ParseCommandLine("-t_srs EPSG:3035"))
    gdal.Warp(datadir_NA + 'WUDAPT_EU_LCZ_' + city + '_LAEA.tif', cropped_LCZ_city_WGS84, format = 'GTiff',
              options = WarpOptions_LAEA)


LCZ_Tiff_LAEA = xr.open_rasterio(datadir_NA + 'WUDAPT_EU_LCZ_' + city + '_LAEA.tif')
LCZ_Tiff  = xr.open_rasterio(datadir_NA + 'WUDAPT_EU_LCZ_' + city + '.tif')
LCZ = LCZ_Tiff.variable.data[0]
lat_LCZ = LCZ_Tiff.y.values
lon_LCZ = LCZ_Tiff.x.values
    
def find_nearest(latitudes_LCZ, lat_station, longitudes_LCZ, lon_station):
    idx_lat = (np.abs(latitudes_LCZ-lat_station)).argmin()
    idx_lon = (np.abs(longitudes_LCZ-lon_station)).argmin()
    return idx_lat, idx_lon
    
######################
##  DATA TREATMENT  ##
######################

#######
### NetAtmo
#######

dates_to_conc = []
for yr in range(2015,2021):
    for mon in range(1,13):
        date_tmp = str(yr) + '-{0:02d}'.format(mon)
        print(date_tmp)
        dates_to_conc.append(date_tmp)

for month in dates_to_conc:
    f_data_csv = 'Netatmo_' + city + '_' + month + '_filt.csv'    
  
    df_data = pd.read_csv(datadir_NA + f_data_csv)
    df_data_filtered = df_data[(df_data['m1'] == True) & (df_data['m2'] == True) & (df_data['m3'] == True) & (df_data['m4'] == True)
                                ]
    
    ### Comment out for a much more restrictive filtering.
    # df_data_filtered = df_data[(df_data['m1'] == True) & (df_data['m2'] == True) & (df_data['m3'] == True) & (df_data['m4'] == True)
    #                             & (df_data['o1'] == True) & (df_data['o2'] == True) & (df_data['o3'] == True)]
    
    df_data_filtered = df_data_filtered[(df_data_filtered['lon'] > llon) & (df_data_filtered['lon'] < ulon)
                                    & (df_data_filtered['lat'] > llat) & (df_data_filtered['lat'] < ulat)]
    
    ID_stations  = list(dict.fromkeys(df_data_filtered['p_id']))
    lon_NetAtmo = list(dict.fromkeys(df_data_filtered['lon']))
    lat_NetAtmo = list(dict.fromkeys(df_data_filtered['lat']))
    dates = list(dict.fromkeys(df_data_filtered['time']))
    
    temp_by_ID_df = pd.DataFrame(index=pd.to_datetime(dates))
    spat_attributes_NetAtmo_filt = pd.DataFrame(index=['Lon', 'Lat', 'Height'])

    for ids in ID_stations:
        lon_id = np.float32(list(dict.fromkeys(df_data_filtered['lon'][df_data_filtered['p_id']==ids])))
        lat_id = np.float32(list(dict.fromkeys(df_data_filtered['lat'][df_data_filtered['p_id']==ids])))
        height_id = np.float32(list(dict.fromkeys(df_data_filtered['z'][df_data_filtered['p_id']==ids])))
        spat_attributes_NetAtmo_filt[ids] = [lon_id[0], lat_id[0], height_id[0]]
        dates_tmp = list(dict.fromkeys(df_data_filtered['time'][df_data_filtered['p_id']==ids]))
        temperatures = pd.DataFrame(data=np.array(df_data_filtered['ta'][df_data_filtered['p_id']==ids]), 
                                    index=pd.to_datetime(dates_tmp),
                                    columns=[ids])
        temp_by_ID_df=pd.concat([temp_by_ID_df,temperatures], axis=1)
    
    temp_by_ID_df.to_csv(datadir_NA + 'Netatmo_' + city + '_' + month + '_filt_temp_by_ID.csv')
    # if month == dates_to_conc[0]:
    #     break
    del temp_by_ID_df, df_data, df_data_filtered


LCZ_NetAtmo = []
for i in range(len(lon_NetAtmo)):
    idy, idx = find_nearest(lat_LCZ, lat_NetAtmo[i], lon_LCZ, lon_NetAtmo[i])
    LCZ_NetAtmo.append(LCZ[idy, idx])

### Add the LCZ per NetAtmo CWS

spat_attributes_NetAtmo_filt.loc['LCZ'] = LCZ_NetAtmo

### Save the spatial attributes of each NetAtmo CWS
spat_attributes_NetAtmo_filt.to_csv(datadir_NA + 'List_Netatmo_stations_' + city + '_Filt.csv')
