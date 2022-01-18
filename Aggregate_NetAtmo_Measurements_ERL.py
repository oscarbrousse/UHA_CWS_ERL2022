# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:36:04 2021

@author: Oscar Brousse

INFO:
This script has to be run once all the Netatmo data has been downloaded
It is used to create Netatmo monthly CSV suitable for the CrowdQC quality-check algorithm
It also aggregates each monthly CSV in one big CSV for the whole period of interest
"""

import pandas as pd
import numpy as np
import xarray as xr


city = 'London'
datadir = '' ### Data where the Netatmo data has been downloaded
f_srtm_data_tif = datadir + 'SRTMv4_30_' + city +'.tif' ### Data where a digital elevation model product has been downloaded
f_name_root = 'netatmo_output_' + city + '_'
f_name_out = 'Netatmo_Filter_Ready_' + city + '_'
Merge_All_DF = True   # Switch to create a merged dataset

dates_to_conc = []
for yr in range(2015,2021):
    for mon in range(1,13):
        date_tmp = str(yr) + '-{0:02d}'.format(mon)
        dates_to_conc.append(date_tmp)
### This last step is just added to check if unreliable date was added during download
dates_to_conc.append('2021-01')  

## We will create a new dataframe in the format required by Meier et al. (2018) to clean unrealistic NetAtmo measurements

### ======================================= ###
### REORGANIZING AND FILTERING NETATMO DATA ###
### ======================================= ###

## First we want to get rid of stations that do not cover a certain period of the period of interest (eg. 30% of it)
NaN_tolerance = 0 # Ratio from 0 to 1, 1 means 0 % tolerance and 0 means 100% tolerance

for t in range(len(dates_to_conc) - 1):
    print(dates_to_conc[t])
    act_yr  = dates_to_conc[t][:4]
    act_mon = dates_to_conc[t][-2:]
    nxt_yr  = dates_to_conc[t+1][:4]
    nxt_mon = dates_to_conc[t+1][-2:]
    period_interest = pd.date_range(start = act_mon + '-01-' + act_yr + ' 00:30:00+00:00', 
                                    end = nxt_mon + '-01-' + nxt_yr + ' 00:30:00+00:00', 
                                    freq = '1H', closed = 'right')
    new_dataframe = pd.read_csv(datadir + f_name_root + dates_to_conc[t] + '.csv')
    new_dataframe = new_dataframe.sort_values(['ID', 'time'])
    
    ### For some reason some stations have the first time step of the next month: Delete 
    new_dataframe.drop(new_dataframe[new_dataframe['time'] == str(period_interest[-1])].index, 
                                       inplace=True)
    
    ## Gets rid of the dupplicated timestamps at some location due to change in stations' ID
    new_dataframe = new_dataframe[new_dataframe['time'] != new_dataframe['time'].shift(-1)]
    
    ID_stations  = list(dict.fromkeys(new_dataframe['ID']))
    dates = list(dict.fromkeys(new_dataframe['time']))
    pot_loc = new_dataframe.groupby(['Lat','Lon']).size().reset_index().rename(columns={0:'count'})
    # bad_loc = pot_loc[pot_loc.Lon.duplicated(keep=False)]
    # bad_df = new_dataframe[(new_dataframe['Lon'].isin(bad_loc['Lon'])) & 
    #                        (new_dataframe['Lat'].isin(bad_loc['Lat']))]
    
    temp_by_ID_df = pd.DataFrame(index=dates)
    for ids in ID_stations:
        dates_tmp = list(dict.fromkeys(new_dataframe['time'][new_dataframe['ID']==ids]))
        temperatures = pd.DataFrame(data=np.array(new_dataframe['Temperature'][new_dataframe['ID']==ids]), index=dates_tmp,
                                    columns=[ids])
        temp_by_ID_df=pd.concat([temp_by_ID_df,temperatures], axis=1)
     
    ## Suppress only stations with a minimum amount of NaN
    filter_df = temp_by_ID_df.dropna(axis = 'columns', thresh=round(float(len(temp_by_ID_df))*NaN_tolerance))
    
    ### ====================================== ###
    ### OBTAIN THE ALTITUDE BASED ON SRTM DATA ###
    ### ====================================== ###
    
    ## I use XArray to obtain lat and lon coordinates as this eases substantially the processing
    ## but there are many other ways to do it
    
    SRTM_tif = xr.open_rasterio(f_srtm_data_tif)
    
    lat_SRTM = SRTM_tif.y.values
    lon_SRTM = SRTM_tif.x.values
    z_altitude_SRTM = SRTM_tif.variable.data[0]
    
    def find_nearest(latitudes_SRTM, lat_station,longitudes_SRTM, lon_station):
        idx_lat = (np.abs(latitudes_SRTM-lat_station)).argmin()
        idx_lon = (np.abs(longitudes_SRTM-lon_station)).argmin()
        return idx_lat, idx_lon
    
    z_altitude_NetAtmo = []
    IDs_lonlat = []
    for i in range(len(pot_loc)):
        idy, idx = find_nearest(lat_SRTM, pot_loc['Lat'].iloc[i], lon_SRTM, pot_loc['Lon'].iloc[i])
        z_altitude_NetAtmo.append(z_altitude_SRTM[idy, idx])
        
        # Sometimes multiple IDs are at the same location. 
        # We need to make sure we refer to them before restructuring the DataFrame
        IDs_tmp = list(dict.fromkeys(new_dataframe['ID'][(new_dataframe['Lat']== pot_loc['Lat'].iloc[i]) & 
                                                         (new_dataframe['Lon']== pot_loc['Lon'].iloc[i])]))
        IDs_lonlat.append(IDs_tmp)
    
        
    
    unvariant_attributes_df = pd.DataFrame({'IDs': IDs_lonlat,
                                            'Lon': list(pot_loc['Lon']),
                                            'Lat': list(pot_loc['Lat']),
                                            'Altitude': z_altitude_NetAtmo
                                            })
    
    ID_stations_filter = list(filter_df.columns)
        
    CrowdQC_filter_rdy_var = ['p_id','time','ta','lon','lat','z']
    CrowdQC_rdy_df = pd.DataFrame(columns=CrowdQC_filter_rdy_var)
    for ids in ID_stations_filter:
        tmp_df = pd.DataFrame(columns=CrowdQC_filter_rdy_var)
        tmp_df['ta'] = np.array(filter_df[ids])
        tmp_df['p_id'] = ids
        tmp_df['time'] = filter_df.index
        # Locates the right ID through multiple IDs at the same location if any
        sub_unvariant_df = unvariant_attributes_df.IDs.apply(lambda x: any(item for item in [ids] if item in x))
        tmp_df['lon'] = float(unvariant_attributes_df['Lon'][sub_unvariant_df])
        tmp_df['lat'] = float(unvariant_attributes_df['Lat'][sub_unvariant_df])
        tmp_df['z'] = float(unvariant_attributes_df['Altitude'][sub_unvariant_df])
        
        CrowdQC_rdy_df = pd.concat([CrowdQC_rdy_df, tmp_df], axis=0)

                
    CrowdQC_rdy_df.to_csv(datadir + f_name_out + dates_to_conc[t] + '.csv', index=False, sep = '|')
    del CrowdQC_rdy_df, tmp_df, new_dataframe

if Merge_All_DF == True:
    CrowdQC_rdy_Merged_df = pd.DataFrame(columns=CrowdQC_filter_rdy_var)
    for t in range(len(dates_to_conc) - 1):
        print(dates_to_conc[t])
        tmp_df_rdy = pd.read_csv(datadir + f_name_out + dates_to_conc[t] + '.csv', sep = '|')
        CrowdQC_rdy_Merged_df = pd.concat([CrowdQC_rdy_Merged_df, tmp_df_rdy], axis = 0, ignore_index=True)
        del tmp_df_rdy
    CrowdQC_rdy_Merged_df.to_csv(datadir + f_name_out + dates_to_conc[0] + '-' + dates_to_conc[-2] + '.csv', 
                                 index=False, sep = '|')
