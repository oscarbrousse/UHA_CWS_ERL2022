# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:56:17 2021

@author: Oscar Brousse

INFO:
This script is the first to run prior to Collect_NetAtmo_Data_ERL.py
It is adapted from the script by Vanter et al. (2020)
It compiles a list of available stations in the domain of interest
"""

import patatmo as patatmo
import pandas as pd
import numpy as np
from time import sleep

### Change as necessary

city = 'London'
datadir = ''
date_range = pd.date_range(start='01-01-2015 00:30:00+00:00', end='01-01-2021 00:30:00+00:00', 
                           freq='1H', closed='left')

# your patatmo connect developer credentials
credentials = {
    "password":"",
    "username":"",
    "client_id":"",
    "client_secret":""
}
# create an api client
client = patatmo.api.client.NetatmoClient()

# tell the client's authentication your credentials
client.authentication.credentials = credentials

## Define the limits of the domain of interest. Default is London domain for the ERL paper.

ulon = 1.3
llon = -1.9
ulat = 52.4
llat = 50.5

lon_tiles = np.append(np.around(np.arange(llon, ulon, 0.2), decimals = 2), np.array(ulon))
lat_tiles = np.append(np.around(np.arange(llat, ulat, 0.2), decimals = 2), np.array(ulat))

list_all_stations = pd.DataFrame(columns=['Lon', 'Lat', 'ID','moduleID', 'index']) 
for x in range(len(lon_tiles)-1):
    print(lon_tiles[x])
    for y in range(len(lat_tiles)-1):
        print(lat_tiles[y])
    
        region = {
            "lat_ne" : lat_tiles[y+1],
            "lat_sw" : lat_tiles[y],
            "lon_ne" : lon_tiles[x+1],
            "lon_sw" : lon_tiles[x],
        }
        # issue the API request
        output = client.Getpublicdata(region = region, filter=True)
        print(len(output.response["body"]))
        
        ### Acquiring the names and places of all stations available in the domain
        
        stations = output.response["body"]
        if len(stations) == 0:
            print('No Stations in this Grid')
            continue

        df = pd.DataFrame(columns=['Lon', 'Lat', 'ID','moduleID', 'index'])
        
        for i in range(0,len(stations)):
          lon = stations[i]["place"]["location"][0]
          lat = stations[i]["place"]["location"][1]
          device = stations[i]["_id"]
          module_id = tuple(stations[i]["measures"].keys())[0]
          new = pd.DataFrame(np.array([[lon, lat, device, module_id,i]]), 
                              columns=['Lon', 'Lat', 'ID','moduleID', 'index'])
          
          df = df.append(new)
        df_filt = df[(df['Lon'].astype('float32') > lon_tiles[x]) & (df['Lon'].astype('float32') < lon_tiles[x+1])
                      & (df['Lat'].astype('float32') > lat_tiles[y]) & (df['Lat'].astype('float32') < lat_tiles[y+1])]
        print(df_filt)

        list_all_stations = list_all_stations.append(df_filt)
            
        del df_filt, df
        sleep(100)

list_all_stations.to_csv(datadir + 'List_Netatmo_stations_' + city + '_2015-2020.csv')
