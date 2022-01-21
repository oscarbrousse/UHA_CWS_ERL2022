# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:56:17 2021

@author: Oscar Brousse

INFO:
This script is the second to run after Collect_NetAtmo_Data_Tile_ERL.py
It is adapted from the script by Vanter et al. (2020)
It will download hourly measurements of Citizen Weather Stations repertoriated in the list compiled in the prior script
"""

import patatmo as patatmo
import pandas as pd
from time import sleep
import datetime

### Change as necessary

city = 'London'
datadir = ''
date_range = pd.date_range(start='02-01-2019 00:30:00+00:00', end='03-01-2019 00:30:00+00:00', 
                           freq='1H', closed='left')
list_stations = pd.read_csv(datadir + 'List_Netatmo_stations_London_WRF_2015-2020.csv')

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

### Gather measurements over a certain period for all stations

epoch = datetime.datetime.utcfromtimestamp(0).replace(tzinfo=datetime.timezone.utc)

def unix_time_millis(dt):
    return (dt - epoch).total_seconds()

### ============== ###
### THIS IS A TEST ###
### ============== ###

### To be used if you want to test that the server is online and that connections are possible

# startDate_dt = datetime.datetime(2018, 6, 21, 0, 0, 0, 0, tzinfo=datetime.timezone.utc)
# endDate_dt = datetime.datetime(2018, 7, 1, 0, 0, 0, 0, tzinfo=datetime.timezone.utc)
# startDate = str(int(unix_time_millis(startDate_dt)))
# endDate =  str(int(unix_time_millis(endDate_dt)))

# ### To do the test randomly change the index over few numbers. Possible no data for certain devices.
# device_id = list_stations.iloc[4]['ID']
# module_id = list_stations.iloc[4]['moduleID']

# test = client.Getmeasure(device_id=device_id,
#                         module_id=module_id,
#                         type=['Temperature'],
#                         scale='1hour',
#                         date_begin=startDate,
#                         date_end=endDate).dataframe()

### ============== ###
### LOOP PER MONTH ###
### ============== ###

years = list(dict.fromkeys(date_range.year))
months = list(dict.fromkeys(date_range.month))
for yr in years:
    for mon in months:
        if mon == 12:
            follow_mon = 1
            follow_yr  = yr + 1
        else:
            follow_mon = mon + 1
            follow_yr  = yr
        startDate_dt = datetime.datetime(yr, mon, 1, 0, 0, 0, 0, tzinfo=datetime.timezone.utc)
        endDate_dt = datetime.datetime(follow_yr, follow_mon, 1, 0, 0, 0, 0, tzinfo=datetime.timezone.utc)
            
        startDate = str(int(unix_time_millis(startDate_dt)))
        endDate =  str(int(unix_time_millis(endDate_dt)))
        # Define start and end date for collection

        dfNetatmo_big = pd.DataFrame()
        step = 200
        for y in range (0,len(list_stations),step):
          dfNetatmo = pd.DataFrame()
          for x in range(y,y+step):
            errorcount = 0
            if x == len(list_stations):
              break
            while True:
              try:
                sleep(8) # need to play around with this to get optimal sleep time (check Netatmo website for downloading limits per hour)
                lat = list_stations.iloc[x]["Lat"]
                lon = list_stations.iloc[x]["Lon"]
                device_id = list_stations.iloc[x]["ID"]
                module_id = list_stations.iloc[x]["moduleID"]
                index = x
        
                payload = client.Getmeasure(device_id=device_id,module_id=module_id,
                                            type=['Temperature'],
                                            scale='1hour',
                                            date_begin=startDate,
                                            date_end=endDate).dataframe()
        
                while payload is None:
                  print('waiting for payload') # sometimes the request to Netatmo servers fails on first attempt
                  sleep(5)
                print(x)
        
                payload['ID'] = device_id
                payload['index'] = index
                payload['Lat'] = lat
                payload['Lon'] = lon
                dfNetatmo = dfNetatmo.append(payload)
                
                break
                
              except BaseException:
                print('Error!!')
                errorcount = errorcount + 1
                print(errorcount)
                sleep(10)
                if (errorcount <3): # skip station if more than three errors
                  continue
                else:
                  break
          print('The temporary dataframe contains ' + str(len(dfNetatmo)) + ' measurements')
          dfNetatmo_big = pd.concat([dfNetatmo_big, dfNetatmo])
          print('The dataframe now contains ' + str(len(dfNetatmo_big)) + ' measurements')
          
          fileName = 'netatmo_output_' + city + '_' + str(yr) + '-{0:02d}'.format(mon) + '.csv'
          dfNetatmo_big.to_csv(datadir + fileName)
         
