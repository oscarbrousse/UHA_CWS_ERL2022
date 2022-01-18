# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 14:16:52 2021

@author: Oscar Brousse

INFO:
This script is where the UHA and urban heat analyses are done
The UHA is calculated in this script using both MIDAS and CWS measurements
Most of the plots and tables in the ERL paper are made here
Some information written in the manuscript are simply printed by the script
"""
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box
import numpy as np
# import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import cartopy.crs as ccrs
from fiona.crs import from_epsg
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from scipy import stats 
import matplotlib.colors as clrs
import geopy.distance
from matplotlib import markers
from matplotlib.path import Path

#########################
###    ATTRIBUTES     ###
#########################

### City of interest

city = 'London'
reshape_na = False ## Only use if working with NetCDF file. Switch to make a df out of NA data. Takes time and memory /!\

### Bounding box to study intra-urban temperature variability: 
###     - default (Gherkin): 51.1,-0.8 ; 51.9,0.6
###     - centered large (British Museum): 51.12,-0.73 ; 51.92,0.47
###     - centered large (Trafalgar Square): 51.1,-0.72 ; 51.9,0.48

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

### Which figures to plot (/!\ not the same number as in the ERL paper since supplementary figures are also included)

plot_fig1 = True
plot_fig2 = True
plot_fig3 = True
plot_fig4 = True
plot_fig5 = True
plot_fig6 = True
plot_fig7 = True
plot_fig8 = True
plot_fig9 = True
plot_tab1 = True
plot_tab2 = True

### Plotting variables
seasons = ['DJF', 'MAM', 'JJA', 'SON']

### Different colormaps for the AVG T, DTR and number of available measurements per CWS
min_val, max_val = (0.1,1)
n = 10

cmap_temp_orig = plt.cm.Reds
colors_temp = cmap_temp_orig(np.linspace(min_val, max_val, n))
cmap_temp = clrs.LinearSegmentedColormap.from_list("mycmap", colors_temp)
cmap_temp.set_bad('white', alpha=0)

cmap_dtr_orig = plt.cm.Greens
colors_dtr = cmap_dtr_orig(np.linspace(min_val, max_val, n))
cmap_dtr = clrs.LinearSegmentedColormap.from_list("mycmap", colors_dtr)
cmap_dtr.set_bad('white', alpha=0)

cmap_prc_orig = plt.cm.Greys
colors_prc = cmap_prc_orig(np.linspace(min_val, max_val, n))
cmap_prc = clrs.LinearSegmentedColormap.from_list("mycmap", colors_prc)
cmap_prc.set_bad('white', alpha=0)

cmap_uha = plt.cm.RdBu_r
cmap_uha.set_bad('white', alpha=0)

## Modulable years and months
startyear = '2015'
startmon = '01'
startday = '01'
endyear = '2021'
endmon = '01'
endday = '01'

## Exact dates for time slicing
startdate = startyear + '-' + startmon + '-' + startday
enddate = endyear + '-' + endmon + '-' + endday

## List of covered dates for plotting labels
dates_list = [d.strftime('%Y-%m-%d') for d in pd.date_range(startdate, enddate, freq='1d').to_list()]
years = [2015, 2016, 2017, 2018, 2019, 2020]

### LCZ color bars and names

lcz_colors_dict =  {0:'#FFFFFF', 1:'#910613', 2:'#D9081C', 3:'#FF0A22', 4:'#C54F1E', 5:'#FF6628', 6:'#FF985E', 
                    7:'#FDED3F', 8:'#BBBBBB', 9:'#FFCBAB',10:'#565656', 11:'#006A18', 12:'#00A926', 
                    13:'#628432', 14:'#B5DA7F', 15:'#000000', 16:'#FCF7B1', 17:'#656BFA', 18:'#00ffff'}
cmap_lcz = mpl.colors.ListedColormap(list(lcz_colors_dict.values()))
lcz_classes = list(lcz_colors_dict.keys()); lcz_classes.append(19)
norm_lcz = mpl.colors.BoundaryNorm(lcz_classes, cmap_lcz.N)

lcz_labels = ['Mask', 'Compact High Rise: LCZ 1', 'Compact Mid Rise: LCZ 2', 'Compact Low Rise: LCZ 3', 
              'Open High Rise: LCZ 4', 'Open Mid Rise: LCZ 5', 'Open Low Rise: LCZ 6',
              'Lighweight Lowrise: LCZ 7', 'Large Lowrise: LCZ 8',
              'Sparsely Built: LCZ 9', 'Heavy Industry: LCZ 10',
              'Dense Trees: LCZ A', 'Sparse Trees: LCZ B', 'Bush - Scrubs: LCZ C',
              'Low Plants: LCZ D', 'Bare Rock - Paved: LCZ E', 'Bare Soil - Sand: LCZ F',
              'Water: LCZ G', 'Wetlands: LCZ W']
lcz_labels_dict = dict(zip(list(lcz_colors_dict.keys()),lcz_labels))

#########################
### AWS OBSERVATIONS  ###
#########################

datadir_MIDAS = '' + city + '/Filtered/'  ### Directory where MIDAS standardized data is located

df = pd.read_csv(datadir_MIDAS + 'Heathrow_' + str(years[0]) + '.csv', index_col=0)
for yr in years[1::]:
    df_tmp = pd.read_csv(datadir_MIDAS + 'Heathrow_' + str(yr) + '.csv',  index_col=0)
    df = df.append(df_tmp)
    del df_tmp
df = df.set_index(pd.DatetimeIndex(df.index))

### Normalize wind speed in m/s

df['normalized_wind_speed'] = df.wind_speed
df['normalized_wind_speed'][df.wind_speed_unit_id == 4] = df.wind_speed*0.51444444

### Filter hours where wind direction is not in the same quadrant for at least 3 hours

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

### Calculate daily averages 

df_day = df.resample('D').apply(np.nanmean)

#########################
### CWS OBSERVATIONS  ###
#########################

### General attributes per station
datadir_NA = ''  ### Directory where the filtered and quality-checked Netatmo data is located 
savedir = datadir_NA + 'Figures/'
spat_attributes_by_ID = pd.read_csv(datadir_NA + 'List_Netatmo_stations_' + city + '_2015-2020.csv', index_col = 3)
spat_attributes_by_ID = spat_attributes_by_ID.drop('Unnamed: 0', axis = 1)

### LCZ map

LCZ_Tiff  = xr.open_rasterio(datadir_NA + 'WUDAPT_EU_LCZ_' + city + '.tif')
LCZ = LCZ_Tiff.variable.data[0]
LCZ = np.nan_to_num(LCZ, nan = 0)
LCZ[LCZ == 0] = 17
LCZ_Urb = np.where(LCZ > 10, 0, 1)
lat_LCZ = LCZ_Tiff.y.values
lon_LCZ = LCZ_Tiff.x.values

### Count how many LCZ in each quadrant

def getFeatures(gdf):
        """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
        import json
        return [json.loads(gdf.to_json())['features'][0]['geometry']]

lcz_prop = []
for quad_i in range(4):
    quad_name = list(quadrants.keys())[quad_i]
    bbox_tmp = quadrants.get(quad_name)
    bbox = box(bbox_tmp[0], bbox_tmp[1], bbox_tmp[2], bbox_tmp[3])
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
    geo = geo.to_crs(crs=LCZ_Tiff.crs)
    coords = getFeatures(geo)
    LCZ_quad = LCZ_Tiff.rio.clip(coords)
    tmp_prop = [[lcz_labels_dict.get(lcz), np.around(np.sum(np.where(LCZ_quad.data[0,:,:] == lcz, 1, 0)) / 
                                                     np.sum(np.ones(np.shape(LCZ_quad.data[0,:,:]))) *
                                                     100, decimals = 2)] for lcz in range(1,17)
                if np.sum(np.where(LCZ_Tiff.data[0,:,:] == lcz, 1, 0)) > 0]
    lcz_prop.append([quad_name, tmp_prop])
    del tmp_prop, bbox_tmp

### Add the LCZ of each station
with rasterio.Env():
    lcz = rasterio.open(datadir_NA + 'WUDAPT_EU_LCZ_' + city + '.tif')
    srtm = rasterio.open(datadir_NA + 'SRTMv4_30_' + city +'.tif')
    xy = [(row.Lon, row.Lat) for index, row in spat_attributes_by_ID.iterrows()]
    sample_z = rasterio.sample.sample_gen(srtm, xy)
    sample = rasterio.sample.sample_gen(lcz, xy)
    spat_attributes_by_ID["LCZ"] = [x[0] for x in list(sample)]
    spat_attributes_by_ID["Altitude"] = [x[0] for x in list(sample_z)]


del lcz

spat_attributes_by_ID['LCZ_names'] = [lcz_labels_dict.get(x) for x in spat_attributes_by_ID['LCZ']]

### Aggregate monthly CSV file of temperatures per station

NA_filt_f = glob.glob(datadir_NA + "/Netatmo_London_*_filt_temp_by_ID.csv")

library = []

for filename in NA_filt_f:
    NA_df_mon = pd.read_csv(filename, index_col=0, header=0)
    library.append(NA_df_mon)
    del NA_df_mon

NA_filt_df = pd.concat(library, axis=0, ignore_index=True)
NA_filt_df = NA_filt_df.set_index(pd.DatetimeIndex(df.index))
for yr in years:
    NA_1yr_filt = (NA_filt_df[NA_filt_df.index.year == yr].T.count(axis=1) / 
                len(NA_filt_df[NA_filt_df.index.year == yr]) * 100)
    NA_filt_df.loc[NA_filt_df[NA_filt_df.index.year == yr].index, 
                   NA_1yr_filt[NA_1yr_filt < 80].index.to_list()] = np.nan
NA_filt_df = NA_filt_df.dropna(axis=1, how='all')

## Remove filtered stations and reorder index based on T2M df
spat_attributes_by_ID = spat_attributes_by_ID.reindex(NA_filt_df.columns)

### Normalize the temperatures by height against the averaged temperature at the mean height
NA_filt_df_norm = NA_filt_df + (0.0065 * (spat_attributes_by_ID.Altitude - spat_attributes_by_ID.Altitude.mean()))

NA_d_filt_df_avg = NA_filt_df_norm.resample("D").apply(np.mean)
NA_d_filt_df_dtr = NA_filt_df_norm.resample("D").apply(np.max) - NA_filt_df.resample("D").apply(np.min)

### Allocate statistics on averaged T2M and DTR for whole 6 years and per season
### Intra-urban variability without influence of elevation

## 6 years average
spat_attributes_by_ID["T2_AVG"] = NA_d_filt_df_avg.mean()
spat_attributes_by_ID["T2_DTR_AVG"] = NA_d_filt_df_dtr.mean()

## 6 years seasonal average (only used for plotting seasonal temperatures and DTR - no intra-urban comparison)
## DJF
spat_attributes_by_ID["T2_AVG_DJF"] = NA_d_filt_df_avg[(NA_d_filt_df_avg.index.month == 12) |
                                                       (NA_d_filt_df_avg.index.month == 1) |
                                                       (NA_d_filt_df_avg.index.month == 2)].mean()
spat_attributes_by_ID["T2_DTR_AVG_DJF"] = NA_d_filt_df_dtr[(NA_d_filt_df_dtr.index.month == 12) |
                                                           (NA_d_filt_df_dtr.index.month == 1) |
                                                           (NA_d_filt_df_dtr.index.month == 2)].mean()
## MAM
spat_attributes_by_ID["T2_AVG_MAM"] = NA_d_filt_df_avg[(NA_d_filt_df_avg.index.month == 3) |
                                                       (NA_d_filt_df_avg.index.month == 4) |
                                                       (NA_d_filt_df_avg.index.month == 5)].mean()
spat_attributes_by_ID["T2_DTR_AVG_MAM"] = NA_d_filt_df_dtr[(NA_d_filt_df_dtr.index.month == 3) |
                                                           (NA_d_filt_df_dtr.index.month == 4) |
                                                           (NA_d_filt_df_dtr.index.month == 5)].mean()

## JJA
spat_attributes_by_ID["T2_AVG_JJA"] = NA_d_filt_df_avg[(NA_d_filt_df_avg.index.month == 6) |
                                                       (NA_d_filt_df_avg.index.month == 7) |
                                                       (NA_d_filt_df_avg.index.month == 8)].mean()
spat_attributes_by_ID["T2_DTR_AVG_JJA"] = NA_d_filt_df_dtr[(NA_d_filt_df_dtr.index.month == 6) |
                                                           (NA_d_filt_df_dtr.index.month == 7) |
                                                           (NA_d_filt_df_dtr.index.month == 8)].mean()

## SON
spat_attributes_by_ID["T2_AVG_SON"] = NA_d_filt_df_avg[(NA_d_filt_df_avg.index.month == 9) |
                                                       (NA_d_filt_df_avg.index.month == 10) |
                                                       (NA_d_filt_df_avg.index.month == 11)].mean()
spat_attributes_by_ID["T2_DTR_AVG_SON"] = NA_d_filt_df_dtr[(NA_d_filt_df_dtr.index.month == 9) |
                                                           (NA_d_filt_df_dtr.index.month == 10) |
                                                           (NA_d_filt_df_dtr.index.month == 11)].mean()


### For intra-urban analysis: 
### Keep only stations within a bounding box defined out of preliminary analyses (See Domain 1 analysis and Domain 2 definition)
### to avoid including stations influenced by the sea-breezes and focus on meaningful stations around Heathrow

NA_d_GL_df_dtr = NA_d_filt_df_dtr[spat_attributes_by_ID[(spat_attributes_by_ID.Lat > bbox_llat) &
                                                         (spat_attributes_by_ID.Lat < bbox_ulat) &
                                                         (spat_attributes_by_ID.Lon > bbox_llon) &
                                                         (spat_attributes_by_ID.Lon < bbox_ulon)].index]

NA_d_GL_df_avg = NA_d_filt_df_avg[spat_attributes_by_ID[(spat_attributes_by_ID.Lat > bbox_llat) &
                                                         (spat_attributes_by_ID.Lat < bbox_ulat) &
                                                         (spat_attributes_by_ID.Lon > bbox_llon) &
                                                         (spat_attributes_by_ID.Lon < bbox_ulon)].index]

NA_filt_GL_df = NA_filt_df_norm[spat_attributes_by_ID[(spat_attributes_by_ID.Lat > bbox_llat) &
                                                      (spat_attributes_by_ID.Lat < bbox_ulat) &
                                                      (spat_attributes_by_ID.Lon > bbox_llon) &
                                                      (spat_attributes_by_ID.Lon < bbox_ulon)].index]

spat_attributes_by_ID_GL = spat_attributes_by_ID[(spat_attributes_by_ID.Lat > bbox_llat) &
                                                 (spat_attributes_by_ID.Lat < bbox_ulat) &
                                                 (spat_attributes_by_ID.Lon > bbox_llon) &
                                                 (spat_attributes_by_ID.Lon < bbox_ulon)]

### Define in which quadrant the station is for urban heat advection calculation

def quadrant_loc(df):
    if ((df.Lat > bbox_llat) and (df.Lat < ((bbox_llat + bbox_ulat) / 2)) and
        (df.Lon > bbox_llon) and (df.Lon < ((bbox_llon + bbox_ulon) / 2))):
        return 'SW'
    elif ((df.Lat > bbox_llat) and (df.Lat < ((bbox_llat + bbox_ulat) / 2)) and
          (df.Lon > ((bbox_llon + bbox_ulon) / 2)) and (df.Lon < bbox_ulon)):
        return 'SE'
    elif ((df.Lat > ((bbox_llat + bbox_ulat) / 2)) and (df.Lat < bbox_ulat) and
          (df.Lon > ((bbox_llon + bbox_ulon) / 2)) and (df.Lon < bbox_ulon)):
        return 'NE'
    elif ((df.Lat > ((bbox_llat + bbox_ulat) / 2)) and (df.Lat < bbox_ulat) and
          (df.Lon > bbox_llon) and (df.Lon < bbox_ulon)):
        return 'NW'

def dist_calc(df):
    return geopy.distance.distance((df.Lon, df.Lat), ((bbox_llon + bbox_ulon) / 2, (bbox_llat + bbox_ulat) / 2)).km

### Attribute a quadrant to each CWS
spat_attributes_by_ID_GL['Quadrant'] = spat_attributes_by_ID_GL.apply(quadrant_loc, axis = 1)
### Calculate CWS distance to the center in km
spat_attributes_by_ID_GL['Distance_Center'] = spat_attributes_by_ID_GL.apply(dist_calc, axis = 1)

### Count proportion of each CWS in each LCZ per quadrant

lcz_frac_quad = [[lcz_labels_dict.get(lcz), 
                  np.around(spat_attributes_by_ID_GL[spat_attributes_by_ID_GL.LCZ == lcz].Quadrant.value_counts() / 
                  spat_attributes_by_ID_GL.Quadrant.value_counts() * 100, decimals = 2)] for lcz in range(1,17)
                 if len(spat_attributes_by_ID_GL[spat_attributes_by_ID_GL.LCZ == lcz]) > 0]
lcz_count_quad = [[lcz_labels_dict.get(lcz), spat_attributes_by_ID_GL[spat_attributes_by_ID_GL.LCZ == lcz].Quadrant.value_counts()] for lcz in range(1,17) 
                 if len(spat_attributes_by_ID_GL[spat_attributes_by_ID_GL.LCZ == lcz]) > 0]

##########################
### WIND REGIMES STUDY ###
##########################

### Count how many stations recorded specific wind regimes 
### Wind speeds are always higher than the bottom limit and lower than or equal to the upper limit of the class
### Wind directions are always higher than or equal to the bottom limit and lower than the upper limit of the quadrant

count_stations_per_days = []

for wd_i in range(len(wind_dir)):
    for bc_i in range(len(breeze_classes)):          
        if ((bc_i == len(breeze_classes) - 1) & (wd_i == len(wind_dir) - 1)):
            df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                        (df.wind_direction >= wind_dir[wd_i])].index.strftime("%Y-%m-%d %H:%M:%S") 
        elif ((bc_i == len(breeze_classes) - 1) & (wd_i != len(wind_dir) - 1)):
            df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                        (df.wind_direction >= wind_dir[wd_i]) &
                        (df.wind_direction < wind_dir[wd_i + 1])].index.strftime("%Y-%m-%d %H:%M:%S") 
        elif ((bc_i != len(breeze_classes) - 1) & (wd_i == len(wind_dir) - 1)):
            df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                        (df.normalized_wind_speed <= breeze_classes[bc_i + 1]) &
                        (df.wind_direction >= wind_dir[wd_i])].index.strftime("%Y-%m-%d %H:%M:%S")  
        else:
            df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                        (df.normalized_wind_speed <= breeze_classes[bc_i + 1]) &
                        (df.wind_direction >= wind_dir[wd_i]) &
                        (df.wind_direction < wind_dir[wd_i + 1])].index.strftime("%Y-%m-%d %H:%M:%S")  
            
        tmp_df_wd_bc = NA_filt_GL_df.loc[df_h_i.tolist()]

        count_stations = [np.around(tmp_df_wd_bc[(tmp_df_wd_bc.index.month == 12) |
                                       (tmp_df_wd_bc.index.month == 1) |
                                       (tmp_df_wd_bc.index.month == 2)].mean(axis=0).count() / len(spat_attributes_by_ID_GL) * 100,
                                    decimals = 2),
                          np.around(tmp_df_wd_bc[(tmp_df_wd_bc.index.month == 3) |
                                       (tmp_df_wd_bc.index.month == 4) |
                                       (tmp_df_wd_bc.index.month == 5)].mean(axis=0).count() / len(spat_attributes_by_ID_GL) * 100,
                                    decimals = 2),
                          np.around(tmp_df_wd_bc[(tmp_df_wd_bc.index.month == 6) |
                                       (tmp_df_wd_bc.index.month == 7) |
                                       (tmp_df_wd_bc.index.month == 8)].mean(axis=0).count() / len(spat_attributes_by_ID_GL) * 100,
                                    decimals = 2),
                          np.around(tmp_df_wd_bc[(tmp_df_wd_bc.index.month == 9) |
                                       (tmp_df_wd_bc.index.month == 10) |
                                       (tmp_df_wd_bc.index.month == 11)].mean(axis=0).count() / len(spat_attributes_by_ID_GL) * 100,
                                    decimals = 2)]
        count_hours = [len(tmp_df_wd_bc[(tmp_df_wd_bc.index.month == 12) |
                                       (tmp_df_wd_bc.index.month == 1) |
                                       (tmp_df_wd_bc.index.month == 2)]),
                      len(tmp_df_wd_bc[(tmp_df_wd_bc.index.month == 3) |
                                       (tmp_df_wd_bc.index.month == 4) |
                                       (tmp_df_wd_bc.index.month == 5)]),
                      len(tmp_df_wd_bc[(tmp_df_wd_bc.index.month == 6) |
                                       (tmp_df_wd_bc.index.month == 7) |
                                       (tmp_df_wd_bc.index.month == 8)]),
                      len(tmp_df_wd_bc[(tmp_df_wd_bc.index.month == 9) |
                                       (tmp_df_wd_bc.index.month == 10) |
                                       (tmp_df_wd_bc.index.month == 11)])]
        count_stations_per_days.append([wind_dir_name[wd_i], breezes[bc_i], count_hours, count_stations])
        
cws_day_rec_df = pd.DataFrame(count_stations_per_days, columns=[['Wind_Quadrant', 'Wind_Speed_Type', 'Hours_Rec', 'Max_N_Stations']])
cws_day_rec_df.to_csv(savedir + 'Nbr_Stations_Hrs_Records_CWS_2015-2020.csv')

#######
### Create a dataframe of hourly UHA per station in Domain 2
#######

df_UHA = pd.DataFrame(index = NA_filt_GL_df.index, columns = list(NA_filt_GL_df.columns))
    
for wd_i in range(len(wind_dir)):
    for bc_i in range(len(breeze_classes)):          
        ## Allocate Y locations for each breeze class per LCZ
        if bc_i == 0:
            breeze_y = breeze_y_classes[0]
            b_marker = 's'
        elif bc_i == 1:
            breeze_y = breeze_y_classes[1]
            b_marker = 'o'
        elif bc_i == 2:
            breeze_y = breeze_y_classes[2]
            b_marker = 'd'
        elif bc_i == 3:
            breeze_y = breeze_y_classes[-1]
            b_marker = '*'   
        if ((bc_i == len(breeze_classes) - 1) & (wd_i == len(wind_dir) - 1)):
            df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                                   (df.wind_direction > wind_dir[wd_i])].index.strftime("%Y-%m-%d %H:%M:%S") 
        elif ((bc_i == len(breeze_classes) - 1) & (wd_i != len(wind_dir) - 1)):
            df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                                   (df.wind_direction > wind_dir[wd_i]) &
                                   (df.wind_direction <= wind_dir[wd_i + 1])].index.strftime("%Y-%m-%d %H:%M:%S") 
        elif ((bc_i != len(breeze_classes) - 1) & (wd_i == len(wind_dir) - 1)):
            df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                                   (df.normalized_wind_speed <= breeze_classes[bc_i + 1]) &
                                   (df.wind_direction > wind_dir[wd_i])].index.strftime("%Y-%m-%d %H:%M:%S")  
        else:
            df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                                   (df.normalized_wind_speed <= breeze_classes[bc_i + 1]) &
                                   (df.wind_direction > wind_dir[wd_i]) &
                                   (df.wind_direction <= wind_dir[wd_i + 1])].index.strftime("%Y-%m-%d %H:%M:%S")   
            
        tmp_df_wd_bc = NA_filt_GL_df.loc[df_h_i.tolist()] - NA_filt_GL_df.apply(np.nanmean, axis=0)

        ### Record which LCZ have at least one station in two opposite quadrants
        lcz_ploted = []
        for lcz in range(1,17):
            quad_unique = 0
            ### Count how many LCZs are in
            lcz_num = len(spat_attributes_by_ID_GL[spat_attributes_by_ID_GL['LCZ'] == lcz])
            ### If there are no more than 1 station, stop the loop
            if lcz_num <= 1:
                continue
            spat_lcz = spat_attributes_by_ID_GL[spat_attributes_by_ID_GL['LCZ'] == lcz]
            for e_quad in ['NE', 'SE']:
                op_w_quad = spat_attributes_by_ID_GL[(~spat_attributes_by_ID_GL.Quadrant.str.contains(e_quad[0])) &
                                                     (~spat_attributes_by_ID_GL.Quadrant.str.contains(e_quad[1]))].Quadrant.unique()[0] 
                quad_unique += spat_lcz[(spat_attributes_by_ID_GL.Quadrant == e_quad) | 
                                       (spat_attributes_by_ID_GL.Quadrant == op_w_quad)].Quadrant.nunique()
            if quad_unique < 2:
                continue
            ### Store the LCZ that will be plotted
            lcz_ploted.append(lcz)
        
        count_lczs = len(lcz_ploted)
        for lcz in lcz_ploted:
            lcz_label = lcz_labels_dict.get(lcz)
            lcz_color = lcz_colors_dict.get(lcz)
            
            spat_lcz = spat_attributes_by_ID_GL[spat_attributes_by_ID_GL['LCZ'] == lcz]
            tmp_df_wd_bc_lcz = tmp_df_wd_bc[spat_lcz.index.tolist()]
            ### Calculate the anomaly to the average temperature of same LCZ located upwind
            NA_id_dow = spat_lcz[(~spat_lcz.Quadrant.str.contains(wind_dir_name[wd_i][0])) &
                                 (~spat_lcz.Quadrant.str.contains(wind_dir_name[wd_i][1]))].index.tolist()
            NA_id_upw = spat_lcz[spat_lcz.Quadrant == wind_dir_name[wd_i]].index.tolist()
            
            if (len(NA_id_dow) == 0) | (len(NA_id_upw) == 0):
                ### Augment by 1 the Y locator for each new LCZ
                breeze_y = breeze_y + 1
                continue

            tmp_df_wd_bc_anom = tmp_df_wd_bc_lcz[NA_id_dow].subtract(tmp_df_wd_bc_lcz[NA_id_upw].mean(axis = 1), axis = 0)
            df_UHA.loc[tmp_df_wd_bc_anom.index, tmp_df_wd_bc_anom.columns] = tmp_df_wd_bc_anom.values

#########################
###       PLOTS       ###
#########################

### Fig 1 : Number of available CWS per day (2015-2020)

if plot_fig1 == True:

    fig, ax = plt.subplots(figsize = (9,7))
    
    ax.plot(NA_d_filt_df_avg.count(axis=1), color='darkgrey')
    ax.set_xlabel('Time', fontsize = 12, color = 'dimgrey')
    ax.tick_params(labelcolor='dimgrey', labelsize = 10)
    ax.tick_params(labelcolor='dimgrey', labelsize = 10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    
    ax.set_title('Number of available and active Netatmo CWS per day', 
                 fontsize = 14, color = 'dimgrey')
    fig.savefig(savedir + 'Nbr_CWS_London_2015-2020.png', dpi=600)
    fig.savefig(savedir + 'Nbr_CWS_London_2015-2020.pdf')

### Fig 2 : Averaged DTR and Averaged temperature per Season

if plot_fig2 == True:

    col = 2; row = 4
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(row, col, figsize = (12,4*row), subplot_kw=dict(projection=proj))
    ax = ax.flatten()
    
    season_i = 0
    for ax_i in ax[0::2]:
        order_t = np.argsort(spat_attributes_by_ID['T2_AVG_' + seasons[season_i]])
        ax_i.coastlines(resolution='10m', alpha=0.1)
        ax_i.contour(lon_LCZ, lat_LCZ, LCZ_Urb, levels=[0.1, 1.1], colors = 'black', linewidths = 0.1)
        ax_i.set_extent([min(LCZ_Tiff.x), max(LCZ_Tiff.x), min(LCZ_Tiff.y), max(LCZ_Tiff.y)])
        im = ax_i.scatter(np.array(spat_attributes_by_ID.Lon)[order_t],
                          np.array(spat_attributes_by_ID.Lat)[order_t],
                          c = np.array(spat_attributes_by_ID['T2_AVG_' + seasons[season_i]])[order_t], 
                          cmap = cmap_temp, 
                          vmin = spat_attributes_by_ID['T2_AVG_' + seasons[season_i]].quantile(q = 0.05),
                          vmax = spat_attributes_by_ID['T2_AVG_' + seasons[season_i]].quantile(q = 0.95),
                          s=20)
        ax_i.set_title('Averaged Temperature (°C) [' + seasons[season_i] + ']', color='dimgrey', fontsize = 14)
        divider = make_axes_locatable(ax_i)
        cax = divider.append_axes('right', size='5%', pad=0.1, axes_class=plt.Axes)
        fig.add_axes(cax)
        fig.colorbar(im, cax=cax, orientation='vertical', extend = 'both')
        season_i = season_i + 1
    
    season_i = 0
    for ax_i in ax[1::2]:
        order_dtr = np.flip(np.argsort(spat_attributes_by_ID['T2_DTR_AVG_' + seasons[season_i]]))
        ax_i.coastlines(resolution='10m', alpha=0.1)
        ax_i.contour(lon_LCZ, lat_LCZ, LCZ_Urb, levels=[0.1, 1.1], colors = 'black', linewidths = 0.1)
        ax_i.set_extent([min(LCZ_Tiff.x), max(LCZ_Tiff.x), min(LCZ_Tiff.y), max(LCZ_Tiff.y)])
        im = ax_i.scatter(np.array(spat_attributes_by_ID.Lon)[order_dtr],
                          np.array(spat_attributes_by_ID.Lat)[order_dtr],
                          c = np.array(spat_attributes_by_ID['T2_DTR_AVG_' + seasons[season_i]])[order_dtr],
                          cmap = cmap_dtr, 
                          vmin = spat_attributes_by_ID['T2_DTR_AVG_' + seasons[season_i]].quantile(q = 0.05),
                          vmax = spat_attributes_by_ID['T2_DTR_AVG_' + seasons[season_i]].quantile(q = 0.95),
                          s=20)
        ax_i.set_title('Daily Temperature Range (°C) [' + seasons[season_i] + ']', color='dimgrey', fontsize = 14)
        divider = make_axes_locatable(ax_i)
        cax = divider.append_axes('right', size='5%', pad=0.1, axes_class=plt.Axes)
        fig.add_axes(cax)
        fig.colorbar(im, cax=cax, orientation='vertical', extend = 'both')
        season_i = season_i + 1
    
    fig.savefig(savedir + 'Netatmo_AVG_T2M_DTR_perSeason.png', dpi=600)
    fig.savefig(savedir + 'Netatmo_AVG_T2M_DTR_perSeason.pdf')

### Fig 3 : Monthly temperatures per LCZ

if plot_fig3 == True:

    months_name = [datetime.strptime(str(mon), "%m").strftime("%b") for mon in range(1,13)]
    col = 1; row = 2
    
    fig, ax = plt.subplots(row, col, figsize = (21.7,20))
    ax = ax.flatten() 
    for lcz in range(1,17):
        ### Count how many LCZs are in
        lcz_num = len(spat_attributes_by_ID[spat_attributes_by_ID['LCZ'] == lcz])
        if lcz_num == 0:
            continue
        lcz_label = lcz_labels_dict.get(lcz)
        lcz_color = lcz_colors_dict.get(lcz)
        
        spat_lcz = spat_attributes_by_ID[spat_attributes_by_ID['LCZ'] == lcz]
        tmp_df_tmp_lcz = NA_filt_df[spat_lcz.index.tolist()]
        tmp_df_dtr_lcz = NA_d_filt_df_dtr[spat_lcz.index.tolist()]
        
        tmp_df_tmp_lcz_mon_avg = [np.nanmean(
                                    np.nanmedian(tmp_df_tmp_lcz[tmp_df_tmp_lcz.index.month == mon], axis = 1)
                                  ) 
                                  for mon in range(1,13)]
        tmp_df_tmp_lcz_mon_q25 = [np.nanpercentile(
                                    np.nanmedian(tmp_df_tmp_lcz[tmp_df_tmp_lcz.index.month == mon], axis = 1),
                                  q=25) 
                                  for mon in range(1,13)]
        tmp_df_tmp_lcz_mon_q75 = [np.nanpercentile(
                                    np.nanmedian(tmp_df_tmp_lcz[tmp_df_tmp_lcz.index.month == mon], axis = 1),
                                  q=75) 
                                  for mon in range(1,13)]
        
        tmp_df_dtr_lcz_mon_avg = [np.nanmean(
                                    np.nanmedian(tmp_df_dtr_lcz[tmp_df_dtr_lcz.index.month == mon], axis = 1)
                                  ) 
                                  for mon in range(1,13)]
        tmp_df_dtr_lcz_mon_q25 = [np.nanpercentile(
                                    np.nanmedian(tmp_df_dtr_lcz[tmp_df_dtr_lcz.index.month == mon], axis = 1),
                                  q=25) 
                                  for mon in range(1,13)]
        tmp_df_dtr_lcz_mon_q75 = [np.nanpercentile(
                                    np.nanmedian(tmp_df_dtr_lcz[tmp_df_dtr_lcz.index.month == mon], axis = 1),
                                  q=75)  
                                  for mon in range(1,13)]
        
        ax[0].plot(tmp_df_tmp_lcz_mon_avg, color = lcz_color, label = lcz_label, marker = 'o', zorder = 1)
        ax[0].plot(tmp_df_tmp_lcz_mon_q25, linestyle = '--', marker = '^', color = lcz_color, zorder = 0)
        ax[0].plot(tmp_df_tmp_lcz_mon_q75, linestyle = '--', marker = 'v', color = lcz_color, zorder = 0)
         
        ax[1].plot(tmp_df_dtr_lcz_mon_avg, color = lcz_color, label = lcz_label, marker = 'o', zorder = 1)
        ax[1].plot(tmp_df_dtr_lcz_mon_q25, linestyle = '--', marker = '^', color = lcz_color, zorder = 0)
        ax[1].plot(tmp_df_dtr_lcz_mon_q75, linestyle = '--', marker = 'v', color = lcz_color, zorder = 0)
    
    for ax_i in ax:
        ax_i.spines['right'].set_visible(False)
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['left'].set_position(('outward', 10))
        ax_i.spines['bottom'].set_position(('outward', 10))
        ax_i.tick_params(axis='y',which='both',left=True,right=False,labelleft=True, 
                         labelcolor = 'dimgrey', labelsize=12)
        ax_i.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True, 
                         labelcolor = 'dimgrey', labelsize=12)
    
    ax[1].set_xticks(np.arange(12))
    ax[1].set_xticklabels(months_name, fontsize = 12)
    ax[1].set_ylabel('DTR [°C]', rotation=0, color = 'dimgrey', fontsize = 12)
    ax[1].yaxis.set_label_coords(0, 1.03)
    
    ax[0].set_ylim(0,25)
    ax[0].set_ylabel('Temperature [°C]', rotation=0, color = 'dimgrey', fontsize = 12)
    ax[0].yaxis.set_label_coords(0, 1.03)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
     
    ### Create temporary objects for the legend
    handles, labels = ax[-1].get_legend_handles_labels()
    # divider = make_axes_locatable(ax[-1])
    # cax = divider.append_axes('bottom', size='20%', pad=0.1, axes_class=plt.Axes)
    # fig.add_axes(cax)
    # fig.legend(handles, labels, cax = cax, ncol=4, fontsize = 14)
    fig.legend(handles, labels, 'lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize = 14)
    fig.suptitle('Monthly averaged temperatures and DTR and their respective 25th and 75th percentiles per LCZ ',
                 color = 'dimgrey', fontsize = 22, y = 0.92)
    fig.savefig(savedir + 'CWS_MonAVG_LCZ.png', dpi=600)
    fig.savefig(savedir + 'CWS_MonAVG_LCZ.pdf')    

### Fig 4 : Averaged DTR and Averaged temperatures for the 6 years, percentage of time measured

if plot_fig4 == True:

    col = 2; row = 2
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(row, col, figsize = (23,8*row), subplot_kw=dict(projection=proj))
    ax = ax.flatten()

    order_t = np.argsort(spat_attributes_by_ID['T2_AVG'])
    order_dtr = np.flip(np.argsort(spat_attributes_by_ID['T2_DTR_AVG']))
    order_prc = np.argsort((NA_filt_df.T.count(axis=1) / len(NA_filt_df) * 100))

    ax[0].coastlines(resolution='10m', alpha=0.1)
    ax[0].contour(lon_LCZ, lat_LCZ, LCZ_Urb, levels=[0.1, 1.1], colors = 'black', linewidths = 0.1)
    ax[0].set_extent([min(LCZ_Tiff.x), max(LCZ_Tiff.x), min(LCZ_Tiff.y), max(LCZ_Tiff.y)])
    im = ax[0].scatter(spat_attributes_by_ID.Lon[order_t],
                       spat_attributes_by_ID.Lat[order_t],
                      c = spat_attributes_by_ID['T2_AVG'][order_t], 
                      cmap = cmap_temp, 
                      vmin = spat_attributes_by_ID['T2_AVG'].quantile(q = 0.05),
                      vmax = spat_attributes_by_ID['T2_AVG'].quantile(q = 0.95),
                      s=20, zorder = 1)
    ax[0].set_title('Averaged Temperature (°C) [A]', color='dimgrey', fontsize = 14)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)
    fig.colorbar(im, cax=cax, orientation='vertical', extend = 'both')
    
    ax[1].coastlines(resolution='10m', alpha=0.1)
    ax[1].contour(lon_LCZ, lat_LCZ, LCZ_Urb, levels=[0.1, 1.1], colors = 'black', linewidths = 0.1)
    ax[1].set_extent([min(LCZ_Tiff.x), max(LCZ_Tiff.x), min(LCZ_Tiff.y), max(LCZ_Tiff.y)])
    im = ax[1].scatter(spat_attributes_by_ID.Lon[order_dtr],
                       spat_attributes_by_ID.Lat[order_dtr],
                      c = spat_attributes_by_ID['T2_DTR_AVG'][order_dtr], 
                      cmap = cmap_dtr, 
                      vmin = spat_attributes_by_ID['T2_DTR_AVG'].quantile(q = 0.05),
                      vmax = spat_attributes_by_ID['T2_DTR_AVG'].quantile(q = 0.95),
                      s=20, zorder = 1)
    ax[1].set_title('Averaged Daily Temperature Range (°C) [B]', color='dimgrey', fontsize = 14)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)
    fig.colorbar(im, cax=cax, orientation='vertical', extend = 'both')
    
    ax[2].coastlines(resolution='10m', alpha=0.1)
    ax[2].contour(lon_LCZ, lat_LCZ, LCZ_Urb, levels=[0.1, 1.1], colors = 'black', linewidths = 0.1)
    ax[2].set_extent([min(LCZ_Tiff.x), max(LCZ_Tiff.x), min(LCZ_Tiff.y), max(LCZ_Tiff.y)])
    im = ax[2].scatter(spat_attributes_by_ID.Lon[order_prc],
                       spat_attributes_by_ID.Lat[order_prc],
                      c = (NA_filt_df.T.count(axis=1) / len(NA_filt_df) * 100)[order_prc], 
                      cmap = cmap_prc, 
                      vmin = 10,
                      vmax = 90,
                      s=20, zorder = 1)
    ax[2].set_title('Percentage of available measurements [C]', color='dimgrey', fontsize = 14)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)
    fig.colorbar(im, cax=cax, orientation='vertical', extend = 'both')
    
    ax[3].remove()
    fig.tight_layout()
    fig.savefig(savedir + 'Netatmo_AVG_T2M_DTR_2015-2020.png', dpi=600)
    fig.savefig(savedir + 'Netatmo_AVG_T2M_DTR_2015-2020.pdf')

    print((stats.pearsonr(np.array(spat_attributes_by_ID['T2_AVG']),
                         (NA_filt_df.T.count(axis=1) / len(NA_filt_df) * 100))[0])**2)
    
### Fig 5 : Distributions per wind regime and lcz of hourly temperatures [whole day, nighttime, daytime]

### Function to have filled markers located at the edge of where the data is plotted (useful for percentile whiskers) 
def align_marker(marker, halign='center', valign='middle',):
    """
    create markers with specified alignment.

    Parameters
    ----------

    marker : a valid marker specification.
      See mpl.markers

    halign : string, float {'left', 'center', 'right'}
      Specifies the horizontal alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'center',
      -1 is 'right', 1 is 'left').

    valign : string, float {'top', 'middle', 'bottom'}
      Specifies the vertical alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'middle',
      -1 is 'top', 1 is 'bottom').

    Returns
    -------

    marker_array : numpy.ndarray
      A Nx2 array that specifies the marker path relative to the
      plot target point at (0, 0).

    Notes
    -----
    The mark_array can be passed directly to ax.plot and ax.scatter, e.g.::

        ax.plot(1, 1, marker=align_marker('>', 'left'))

    """

    if isinstance(halign, (str, str)):
        halign = {'right': -1.,
                  'middle': 0.,
                  'center': 0.,
                  'left': 1.,
                  }[halign]

    if isinstance(valign, (str, str)):
        valign = {'top': -1.,
                  'middle': 0.,
                  'center': 0.,
                  'bottom': 1.,
                  }[valign]

    # Define the base marker
    bm = markers.MarkerStyle(marker)

    # Get the marker path and apply the marker transform to get the
    # actual marker vertices (they should all be in a unit-square
    # centered at (0, 0))
    m_arr = bm.get_path().transformed(bm.get_transform()).vertices

    # Shift the marker vertices for the specified alignment.
    m_arr[:, 0] += halign / 2
    m_arr[:, 1] += valign / 2

    return Path(m_arr, bm.get_path().codes)

if plot_fig5 == True:
        
    col = 4; row = 1
    
    #dayperiod = ['WholeDay']
    dayperiod = ['WholeDay', 'NightTime', 'DayTime']
    for dp in dayperiod:
        fig, ax = plt.subplots(row, col, figsize = (24,15*row))
        ax = ax.flatten()
        ### Create temporary objects for the legend
        ax[-1].scatter(0, 0, color = 'dimgrey', s=130, marker = 's', label = breezes[0])
        ax[-1].scatter(1, 1, color = 'dimgrey', s=130,  marker = 'o', label = breezes[1])
        ax[-1].scatter(2, 2, color = 'dimgrey', s=130, marker = 'd', label = breezes[2])
        ax[-1].scatter(3, 3, color = 'dimgrey', s=130, marker = '*', label = breezes[3])
        ax[-1].scatter(-1, -1, color = 'dimgrey', s=130, marker = 5, label = '25th Percentile')
        ax[-1].scatter(-1, -1, color = 'dimgrey', s=130, marker = 4, label = '75th Percentile')
        handles, labels = ax[-1].get_legend_handles_labels()
        ax[-1].cla() ## Release the memory and wipes out all graphical objects on the axe
        
        for wd_i in range(len(wind_dir)):
            for bc_i in range(len(breeze_classes)):          
                ## Allocate Y locations for each breeze class per LCZ
                if bc_i == 0:
                    breeze_y = breeze_y_classes[0]
                    b_marker = 's'
                elif bc_i == 1:
                    breeze_y = breeze_y_classes[1]
                    b_marker = 'o'
                elif bc_i == 2:
                    breeze_y = breeze_y_classes[2]
                    b_marker = 'd'
                elif bc_i == 3:
                    breeze_y = breeze_y_classes[-1]
                    b_marker = '*'        
                if ((bc_i == len(breeze_classes) - 1) & (wd_i == len(wind_dir) - 1)):
                    df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                                            (df.wind_direction > wind_dir[wd_i])].index.strftime("%Y-%m-%d %H:%M:%S") 
                elif ((bc_i == len(breeze_classes) - 1) & (wd_i != len(wind_dir) - 1)):
                    df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                                            (df.wind_direction > wind_dir[wd_i]) &
                                            (df.wind_direction <= wind_dir[wd_i + 1])].index.strftime("%Y-%m-%d %H:%M:%S") 
                elif ((bc_i != len(breeze_classes) - 1) & (wd_i == len(wind_dir) - 1)):
                    df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                                            (df.normalized_wind_speed <= breeze_classes[bc_i + 1]) &
                                            (df.wind_direction > wind_dir[wd_i])].index.strftime("%Y-%m-%d %H:%M:%S")  
                else:
                    df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                                            (df.normalized_wind_speed <= breeze_classes[bc_i + 1]) &
                                            (df.wind_direction > wind_dir[wd_i]) &
                                            (df.wind_direction <= wind_dir[wd_i + 1])].index.strftime("%Y-%m-%d %H:%M:%S")  
                    
                if dp == 'NightTime':
                    tmp_df_wd_bc = df_UHA.loc[df_h_i.tolist()]
                    time_mask = (tmp_df_wd_bc.index.hour >= 19) | (tmp_df_wd_bc.index.hour < 7)
                    tmp_df_wd_bc = tmp_df_wd_bc[time_mask]
                elif dp == "DayTime":
                    tmp_df_wd_bc = df_UHA.loc[df_h_i.tolist()]
                    time_mask = (tmp_df_wd_bc.index.hour >= 7) & (tmp_df_wd_bc.index.hour < 19)
                    tmp_df_wd_bc = tmp_df_wd_bc[time_mask]
                else:
                    tmp_df_wd_bc = df_UHA.loc[df_h_i.tolist()]

                for lcz in lcz_ploted:
                #for lcz in range(6,7,1):
                    lcz_label = lcz_labels_dict.get(lcz)
                    lcz_color = lcz_colors_dict.get(lcz)
                    
                    spat_lcz = spat_attributes_by_ID_GL[spat_attributes_by_ID_GL['LCZ'] == lcz]
                    tmp_df_wd_bc_lcz = tmp_df_wd_bc[spat_lcz.index.tolist()]       
                    tmp_df_wd_bc_anom_avg = tmp_df_wd_bc_lcz.mean()
                    print(wind_dir_name[wd_i], breeze_classes[bc_i], np.nanmedian(tmp_df_wd_bc_anom_avg), np.nanmean(tmp_df_wd_bc_anom_avg), lcz_label)
                                        
                    ### Create Y axes per breeze characteristics
                    breeze_Y = np.array([breeze_y] * len(tmp_df_wd_bc_lcz.columns))          
                    ax[wd_i].scatter(tmp_df_wd_bc_anom_avg, breeze_Y, color=lcz_color, s=45, marker = b_marker, alpha=0.4)
                    ax[wd_i].scatter(np.nanmedian(tmp_df_wd_bc_anom_avg, axis=0), breeze_Y[0],
                                            marker = b_marker, color=lcz_color, s=130, linewidth = 1, edgecolors = 'dimgrey')
                    ax[wd_i].scatter(np.nanpercentile(tmp_df_wd_bc_anom_avg, axis=0, q=25), breeze_Y[0], 
                                     marker=align_marker('>', halign='right'),
                                     s=390, color=lcz_color, linewidth = 1, edgecolors = 'dimgrey')
                    ax[wd_i].scatter(np.nanpercentile(tmp_df_wd_bc_anom_avg, axis=0, q=75), breeze_Y[0], 
                                     marker=align_marker('<', halign='left'),
                                     s=390, color=lcz_color, linewidth = 1, edgecolors = 'dimgrey')
                    
                    ### Augment by 1 the Y locator for each new LCZ
                    breeze_y = breeze_y + 1
                    
                del tmp_df_wd_bc  
        
        for ax_i in ax:
            ax_i.set_xlim(-3.5,3.5)
            ax_i.vlines(x=0,ymin=breeze_y_classes[-1], ymax=breeze_y, color = 'dimgrey', linestyle = 'dotted')
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
            ax_i.spines['left'].set_visible(False)
            ax_i.spines['bottom'].set_position(('outward', 10))
            ax_i.tick_params(axis='y',which='both',left=False,right=False,labelleft=True)
            ax_i.tick_params(axis='x',which='both',bottom=True,top=False,labelbottom=False, labelsize = 16)
            ax_i.set_ylim(breeze_y_classes[-1] - 0.2, breeze_y_classes[0] + count_lczs - 1 + 0.2)
            ax_i.set_yticks([])
        
        ### Add wind direction in top axes 
        for i in range(4):
            ax[i].set_title(wind_dir_name[i], color = 'dimgrey', fontsize = 22, x=0.5, y=1.03)
            ax[-i -1].tick_params(axis='x',which='both',bottom=True,top=False,labelbottom=True)
        
        
        fig.legend(handles, labels, 'lower center', bbox_to_anchor=(0.75, 0.05), ncol=3, fontsize = 18)
        
        ### Add legend of ploted LCZ (need to recreate a colormap based on the default one)
        list_lcz_leg = list(set(lcz_ploted))
        cmap_lcz_scatter = mpl.colors.ListedColormap([lcz_colors_dict.get(lp) for lp in list_lcz_leg])
        lcz_tick_classes = list(np.arange(0,len(list_lcz_leg)+1))
        norm_lcz_scatter = mpl.colors.BoundaryNorm(lcz_tick_classes, cmap_lcz_scatter.N)
        tmp_ax = plt.axes([0, 0, 0, 0])
        im = tmp_ax.imshow([lcz_tick_classes], vmin = lcz_tick_classes[0], vmax = lcz_tick_classes[-1], 
                           cmap = cmap_lcz_scatter)
        tmp_ax.remove()

        
        cax = plt.axes([0.1, 0.07, 0.3, 0.02])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_ticks(list(np.array(lcz_tick_classes[0:-1]) + 0.5))
        cbar.ax.tick_params(axis='x',which='both',left=False,right=False,labelright=True)
        labels = cbar.ax.set_xticklabels([lcz_labels_dict.get(lp)[-1] for lp in list_lcz_leg], 
                                         color = 'dimgrey', fontsize = 18)
        cbar.ax.set_title(r'LCZ', loc='center', color = 'dimgrey', fontsize = 20)
        
        fig.tight_layout(rect=[0.0, 0.12, 1.0, 0.92])
        fig.suptitle('Averaged Temperature Anomaly driven by Urban Heat Advection (UHA) per LCZ and Wind Regimes [°C]',
                     color = 'dimgrey', fontsize = 26, y = 0.97)
        fig.savefig(savedir + 'Netatmo_T2M_ANOM_' + dp + '_Hourly_DownWindRegimes_LCZ_' 
                                + str(bbox_llat) + '-' + str(bbox_llon) + '_' 
                                + str(bbox_ulat) + '-' + str(bbox_ulon) + '_AvgT.png', dpi=300)
        fig.savefig(savedir + 'Netatmo_T2M_ANOM_' + dp + '_Hourly_DownWindRegimes_LCZ_' 
                                + str(bbox_llat) + '-' + str(bbox_llon) + '_' 
                                + str(bbox_ulat) + '-' + str(bbox_ulon) + 'AvgT.pdf')

### Fig 6 : Distributions per wind regime, season and lcz of hourly temperatures [whole day, nighttime, daytime]

if plot_fig6 == True:
        
    col = 4; row = 4
    
    dayperiod = ['WholeDay']
    # dayperiod = ['WholeDay', 'NightTime', 'DayTime']
    tmp_med_lcz = []
    for dp in dayperiod:
        fig, ax = plt.subplots(row, col, figsize = (21.7,8.5*row))
        ax = ax.flatten()
        ### Create temporary objects for the legend
        ax[-1].scatter(0, 0, color = 'dimgrey', s=130, marker = 's', label = breezes[0])
        ax[-1].scatter(1, 1, color = 'dimgrey', s=130, marker = 'o', label = breezes[1])
        ax[-1].scatter(2, 2, color = 'dimgrey', s=130, marker = 'd', label = breezes[2])
        ax[-1].scatter(3, 3, color = 'dimgrey', s=130, marker = '*', label = breezes[3])
        ax[-1].scatter(-1, -1, color = 'dimgrey', s=130, marker = 5, label = '25th Percentile')
        ax[-1].scatter(-1, -1, color = 'dimgrey', s=130, marker = 4, label = '75th Percentile')
        handles, labels = ax[-1].get_legend_handles_labels()
        ax[-1].cla() ## Release the memory and wipes out all graphical objects on the axe
            
        ax_seas_loc = 0
        for wd_i in range(len(wind_dir)):
            for bc_i in range(len(breeze_classes)):          
                ## Allocate Y locations for each breeze class per LCZ
                if bc_i == 0:
                    breeze_y = breeze_y_classes[0]
                    b_marker = 's'
                elif bc_i == 1:
                    breeze_y = breeze_y_classes[1]
                    b_marker = 'o'
                elif bc_i == 2:
                    breeze_y = breeze_y_classes[2]
                    b_marker = 'd'
                elif bc_i == 3:
                    breeze_y = breeze_y_classes[-1]
                    b_marker = '*'   
                if ((bc_i == len(breeze_classes) - 1) & (wd_i == len(wind_dir) - 1)):
                    df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                                           (df.wind_direction > wind_dir[wd_i])].index.strftime("%Y-%m-%d %H:%M:%S") 
                elif ((bc_i == len(breeze_classes) - 1) & (wd_i != len(wind_dir) - 1)):
                    df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                                           (df.wind_direction > wind_dir[wd_i]) &
                                           (df.wind_direction <= wind_dir[wd_i + 1])].index.strftime("%Y-%m-%d %H:%M:%S") 
                elif ((bc_i != len(breeze_classes) - 1) & (wd_i == len(wind_dir) - 1)):
                    df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                                           (df.normalized_wind_speed <= breeze_classes[bc_i + 1]) &
                                           (df.wind_direction > wind_dir[wd_i])].index.strftime("%Y-%m-%d %H:%M:%S")  
                else:
                    df_h_i = df[(df.normalized_wind_speed > breeze_classes[bc_i]) & 
                                           (df.normalized_wind_speed <= breeze_classes[bc_i + 1]) &
                                           (df.wind_direction > wind_dir[wd_i]) &
                                           (df.wind_direction <= wind_dir[wd_i + 1])].index.strftime("%Y-%m-%d %H:%M:%S")   
                    
                if dp == 'NightTime':
                    tmp_df_wd_bc = df_UHA.loc[df_h_i.tolist()]
                    time_mask = (tmp_df_wd_bc.index.hour >= 19) | (tmp_df_wd_bc.index.hour < 7)
                    tmp_df_wd_bc = tmp_df_wd_bc[time_mask]
                elif dp == "DayTime":
                    tmp_df_wd_bc = df_UHA.loc[df_h_i.tolist()]
                    time_mask = (tmp_df_wd_bc.index.hour >= 7) & (tmp_df_wd_bc.index.hour < 19)
                    tmp_df_wd_bc = tmp_df_wd_bc[time_mask]
                else:
                    tmp_df_wd_bc = df_UHA.loc[df_h_i.tolist()]

                for lcz in lcz_ploted:
                    lcz_label = lcz_labels_dict.get(lcz)
                    lcz_color = lcz_colors_dict.get(lcz)
                    
                    spat_lcz = spat_attributes_by_ID_GL[spat_attributes_by_ID_GL['LCZ'] == lcz]
                    tmp_df_wd_bc_lcz = tmp_df_wd_bc[spat_lcz.index.tolist()]       
                    
                    NA_h_avg_i = pd.DataFrame()
                    NA_h_avg_i["avg_DJF"] = tmp_df_wd_bc_lcz[(tmp_df_wd_bc_lcz.index.month == 12) |
                                                              (tmp_df_wd_bc_lcz.index.month == 1) |
                                                              (tmp_df_wd_bc_lcz.index.month == 2)].mean()
                    NA_h_avg_i["avg_MAM"] = tmp_df_wd_bc_lcz[(tmp_df_wd_bc_lcz.index.month == 3) |
                                                              (tmp_df_wd_bc_lcz.index.month == 4) |
                                                              (tmp_df_wd_bc_lcz.index.month == 5)].mean()
                    NA_h_avg_i["avg_JJA"] = tmp_df_wd_bc_lcz[(tmp_df_wd_bc_lcz.index.month == 6) |
                                                              (tmp_df_wd_bc_lcz.index.month == 7) |
                                                              (tmp_df_wd_bc_lcz.index.month == 8)].mean()
                    NA_h_avg_i["avg_SON"] = tmp_df_wd_bc_lcz[(tmp_df_wd_bc_lcz.index.month == 9) |
                                                              (tmp_df_wd_bc_lcz.index.month == 10) |
                                                              (tmp_df_wd_bc_lcz.index.month == 11)].mean()

                    ### Create Y axes per breeze characteristics
                    breeze_Y = np.array([breeze_y] * len(tmp_df_wd_bc_lcz.columns))          
                    s_loc = ax_seas_loc
                    for s_i in seasons:
                        lcz_sea_avg =  NA_h_avg_i["avg_" + s_i].values.astype(float)
                        ax[s_loc].scatter(lcz_sea_avg, breeze_Y, color=lcz_color, s=45, marker = b_marker, alpha=0.5)
                        ax[s_loc].scatter(np.nanmean(lcz_sea_avg, axis=0), breeze_Y[0],
                                          marker = b_marker, 
                                          linewidth = 1, edgecolors = 'dimgrey', color=lcz_color, s=130)
                        ax[s_loc].scatter(np.nanpercentile(lcz_sea_avg, axis=0, q=25), breeze_Y[0], 
                                          marker=align_marker('>', halign='right'),
                                          s=390, color=lcz_color, linewidth = 1, edgecolors = 'dimgrey')
                        ax[s_loc].scatter(np.nanpercentile(lcz_sea_avg, axis=0, q=75), breeze_Y[0], 
                                          marker=align_marker('<', halign='left'),
                                          s=390, color=lcz_color, linewidth = 1, edgecolors = 'dimgrey')
                        s_loc += 4
                    
                    ### Augment by 1 the Y locator for each new LCZ
                    breeze_y = breeze_y + 1
                    
                del tmp_df_wd_bc
            ### Go to next row of wind direction
            ax_seas_loc += 1
        
        
        for ax_i in ax:
            ax_i.set_xlim(-3.5,3.5)
            ax_i.vlines(x=0,ymin=breeze_y_classes[-1], ymax=breeze_y, color = 'dimgrey', linestyle = 'dotted')
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
            ax_i.spines['left'].set_visible(False)
            ax_i.spines['left'].set_position(('outward', 10))
            ax_i.spines['bottom'].set_position(('outward', 10))
            ax_i.tick_params(axis='y',which='both',left=False,right=False,labelleft=True)
            ax_i.tick_params(axis='x',which='both',bottom=True,top=False,labelbottom=False)
            ax_i.set_ylim(breeze_y_classes[-1] - 0.2, breeze_y_classes[0] + count_lczs - 1 + 0.2)
            ax_i.set_yticks([])
                
        ### Add Seasons in top axes and set x ticks visible bottom axes
        for i in range(4):
            ax[i].set_title(wind_dir_name[i], color = 'dimgrey', fontsize = 18, x=0.5, y=1.03)
            ax[-i -1].tick_params(axis='x',which='both',bottom=True,top=False,labelbottom=True)
        
        season_i = 0
        for i in range(0,16,4):
            ax[i].set_ylabel(seasons[season_i], rotation = 0, color = 'dimgrey', fontsize = 18)
            ax[i].yaxis.set_label_coords(-0.05, 1.04)
            season_i += 1
        
        fig.legend(handles, labels, 'lower center', bbox_to_anchor=(0.75, 0.07), ncol=3, fontsize = 14)
        
        ### Add legend of ploted LCZ (need to recreate a colormap based on the default one)
        list_lcz_leg = list(set(lcz_ploted))
        cmap_lcz_scatter = mpl.colors.ListedColormap([lcz_colors_dict.get(lp) for lp in list_lcz_leg])
        lcz_tick_classes = list(np.arange(0,len(list_lcz_leg)+1))
        norm_lcz_scatter = mpl.colors.BoundaryNorm(lcz_tick_classes, cmap_lcz_scatter.N)
        tmp_ax = plt.axes([0, 0, 0, 0])
        im = tmp_ax.imshow([lcz_tick_classes], vmin = lcz_tick_classes[0], vmax = lcz_tick_classes[-1], 
                           cmap = cmap_lcz_scatter)
        tmp_ax.remove()

        
        cax = plt.axes([0.1, 0.08, 0.3, 0.01])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_ticks(list(np.array(lcz_tick_classes[0:-1]) + 0.5))
        cbar.ax.tick_params(axis='x',which='both',left=False,right=False,labelright=True)
        labels = cbar.ax.set_xticklabels([lcz_labels_dict.get(lp)[-1] for lp in list_lcz_leg], 
                                         color = 'dimgrey', fontsize = 16)
        cbar.ax.set_title(r'LCZ', loc='center', color = 'dimgrey', fontsize = 18)
        
        fig.tight_layout(rect=[0.0, 0.12, 1.0, 0.92])
        fig.suptitle('Seasonality of Averaged Temperature Anomaly driven by Urban Heat Advection (UHA) per LCZ and Wind Regimes [°C]',
                     color = 'dimgrey', fontsize = 22, y = 0.94)
        fig.savefig(savedir + 'Netatmo_T2M_ANOM_' + dp + '_Hourly_perSeason_DownWindRegimes_LCZ_' 
                                + str(bbox_llat) + '-' + str(bbox_llon) + '_' 
                                + str(bbox_ulat) + '-' + str(bbox_ulon) + '_AvgT.png', dpi=300)
        fig.savefig(savedir + 'Netatmo_T2M_ANOM_' + dp + '_Hourly_perSeason_DownWindRegimes_LCZ_' 
                                + str(bbox_llat) + '-' + str(bbox_llon) + '_' 
                                + str(bbox_ulat) + '-' + str(bbox_ulon) + 'AvgT.pdf')

### Fig 7 : Averaged, Minimum and Maximum 6-year observed temperatures per longitude and latitude

if plot_fig7 == True:

    lon_bins = pd.cut(spat_attributes_by_ID.Lon,np.arange(spat_attributes_by_ID.Lon.min(),
                                                          spat_attributes_by_ID.Lon.max(),
                                                          0.1))
    lon_bin_plt = spat_attributes_by_ID.groupby(lon_bins).agg({"T2_AVG": "median", "Lon": "mean"})    

    lat_bins = pd.cut(spat_attributes_by_ID.Lat,np.arange(spat_attributes_by_ID.Lat.min(),
                                                          spat_attributes_by_ID.Lat.max(),
                                                          0.1))
    lat_bin_plt = spat_attributes_by_ID.groupby(lat_bins).agg({"T2_AVG": "median", "Lat": "mean"})
    
    col = 2; row = 1
    fig, ax = plt.subplots(row, col, figsize = (18,9*row))
    ax = ax.flatten()
    
    ax[0].scatter(spat_attributes_by_ID.Lon, spat_attributes_by_ID.T2_AVG,
                      c = 'dimgrey',
                      s=10, zorder = 1)
    ax[0].set_title('Averaged Temperature (°C) by Longitude (°) [A]', color='dimgrey', fontsize = 20)
    ax[0].axvline(bbox_llon, color='purple', linestyle = '--')
    ax[0].axvline(bbox_ulon, color='purple', linestyle = '--')
    ax[0].plot(lon_bin_plt.Lon, lon_bin_plt.T2_AVG, color = 'orange', linewidth = 3)
    ax[0].tick_params(axis='both',which='both', labelsize = 16)
    ax[0].set_xlabel('Longitude', color='dimgrey', fontsize = 18)
    ax[0].set_ylabel('Temperature', color='dimgrey', fontsize = 18)
    
    ax[1].scatter(spat_attributes_by_ID.Lat, spat_attributes_by_ID.T2_AVG,
                      c = 'dimgrey',
                      s=10, zorder = 1)
    ax[1].axvline(bbox_llat, color='purple', linestyle = '--')
    ax[1].axvline(bbox_ulat, color='purple', linestyle = '--')
    ax[1].set_title('Averaged Temperature (°C) by Latitude (°) [B]', color='dimgrey', fontsize = 20)
    ax[1].plot(lat_bin_plt.Lat, lat_bin_plt.T2_AVG, color = 'orange', linewidth = 3)
    ax[1].tick_params(axis='both',which='both', labelsize = 16)
    ax[1].set_xlabel('Latitude', color='dimgrey', fontsize = 18)    

    
    fig.savefig(savedir + 'AvgT_CWS_London_LonLat_2015-2020.png', dpi=300)
    fig.savefig(savedir + 'AvgT_CWS_London_LonLat_2015-2020.pdf')

### Fig 8 : Hour of the day, Distance to center of Domain 2, wind direction and wind speed against UHA.

if plot_fig8 == True:
    
    col = count_lczs; row = 4
    fig, ax = plt.subplots(row, col, figsize = (6*col,8.5*row))
    ax = ax.flatten()
    i = 0
    for lcz in lcz_ploted:
        tmp_df_UHA = df_UHA[spat_attributes_by_ID_GL[spat_attributes_by_ID_GL.LCZ == lcz].index.tolist()]
        tmp_df_UHA_h = tmp_df_UHA.groupby(tmp_df_UHA.index.hour).apply(np.mean)
        ### Hourly UHA is averaged among all CWS in the same LCZ 
        ax[i].scatter(df.wind_direction.values,
                      tmp_df_UHA.mean(axis=1),
                      c = lcz_colors_dict.get(lcz),
                      s=10, alpha = 0.8)
        ax[i+count_lczs].scatter(df.normalized_wind_speed.values,
                                 tmp_df_UHA.mean(axis=1),
                                 c = lcz_colors_dict.get(lcz),
                                 s=10, alpha = 0.8)
        ### Time-mean UHA for each CWS per distance to the city center
        ax[i+(count_lczs*2)].scatter(spat_attributes_by_ID_GL[spat_attributes_by_ID_GL.LCZ == lcz].Distance_Center, 
                                     tmp_df_UHA.mean(axis=0),
                                     c = lcz_colors_dict.get(lcz),
                                     s=10, alpha = 0.8)
        ax[i+(count_lczs*3)].scatter([tmp_df_UHA_h.index.values] * len(tmp_df_UHA.iloc[0,:]), 
                                     tmp_df_UHA_h.values, 
                                     c = lcz_colors_dict.get(lcz),
                                     s=10, alpha = 0.8)
        del tmp_df_UHA, tmp_df_UHA_h
        i +=1
    for ax_i in range(0,8):
        ax[ax_i].set_xlim(0,360)
        ax[ax_i].vlines(x = [90,180,270], 
                        ymin = -6.5, ymax = 6.5, linewidth=1, linestyle = '--', color='dimgrey')
        ax[ax_i].set_ylim(-6.5,6.5)
        ax[ax_i].set_title(lcz_labels_dict.get(lcz_ploted[ax_i]), fontsize = 22, color='dimgrey')
    for ax_i in range(8,16):
        ax[ax_i].set_xlim(0,20)
        ax[ax_i].vlines(x = [3,6,9], 
                        ymin = -6.5, ymax = 6.5, linewidth=1, linestyle = '--', color='dimgrey')
        ax[ax_i].set_ylim(-6.5,6.5)
    for ax_i in range(16,24):
        ax[ax_i].set_xlim(0,75)
        ax[ax_i].set_ylim(-2,2)
    for ax_i in range(24,32):
        ax[ax_i].set_xlim(0,23)
        ax[ax_i].set_ylim(-2.5,2.5)
    for ax_i in range(0,32,8):
        ax[ax_i].set_ylabel('UHA [°C]', fontsize = 22, color='dimgrey')
    for ax_i in range(32):
        ax[ax_i].tick_params(axis='both',which='both', labelsize = 18, color='dimgrey', labelcolor='dimgrey')
        ax[ax_i].spines['right'].set_visible(False)
        ax[ax_i].spines['top'].set_visible(False)       

    ax[0].set_xlabel('Wind Direction [°]', fontsize = 22, color='dimgrey')
    ax[8].set_xlabel('Wind Speed [m/s]', fontsize = 22, color='dimgrey')
    ax[16].set_xlabel('Distance to Domain 2 Center [km]', fontsize = 22, color='dimgrey')
    ax[24].set_xlabel('Hours', fontsize = 22, color='dimgrey')

    
    fig.savefig(savedir + 'UHA_CWS_London_WD_WS_Dist_Hrs_2015-2020.png', dpi=300)
    fig.savefig(savedir + 'UHA_CWS_London_WD_WS_Dist_Hrs_2015-2020.pdf')
    
### Fig 9 : Averaged 6-year observed UHA per longitude and latitude

if plot_fig9 == True:
    
    for w_dir in wind_dir_name:
        spat_attributes_by_ID_GL['UHA_AVG_' + w_dir] = df_UHA[df.Wind_Quadrant == w_dir
                                                                ].mean(axis=0).reindex(spat_attributes_by_ID_GL.index)

    proj = ccrs.PlateCarree()    
    fig, ax = plt.subplots(figsize = (27,8), subplot_kw=dict(projection=proj))
    
    for w_dir in wind_dir_name:
        order_t = np.argsort(spat_attributes_by_ID_GL['UHA_AVG_' + w_dir])

        im = ax.pcolormesh(lon_LCZ, lat_LCZ, LCZ, cmap=cmap_lcz, norm=norm_lcz, alpha = 0.7)
        ax.set_extent([bbox_llon, bbox_ulon, bbox_llat, bbox_ulat])
        im2 = ax.scatter(spat_attributes_by_ID_GL.Lon[order_t],
                          spat_attributes_by_ID_GL.Lat[order_t],
                          c = spat_attributes_by_ID_GL['UHA_AVG_' + w_dir][order_t], 
                          cmap = cmap_uha, 
                          vmin = -1,
                          vmax = 1,
                          s=70, zorder = 2, linewidth = 0.5, edgecolors = 'k')
    ax.set_title('Averaged UHA (°C)', color='dimgrey', fontsize = 20)
    ax.vlines([bbox_llon, bbox_ulon], bbox_llat, bbox_ulat, color='purple')
    ax.vlines((bbox_llon + bbox_ulon)/2, bbox_llat, bbox_ulat, color='purple', linestyle = '--')
    ax.hlines([bbox_llat, bbox_ulat], bbox_llon, bbox_ulon, color='purple')
    ax.hlines((bbox_llat + bbox_ulat)/2, bbox_llon, bbox_ulon, color='purple', linestyle = '--')
    
    ax.set_yticks([bbox_llat,bbox_ulat])
    ax.set_yticklabels([str(bbox_llat), str(bbox_ulat)],
                          color = 'dimgrey', fontsize = 10)
    ax.set_ylabel('Latitude [°]', color = 'dimgrey', fontsize = 10, rotation = 0)
    ax.yaxis.set_label_coords(0, 1.03)
    
    ax.set_xticks([bbox_llon,bbox_ulon])
    ax.set_xticklabels([str(bbox_llon), str(bbox_ulon)],
                          color = 'dimgrey', fontsize = 10)
    ax.set_xlabel('Longitude [°]', color = 'dimgrey', fontsize = 10)
    ax.xaxis.set_label_coords(0.5, -0.02)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)
    cbar_lcz = fig.colorbar(im, cax=cax, orientation='vertical', ticks = [np.arange(0.5,19.5,1)])
    cbar_lcz.ax.tick_params(axis='y',which='both',left=False,right=False,labelright=True)
    labels_lcz = cbar_lcz.ax.set_yticklabels(lcz_labels, color = 'dimgrey', fontsize = 16)
    cbar_lcz.ax.set_title('LCZ', color = 'dimgrey', fontsize = 18)
    cbar_lcz.ax.invert_yaxis()
    
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes('bottom', size='5%', pad=0.40, axes_class=plt.Axes)
    fig.add_axes(cax2)
    cbar_uha = fig.colorbar(im2, cax=cax2, orientation='horizontal', 
                            ticks = np.linspace(-1, 1, 21, endpoint=True), extend = 'both')
    cbar_uha.ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True, 
                            colors = 'dimgrey', labelsize = 12)
    cbar_uha.set_label('UHA [°C]', color = 'dimgrey', fontsize = 18)
    
    fig.tight_layout()
    fig.savefig(savedir + 'AvgUHA_Map_Domain2_London_2015-2020.png', dpi=600)
    # fig.savefig(savedir + 'AvgUHA_Map_Domain2_London_2015-2020.pdf')


### Tab 1 : Statistical test of upwind-downwind temperatures.

if plot_tab1 == True:
          
    tmp_stat_lcz = []
    prevail_winds = df.Wind_Quadrant.mask(df.normalized_wind_speed.isnull())
    
    for wd in wind_dir_name:
        tmp_df_wd_bc_upw = NA_filt_GL_df.loc[prevail_winds == wd] - NA_filt_GL_df.apply(np.nanmean, axis=0)
        tmp_df_wd_bc_dow = NA_filt_GL_df.loc[(~prevail_winds.str.contains(wd[0], na=False)) &
                                             (~prevail_winds.str.contains(wd[1], na=False))] - NA_filt_GL_df.apply(np.nanmean, axis=0)
        for lcz in range(1,17):
            ### Count how many LCZs are in
            lcz_num = len(spat_attributes_by_ID_GL[spat_attributes_by_ID_GL['LCZ'] == lcz])
            if lcz_num == 0:
                continue
            lcz_label = lcz_labels_dict.get(lcz)
            lcz_color = lcz_colors_dict.get(lcz)
            
            spat_lcz = spat_attributes_by_ID_GL[spat_attributes_by_ID_GL['LCZ'] == lcz]
            ### Calculate the anomaly to the average temperature of same LCZ located upwind
            NA_id_quad = spat_lcz[spat_lcz.Quadrant == wd].index.tolist()
            
            ### Check if there are stations in the upwind or downwind quadrant
            if len(NA_id_quad) == 0:
                continue
            
            tmp_df_wd_bc_upw_lcz = tmp_df_wd_bc_upw[NA_id_quad].mean(axis=0)
            tmp_df_wd_bc_dow_lcz = tmp_df_wd_bc_dow[NA_id_quad].mean(axis=0)
            ### Subtract the averaged temperature from the upwind quadrant to each station
            stat_ttest = stats.ttest_rel(tmp_df_wd_bc_upw_lcz, tmp_df_wd_bc_dow_lcz, 
                                         alternative='less')
           
            ### Store median temperatures per season, daytime, and LCZ
            tmp_stat_lcz.append([wd, lcz_label, np.mean(tmp_df_wd_bc_upw_lcz - tmp_df_wd_bc_dow_lcz), stat_ttest[1], np.std(tmp_df_wd_bc_upw_lcz - tmp_df_wd_bc_dow_lcz)])
        del tmp_df_wd_bc_upw, tmp_df_wd_bc_dow

### Tab 2 : Statistical test of upwind-downwind UHA anomalies in the same quadrant.

if plot_tab2 == True:
          
    diff_ar=0
    uha_stat_lcz = []
    std_stack = []
    prevail_winds = df.Wind_Quadrant.mask(df.normalized_wind_speed.isnull())

    for wd in wind_dir_name:
        tmp_df_wd_bc_upw = NA_filt_GL_df.loc[prevail_winds == wd] - NA_filt_GL_df.apply(np.nanmean, axis=0)
        tmp_df_wd_bc_dow = NA_filt_GL_df.loc[(~prevail_winds.str.contains(wd[0], na=False)) &
                                             (~prevail_winds.str.contains(wd[1], na=False))] - NA_filt_GL_df.apply(np.nanmean, axis=0)
        for lcz in range(1,17):
            ### Count how many LCZs are in
            lcz_num = len(spat_attributes_by_ID_GL[spat_attributes_by_ID_GL['LCZ'] == lcz])
            if lcz_num == 0:
                continue
            lcz_label = lcz_labels_dict.get(lcz)
            lcz_color = lcz_colors_dict.get(lcz)
            
            spat_lcz = spat_attributes_by_ID_GL[spat_attributes_by_ID_GL['LCZ'] == lcz]
            
            ### Calculate the anomaly to the average temperature of same LCZ located upwind
            NA_id_opquad = spat_lcz[(~spat_lcz.Quadrant.str.contains(wd)) &
                                 (~spat_lcz.Quadrant.str.contains(wd))].index.tolist()
            NA_id_quad = spat_lcz[spat_lcz.Quadrant == wd].index.tolist()
                
            ### Check if there are stations in the upwind or downwind quadrant
            if (len(NA_id_quad) == 0) | (len(NA_id_opquad) == 0):
                continue
            
            tmp_df_wd_bc_upw_lcz = tmp_df_wd_bc_upw[NA_id_quad].subtract(tmp_df_wd_bc_upw[NA_id_opquad].mean(axis = 1), axis = 0)
            tmp_df_wd_bc_dow_lcz = tmp_df_wd_bc_dow[NA_id_quad].subtract(tmp_df_wd_bc_dow[NA_id_opquad].mean(axis = 1), axis = 0)
            ### Subtract the averaged temperature from the upwind quadrant to each station
            stat_ttest = stats.ttest_rel(tmp_df_wd_bc_upw_lcz.mean(axis=0), tmp_df_wd_bc_dow_lcz.mean(axis=0), 
                                         alternative='less')
           
            ### Store mean temperatures per season, daytime, and LCZ
            uha_stat_lcz.append([wd, lcz_label, 
                                 np.mean(tmp_df_wd_bc_upw_lcz.mean(axis=0) - tmp_df_wd_bc_dow_lcz.mean(axis=0)),
                                 stat_ttest[1], np.std(tmp_df_wd_bc_upw_lcz.mean(axis=0) - tmp_df_wd_bc_dow_lcz.mean(axis=0))])
            diff_ar += np.mean(tmp_df_wd_bc_upw_lcz.mean(axis=0) - tmp_df_wd_bc_dow_lcz.mean(axis=0))
            std_stack.append(np.mean(tmp_df_wd_bc_upw_lcz.mean(axis=0) - tmp_df_wd_bc_dow_lcz.mean(axis=0)))
        del tmp_df_wd_bc_upw, tmp_df_wd_bc_dow
        print(diff_ar / len(uha_stat_lcz), np.std(std_stack))
