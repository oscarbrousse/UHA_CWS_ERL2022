# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:58:17 2021

INFO:
This simple script is used to produce a map of all Netatmo CWS locations over an LCZ map
MIDAS official weather stations can also be added.
"""

import pandas as pd
import numpy as np
# import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable

#########################
###    ATTRIBUTES     ###
#########################

### City of interest

city = 'London'

datadir_NA = r'C:/Users/oscar/Documents/Work/Weather_Stations/Netatmo/London/'
savedir = datadir_NA + 'Figures/'

spat_attributes_by_ID = pd.read_csv(datadir_NA + 'List_Netatmo_stations_London_WRF_2015-2020.csv', index_col = 3)
spat_attributes_by_ID = spat_attributes_by_ID.drop('Unnamed: 0', axis = 1)

### Bounding box to study intra-urban temperature variability: 
###     - default (Gherkin): 51.1,-0.8 ; 51.9,0.6
###     - centered large (British Museum): 51.12,-0.73 ; 51.92,0.47
###     - centered large (Trafalgar Square): 51.1,-0.72 ; 51.9,0.48

bbox_llat = 51.1
bbox_ulat = 51.9
bbox_llon = -0.72
bbox_ulon = 0.48

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

LCZ_Tiff  = xr.open_rasterio(datadir_NA + 'WUDAPT_EU_LCZ_' + city + '_BD.tif')
LCZ = LCZ_Tiff.variable.data[0]
LCZ = np.nan_to_num(LCZ, nan = 0)
LCZ[LCZ == 0] = 17
LCZ_Urb = np.where(LCZ > 10, 0, 1)
lat_LCZ = LCZ_Tiff.y.values
lon_LCZ = LCZ_Tiff.x.values

### Bounding box for CWS acquisition

ulon = 1.3
llon = -1.9
ulat = 52.4
llat = 50.5

### Heathrow station coordinates (To be changed for another official weather station)
lon_h = -0.451
lat_h = 51.479

#########################
###       PLOT        ###
#########################

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize = (24,8), subplot_kw=dict(projection=proj))

im = ax.pcolormesh(lon_LCZ, lat_LCZ, LCZ, cmap=cmap_lcz, norm=norm_lcz)
ax.coastlines(resolution='10m', alpha=0.1)
ax.set_extent([llon, ulon, llat, ulat])
ax.scatter(spat_attributes_by_ID.Lon, spat_attributes_by_ID.Lat,
              color = 'black', s=10, zorder = 1, alpha = 0.8,
              label = 'Citizen Weather Stations (2015-2020)')
ax.scatter(lon_h, lat_h,
           color = 'fuchsia', marker = '^', linewidth = 2, edgecolors = 'black', s=200, zorder = 2, alpha = 1,
           label = 'Heathrow Airport Automatic Weather Station')
ax.legend(loc='upper right', fontsize = 12, facecolor = 'white', framealpha = 1, labelcolor='dimgrey')

ax.vlines([bbox_llon, bbox_ulon], bbox_llat, bbox_ulat, color='purple')
ax.vlines((bbox_llon + bbox_ulon)/2, bbox_llat, bbox_ulat, color='purple', linestyle = '--')
ax.hlines([bbox_llat, bbox_ulat], bbox_llon, bbox_ulon, color='purple')
ax.hlines((bbox_llat + bbox_ulat)/2, bbox_llon, bbox_ulon, color='purple', linestyle = '--')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1, axes_class=plt.Axes)
fig.add_axes(cax)
cbar_lcz = fig.colorbar(im, cax=cax, orientation='vertical', ticks = [np.arange(0.5,19.5,1)])
cbar_lcz.ax.tick_params(axis='y',which='both',left=False,right=False,labelright=True)
labels_lcz = cbar_lcz.ax.set_yticklabels(lcz_labels, color = 'dimgrey', fontsize = 16)
cbar_lcz.ax.set_title('LCZ', color = 'dimgrey', fontsize = 20)
cbar_lcz.ax.invert_yaxis()

ax.set_yticks([llat,ulat])
ax.set_yticklabels([str(llat), str(ulat)],
                      color = 'dimgrey', fontsize = 10)
ax.set_ylabel('Latitude [°]', color = 'dimgrey', fontsize = 10, rotation = 0)
ax.yaxis.set_label_coords(0, 1.03)

ax.set_xticks([llon,ulon])
ax.set_xticklabels([str(llon), str(ulon)],
                      color = 'dimgrey', fontsize = 10)
ax.set_xlabel('Longitude [°]', color = 'dimgrey', fontsize = 10)
ax.xaxis.set_label_coords(0.5, -0.02)
ax.set_title('Local Climate Zones map, CWS and AWS locations in the domains of study', fontsize = 14, color = 'dimgrey')

fig.tight_layout()
fig.savefig(savedir + 'CWSLoc_LCZMap_Domains_HeathrowLoc.png', dpi=600)
fig.savefig(savedir + 'CWSLoc_LCZMap_Domains_HeathrowLoc.pdf')