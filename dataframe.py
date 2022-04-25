import xarray as xr
import numpy as np
import pandas as pd
from eofs.xarray import Eof
from typing import Union,Optional

class dataframe():

    def __init__(self,data_dir:str,features_fname:str,target_fname:str,features_range:tuple=(-30,0,-80,-40,1979,2019),target_range:tuple=(-20,-5,-70,-50,1979,2019)):
        '''
        Initialization of dataframe object

        Parameters
        ----------
        data_dir : file directory of the data folder
        features_fname : netcdf filename for the features dataset
        target_fname : netcdf filename for the target dataset
        features_range : tuple containing values of lat_min,lat_max,lon_min,lon_max,yr_min,yr_max for the features dataset
        target_range : tuple containing values of lat_min,lat_max,lon_min,lon_max,yr_min,yr_max for the target dataset

        Attributes
        ----------
        features_ds : xarray dataset from ERA5 containing the features variables (CAPE,CIN,Geopotential,relative_humidity)
        target_ds : xarray dataset from GPCC containing the target variable (precip)
        '''
        # Open files using xarray
        # -----------------------
        # read in features
        features_ds = xr.open_dataset(data_dir+features_fname)
        features_ds.close()
        # read in target
        target_ds = xr.open_dataset(data_dir+target_fname)
        target_ds.close()
        # Select ranges for datasets
        self.features_ds = self.sel_range(features_ds,features_range)
        self.target_ds = self.sel_range(target_ds,target_range)

    def sel_range(self,ds:xr.Dataset,range:tuple)->xr.Dataset:
        '''
        Helper function to select coordinate ranges and reformat coordinates
        '''
        # Unwrap the coordinate ranges
        lat_min,lat_max,lon_min,lon_max,yr_min,yr_max=range
        # Select years
        ds = ds.sel(time=np.logical_and(ds['time.year'] >= yr_min,ds['time.year']<=yr_max))
        # Reformat latitude and longitude
        lon_attrs = ds.lon.attrs
        ds['lon'] = xr.where(ds['lon']<180,ds['lon'],ds['lon']-360)
        ds.lon.attrs = lon_attrs
        ds = ds.sortby(['lat','lon','time'])
        # Select lat,lon ranges
        ds = ds.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
        return ds
    
    def flatten(self):
        self.features_da = self.features_ds.to_stacked_array('feature',['time'],name='Feature').reset_index('feature')
        self.target_da = self.target_ds.to_stacked_array('feature',['time'],name='Target').reset_index('feature')
    
    def remove_nan(self):
        self.features_ds = self.features_ds.dropna('time','all').fillna(0)
        self.target_ds = self.target_ds.dropna('time','all').fillna(0)

    def detrend(self):
        '''
        Preprocessing function to remove linear trends

        Attributes
        ----------
        trends_features : Stores the linear trends
        '''
        def get_trends(x:xr.Dataset)->xr.Dataset:
            c = x.polyfit('time',1)
            trends = xr.polyval(x.time,c)
            trends = trends.rename({old:new for old,new in zip(trends.data_vars,x.data_vars)})
            return trends
            
        # group by months
        features_monthly = self.features_ds.groupby('time.month')
        target_monthly = self.target_ds.groupby('time.month')

        # Calculate linear trends
        self.trends_features = features_monthly.map(get_trends)
        self.trends_target = target_monthly.map(get_trends)

        # Detrend and use residuals to copy metadata
        res_f = self.features_ds-self.trends_features
        res_t = self.target_ds-self.trends_target

        # Rewrite metadata
        for var in self.features_ds.variables:
            res_f[var].attrs=self.features_ds[var].attrs
        for var in self.target_ds.variables:
            res_t[var].attrs=self.target_ds[var].attrs
        res_f.attrs = self.features_ds.attrs
        res_t.attrs = self.target_ds.attrs
    
        self.features_ds = res_f
        self.target_ds = res_t
        
    def std_anom(self):
        '''
        Preprocessing function to calculate standardized anomalies

        Attributes
        ----------
        clim_f : climatology of the features dataset to be removed
        clim_t : climatology of the target dataset to be removed
        '''
        # group by months
        features_monthly = self.features_ds.groupby('time.month')
        target_monthly = self.target_ds.groupby('time.month')

        # calculate climatology
        self.clim_f = self.climatology(self.features_ds)
        self.clim_t = self.climatology(self.target_ds)

        # calculate standardized anomaly
        get_std_anom = lambda x, m, s: (x - m) / s

        std_anom_f = xr.apply_ufunc(get_std_anom,features_monthly,self.clim_f.mean,self.clim_f.std).drop_vars('month')
        std_anom_t = xr.apply_ufunc(get_std_anom,target_monthly,self.clim_t.mean,self.clim_t.std).drop_vars('month')
        # Rewrite metadata
        for var in self.features_ds.variables:
            std_anom_f[var].attrs=self.features_ds[var].attrs
        for var in self.target_ds.variables:
            std_anom_t[var].attrs=self.target_ds[var].attrs
        std_anom_f.attrs = self.features_ds.attrs
        std_anom_t.attrs = self.target_ds.attrs

        self.features_ds = std_anom_f
        self.target_ds = std_anom_t

    class climatology():
        '''
        Represents the monthly climatology of the dataset
        '''
        def __init__(self,ds:xr.Dataset):
            '''
            Initializes by calculating the climatology on the input dataset

            Parameters
            ----------
            ds : xarray dataset

            Attributes
            ----------
            mean : monthly climatological mean
            std : monthly climatological standard deviation
            '''
            self.mean = ds.groupby('time.month').mean(dim='time',keep_attrs=True)
            self.std = ds.groupby('time.month').std(dim='time',keep_attrs=True)