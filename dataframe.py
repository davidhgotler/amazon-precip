
import xarray as xr
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from eofs.xarray import Eof
from typing import Union,Optional

from logging import exception, warning
# Constants
# ---------
DATA_DIR = 'data/'

class dataframe():

    def __init__(self,predictors_fname:str,predictand_fname:str,data_dir:str=DATA_DIR,predictors_range:tuple=(-30,0,-80,-40,1950,2019),predictand_range:tuple=(-20,-5,-70,-50,1950,2019)):
        '''
        Initialization of dataframe object

        Parameters
        ----------
        data_dir : file directory of the data folder
        predictors_fname : netcdf filename for the predictors dataset
        predictand_fname : netcdf filename for the predictand dataset
        predictors_range : tuple containing values of lat_min,lat_max,lon_min,lon_max,yr_min,yr_max for the predictors dataset
        predictand_range : tuple containing values of lat_min,lat_max,lon_min,lon_max,yr_min,yr_max for the predictand dataset

        Attributes
        ----------
        predictors_ds : xarray dataset from ERA5 containing the predictors variables (CAPE,CIN,Geopotential,relative_humidity)
        predictand_ds : xarray dataset from GPCC containing the predictand variable (precip)
        '''
        # Open files using xarray
        # -----------------------
        # read in predictors
        predictors_ds = xr.open_dataset(data_dir+predictors_fname)
        predictors_ds.close()
        # read in predictand
        predictand_ds = xr.open_dataset(data_dir+predictand_fname)
        predictand_ds.close()
        # Select ranges for datasets
        self.predictors_ds = self.sel_range(predictors_ds,predictors_range)
        self.predictand_ds = self.sel_range(predictand_ds,predictand_range)

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
        self.predictors_da = self.predictors_ds.to_stacked_array('feature',['time'],name='Feature').reset_index('feature')
        self.predictand_da = self.predictand_ds.to_stacked_array('feature',['time'],name='predictand').reset_index('feature')
    
    def remove_nan(self):
        if not (hasattr(self,'predictors_da') or hasattr(self,'predictand_da')):
            self.flatten()
        self.predictors_da = self.predictors_da.dropna('time','all').fillna(0.)
        self.predictand_da = self.predictand_da.dropna('time','all').fillna(0.)
        

    def detrend(self):
        '''
        Preprocessing function to remove linear trends

        Attributes
        ----------
        trends_predictors : Stores the linear trends
        '''
        def get_trends(x:xr.Dataset)->xr.Dataset:
            c = x.polyfit('time',1)
            trends = xr.polyval(x.time,c)
            trends = trends.rename({old:new for old,new in zip(trends.data_vars,x.data_vars)})
            return trends
            
        # group by months
        predictors_monthly = self.predictors_ds.groupby('time.month')
        predictand_monthly = self.predictand_ds.groupby('time.month')

        # Calculate linear trends
        self.predictors_trends = predictors_monthly.map(get_trends)
        self.predictand_trends = predictand_monthly.map(get_trends)

        # Detrend and use residuals to copy metadata
        res_f = self.predictors_ds-self.trends_predictors
        res_t = self.predictand_ds-self.trends_predictand

        # Rewrite metadata
        for var in self.predictors_ds.variables:
            res_f[var].attrs=self.predictors_ds[var].attrs
        for var in self.predictand_ds.variables:
            res_t[var].attrs=self.predictand_ds[var].attrs
        res_f.attrs = self.predictors_ds.attrs
        res_t.attrs = self.predictand_ds.attrs
    
        self.predictors_ds = res_f
        self.predictand_ds = res_t
        
    def std_anom(self):
        '''
        Preprocessing function to calculate standardized anomalies

        Attributes
        ----------
        clim_f : climatology of the predictors dataset to be removed
        clim_t : climatology of the predictand dataset to be removed
        '''
        # group by months
        predictors_monthly = self.predictors_ds.groupby('time.month')
        predictand_monthly = self.predictand_ds.groupby('time.month')

        # calculate climatology
        self.clim_f = self.climatology(self.predictors_ds)
        self.clim_t = self.climatology(self.predictand_ds)

        # calculate standardized anomaly
        get_std_anom = lambda x, m, s: (x - m) / s

        std_anom_f = xr.apply_ufunc(get_std_anom,predictors_monthly,self.clim_f.mean,self.clim_f.std).drop_vars('month')
        std_anom_t = xr.apply_ufunc(get_std_anom,predictand_monthly,self.clim_t.mean,self.clim_t.std).drop_vars('month')
        # Rewrite metadata
        for var in self.predictors_ds.variables:
            std_anom_f[var].attrs=self.predictors_ds[var].attrs
        for var in self.predictand_ds.variables:
            std_anom_t[var].attrs=self.predictand_ds[var].attrs
        std_anom_f.attrs = self.predictors_ds.attrs
        std_anom_t.attrs = self.predictand_ds.attrs

        self.predictors_ds = std_anom_f
        self.predictand_ds = std_anom_t

    def get_pcs(
        self,
        n:int=10,
        method:str='meof',
    ):
        '''
        Helper function to do eof analysis using eofs.xarray
        '''
        if not (hasattr(self,'predictors_da') and hasattr(self,'predictand_da')):
            raise exception('MEOF requires stacked arrays')
        # For meof use the stacked array
        if method=='meof':
            # Perform meof analysis on stacked arrays grouped by month
            
            self.eofs = {
                'predictors_eofs':Eof(self.predictors_da,np.sqrt(np.abs(np.cos(np.deg2rad(self.predictors_da.lat.values))))),
                'predictand_eofs':Eof(self.predictand_da,np.sqrt(np.abs(np.cos(np.deg2rad(self.predictand_da.lat.values))))),
            }
            # Get pcs and return
            predictors_pcs = self.eofs['predictors_eofs'].pcs(pcscaling=1,npcs=n)
            predictand_pcs = self.eofs['predictand_eofs'].pcs(pcscaling=1,npcs=n)
            return predictors_pcs,predictand_pcs
        elif method=='variables':


        # Save eof object to access eof analysis data later

        # Return reduced array with pc predictors
            return self.eofs.pcs(pcscaling=1,npcs=n)

    def eof_plots(
        self,
        n:int,
        plot_var:bool=True,
        plot_1st_pc:bool=True,
        plot_1st_eof:bool=True,
    ):
        '''
        Plot eof/pc analysis information
        '''
        if not hasattr(self,'eofs'):
            self.eofs = {'predictors_eofs':Eof(self.predictors_da,np.sqrt(np.abs(np.cos(np.deg2rad(self.predictors_da.lat.values)))))}
            self.eofs['predictand_eofs'] = Eof(self.predictand_da,np.sqrt(np.abs(np.cos(np.deg2rad(self.predictand_da.lat.values)))))
        if plot_var:
            eofs_variance = {name.removesuffix('_eofs')+'_var':eofs['predictors_eofs'].varianceFraction(neigs=n) for name,eofs in self.eofs.items()}
            fig,axs = plt.subplots(len(eofs_variance),2,figsize=(12,6*len(eofs_variance)),constrained_layout=True)
            for ax,(name,variance) in zip(axs,eofs_variance.items()):
                variance.plot(ax=ax[0])
                variance.cumsum().plot(ax=ax[1])

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