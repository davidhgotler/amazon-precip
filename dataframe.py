import xarray as xr
import numpy as np
import pandas as pd

from eofs.xarray import Eof
from sklearn.decomposition import FactorAnalysis as FA

from matplotlib import pyplot as plt

from typing import Tuple,List,Union

from logging import exception, warning

# Constants
# ---------
DATA_DIR = 'data/'
FIG_DIR = 'Figures/'
# Useful functions
# ----------------
def sel_months(
    X:Union[xr.DataArray,xr.Dataset],
    months:Union[int,list[int],np.ndarray],
):
    return X.sel(time=X['time.month'].isin(months))

def year2date(
    yrs:Union[int,List[int],
    np.ndarray],
):
    if isinstance(yrs,int):
        return np.datetime64('{}'.format(yrs),'Y')
    elif isinstance(yrs,np.ndarray) or isinstance(yrs,list):
        yrs_list = [np.datetime64('{}'.format(y),'Y') for y in yrs]
        return np.array(yrs_list)

def weight(X:xr.DataArray,weight:str='cos_lat'):
    if weight=='cos_lat':
        weights_arr = np.sqrt(np.cos(np.abs(np.deg2rad(X.lat))))
    return X*weights_arr

def categorise_ndcoords(array, time_name):
    """
    Categorise all the non-dimension coordinates of an
    `xarray.DataArray` into those that span only time, those that span
    only space, and those that span both time and space.

    Parameters
    ----------
    `array` : An `xarray.DataArray`.
    `time_name` : Name of the time dimension coordinate in the input `array`.

    Returns
    -------
    `time_coords` : A list of coordinates that span only the time dimension.
    `space_coords` : A list of coordinates that span only the space dimensions.
    `time_space_coords` : A list of coordinates that span both the time and space coordinates.

    ref:
    Dawson, A. (2016). eofs: A Library for EOF Analysis of Meteorological, Oceanographic, and Climate Data. Journal of Open Research Software, 4(1), e14. DOI: http://doi.org/10.5334/jors.122
    """
    ndcoords = [(name,coord) for name, coord in array.coords.items()
                if name not in array.dims]
    time_ndcoords = {}
    space_ndcoords = {}
    time_space_ndcoords = {}
    for name,coord in ndcoords:
        if coord.dims == (time_name,):
            time_ndcoords[name] = coord
        elif coord.dims:
            if time_name in coord.dims:
                time_space_ndcoords[name] = coord
            else:
                space_ndcoords[name] = coord
    return dict(time_ndcoords=time_ndcoords,space_ndcoords=space_ndcoords,time_space_ndcoords=time_space_ndcoords)

def get_dataset(
    array:xr.DataArray
)->xr.Dataset:
    '''
    Helper function to unstack a flattened data array to a dataset.

    Parameters
    ----------
    `array` : Data Array containing variable(s) that have been stacked/flattened into (sample,predictors)
    '''
    ft_dim = array.dims[1]
    coords = categorise_ndcoords(array,array.dims[0])
    predictors_coord_names = list(coords['space_ndcoords'].keys())
    # Check if the predictors dimension is a multi-index
    if not ft_dim in array.indexes.keys():
        array = array.set_index({ft_dim:predictors_coord_names})
    return array.unstack(ft_dim).to_dataset('variable')

class dataframe():

    def __init__(
        self,
        predictors_fname:str,
        predictand_fname:str,
        data_dir:str=DATA_DIR,
        predictors_range:Tuple[int]=(-30,0,-80,-40,1950,2019),
        predictand_range:Tuple[int]=(-20,-5,-70,-50,1950,2019),
        predictors_months:Tuple[int]=(6,7,8),
        predictand_months:Tuple[int]=(9,10,11),
        predictors_coarsen:Tuple[int]=(3,2.5,2.5),
        predictand_coarsen:Tuple[int]=(1,1,1)
    ):
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
        # Select ranges for datasets as spanning both ranges
        self.predictors_ds = self.sel_range(predictors_ds,predictors_range,predictors_months,predictors_coarsen)
        self.predictand_ds = self.sel_range(predictand_ds,predictand_range,predictand_months,predictand_coarsen)

    def sel_range(self,ds:xr.Dataset,range:Tuple[int],months:Tuple[int],coarsen:Tuple[int])->xr.Dataset:
        '''
        Helper function to select coordinate ranges and reformat coordinates
        '''
        # Unwrap the coordinate ranges
        lat_min,lat_max,lon_min,lon_max,yr_min,yr_max=range
        dt,dlat,dlon = coarsen
        # Select years
        ds = ds.sel(time=np.logical_and(ds['time.year'] >= yr_min,ds['time.year']<=yr_max))
        ds = ds.sel(time=ds['time.month'].isin(months))

        # Reformat lat,lon
        lon_attrs = ds.lon.attrs
        ds['lon'] = xr.where(ds['lon']<180,ds['lon'],ds['lon']-360)
        ds.lon.attrs = lon_attrs
        ds = ds.sortby(['lat','lon','time'])    
        # coarsen predictors
        lat_bins = np.arange(lat_min-dlat/2,lat_max+dlat,dlat)
        lon_bins = np.arange(lon_min-dlon/2,lon_max+dlon,dlon)
        lat_center = np.arange(lat_min,lat_max+dlat,dlat)
        lon_center = np.arange(lon_min,lon_max+dlon,dlon)
        # Select lat,lon ranges
        ds = ds.groupby_bins('lat',lat_bins,labels=lat_center).mean(dim='lat')
        ds = ds.groupby_bins('lon',lon_bins,labels=lon_center).mean(dim='lon')
        ds = ds.rename(lat_bins='lat',lon_bins='lon')
        if dt>1:
            ds=ds.coarsen(time=dt).mean()

        return ds
    
    def flatten(self):
        self.predictors_da = self.predictors_ds.to_stacked_array('feature',['time'],name='predictors').reset_index('feature')
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
        res_f = self.predictors_ds-self.predictors_trends
        res_t = self.predictand_ds-self.predictand_trends

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

    def get_pcs(self,X,n=10,method='varimax'):
        # Get unrotated eof's
        weights_arr = np.sqrt(np.cos(np.abs(np.deg2rad(X.lat))))
        self.eof=Eof(X,weights=weights_arr)
        X_pcs = self.eof.pcs(pcscaling=1,npcs=n)
        X_eofs = self.eof.eofs(eofscaling=2,neofs=n)
        X_var = self.eof.varianceFraction(neigs=n)
        tot_var = self.eof.totalAnomalyVariance()
        # Find rotated eof's using varimax
        self.rot_eof = FA(n_components=n,rotation='varimax',max_iter=10000)
        X_rot_pcs = xr.DataArray(self.rot_eof.fit_transform(weight(X)),coords=X_pcs.coords)
        X_rot_eofs = xr.DataArray(self.rot_eof.components_,coords=X_eofs.coords)
        X_rot_var = ((X_rot_eofs**2).sum(dim='feature')/tot_var).rename('variance_fractions')

        X_tot_var = ((X_eofs**2).sum()/tot_var).values
        X_rot_tot_var = ((X_rot_eofs**2).sum()/tot_var).values
        print('Total variance explained by {} eof modes: {:.3f}'.format(n,X_tot_var))
        print('Total variance explained by {} rotated eof modes: {:.3f}'.format(n,X_rot_tot_var))
        self.plot_eofs(X_pcs,X_eofs,X_var,title='PCA')
        self.plot_eofs(X_rot_pcs,X_rot_eofs,X_rot_var)
        if method=='PCA':
            return X_pcs
        elif method=='varimax':
            return X_rot_pcs
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