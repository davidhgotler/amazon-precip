from cProfile import label
import logging

import numpy as np
import pandas as pd
import xarray as xr

import holoviews as hv
import hvplot.xarray

import matplotlib as mpl
from matplotlib import axes, pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap,ScalarMappable

import sklearn as skl
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from eofs.xarray import Eof

from abc import ABC, abstractmethod
from copy import copy, deepcopy
from typing import List,Tuple,Union,Optional, final

# Constants
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
    `array` : Data Array containing variable(s) that have been stacked/flattened into (sample,feature)
    '''
    ft_dim = array.dims[1]
    coords = categorise_ndcoords(array,array.dims[0])
    feature_coord_names = list(coords['space_ndcoords'].keys())
    # Check if the feature dimension is a multi-index
    if not ft_dim in array.indexes.keys():
        array = array.set_index({ft_dim:feature_coord_names})
    return array.unstack(ft_dim).to_dataset('variable')

# Forecaster base class
# ---------------------
class ForecasterBase(ABC):
    @abstractmethod
    def train_test_split(
        self,
        test_yrs:Union[int,List[int]],
        autoreg:xr.DataArray,
        exog:Union[xr.DataArray,xr.Dataset]=None,
    ) -> Tuple[xr.DataArray]:
        '''
        Select,flatten and concatenate autoregressive and exogenous features then split into training, test datasets.

        Parameters
        ----------
        `test_yrs` : List of years to include in the test set. Or an integer number of years to include. The most recent years will be used for the test set.
        `autoreg` : DataArray containing target data. Split into features (if autoreg features are included)/targets, and training/test sets.
        `exog` : DataArray or Dataset containing exogeneous features data. Split into training/test sets.
        
        Returns
        -------
        `train_data` : DataArray(s) for the training set years: X_train,y_train
        `test_data` : DataArray(s) for the test set years: X_test,y_test
        '''
        pass
    
    @abstractmethod
    def get_training_matrix(
        self,
        m:int,
        s:int,
        X:xr.DataArray,
        y:xr.DataArray,
    ) -> Tuple[xr.DataArray]:
        '''
        Selects the correct training matrix for the current step, s and stacks lag variables.

        Parameters
        ----------
        `m` : Current target month.
        `s` : Current step in training/predicting used as an index. s should be on the range [0,steps-1]
        `X` : Features dataset
        `y` : Target dataset

        Returns
        -------
        X_m : DataArray containing the lag features for the current step, s
        y_m : DataArray from the target month for the current step, s
        '''
        pass

    @abstractmethod
    def fit(
        self,
        X:xr.DataArray,
        y:xr.DataArray,
    ):
        '''
        Fit the data using regressor sklearn model
        '''
        pass

    @abstractmethod
    def predict(
        self,
        X:xr.DataArray,
        y:xr.DataArray,
    ) -> None:
        '''
        Predict the data using autoregressive forecast methods (direct or recursive)
        '''
        pass

# Direct forecaster class
# -----------------------
class DirectForecaster(ForecasterBase):
    '''
    Class for direct multioutput forecast prediction. N steps are predicted independently from L lags using autoregressive and/or exogeneous features.
    '''
    def __init__(
        self,
        regressor,
        lags:Optional[Union[int,list,np.ndarray]]=1,
        target_months:Optional[Union[list,np.ndarray]]=None,
        steps:Optional[int]=1,
        include_autoreg:bool=False,
        include_exog:bool=True,
        avg_lags:bool=True,
        pca_features:bool=True,
        pca_target:bool=False,
    ):
        '''
        Parameters
        ----------
        `regressor` : Sklearn compatible regression model used for prediction
        `lags` : Number of lags to include
        `target_months` : List of target months. If none are defined defaults to all 12.
        `steps` : Number of steps to predict at a time. Length of target_months should be a multiple of steps. If not, steps will be set to the length of target_months.
        '''
        self.regressor = regressor
        if target_months==None:
            self.target_months = np.arange(12)+1
        elif not isinstance(target_months,np.ndarray):
            self.target_months = np.array(target_months)
        else:
            self.target_months = target_months

        if len(self.target_months)%steps == 0:
            self.steps = steps
        else:
            self.steps = len(self.target_months)

        if (not include_autoreg and not include_exog) or (lags==None and not include_exog):
            raise Exception('No Features???')
        else:
            self.lags = lags
            self.include_autoreg = include_autoreg
            self.include_exog = include_exog
        self.avg_lags = avg_lags
        self.pca_features = pca_features
        self.fitted = False
    
    def train_test_split(
        self,
        test_yrs: Union[int, List[int],np.ndarray],
        autoreg: xr.DataArray,
        exog: xr.DataArray,
    ) -> Tuple[xr.DataArray]:
        '''
        Select,flatten and concatenate autoregressive and exogenous features then split into training, test datasets. For direct prediction feature months are fixed.

        Parameters
        ----------
        `test_yrs` : List of years to include in the test set. Or an integer number of years to include. The most recent years will be used for the test set.
        `autoreg` : Dataset containing target data. Split into features (if autoreg features are included)/targets, and training/test sets.
        `exog` : Dataset or Dataset containing exogeneous features data. Split into training/test sets.
        
        Returns
        -------
        `train_data` : Dataset(s) for the training set years: X_train,y_train
        `test_data` : Dataset(s) for the test set years: X_test,y_test
        '''
        if isinstance(test_yrs,np.ndarray):
            self.test_yrs = test_yrs
        elif isinstance(test_yrs,int):
            test_yrs = autoreg['time.year'].values[-1]-np.arange(test_yrs)
            self.test_yrs = test_yrs
        else:
            test_yrs = np.array(test_yrs)
            self.test_yrs = test_yrs
        
        # Select target months
        y = sel_months(autoreg,self.target_months)

        # Select feature months for direct prediction on a range of (1,12)
        feature_months = np.arange(self.target_months.min()-self.lags,self.target_months.max()-self.steps+1)
        feature_months = np.where(feature_months>0,feature_months,feature_months+12)

        if self.include_autoreg and self.include_exog:
            X_autoreg = sel_months(autoreg,feature_months)
            X_exog = sel_months(exog,feature_months)
            X = xr.concat([X_autoreg,X_exog],'feature',coords='all').rename('Features')
          
        elif self.include_autoreg:
            X = sel_months(autoreg,feature_months)
        elif self.include_exog:
            X = sel_months(exog,feature_months)

        # Check if some lag features should be from the previous year
        # This is probably bugged or logically wrong but it won't be used
        if self.target_months.min()-self.lags < 1:
            test_yrs_ft = np.concatenate(self.test_yrs,self.test_yrs.min()-1)

            X_train = X.drop_sel(time=X['time'][X['time.year'].isin(test_yrs_ft)])
            y_train = y.drop_sel(time=y['time'][y['time.year'].isin(test_yrs)])
            X_test = X.sel(time=X['time.year'].isin(test_yrs_ft))
            y_test = y.sel(time=y['time.year'].isin(test_yrs))
        # Otherwise split normally
        else:
            X_train = X.drop_sel(time=X['time'][X['time.year'].isin(test_yrs)])
            y_train = y.drop_sel(time=y['time'][y['time.year'].isin(test_yrs)])
            X_test = X.sel(time=X['time.year'].isin(test_yrs))
            y_test = y.sel(time=y['time.year'].isin(test_yrs))
        return X_train,y_train,X_test,y_test

    def get_training_matrix(
        self,
        m: int,
        s: int, 
        X: xr.Dataset, 
        y: xr.Dataset,
    ) -> Tuple[xr.DataArray]:
        '''
        Selects the correct training matrix for the current step, s and stacks lag variables.

        Parameters
        ----------
        `m` : Current target month.
        `s` : Current step in training/predicting used as an index. s should be on the range [0,steps-1]
        `X` : Features dataset
        `y` : Target dataset

        Returns
        -------
        X_m : DataArray containing the lag features for the current step, s
        y_m : DataArray from the target month for the current step, s
        '''
        # Select feature months for direct prediction on a range of (1,12)
        feature_months = np.arange(m-self.lags,m)-s
        feature_months = np.where(feature_months>0,feature_months,feature_months+12)

        # Select target month and make sample index as the year
        y_m = sel_months(y,m)
        # y_m = y_m.assign_coords({'sample':('time',y_m['time.year'].data)})
        # y_m = y_m.swap_dims(time='sample')

        X_m = sel_months(X,feature_months)
        if self.avg_lags:
            X_m = X_m.coarsen(time=len(feature_months)).mean()
        else:
            X_m = X_m.coarsen(time=len(feature_months)).construct(time=('sample','month'),keep_attrs=True)
            X_m = X_m.assign_coords({'sample':year2date(X_m['time.year'][:,0].values),'month':X_m['time.month'][0,:].values})
            X_m = X_m.stack(lag_feature=X_m.dims[1:])
        if self.pca_features:
            if not self.fitted:
                self.eof = Eof(X_m,np.sqrt(np.cos(np.deg2rad(X_m.lat))))
                X_m_pc = self.eof.pcs(pcscaling=1,npcs=10)
                return X_m_pc,y_m
            else:
                X_m_pc = self.eof.projectField(X_m,neofs=10,eofscaling=1)
                return X_m_pc,y_m
        else: 
            return X_m,y_m
    
    def fit(self, X: xr.DataArray, y: xr.DataArray):
        steps_arr = np.arange(len(self.target_months))%self.steps
        forecaster = {}
        for m,s in zip(self.target_months,steps_arr):
            X_m,y_m = self.get_training_matrix(m,s,X,y)
            forecaster[m] = deepcopy(self.regressor.fit(X_m,y_m))
        self.forecaster = forecaster
        self.fitted = True

    def predict(self, X: xr.DataArray, y: xr.DataArray) -> None:
        steps_arr = np.arange(len(self.target_months))%self.steps
        y_pred_list = []
        for m,s in zip(self.target_months,steps_arr):
            X_m,y_m = self.get_training_matrix(m,s,X,y)
            y_pred_list.append(xr.DataArray(self.forecaster[m].predict(X_m),coords=y_m.coords))
        y_pred = xr.concat(y_pred_list,'time')
        y_pred = y_pred.sortby('time')
        # y_pred = y_pred.swap_dims(sample='time').sortby('time')
        return y_pred

    def get_pcs(
        self,
        X:xr.DataArray,
        n:int=10,
        name:str='features_eofs',
    ):
        '''
        Helper function to do eof analysis using eofs.xarray
        '''
        # Save eof object to access eof analysis data later
        if not hasattr(self,'eofs'):
            self.eofs = {name:Eof(X,np.sqrt(np.abs(np.cos(np.deg2rad(X.lat.values)))))}
        else:
            self.eofs[name] = Eof(X,np.sqrt(np.abs(np.cos(np.deg2rad(X.lat.values)))))
        # Return reduced array with pc features
        return self.eofs.pcs(pcscaling=1,npcs=n)
    
    def cross_validation(
        self,
        X:xr.DataArray,
        y:xr.DataArray,
        params:dict,
    ):
        pass

    def data_plots(
        self,
        y_true:Union[xr.DataArray,xr.Dataset],
        y_pred:Union[xr.DataArray,xr.Dataset],
        plot_facet:bool=True,
        plot_timeseries:bool=False,
        plot_corr_map:bool=False,
        plot_error:bool=False,
        individual:bool=True,
        save_figs:bool=True,
        model_name:str=None,
        forecaster_name:str=None,
    ):
        '''
        Plotting function for comparing true field values to predicted

        Parameters
        ----------
        `y_true` : DataArray containing true values
        `y_pred` : DataArray contatining predicted values
        `plot_facet` : If true - facet plot of the grid data
        `plot_timeseries` : If true - plot an avg timeseries using cos(lat) weights
        `plot_error` : If true - plot the error
        `plot_corr_map` : If true - plot a correlation map of the region
        '''

        if isinstance(y_true,xr.DataArray):
            y_true = get_dataset(y_true)
        if isinstance(y_pred,xr.DataArray):
            y_pred = get_dataset(y_pred)
        plots = {}
        if plot_facet:
            # Get aspect ratio of grid
            lon_range = y_true.lon.max() - y_true.lon.min()
            lat_range = y_true.lat.max() - y_true.lat.min()
            aspect = lon_range / lat_range
            bounds = np.linspace(-2,2,10)
            cmap = mpl.cm.Spectral
            norm = BoundaryNorm(bounds, cmap.N, extend='both')
            sm = ScalarMappable(norm=norm, cmap=cmap)
            if individual:
                # Plot individual variables using facetgrid
                p_true = y_true.precip.plot(col='time',col_wrap=self.steps,vmin=-2,vmax=2,cmap=cmap,aspect=aspect,cbar_kwargs={'label':'precip standard anomaly','shrink':0.6})
                p_true.map_dataarray(xr.plot.contour,x='lon',y='lat',levels=bounds,colors='k',add_colorbar=False)
                if save_figs:
                    plt.savefig(f'{FIG_DIR}facet_precip_true.png',facecolor='white',transparent=False)
                else:
                    plots['facet_true']=(p_true.fig,p_true.axes)
                p_pred = y_pred.precip.plot(col='time',col_wrap=self.steps,vmin=-2,vmax=2,cmap=cmap,aspect=aspect,cbar_kwargs={'label':'predicted precip standard anomaly','shrink':0.6})
                p_pred.map_dataarray(xr.plot.contour,x='lon',y='lat',levels=bounds,colors='k',add_colorbar=False)
                if save_figs:
                    plt.savefig(f'{FIG_DIR}facet_precip_pred_{forecaster_name}_{model_name}.png',facecolor='white',transparent=False)
                else:
                    plots['facet_pred']=(p_pred.fig,p_true.axes)
            else:
                fig = plt.figure(constrained_layout=True,figsize=(int(aspect*3),3))
                fig_true,fig_pred = fig.subfigures(1,2,width_ratios=[1,1.1])

                fig_true.suptitle('ground truth',fontsize='x-large')
                fig_pred.suptitle('predicted',fontsize='x-large')
                fig.suptitle('{} of precipitation using {}'.format(forecaster_name,model_name),fontsize='xx-large')

                axs_true = fig_true.subplots(int(len(y_true.time)/self.steps),self.steps)
                axs_pred = fig_pred.subplots(int(len(y_true.time)/self.steps),self.steps)

                # Plot true values
                for t,ax in enumerate(axs_true.flatten()):
                    y_true.precip.isel(time=t).plot(ax=ax,vmin=-2,vmax=2,cmap=cmap,add_colorbar=False)
                    y_true.precip.isel(time=t).plot(ax=ax,vmin=-2,vmax=2,colors='k',add_colorbar=False)
                # Plot predicted values
                for t,ax in enumerate(axs_pred.flatten()):
                    y_pred.precip.isel(time=t).plot(ax=ax,vmin=-2,vmax=2,cmap=cmap,add_colorbar=False)
                    y_pred.precip.isel(time=t).plot(ax=ax,vmin=-2,vmax=2,colors='k',add_colorbar=False)

                fig.colorbar(sm,ax=axs_pred,shrink=0.6)
                if save_figs:
                    fig.save_fig(f'{FIG_DIR}facet_precip_{forecaster_name}_{model_name}.png',facecolor='white',transparent=False)
                else:
                    return fig
        if plot_timeseries:
            # Get cos(lat) weights
            weights = np.cos(np.deg2rad(y_true.lat))

            # Group by year for subplots
            y_true_group = y_true.groupby('time.year')
            y_pred_group = y_pred.groupby('time.year')

            fig,axs = plt.subplots(2,2,figsize=(12,8))
            for ax,(year,y_true_n),(_,y_pred_n) in zip(axs.flatten(),list(y_true_group),list(y_pred_group)):
                # Calculate weighted mean
                y_true_mean = y_true_n.precip.weighted(weights).mean(dim=['lat','lon']).to_series()
                y_pred_mean = y_pred_n.precip.weighted(weights).mean(dim=['lat','lon']).to_series()
                # Calculate weighted std for error bars
                y_true_std = y_true_n.precip.std(dim=['lat','lon']).to_series()
                y_pred_std = y_pred_n.precip.std(dim=['lat','lon']).to_series()
                ax.errorbar(y_true_mean.index,y_true_mean,yerr=y_true_std,fmt='o',capsize=10,color='darkcyan',label='true')
                ax.errorbar(y_pred_mean.index,y_pred_mean,yerr=y_pred_std,fmt='v',capsize=10,color='darkblue',label='predicted')
                ax.axhline(y=0, color="black", linestyle="--",linewidth=0.6)

                ax.set_ylim(-2,2)
                ax.set_xticks(y_true_mean.index)
                ax.set_xticklabels(y_true_mean.index.month_name())
                ax.set_title(year,loc='left')
                
                # ax.legend(loc='upper right')
                # ax.grid()

            fig.suptitle('Weighted Mean Precipitation Standardized Anomaly on season: SON')
            handles,labels = axs[0,0].get_legend_handles_labels()
            fig.legend(handles,labels)

            if save_figs:
                plt.savefig(f'{FIG_DIR}timeseries_precip_{forecaster_name}_{model_name}.png',facecolor='white',transparent=False)
            else:
                return fig,ax

            # p_pred_agg = y_pred.precip.hvplot(x='time',aggregate=['lat','lon'])
            # p_pred_mean = y_pred.precip.weighted(weights).mean(dim=['lat','lon']).hvplot(x='time')
            # p_pred = p_pred_agg*p_pred_mean

            # p_ts_pred = y_pred.precip.weighted(weights).mean(dim=['lat','lon']).plot()
        return plots

    def eof_plots(
        self,
        eofs:Eof,
        n:int,
        plot_var:bool=True,
        plot_1st_pc:bool=True,
        plot_1st_eof:bool=True,
    ):
        '''
        Plot eof/pc analysis information
        '''
        


class RecursiveForecaster(ForecasterBase):
    def __init__(self) -> None:
        pass

    def train_test_split(self, test_yrs: Union[int, List[int]], autoreg: xr.DataArray, exog: Union[xr.DataArray, xr.Dataset] = None) -> Tuple[xr.DataArray]:
        return super().train_test_split(test_yrs, autoreg, exog)
    
    def get_training_matrix(self, m: int, s: int, X: xr.DataArray, y: xr.DataArray) -> Tuple[xr.DataArray]:
        return super().get_training_matrix(m, s, X, y)
    
    def fit(self, X: xr.DataArray, y: xr.DataArray):
        return super().fit(X, y)
    
    def predict(self, X: xr.DataArray, y: xr.DataArray) -> None:
        return super().predict(X, y)


