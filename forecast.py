from lib2to3.refactor import get_all_fix_names
from pkgutil import get_data
from unittest import result
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error,mean_squared_error, precision_score, r2_score, recall_score
from sklearn.preprocessing import StandardScaler
import xarray as xr

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap,ScalarMappable

from dataframe import *
import datetime

from eofs.xarray import Eof

import sklearn as skl
from sklearn.decomposition import PCA,FactorAnalysis as FA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid


from logging import exception
from inspect import isfunction,signature
from collections import defaultdict
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from typing import Callable, List,Tuple,Type,Union,Optional
from warnings import warn
from joblib import Parallel, delayed

# Forecaster base class
# ---------------------
class ForecasterBase(ABC):

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    @abstractmethod
    def train_test_split(
        self,
        test_yrs:Union[int,List[int]],
        predictand:xr.DataArray,
        predictors:Union[xr.DataArray,xr.Dataset]=None,
    ) -> Tuple[xr.DataArray]:
        '''
        Select,flatten and concatenate autoregressive (predictand) and predictorsenous predictors then split into training, test datasets.

        Parameters
        ----------
        `test_yrs` : List of years to include in the test set. Or an integer number of years to include. The most recent years will be used for the test set.
        `predictand` : DataArray containing predictand data. Split into predictors (if predictand predictors are included)/predictands, and training/test sets.
        `predictors` : DataArray or Dataset containing exogenous (predictors) predictors data. Split into training/test sets.
        
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
        `m` : Current predictand month.
        `s` : Current step in training/predicting used as an index. s should be on the range [0,steps-1]
        `X` : predictors dataset
        `y` : predictand dataset

        Returns
        -------
        X_m : DataArray containing the lag predictors for the current step, s
        y_m : DataArray from the predictand month for the current step, s
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
        Predict the data using autoregressive (predictand) forecast methods (direct or recursive)
        '''
        pass

# Direct forecaster class
# -----------------------
class DirectForecaster(ForecasterBase):
    '''
    Class for direct multioutput forecast prediction. N steps are predicted independently from L lags using autoregressive (predictand) and/or exogenous (predictors) predictors.
    '''
    def __init__(
        self,
        reg,
        predictors_months:Tuple[int]=(6,7,8),
        predictand_months:Tuple[int]=(9,10,11),
        lags:int=None,
        steps:int=None,
        lead:int=None,
        include_predictand:bool=False,
        include_predictors:bool=True,
    ):
        '''
        Parameters
        ----------
        `reg` : Sklearn compatible regression model used for prediction
        `lags` : Number of lags to include
        `predictand_months` : List of predictand months. If none are defined defaults to all 12.
        `steps` : Number of steps to predict at a time. Length of predictand_months should be a multiple of steps. If not, steps will be set to the length of predictand_months.
        '''

        self.reg = reg
        
        if isinstance(predictors_months,np.ndarray):
            self.predictors_months = predictors_months
        else:
            self.predictors_months = np.array(predictors_months)
        
        if isinstance(predictand_months,np.ndarray):
            self.predictand_months = predictand_months
        else:
            self.predictand_months = np.array(predictand_months)
        
        if lags==None:
            self.lags=len(predictors_months)
        else: self.lags = lags
        
        if len(self.predictand_months)%steps == 0:
            self.steps = steps
        else:
            self.steps = len(self.predictand_months)
        
        if lead==None:
            self.lead=1
        else:
            self.lead=lead

        if (not include_predictand and not include_predictors) or (lags==None and not include_predictors):
            raise Exception('No predictors???')
        else:
            self.include_predictand = include_predictand
            self.include_predictors = include_predictors
        
        self.fitted = False
    
    def reset(self):
        if hasattr(self,'forecaster_'):
            del self.forecaster_
        self.fitted=False
    
    def train_test_split(
        self,
        predictors: xr.DataArray,
        predictand: xr.DataArray,
        test_yrs: Union[int, List[int],np.ndarray],
    ) -> Tuple[xr.DataArray]:
        '''
        Select,flatten and concatenate autoregressive (predictand) and predictorsenous predictors then split into training, test datasets. For direct prediction predictors months are fixed.

        Parameters
        ----------
        `predictors` : Dataset or Dataset containing exogenous (predictors) predictors data. Split into training/test sets.
        `predictand` : Dataset containing predictand data. Split into predictors (if predictand predictors are included)/predictands, and training/test sets.
        `test_yrs` : List of years to include in the test set. Or an integer number of years to include. The most recent years will be used for the test set.
        
        Returns
        -------
        `train_data` : Dataset(s) for the training set years: X_train,y_train
        `test_data` : Dataset(s) for the test set years: X_test,y_test
        '''
        # if isinstance(test_yrs,np.ndarray):
        #     self.test_yrs = test_yrs
        # elif isinstance(test_yrs,int):
        #     test_yrs = predictand['time.year'].values[-1]-np.arange(test_yrs)
        #     self.test_yrs = test_yrs
        # else:
        #     self.test_yrs = np.array(test_yrs)
        
        # Select predictand months
        y = sel_months(predictand,self.predictand_months)

        # Select predictor months and include autoregressive and/or exogenous variables
        if self.include_predictand and self.include_predictors:
            X_predictand = sel_months(predictand,self.predictors_months)
            X_predictors = sel_months(predictors,self.predictors_months)
            X = xr.concat([X_predictand,X_predictors],'predictors',coords='all').rename('predictors')
        elif self.include_predictand:
            X = sel_months(predictand,self.predictors_months)
        elif self.include_predictors:
            X = sel_months(predictors,self.predictors_months)

        # Check if some lag predictors should be from the previous year
        # This is probably bugged or logically wrong but it won't be used
        if self.predictand_months.min()-self.lags < 1:
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
        y: xr.Dataset=None,
    ) -> Tuple[xr.DataArray]:
        '''
        Selects the correct training matrix for the current step, s and stacks lag variables.

        Parameters
        ----------
        `m` : Current predictand month.
        `s` : Current step in training/predicting used as an index. s should be on the range [0,steps-1]
        `X` : predictors dataset
        `y` : predictand dataset

        Returns
        -------
        X_m : DataArray containing the lag predictors for the current step, s
        y_m : DataArray from the predictand month for the current step, s
        '''

        # Select predictand month and make sample index as the year
        if (not self.fitted) or (not y is None):
            y_m = sel_months(y,m)
            # y_m = y_m.assign_coords({'sample':('time',y_m['time.year'].data)})
            # y_m = y_m.swap_dims(time='sample')
            if len(X)==len(y_m):
                return X,y_m
            else:
                if self.lags!=len(self.predictors_months):
                    predictors_months = np.arange(m-self.lags-self.lead,m-self.lead)+s+1
                else: 
                    predictors_months = self.predictors_months
                X_m = sel_months(X,predictors_months)
                if self.avg_lags:
                    X_m = X_m.coarsen(time=self.lags).mean()
                else:
                    X_m = X_m.coarsen(time=self.lags).construct(time=('sample','month'),keep_attrs=True)
                    X_m = X_m.assign_coords({'sample':X_m['time.year'][:,0].values,'month':X_m['time.month'][0,:].values})
                    X_m = X_m.stack(lag_predictors=X_m.dims[1:])
                return X_m,y_m
        else:
            t = X.time.to_index()
            if len(t.month.unique())==1:
                return X
            else:
                if self.lags!=len(self.predictors_months):
                    predictors_months = np.arange(m-self.lags-self.lead,m-self.lead)+s+1
                else: 
                    predictors_months = self.predictors_months
                X_m = sel_months(X,predictors_months)
                if self.avg_lags:
                    X_m = X_m.coarsen(time=self.lags).mean()
                else:
                    X_m = X_m.coarsen(time=self.lags).construct(time=('sample','month'),keep_attrs=True)
                    X_m = X_m.assign_coords({'sample':X_m['time.year'][:,0].values,'month':X_m['time.month'][0,:].values})
                    X_m = X_m.stack(lag_predictors=X_m.dims[1:])
                return X_m

    def fit(self, X: xr.DataArray, y: xr.DataArray):
        if self.fitted:
            self.reset()
        steps_arr = np.arange(len(self.predictand_months))%self.steps
        forecaster = {}
        for m,s in zip(self.predictand_months,steps_arr):
            X_m,y_m = self.get_training_matrix(m,s,X,y=y)
            forecaster[m] = deepcopy(self.reg.fit(X_m,y_m))
        self.forecaster_ = forecaster
        self.fitted = True
        self.predictand_coords_=y.coords.to_dataset()

    def predict(self, X: xr.DataArray) -> None:
        if not self.fitted:
            raise exception('Forecaster is not fitted.')
        steps_arr = np.arange(len(self.predictand_months))%self.steps 
        y_pred_list = []
        for m,s in zip(self.predictand_months,steps_arr):
            X_m = self.get_training_matrix(m,s,X)
            if m<10:
                time_coords = np.array([np.datetime64(f'{yr}-0{m}-01') for yr in X_m['time.year'].values])
            else: 
                time_coords = np.array([np.datetime64(f'{yr}-{m}-01') for yr in X_m['time.year'].values])
            self.predictand_coords_['time']=xr.DataArray(time_coords,coords={'time':time_coords})
            y_pred_list.append(xr.DataArray(self.forecaster_[m].predict(X_m),coords=self.predictand_coords_.coords))
        y_pred = xr.concat(y_pred_list,'time')
        y_pred = y_pred.sortby('time')
        return y_pred

    def data_plots(
        self,
        y_true:Union[xr.DataArray,xr.Dataset],
        y_pred:Union[xr.DataArray,xr.Dataset],
        plot_facet:bool=True,
        plot_true:bool=True,
        plot_timeseries:bool=False,
        plot_val:bool=False,
        model_name:str=None,
        forecaster_name:str=None,
        set_name:str=None,
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

        # Get aspect ratio of grid
        lon_range = y_true.lon.max() - y_true.lon.min()
        lat_range = y_true.lat.max() - y_true.lat.min()
        aspect = lon_range / lat_range

        plots={}
        if plot_facet:
            bounds = np.linspace(-2,2,10)
            cmap = mpl.cm.Spectral
            norm = BoundaryNorm(bounds, cmap.N, extend='both')
            sm = ScalarMappable(norm=norm, cmap=cmap)
            vmin=-2
            vmax=2
            if plot_true:
                # Plot individual variables using facetgrid
                p_true = y_true.precip.plot(x='lon',y='lat',col='time',col_wrap=self.steps,vmin=vmin,vmax=vmax,cmap=cmap,aspect=aspect,cbar_kwargs={'label':'precip standard anomaly','shrink':0.5})
                p_true.map_dataarray(xr.plot.contour,x='lon',y='lat',levels=bounds,colors='k',add_colorbar=False)
                plt.savefig(f'{FIG_DIR}facet_precip_true_{set_name}.png',facecolor='white',transparent=False)
                plots['facet_true']=p_true
            p_pred = y_pred.precip.plot(x='lon',y='lat',col='time',col_wrap=self.steps,vmin=vmin,vmax=vmax,cmap=cmap,aspect=aspect,cbar_kwargs={'label':'predicted precip standard anomaly','shrink':0.6})
            p_pred.map_dataarray(xr.plot.contour,x='lon',y='lat',levels=bounds,colors='k',add_colorbar=False)
            plt.savefig(f'{FIG_DIR}facet_precip_pred_{set_name}_{model_name}.png',facecolor='white',transparent=False)
            # elif not plot_individual:
            #     fig = plt.figure(constrained_layout=True,figsize=(int(aspect*3),3))
            #     fig_true,fig_pred = fig.subfigures(1,2,width_ratios=[1,1.1])

            #     fig_true.suptitle('ground truth',fontsize='x-large')
            #     fig_pred.suptitle('predicted',fontsize='x-large')
            #     fig.suptitle('{} of precipitation using {}'.format(set_name,model_name),fontsize='xx-large')

            #     axs_true = fig_true.subplots(int(len(y_true.time)/self.steps),self.steps)
            #     axs_pred = fig_pred.subplots(int(len(y_true.time)/self.steps),self.steps)

            #     # Plot true values
            #     for t,ax in enumerate(axs_true.flatten()):
            #         y_true.precip.isel(time=t).plot(ax=ax,vmin=-2,vmax=2,cmap=cmap,add_colorbar=False)
            #         y_true.precip.isel(time=t).plot.contour(ax=ax,vmin=-2,vmax=2,colors='k',add_colorbar=False)
            #     # Plot predicted values
            #     for t,ax in enumerate(axs_pred.flatten()):
            #         y_pred.precip.isel(time=t).plot(ax=ax,vmin=-2,vmax=2,cmap=cmap,add_colorbar=False)
            #         y_pred.precip.isel(time=t).plot.contour(ax=ax,vmin=-2,vmax=2,colors='k',add_colorbar=False)

            #     fig.colorbar(sm,ax=axs_pred,shrink=0.6)
            #     fig.save_fig(f'{FIG_DIR}facet_precip_{set_name}_{model_name}.png',facecolor='white',transparent=False)
            plots['facet_pred']=p_pred
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
                ax.set_xticklabels(y_true_mean.index.strftime('%b'))
                ax.set_title(year,loc='left')
                
                # ax.legend(loc='upper right')
                # ax.grid()

            fig.suptitle('Weighted Mean Precipitation Standardized Anomaly on season: SON')
            handles,labels = axs[0,0].get_legend_handles_labels()
            fig.legend(handles,labels)

            plt.savefig(f'{FIG_DIR}timeseries_precip_{set_name}_{model_name}.png',facecolor='white',transparent=False)


        if plot_val:
            # Group by month for subplots
            y_true_group = y_true.groupby('time.month')
            y_pred_group = y_pred.groupby('time.month')

            corr_list = []
            for (m,y_true_m),(_,y_pred_m) in zip(list(y_true_group),list(y_pred_group)):
                y_true_arr = y_true_m.precip.values
                y_pred_arr = y_pred_m.precip.values
                shape = y_true_arr.shape
                corr = np.corrcoef(y_true_arr.reshape(shape[0],-1),y_pred_arr.reshape(shape[0],-1),rowvar=False)
                corr = corr[np.arange(shape[1]*shape[2]),np.arange(shape[1]*shape[2],2*shape[1]*shape[2])].reshape([shape[1],shape[2]])
                corr = xr.DataArray(corr,coords=y_true_m.precip.mean(dim='time').coords,name='correlation')
                corr = corr.assign_coords(month=m)
                corr_list.append(corr)
            corr = xr.concat(corr_list,'month')
            corr_plot = corr.plot(x='lon',y='lat',col='month')
            plt.savefig(f'{FIG_DIR}corr_precip_{set_name}_{model_name}.png',facecolor='white',transparent=False)
        
            time = y_true.time.to_index()
            years = time.strftime('%Y')
            months = time.strftime('%b')
            fig,ax = plt.subplots(int(len(y_true.time)/self.steps),self.steps,constrained_layout=True,figsize=(3*self.steps,3*int(len(y_true.time)/self.steps)),sharex=True,sharey=True)

            for t,(ax,yr,m) in enumerate(zip(axs.flatten(),years,months)):
                y_true_n = y_true.precip.isel(time=t).values.ravel()
                y_pred_n = y_pred.precip.isel(time=t).values.ravel()
                R_n = np.corrcoef(y_true_n,y_pred_n)[0,1]
            
                ax.scatter(y_true_n,y_pred_n,s=1)
                ax.axline([0,0],[1,1],linestyle='--',color='k')
                ax.annotate('$R = {:.2f}$'.format(R_n),(0,2.5),size=14)

                ax.set_xlim(-3,3)
                ax.set_ylim(-3,3)
                ax.set_title(f'{yr}',loc='left')
                ax.set_title(f'{m}',loc='right')
            for ax in axs[-1,:]:
                ax.set_xlabel('y true')
            for ax in axs[:,0]:
                ax.set_ylabel('y pred')
            plt.savefig(f'{FIG_DIR}error_precip_{set_name}_{model_name}.png',facecolor='white',transparent=False)
        return plots

class RecursiveForecaster(ForecasterBase):
    def __init__(self) -> None:
        pass

    def train_test_split(self, test_yrs: Union[int, List[int]], predictand: xr.DataArray, predictors: Union[xr.DataArray, xr.Dataset] = None) -> Tuple[xr.DataArray]:
        return super().train_test_split(test_yrs, predictand, predictors)
    
    def get_training_matrix(self, m: int, s: int, X: xr.DataArray, y: xr.DataArray) -> Tuple[xr.DataArray]:
        return super().get_training_matrix(m, s, X, y)
    
    def fit(self, X: xr.DataArray, y: xr.DataArray):
        return super().fit(X, y)
    
    def predict(self, X: xr.DataArray, y: xr.DataArray) -> None:
        return super().predict(X, y)


class kfold_grid_search():
    '''
    Cross validation function for forecasters
    '''
    def __init__(self,
        frc,
        param_grid,
        k=12,
        n_yrs=3,
        scoring=None,
        metric=None,
        n_jobs=None,
        refit=True,
        error_score=np.nan,
        return_train_score=False,
    ):
        self.frc=frc
        self.param_grid=param_grid
        self.k=k
        self.n_yrs=n_yrs
        self.scoring=scoring
        self.metric=metric
        self.n_jobs=n_jobs
        self.refit=refit
        self.error_score=error_score
        self.return_train_score=return_train_score
    
    def get_cv(self,yrs):
        cv_size = self.k*self.n_yrs
        val_yrs = yrs[-cv_size:]
        self._val_yrs = val_yrs

        cv = [
            np.arange(yr_start,yr_end)
            for yr_start,yr_end in zip(val_yrs[::self.n_yrs],val_yrs[self.n_yrs-1::self.n_yrs]+1)
        ]
        return cv
        

    def _fit_predict(self,frc,X_train,y_train,X_val):
        if frc.fitted:
            frc.reset()
        frc.fit(X_train,y_train)
        return frc.predict(X_val)

    def fit_predict(self,frc,X,y):
        # Get validation set
        _,_,_,y_val = frc.train_test_split(X,y,self._val_yrs)

        # Fit and predict on the validation set
        y_pred = [self._fit_predict(frc,X_train,y_train,X_val)
                for X_train,y_train,X_val,_ in [frc.train_test_split(X,y,yrs)
                    for yrs in self._cv
                ]
        ]
        y_pred = xr.concat(y_pred,dim='time')
        return y_val,y_pred

    def _score(self,y_val,y_pred):
        def get_label(metric):
            return metric.__name__
        def get_binary(X,dec_func=0):
            return (X>dec_func).astype(int)
        if len(self.scoring)>1:
            metrics = [get_label(metric) for metric in self.scoring]
            score = {}
            for metric,scorer in zip(metrics,self.scoring):
                if scorer in [precision_score,accuracy_score,recall_score,f1_score]:
                    score[metric]=scorer(get_binary(y_val),get_binary(y_pred),average='samples')
                else:
                    score[metric]=scorer(y_val,y_pred)
        elif type(self.scoring)==list and len(self.scoring)==1:
            metric = get_label(self.scoring[0])
            scorer = self.scoring[0]
            if scorer in [precision_score,accuracy_score,recall_score,f1_score]:
                score={metric:scorer(get_binary(y_val),get_binary(y_pred))}
            else:
                score={metric:scorer(y_val,y_pred)}
        elif isfunction(self.scoring):
            metric=get_label(self.scoring)
            if self.scoring in [precision_score,accuracy_score,recall_score,f1_score]:
                score={metric:self.scoring(get_binary(y_val),get_binary(y_pred))}
            else:
                score={metric:self.scoring(y_val,y_pred)}
        return score

    def fit(self,X,y):
        # Get cv and validation years
        self._cv = self.get_cv(X['time.year'].values)
        # Get parameter grid (all permutations)
        param_grid = list(ParameterGrid(self.param_grid))

        # Parallel loop through all parameters
        with Parallel(n_jobs=self.n_jobs) as parallel:
            out = parallel(delayed(self._evaluate)(params,X,y) for params in param_grid)
        results = pd.DataFrame(out)
        if self.metric.endswith('error'):
            self.best_fit_results = results.loc[results[self.metric].idxmin()]
        elif self.metric in ['precision_score','accuracy_score','recall_score','f1_score','r2_score']:
            self.best_fit_results = results.loc[results[self.metric].idxmax()]
        else:
            warn("couldn't tell if metric should be ascending or descending",category='BAD CODING')
        self.best_estimator = self.best_fit_results['frc']
        print(self.best_estimator.get_params())
        self.results = results
        if self.refit:
            self.best_estimator.fit(X,y)

    def _evaluate(self,fit_params,X,y):
        # Set parameters
        frc = deepcopy(self.frc)
        frc.reset()
        frc.set_params(**fit_params)
        # Fit and predict on the validation set
        y_val,y_pred = self.fit_predict(frc,X,y)

        # Score model
        score = self._score(y_val,y_pred)
        result = deepcopy(fit_params)|score|{'frc':frc}
        return result

        
