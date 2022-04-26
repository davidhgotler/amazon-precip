import logging

import numpy as np
import pandas as pd
import xarray as xr
import hvplot.xarray

from matplotlib import pyplot as plt

import sklearn as skl
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from eofs.xarray import Eof

from abc import ABC, abstractmethod
from copy import copy
from typing import List,Tuple,Union,Optional


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
        y = self.sel_months(autoreg,self.target_months)

        # Select feature months for direct prediction on a range of (1,12)
        feature_months = np.arange(self.target_months.min()-self.lags,self.target_months.max()-self.steps+1)
        feature_months = np.where(feature_months>0,feature_months,feature_months+12)

        if self.include_autoreg and self.include_exog:
            X_autoreg = self.sel_months(autoreg,feature_months)
            X_exog = self.sel_months(exog,feature_months)
            X = xr.concat([X_autoreg,X_exog],'feature',coords='all').rename('Features')
          
        elif self.include_autoreg:
            X = self.sel_months(autoreg,feature_months)
        elif self.include_exog:
            X = self.sel_months(exog,feature_months)

        # Split into training and test sets
        # ---------------------------------

        # Check if January is a target month to include correct lag years in features
        if np.isin(12,self.target_months):
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
        y_m = self.sel_months(y,m)
        # y_m = y_m.assign_coords({'sample':('time',y_m['time.year'].data)})
        # y_m = y_m.swap_dims(time='sample')

        X_m = self.sel_months(X,feature_months)
        if self.avg_lags:
            X_m = X_m.coarsen(time=len(feature_months)).mean()
        else:
            X_m = X_m.coarsen(time=len(feature_months)).construct(time=('sample','month'),keep_attrs=True)
            X_m = X_m.assign_coords({'sample':self.year2date(X_m['time.year'][:,0].values),'month':X_m['time.month'][0,:].values})
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
            forecaster[m] = self.regressor.fit(X_m,y_m)
        self.forecaster = forecaster
        self.fitted = True

    def predict(self, X: xr.DataArray, y: xr.DataArray) -> None:
        steps_arr = np.arange(len(self.target_months))%self.steps
        y_pred_list = []
        for m,s in zip(self.target_months,steps_arr):
            X_m,y_m = self.get_training_matrix(m,s,X,y)
            y_pred_list.append(xr.DataArray(self.forecaster[m].predict(X_m),coords=y_m.coords))
        y_pred = xr.concat(y_pred_list,'time',coords='all')
        # y_pred = y_pred.swap_dims(sample='time').sortby('time')
        return y_pred

    def year2date(self,yrs:Union[int,List[int],np.ndarray]):
        if isinstance(yrs,int):
            return np.datetime64('{}'.format(yrs),'Y')
        elif isinstance(yrs,np.ndarray) or isinstance(yrs,list):
            yrs_list = [np.datetime64('{}'.format(y),'Y') for y in yrs]
            return np.array(yrs_list)

    def sel_months(
        self,
        X:Union[xr.DataArray,xr.Dataset],
        months:Union[list,np.ndarray],
    ):
        return X.sel(time=X['time.month'].isin(months))

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
        y_true:xr.DataArray,
        y_pred:xr.DataArray,
        plot_facet:bool=True,
        plot_timeseries:bool=False,
        plot_corr_map:bool=False,
        plot_error:bool=False,
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
        plots = {}
        if plot_facet:
            p_true = y_true.hvplot(x='lon',y='lat',col='time').cols(self.steps)
            p_pred = y_pred.hvplot(x='lon',y='lat',col='time').cols(self.steps)
            plots['facet'] = p_true+p_pred
        
        

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

