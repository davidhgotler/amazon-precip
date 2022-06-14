from calendar import month
import enum
from operator import imod
import numpy as np
from sklearn import multioutput
import xarray as xr

from matplotlib import pyplot as plt

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA,FactorAnalysis as FA
from sklearn.metrics import mean_absolute_error,mean_squared_error

import datetime
    
from dataframe import *
from forecast import *

def get_corr_map(y_true_group,y_pred_group):
    corr_list = []
    # Calculate correlation at each gridpoint
    for (m,y_true_m),(_,y_pred_m) in zip(list(y_true_group),list(y_pred_group)):
        y_true_arr = y_true_m.precip.values
        y_pred_arr = y_pred_m.precip.values
        shape = y_true_arr.shape
        corr = np.corrcoef(y_true_arr.reshape(shape[0],-1),y_pred_arr.reshape(shape[0],-1),rowvar=False)
        corr = corr[np.arange(shape[1]*shape[2]),np.arange(shape[1]*shape[2],2*shape[1]*shape[2])].reshape([shape[1],shape[2]])
        corr = xr.DataArray(corr,coords=y_true_m.precip.mean(dim='time').coords,name='correlation')
        corr = corr.assign_coords(month=datetime.datetime.strptime(f'{m}', "%m").strftime("%b"))
        corr_list.append(corr)
    return xr.concat(corr_list,'month')

def get_MSE_map(y_true_group,y_pred_group):
    MSE_list = []
    # Calculate correlation at each gridpoint
    for (m,y_true_m),(_,y_pred_m) in zip(list(y_true_group),list(y_pred_group)):
        y_true_arr = y_true_m.precip.values
        y_pred_arr = y_pred_m.precip.values
        shape = y_true_arr.shape
        with Parallel(n_jobs=-1) as parallel:
            MSE = parallel( delayed(mean_squared_error)(y_true_arr[:,i,j],y_pred_arr[:,i,j],squared=True)
                for i in np.arange(shape[1])
                for j in np.arange(shape[2])
            )
        MSE = np.array(MSE).reshape(shape[1],shape[2])
        MSE = xr.DataArray(MSE,coords=y_true_m.precip.mean(dim='time').coords,name='mean_squared_error')
        MSE = MSE.assign_coords(month=datetime.datetime.strptime(f'{m}', "%m").strftime("%b"))
        MSE_list.append(MSE)
    return xr.concat(MSE_list,dim='month')

def get_F1_map(y_true_group,y_pred_group):
    F1_list = []
    # Calculate correlation at each gridpoint
    for (m,y_true_m),(_,y_pred_m) in zip(list(y_true_group),list(y_pred_group)):
        y_true_arr = (y_true_m.precip.values>0).astype(int)
        y_pred_arr = (y_pred_m.precip.values>0).astype(int)
        shape = y_true_arr.shape
        with Parallel(n_jobs=-1) as parallel:
            F1 = parallel( delayed(f1_score)(y_true_arr[:,i,j],y_pred_arr[:,i,j])
                for i in np.arange(shape[1])
                for j in np.arange(shape[2])
            )
        F1 = np.array(F1).reshape(shape[1],shape[2])
        F1 = xr.DataArray(F1,coords=y_true_m.precip.mean(dim='time').coords,name='balanced_f_score')
        F1 = F1.assign_coords(month=datetime.datetime.strptime(f'{m}', "%m").strftime("%b"))
        F1_list.append(F1)
    return xr.concat(F1_list,dim='month')

def spatial_plots(y_true,y_pred,dataset='val',model='ridge',reg=1):
    # Group by month
    y_true_group = y_true.groupby('time.month')
    y_pred_group = y_pred.groupby('time.month')
    corr = get_corr_map(y_true_group,y_pred_group)
    MSE = get_MSE_map(y_true_group,y_pred_group)
    F1 = get_F1_map(y_true_group,y_pred_group)
    fig = plt.figure(constrained_layout=True,figsize=(12,12))
    fig_corr,fig_MSE,fig_f1 = fig.subfigures(3,1)
    axs_corr = fig_corr.subplots(1,3)
    for i,ax in enumerate(axs_corr):
        if i<2:
            corr.isel(month=i).plot(x='lon',y='lat',ax=ax,cmap='seismic',vmin=-1,vmax=1,add_colorbar=False)
        else:
            corr.isel(month=i).plot(x='lon',y='lat',ax=ax,cmap='seismic',vmin=-1,vmax=1)
    axs_MSE = fig_MSE.subplots(1,3)
    for i,ax in enumerate(axs_MSE):
        if i<2:
            MSE.isel(month=i).plot(x='lon',y='lat',ax=ax,cmap='seismic',vmin=0,vmax=2,add_colorbar=False)
        else:
            MSE.isel(month=i).plot(x='lon',y='lat',ax=ax,cmap='seismic',vmin=0,vmax=2)
    axs_f1 = fig_f1.subplots(1,3)
    for i,ax in enumerate(axs_f1):
        if i<2:
            F1.isel(month=i).plot(x='lon',y='lat',ax=ax,cmap='binary_r',vmin=0,vmax=1,add_colorbar=False)
        else:
            F1.isel(month=i).plot(x='lon',y='lat',ax=ax,cmap='binary_r',vmin=0,vmax=1)

    plt.savefig(f'{FIG_DIR}/{dataset}_{model}_{reg}.png')

def main():
    # directories
    predictors_fname = 'ERA5.predictors.1950-2019.nc'
    predictand_fname = 'precip.mon.total.1x1.v2020.nc'

    # input regions
    predictors_lat = -30,0
    predictors_lon = -80,-40

    predictand_lat = -20,-5
    predictand_lon = -65,-45

    yrs = 1950,2019
    # pack tuple
    predictors_range =  predictors_lat+predictors_lon+yrs
    predictand_range = predictand_lat+predictand_lon+yrs

    # Define month range of datasets to cut down on computation time
    predictors_months = 6,7,8
    predictand_months = 9,10,11

    lags=3
    steps=3

    # Coarsen data according to
    predictors_coarsen = lags,2.5,2.5
    predictand_coarsen = 1,1,1

    # Create dataframe object to store our feature and predictand datasets
    data = dataframe(
        predictors_fname,
        predictand_fname,
        predictors_range=predictors_range,
        predictand_range=predictand_range,
        predictors_months=predictors_months,
        predictand_months=predictand_months,
        predictors_coarsen=predictors_coarsen,
        predictand_coarsen=predictand_coarsen
    )

    data.detrend()
    data.std_anom()
    data.flatten()
    data.remove_nan()
    X = data.get_pcs(data.predictors_da,n=20,method='varimax')

    forecaster = DirectForecaster(
        Ridge(alpha=1),
        lags=lags,
        steps=steps,
        lead=1,
        include_predictand=False,
        include_predictors=True,
    )

    X_train,y_train,X_test,y_test = forecaster.train_test_split(X,data.predictand_da,[2009,2010,2011])

    for model,param_grid in {'ridge':{'reg':[Ridge()],'reg__alpha':np.arange(0,1000)},'lasso':{'reg':[Lasso()],'reg__alpha':np.arange(0,1,.001)}}.items():
        forecaster.set_params(reg=param_grid['reg'][0])
        frc = kfold_grid_search(
            frc=forecaster,
            param_grid=param_grid,
            scoring=f1_score,
            n_jobs=-1
        )
        y_val,y_val_pred = frc.cv_predict(X_train,y_train)
        y_val_ds = get_dataset(y_val)
        y_val_pred_ds = get_dataset(y_val_pred)

        spatial_plots(y_val_ds,y_val_pred_ds,dataset='val',model=model,reg=frc.best_fit_results['reg__alpha'])

    
    # forecaster.fit(X_train,y_train)

    # y_pred = forecaster.predict(X_test)

    # weights = np.sqrt(np.abs(np.cos(np.deg2rad(y_test.lat))))
    # print('mean absolute error on the test set: {:.3f}'.format(mean_squared_error(y_test,y_pred),multioutput=weights))
    
    # y_true_ds = get_dataset(y_test)
    # y_pred_ds = get_dataset(y_pred)
    # plots = forecaster.data_plots(
    #     y_true_ds,
    #     y_pred_ds,
    #     plot_facet=True,
    #     plot_true=True,
    #     plot_timeseries=True,
    #     plot_corr_map=True,
    #     plot_error=True,
    #     model_name='varimax_cringe',
    #     forecaster_name='direct'
    # )

if __name__ == '__main__':
    main()