import numpy as np
from sklearn import multioutput
from sklearn import linear_model
import xarray as xr

from matplotlib import pyplot as plt

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA,FactorAnalysis as FA
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

import datetime
    
from dataframe import *
from forecast import *

def get_corr_map(y_true_group,y_pred_group):
    corr_list = []
    corr_m_list = []
    # Calculate correlation at each gridpoint
    for (m,y_true_m),(_,y_pred_m) in zip(list(y_true_group),list(y_pred_group)):
        # Get arrays
        y_true_arr = y_true_m.precip.values
        y_pred_arr = y_pred_m.precip.values
        # Save shape
        shape = y_true_arr.shape
        # Calculate correlation map for each month
        corr = np.corrcoef(y_true_arr.reshape(shape[0],-1),y_pred_arr.reshape(shape[0],-1),rowvar=False)
        corr = corr[np.arange(shape[1]*shape[2]),np.arange(shape[1]*shape[2],2*shape[1]*shape[2])].reshape([shape[1],shape[2]])
        
        corr = xr.DataArray(corr,coords=y_true_m.precip.mean(dim='time').coords,name='correlation')
        corr = corr.assign_coords(month=datetime.datetime.strptime(f'{m}', "%m").strftime("%b"))
        corr_list.append(corr)

        # Calculate total correlation for that month
        corr_m = np.corrcoef(y_true_arr.ravel(),y_pred_arr.ravel())[0,1]
        corr_m_list.append(corr_m)
    return xr.concat(corr_list,'month'),np.array(corr_m_list)

def get_MSE_map(y_true_group,y_pred_group):
    MSE_list = []
    MSE_m_list = []
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

        MSE_m = mean_squared_error(y_true_arr.reshape(shape[0],-1),y_pred_arr.reshape(shape[0],-1))
        MSE_m_list.append(MSE_m)

    return xr.concat(MSE_list,dim='month'),np.array(MSE_m_list)

def get_f1_map(y_true_group,y_pred_group):
    f1_list = []
    f1_m_list = []
    # Calculate f score at each gridpoint
    for (m,y_true_m),(_,y_pred_m) in zip(list(y_true_group),list(y_pred_group)):
        # y_true_arr = y_true_m.precip.values
        # y_pred_arr = y_pred_m.precip.values
        y_true_arr = (y_true_m.precip.values>0).astype(int)
        y_pred_arr = (y_pred_m.precip.values>0).astype(int)
        shape = y_true_arr.shape
        with Parallel(n_jobs=-1) as parallel:
            f1 = parallel( delayed(f1_score)(y_true_arr[:,i,j],y_pred_arr[:,i,j])
                for i in np.arange(shape[1])
                for j in np.arange(shape[2])
            )
        f1 = np.array(f1).reshape(shape[1],shape[2])
        f1 = xr.DataArray(f1,coords=y_true_m.precip.mean(dim='time').coords,name='balanced_f_score')
        f1 = f1.assign_coords(month=datetime.datetime.strptime(f'{m}', "%m").strftime("%b"))
        f1_list.append(f1)

        # Calculate f score for entire month
        f1_m = f1_score(y_true_arr.reshape(shape[0],-1),y_pred_arr.reshape(shape[0],-1),average='samples')
        f1_m_list.append(f1_m)
    return xr.concat(f1_list,dim='month'),np.array(f1_m_list)

def get_r2_map(y_true_group,y_pred_group):
    r2_list = []
    r2_m_list = []
    # Calculate r2 at each gridpoint
    for (m,y_true_m),(_,y_pred_m) in zip(list(y_true_group),list(y_pred_group)):
        y_true_arr = y_true_m.precip.values
        y_pred_arr = y_pred_m.precip.values

        shape = y_true_arr.shape
        with Parallel(n_jobs=-1) as parallel:
            r2 = parallel( delayed(r2_score)(y_true_arr[:,i,j],y_pred_arr[:,i,j])
                for i in np.arange(shape[1])
                for j in np.arange(shape[2])
            )
        r2 = np.array(r2).reshape(shape[1],shape[2])
        r2 = xr.DataArray(r2,coords=y_true_m.precip.mean(dim='time').coords,name='coef_of_determination')
        r2 = r2.assign_coords(month=datetime.datetime.strptime(f'{m}', "%m").strftime("%b"))
        r2_list.append(r2)

        # Calculate r2 score for entire month
        r2_m = r2_score(y_true_arr.reshape(shape[0],-1),y_pred_arr.reshape(shape[0],-1))
        r2_m_list.append(r2_m)
    return xr.concat(r2_list,dim='month'),np.array(r2_m_list)

def validation_plots(y_true,y_pred,dataset='val',model='ridge',alpha=1,results=None,best_results=None):

    # Group by month
    y_true_group = y_true.groupby('time.month')
    y_pred_group = y_pred.groupby('time.month')
    # Plot spatial maps
    aspect = (y_true.lat.max() - y_true.lat.min())/(y_true.lon.max() - y_true.lon.min())
    # Get correlation
    corr_sp,corr_m = get_corr_map(y_true_group,y_pred_group)
    MSE_sp,MSE_m = get_MSE_map(y_true_group,y_pred_group)
    f1_sp,f1_m = get_f1_map(y_true_group,y_pred_group)
    r2_sp,r2_m = get_r2_map(y_true_group,y_pred_group)
    fig = plt.figure(constrained_layout=True,figsize=(12,int(12*aspect)))
    fig_corr,fig_MSE,fig_f1,fig_r2 = fig.subfigures(4,1)
    axs_corr = fig_corr.subplots(1,4)
    
    axs_corr[0].plot(corr_sp.month,corr_m,'o-')
    for i,ax in enumerate(axs_corr[1:]):
        if i<2:
            corr_sp.isel(month=i).plot(x='lon',y='lat',ax=ax,cmap='seismic',vmin=-1,vmax=1,add_colorbar=False)
        else:
            corr_sp.isel(month=i).plot(x='lon',y='lat',ax=ax,cmap='seismic',vmin=-1,vmax=1)
        
    axs_MSE = fig_MSE.subplots(1,4)
    axs_MSE[0].plot(MSE_sp.month,MSE_m,'o-')
    for i,ax in enumerate(axs_MSE[1:]):
        if i<2:
            MSE_sp.isel(month=i).plot(x='lon',y='lat',ax=ax,cmap='bone',vmin=0,vmax=2,add_colorbar=False)
        else:
            MSE_sp.isel(month=i).plot(x='lon',y='lat',ax=ax,cmap='bone',vmin=0,vmax=2)
    
    axs_f1 = fig_f1.subplots(1,4)
    axs_f1[0].plot(f1_sp.month,f1_m,'o-')
    for i,ax in enumerate(axs_f1[1:]):
        if i<2:
            f1_sp.isel(month=i).plot(x='lon',y='lat',ax=ax,cmap='bone',vmin=0,vmax=2,add_colorbar=False)
        else:
            f1_sp.isel(month=i).plot(x='lon',y='lat',ax=ax,cmap='bone',vmin=0,vmax=2)

    axs_r2 = fig_r2.subplots(1,4)
    axs_r2[0].plot(r2_sp.month,r2_m,'o-')
    for i,ax in enumerate(axs_r2[1:]):
        if i<2:
            r2_sp.isel(month=i).plot(x='lon',y='lat',ax=ax,cmap='seismic',vmin=-1,vmax=1,add_colorbar=False)
        else:
            r2_sp.isel(month=i).plot(x='lon',y='lat',ax=ax,cmap='seismic',vmin=-1,vmax=1)

    if alpha is None:
        plt.savefig(f'{FIG_DIR}/{dataset}_{model}.png')
    else:
        plt.savefig(f'{FIG_DIR}/{dataset}_{model}_{alpha}.png')


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
    X = data.predictors_da
    y = data.predictand_da
    # X = data.get_pcs(data.predictors_da,n=None,method='varimax')

    forecaster = DirectForecaster(
        Ridge(alpha=1),
        lags=lags,
        steps=steps,
        lead=1,
        include_predictand=False,
        include_predictors=True,
    )

    X_train,y_train,X_test,y_test = forecaster.train_test_split(X,y,[2009,2010,2011])

    # Cross-validation
    # max length of val. training set is # train. yr.s - # val. yrs. per k-fold
    n_val_yrs = 3
    k = 12
    n_max = len(X_train)-3

    for param_grid in [
        {
            'reg':[Pipeline([('fa',FA(rotation='varimax',max_iter=10000)),('linreg',LinearRegression(fit_intercept=False))])],
            'reg__fa__n_components':np.arange(0,n_max,5)+1,
        },
        {
            'reg':[Pipeline([('fa',FA(rotation='varimax',max_iter=10000)),('ridge',Ridge(fit_intercept=False))],verbose=True)],
            'reg__fa__n_components':np.arange(0,n_max,5)+1,
            'reg__ridge__alpha':np.logspace(-4,1,6),
        },
        {
            'reg':[Pipeline([('fa',FA(rotation='varimax',max_iter=10000)),('lasso',Lasso(fit_intercept=False))],verbose=True)],
            'reg__fa__n_components':np.arange(0,n_max,5)+1,
            'reg__ridge__alpha':np.logspace(-4,1,6),
        },
    ]:

        model=param_grid['reg'][0]._final_estimator.__class__.__name__

        print(f'\nrunning cross-validation on cv on {model}\n')

        kfold = kfold_grid_search(
            frc=forecaster,
            param_grid=param_grid,
            scoring=[mean_squared_error,f1_score,r2_score],
            metric='mean_squared_error',
            n_jobs=-1,
        )

        kfold.fit(X_train,y_train)
        print(kfold.results.drop(columns='frc'))
        y_val,y_val_pred = kfold.fit_predict(kfold.best_estimator,X_train,y_train)
        y_val = get_dataset(y_val)
        y_val_pred = get_dataset(y_val_pred)

        if 'reg__alpha' in param_grid:
            validation_plots(
                y_val,
                y_val_pred,
                dataset='val',
                model=model,
                alpha=kfold.best_fit_results['reg__alpha'],
                results=kfold.results,
                best_results=kfold.best_fit_results,
            )
        else:
            validation_plots(
                y_val,
                y_val_pred,
                dataset='val',
                model=model,
                alpha=None,
                results=kfold.results,
                best_results=kfold.best_fit_results,
            )
        
        # kfold.best_estimator.data_plots(y_val,
        # y_val_pred,
        # plot_facet=True,
        # plot_true=True,
        # plot_timeseries=False,
        # plot_val=False,
        # model_name=model,
        # forecaster_name='direct',
        # set_name='val',
        # )

        
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