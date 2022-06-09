import numpy as np
import xarray as xr

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
    
from dataframe import dataframe as df
from forecast import DirectForecaster,RecursiveForecaster
from forecast import get_dataset

import logging

def main():
    # directories
    features_fname = 'ERA5.features.1950-2019.nc'
    target_fname = 'precip.mon.total.1x1.v2020.nc'

    # input regions
    lat_ft = -30,0
    lon_ft = -80,-40

    lat_t = -20,-5
    lon_t = -65,-45

    yrs = 1950,2019
    # pack tuple
    range_ft =  lat_ft+lon_ft+yrs
    range_t = lat_t+lon_t+yrs

    # Create df object to store our feature and target datasets
    data = df(features_fname,target_fname,features_range=range_ft,target_range=range_t)
    data.detrend()
    data.std_anom()
    data.flatten()
    data.remove_nan()

    forecaster = DirectForecaster(LinearRegression(),lags=3,target_months=[9,10,11],steps=3,include_autoreg=True,avg_lags=False,pca_features=False)
    
    X_train,y_train,X_test,y_test = forecaster.train_test_split([2016,2017,2018,2019],df.target_da,df.features_da)
    forecaster.fit(X_train,y_train)

    y_pred = forecaster.predict(X_test,y_test)

    weights = np.sqrt(np.abs(np.cos(np.deg2rad(y_test.lat.values))))
    print('mean absolute error on the test set: {:.3f}'.format(mean_absolute_error(y_test,y_pred,multioutput=weights)))
    
    y_true_ds = get_dataset(y_test)
    y_pred_ds = get_dataset(y_pred)
    plots = forecaster.data_plots(y_true_ds,y_pred_ds)
    plots.show()

if __name__ == '__main__':
    main()