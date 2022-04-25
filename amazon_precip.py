import numpy as np
import xarray as xr

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
    
from dataframe import dataframe
from forecast import DirectForecaster,RecursiveForecaster

def main():
    # directories
    data_dir = 'data_store/'
    features_fname = 'ERA5.features.1979-2021.nc'
    target_fname = 'precip.mon.total.1x1.v2020.nc'

    # Create df object to store our feature and target datasets
    df = dataframe(data_dir,features_fname,target_fname)
    df.detrend()
    df.std_anom()
    df.remove_nan()
    df.flatten()

    forecaster = DirectForecaster(MLPRegressor((5,)),lags=3,target_months=[9,10,11],steps=3,include_autoreg=True,avg_lags=True,pca_features=True)
    X_train,y_train,X_test,y_test = forecaster.train_test_split([2016,2017,2018,2019],df.target_da,df.features_da)
    forecaster.fit(X_train,y_train)
    y_pred = forecaster.predict(X_test,y_test)
    print('mean absolute error on the test set: {:.3f}'.format(mean_absolute_error(y_test,y_pred,multioutput=np.sqrt(np.abs(np.cos(np.deg2rad(y_test.lat.values)))))))

if __name__ == '__main__':
    main()