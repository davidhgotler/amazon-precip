from fileinput import filename
import xarray as xr
import numpy as np
from glob import glob

def main():
    data_dir = 'data'
    dataset = 'ERA5'
    years = 1950,1978,1979,2019
    ft_filename = f'{data_dir}\{dataset}.predictors.{years[0]}-{years[-1]}.nc'

    # Merge variables for each time period
    ds_list=[xr.merge([xr.open_dataset(f) for f in glob(f'{data_dir}\{dataset}*{yrs[0]}-{yrs[1]}.nc')]) for yrs in [years[0:2],years[2:]]]


    # Select correct dataset to make prelim compatible
    for ds in ds_list:
        if 'expver' in ds.dims:
            ds = ds.sel(expver=1)
        

    # Concatenate two time periods
    ds = xr.concat(ds_list,'time')

    # Reformat latitude and longitude
    ds = ds.rename(longitude='lon',latitude='lat')
    lon_attrs = ds.lon.attrs
    ds['lon'] = xr.where(ds['lon']<180,ds['lon'],ds['lon']-360)
    ds.lon.attrs = lon_attrs
    ds = ds.sortby(['lat','lon','time'])
    print(ds)
    ds.to_netcdf(ft_filename)
if __name__ == '__main__':
    main()