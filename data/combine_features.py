import xarray as xr
import numpy as np
import pandas as pd

def main():
    data_dir = 'data'

    dataset = 'ERA5'
    vars = ['relative_humidity','geopotential','cape','cin']
    years = ['1950-1978','1979-2019']
    tot_years = '1950-2019'

    ds_list = []
    for y in years:
        filenames = ['{}/{}.{}.{}.nc'.format(data_dir,dataset,v,y) for v in vars]
        merge_list = []
        for f in filenames:
            ds = xr.open_dataset(f)
            merge_list.append(ds)
            ds.close()
        ds_list.append(xr.merge(merge_list))
    for i,ds in enumerate(ds_list):
        if 'expver' in ds.dims:
            ds_list[i] = ds.sel(expver=1)
    ft_filename = '{}/{}.features.{}.nc'.format(data_dir,dataset,tot_years)

    ds = xr.concat(ds_list,'time')
    ds = ds.rename(longitude='lon',latitude='lat')

    ds.to_netcdf(ft_filename)
if __name__ == '__main__':
    main()