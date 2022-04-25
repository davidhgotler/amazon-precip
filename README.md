# Predicting Rainfall in the Amazon Using Statistical Forecast
This project is focused on predicting precipitation in the Amazon region, using machine learning techniques. Code is written in python, built on xarray and sci-kit learn libraries.
## Data
### features/predictors: ERA5 Monthly Reanalysis
- Convective Available Potential Energy
- Convective Inhibition
- Geopotential at 200 hPa
- Relative Humidity at 1000 hPa

To download data run the cdsapi_request.py file. Then, run combine_features.py to make a single features dataset file. Or download from the climate data store https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels-monthly-means?tab=overview
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview.
### target/predictand: GPCC Monthly Precipitation 1x1 v2020
- precip

To download go to https://opendata.dwd.de/climate_environment/GPCC/html/fulldata-daily_v2020_doi_download.html or download from noaa catalog https://psl.noaa.gov/thredds/fileServer/Datasets/gpcc/full_v2020/precip.mon.total.1x1.v2020.nc