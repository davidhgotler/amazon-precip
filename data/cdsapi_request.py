from cdsapi import Client

def main():
    c = Client()

    data_dir = 'data/'
    prelim_years = [
        '1950', '1951', '1952', '1953', '1954', '1955', '1956', '1957', '1958', '1959',
        '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969',
        '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978',
    ]
    years = [
        '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', 
        '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998',
        '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008',
        '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018',
        '2019',
    ]
    months = [
        '01', '02', '03',
        '04', '05', '06',
        '07', '08', '09',
        '10', '11', '12',
    ]

    # Download current ERA5 dataset
    # -----------------------------

    var = 'relative_humidity'
    c.retrieve(
        'reanalysis-era5-pressure-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': 'relative_humidity',
            'pressure_level': '1000',
            'year': years,
            'month': months,
            'time': '00:00',
            'area': [
                0, -90, -90,
                0,
            ],
            'format': 'netcdf',
            'expver': '1',
        },
        data_dir+'ERA5.{}.{}-{}.nc'.format(var,years[0],years[-1]),
    )

    var='geopotential'
    c.retrieve(
        'reanalysis-era5-pressure-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': var,
            'pressure_level': '200',
            'year': years,
            'month': months,
            'time': '00:00',
            'area': [
                0, -90, -90,
                0,
            ],
            'format': 'netcdf',
            'expver': '1',
        },
        data_dir+'ERA5.{}.{}-{}.nc'.format(var,years[0],years[-1]),
    )

    var = 'convective_available_potential_energy'
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': var,
            'year': years,
            'month': months,
            'time': '00:00',
            'area': [
                0, -90, -90,
                0,
            ],
            'format': 'netcdf',
            'expver': '1',
        },
        data_dir+'ERA5.cape.{}-{}.nc'.format(years[0],years[-1]),
    )

    var = 'convective_inhibition'
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': var,
            'year': years,
            'month': months,
            'time': '00:00',
            'area': [
                0, -90, -90,
                0,
            ],
            'format': 'netcdf',
            'expver': '1',
        },
        data_dir+'ERA5.cin.{}-{}.nc'.format(years[0],years[-1]),
        )

    # Download preliminary data version
    # -------------------------
    var = 'relative_humidity'
    c.retrieve(
        'reanalysis-era5-pressure-levels-monthly-means-preliminary-back-extension',
        {
            'product_type': 'reanalysis-monthly-means-of-daily-means',
            'variable': var,
            'pressure_level': '1000',
            'year': prelim_years,
            'month': months,
            'time': '00:00',
            'area': [
                0, -90, -90,
                0,
            ],
            'format': 'netcdf',
            'expver': '1',
        },
        data_dir+'ERA5.{}.{}-{}.nc'.format(var,prelim_years[0],prelim_years[-1]),
    )

    var='geopotential'
    c.retrieve(
        'reanalysis-era5-pressure-levels-monthly-means-preliminary-back-extension',
        {
            'product_type': 'reanalysis-monthly-means-of-daily-means',
            'variable': var,
            'pressure_level': '200',
            'year': prelim_years,
            'month': months,
            'time': '00:00',
            'area': [
                0, -90, -90,
                0,
            ],
            'format': 'netcdf',
            'expver': '1',
        },
        data_dir+'ERA5.{}.{}-{}.nc'.format(var,prelim_years[0],prelim_years[-1]),
    )

    var = 'convective_available_potential_energy'
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means-preliminary-back-extension',
        {
            'product_type': 'reanalysis-monthly-means-of-daily-means',
            'variable': var,
            'year': prelim_years,
            'month': months,
            'time': '00:00',
            'area': [
                0, -90, -90,
                0,
            ],
            'format': 'netcdf',
            'expver': '1',
        },
        data_dir+'ERA5.cape.{}-{}.nc'.format(prelim_years[0],prelim_years[-1]),
    )

    var = 'convective_inhibition'
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means-preliminary-back-extension',
        {
            'product_type': 'reanalysis-monthly-means-of-daily-means',
            'variable': var,
            'year': prelim_years,
            'month': months,
            'time': '00:00',
            'area': [
                0, -90, -90,
                0,
            ],
            'format': 'netcdf',
            'expver': '1',
        },
        data_dir+'ERA5.cin.{}-{}.nc'.format(prelim_years[0],prelim_years[-1]),
    )

if __name__ == '__main__':
    main()
