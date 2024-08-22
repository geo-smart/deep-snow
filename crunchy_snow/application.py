import numpy as np
import xarray as xr
import pystac_client
import planetary_computer
import rasterio as rio
import rioxarray as rxr
from rioxarray.merge import merge_arrays
from urllib.request import urlretrieve
from pyproj import Proj, transform
from os.path import basename, exists, expanduser, join
import geopandas as gpd
from shapely.geometry import shape
import odc.stac
from datetime import datetime, timedelta

from crunchy_snow.utils import calc_norm, undo_norm, db_scale
from crunchy_snow.dataset import norm_dict
import crunchy_snow.models

def parse_bounding_box(value):
    try:
        minlon, minlat, maxlon, maxlat = map(float, value.split())
        return {'minlon':minlon, 'minlat':minlat, 'maxlon':maxlon, 'maxlat':maxlat}
    except ValueError:
        raise argparse.ArgumentTypeError("Bounding box must be in format 'minlon minlat maxlon maxlat' with float values.")

def get_parser():
    parser = argparse.ArgumentParser(description="CNN predictions of snow depth from remote sensing data")
    parser.add_argument("target_date", type=str, help="target date for snow depths with format YYYYmmdd")
    parser.add_argument("snow_off_date", type=str, help="snow off date (perhaps previous late summer) with format YYYYmmdd")
    parser.add_argument("aoi", type=parse_bounding_box, help="area of interest in format 'minlon minlat maxlon maxlat'")
    parser.add_argument("cloud_cover", type=float, help="percent cloud cover allowed in Sentinel-2 images (0-100)")
    parser.add_argument("delete_inputs", type=str, help="if True, delete input dataset from disk after processing")
    
    return parser

def date_range(date_str, padding):
    # Convert the string to a datetime object
    date = datetime.strptime(date_str, "%Y%m%d")
    
    # Calculate the dates n days before and after
    start_date = date - timedelta(days=padding)
    end_date = date + timedelta(days=padding)
    
    # Convert the dates back to strings in the desired format
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")
    
    # Return the date range string
    return f"{start_date_str}/{end_date_str}"

def url_download(url, out_fp, overwrite = False):
    # check if file already exists
    if not exists(out_fp) or overwrite == True:
            urlretrieve(url, out_fp)
    # if already exists. skip download.
    else:
        print('file already exists, skipping')

def download_fcf(out_fp):
    # this is the url from Lievens et al. 2021 paper
    fcf_url = 'https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif'
    # download just forest cover fraction to out file
    url_download(fcf_url, out_fp)

def download_data(aoi, target_date, snowoff_date):

    aoi = {
    "type": "Polygon",
    "coordinates":  [
          [
            [aoi['minlon'], aoi['maxlat']],
            [aoi['minlon'], aoi['minlat']],
            [aoi['maxlon'], aoi['minlat']],
            [aoi['maxlon'], aoi['maxlat']],
            [aoi['minlon'], aoi['maxlat']]
          ]
        ]
    }

    aoi_gpd = gpd.GeoDataFrame({'geometry':[shape(aoi)]}).set_crs(crs="EPSG:4326")
    crs = aoi_gpd.estimate_utm_crs()

    snowon_date_range = date_range(target_date, 6)
    snowoff_date_range = date_range(snowoff_date, 6)

    stac = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace)

    # Search for snow-on Sentinel-1 data
    search = stac.search(
        intersects=aoi,
        datetime=snowon_date_range,
        collections=["sentinel-1-rtc"])
    items = search.item_collection()
    
    snowon_s1_ds = odc.stac.load(items,chunks={"x": 2048, "y": 2048},
                                 crs=crs,
                                 resolution=50,
                                 bbox=aoi_gpd.total_bounds,
                                 groupby='sat:absolute_orbit')
    print(f"Returned {len(snowon_s1_ds.time)} acquisitions")
    
    # limit to morning acquisitions
    snowon_s1_ds = snowon_s1_ds.where(snowon_s1_ds.time.dt.hour > 11, drop=True)
    # compute median
    snowon_s1_ds = snowon_s1_ds.median(dim='time').squeeze().compute()
    # rename variables
    snowon_s1_ds = snowon_s1_ds.rename({'vv': 'snowon_vv', 'vh': 'snowon_vh'})

    # Search for snow-off Sentinel-1 data
    search = stac.search(
        intersects=aoi,
        datetime=snowoff_date_range,
        collections=["sentinel-1-rtc"])
    items = search.item_collection()
    
    snowoff_s1_ds = odc.stac.load(items,chunks={"x": 2048, "y": 2048},
                                  like=snowon_s1_ds,
                                  groupby='sat:absolute_orbit')
    print(f"Returned {len(snowoff_s1_ds.time)} acquisitions")
    
    # limit to morning acquisitions
    snowoff_s1_ds = snowoff_s1_ds.where(snowoff_s1_ds.time.dt.hour > 11, drop=True)
    # compute median
    snowoff_s1_ds = snowoff_s1_ds.median(dim='time').squeeze().compute()
    snowoff_s1_ds = snowoff_s1_ds.rename({'vv': 'snowoff_vv', 'vh': 'snowoff_vh'})

    # search for Sentinel-2 data
    search = stac.search(
    intersects=aoi,
    datetime=snowon_date_range,
    collections=["sentinel-2-l2a"],
    query={"eo:cloud_cover": {"lt": 25}})
    items = search.item_collection()
    
    s2_ds = odc.stac.load(items,chunks={"x": 2048, "y": 2048},
                          like=snowon_s1_ds,
                          groupby='solar_day').where(lambda x: x > 0, other=np.nan)
    print(f"Returned {len(s2_ds.time)} acquisitions")
    
    # compute median
    s2_ds = s2_ds.median(dim='time').squeeze().compute()

    # Search for COP30 elevation date
    search = stac.search(
    collections=["cop-dem-glo-30"],
    intersects=aoi)
    items = search.item_collection()
        
    data = []
    for item in items:
        dem_path = planetary_computer.sign(item.assets['data']).href
        data.append(rxr.open_rasterio(dem_path))
    cop30_da = merge_arrays(data)
    cop30_ds = cop30_da.rename('elevation').squeeze().to_dataset()
    
    # clip to aoi
    cop30_ds = cop30_ds.rio.clip_box(*aoi_gpd.total_bounds, crs="EPSG:4326").compute()
    # reproject to match radar dataset
    cop30_ds = cop30_ds.rio.reproject_match(snowon_s1_ds, resampling=rio.enums.Resampling.bilinear)

    # download fractional forest cover data
    fcf_path ='/tmp/fcf_global.tif'
    download_fcf(fcf_path)
    
    # open as dataArray and return
    fcf_ds = rxr.open_rasterio(fcf_path)
    
    # clip to aoi
    fcf_ds = fcf_ds.rio.clip_box(*aoi_gpd.total_bounds,crs="EPSG:4326") 
    # promote to dataset
    fcf_ds = fcf_ds.rename('fcf').squeeze().to_dataset()
    # reproject to match radar dataset
    fcf_ds = fcf_ds.rio.reproject_match(snowon_s1_ds, resampling=rio.enums.Resampling.bilinear)


    # combine datasets
    ds_list = [snowon_s1_ds, snowoff_s1_ds, s2_ds, cop30_ds, fcf_ds]
    ds = xr.merge(ds_list, compat='override', join='override').squeeze()

    # radar data variables
    # convert to decibels
    ds['snowon_vv'] = (('y', 'x'), db_scale(ds['snowon_vv']))
    ds['snowon_vh'] =  (('y', 'x'),db_scale(ds['snowon_vh']))
    ds['snowoff_vv'] =  (('y', 'x'),db_scale(ds['snowoff_vv']))
    ds['snowoff_vh'] =  (('y', 'x'),db_scale(ds['snowoff_vh']))
    
    # calculate variables
    ds['snowon_cr'] = ds['snowon_vh'] - ds['snowon_vv']
    ds['snowoff_cr'] = ds['snowoff_vh'] - ds['snowoff_vv']
    ds['delta_cr'] = ds['snowon_cr'] - ds['snowoff_cr']

    # s2 band indices
    ds['ndvi'] = (ds['B08'] - ds['B04'])/(ds['B08'] + ds['B04'])
    ds['ndsi'] = (ds['B03'] - ds['B11'])/(ds['B03'] + ds['B11'])
    ds['ndwi'] = (ds['B03'] - ds['B08'])/(ds['B03'] + ds['B08'])

    # latitude, longitude
    # define projections
    utm_proj = Proj(proj='utm', zone=crs.name[-3:-1], ellps='WGS84') ## NOTE hardcoded utm for now, adjust before use
    wgs84_proj = Proj(proj='latlong', datum='WGS84')
    
    x, y = np.meshgrid(ds['x'].values, ds['y'].values)
    lon, lat = transform(utm_proj, wgs84_proj, x, y)
    ds['latitude'] = (('y', 'x'), lat)
    ds['longitude'] = (('y', 'x'), lon)

    ds.to_netcdf('../data/model_inputs.nc')

    

def main():
    parser = get_parser()
    args = parser.parse_args()

    # download data
    download_data(args.aoi, args.target_date, args.snowoff_date)
    if args.delete_inputs == True:
        !rm ../data/model_inputs.nc

    
    

    
    
    
    
        
        
        



