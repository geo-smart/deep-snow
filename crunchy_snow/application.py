#! /usr/bin/env python

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
import os
import geopandas as gpd
from shapely.geometry import shape
import odc.stac
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import torch
from glob import glob
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

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
    parser.add_argument("out_dir", type=str, help="directory to write inputs and outputs to")
    parser.add_argument("model_path", type=str, help="path to model weights to use")
    parser.add_argument("out_name", type=str, help="name for predicted snow depth geotif")
    
    return parser

def date_range(date_str, padding):
    # Convert the string to a datetime object
    date = datetime.strptime(date_str, "%Y%m%d")
    
    # Calculate the dates n days before and after
    start_date = date - timedelta(days=padding)
    end_date = date + timedelta(days=padding)
    
    # Convert the dates back to strings in the desired format
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
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

def download_data(aoi, target_date, snowoff_date,  out_dir, cloud_cover=25):

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
    print('searching for Sentinel-1 snow-on data')
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
    print('searching for Sentinel-1 snow-off data')
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
    print('searching for Sentinel-2 snow-on data')
    search = stac.search(
    intersects=aoi,
    datetime=snowon_date_range,
    collections=["sentinel-2-l2a"],
    query={"eo:cloud_cover": {"lt": cloud_cover}})
    items = search.item_collection()
    
    s2_ds = odc.stac.load(items,chunks={"x": 2048, "y": 2048},
                          like=snowon_s1_ds,
                          groupby='solar_day').where(lambda x: x > 0, other=np.nan)
    print(f"Returned {len(s2_ds.time)} acquisitions")
    
    # compute median
    s2_ds = s2_ds.median(dim='time').squeeze().compute()

    # Search for COP30 elevation date
    print('searching for COP30 dem data')
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
    
    # reproject to match radar dataset
    cop30_ds = cop30_ds.rio.reproject_match(snowon_s1_ds, resampling=rio.enums.Resampling.bilinear).compute()

    # download fractional forest cover data
    print('downloading fractional forest cover data')
    fcf_path ='/tmp/fcf_global.tif'
    download_fcf(fcf_path)
    
    # open as dataArray and return
    fcf_ds = rxr.open_rasterio(fcf_path)
    
    # clip to aoi
    ## MAKE LARGER BBOX TO AVOID EDGE PROBLEMS
    fcf_ds = fcf_ds.rio.clip_box(*aoi_gpd.total_bounds,crs="EPSG:4326") 
    # promote to dataset
    fcf_ds = fcf_ds.rename('fcf').squeeze().to_dataset()
    # reproject to match radar dataset
    fcf_ds = fcf_ds.rio.reproject_match(snowon_s1_ds, resampling=rio.enums.Resampling.bilinear)
    # set values above 100 to nodata
    fcf_ds['fcf'] = fcf_ds['fcf'].where(fcf_ds['fcf'] <= 100, np.nan)/100

    # combine datasets
    print('combining datasets')
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

    data_fn = f'{out_dir}/model_inputs.nc'
    print('writing input data')
    ds.to_netcdf(data_fn)
    print('done!')

    return crs

def apply_model(out_dir, model_path, crs, out_name='crunchy-snow_sd.tif', delete_inputs=False):
    data_fn = f'{out_dir}/model_inputs.nc'
    print('reading input data')
    ds = xr.open_dataset(data_fn)
    ds = ds.fillna(0)

    data_dict = {}
    # normalize layers 
    data_dict['snowon_vv'] = calc_norm(torch.Tensor(ds['snowon_vv'].values), norm_dict['vv'])
    data_dict['snowon_vh'] = calc_norm(torch.Tensor(ds['snowon_vh'].values), norm_dict['vh'])
    data_dict['snowoff_vv'] = calc_norm(torch.Tensor(ds['snowoff_vv'].values), norm_dict['vv'])
    data_dict['snowoff_vh'] = calc_norm(torch.Tensor(ds['snowoff_vh'].values), norm_dict['vh'])
    data_dict['aerosol_optical_thickness'] = calc_norm(torch.Tensor(ds['AOT'].values), norm_dict['AOT'])
    data_dict['coastal_aerosol'] = calc_norm(torch.Tensor(ds['B01'].values), norm_dict['coastal'])
    data_dict['blue'] = calc_norm(torch.Tensor(ds['B02'].values), norm_dict['blue'])
    data_dict['green'] = calc_norm(torch.Tensor(ds['B03'].values), norm_dict['green'])
    data_dict['red'] = calc_norm(torch.Tensor(ds['B04'].values), norm_dict['red'])
    data_dict['red_edge1'] = calc_norm(torch.Tensor(ds['B05'].values), norm_dict['red_edge1'])
    data_dict['red_edge2'] = calc_norm(torch.Tensor(ds['B06'].values), norm_dict['red_edge2'])
    data_dict['red_edge3'] = calc_norm(torch.Tensor(ds['B07'].values), norm_dict['red_edge3'])
    data_dict['nir'] = calc_norm(torch.Tensor(ds['B08'].values), norm_dict['nir'])
    data_dict['water_vapor'] = calc_norm(torch.Tensor(ds['B09'].values), norm_dict['water_vapor'])
    data_dict['swir1'] = calc_norm(torch.Tensor(ds['B11'].values), norm_dict['swir1'])
    data_dict['swir2'] = calc_norm(torch.Tensor(ds['B12'].values), norm_dict['swir2'])
    data_dict['scene_class_map'] = calc_norm(torch.Tensor(ds['SCL'].values), norm_dict['scene_class_map'])
    data_dict['water_vapor_product'] = calc_norm(torch.Tensor(ds['WVP'].values), norm_dict['water_vapor_product'])
    data_dict['elevation'] = calc_norm(torch.Tensor(ds['elevation'].values), norm_dict['elevation'])
    data_dict['latitude'] = calc_norm(torch.Tensor(ds['latitude'].values), norm_dict['latitude'])
    data_dict['longitude'] = calc_norm(torch.Tensor(ds['longitude'].values), norm_dict['longitude'])
    # data_dict['dowy'] = calc_norm(torch.Tensor(dowy, [0, 365])
    data_dict['ndvi'] = calc_norm(torch.Tensor(ds['ndvi'].values), [-1, 1])
    data_dict['ndsi'] = calc_norm(torch.Tensor(ds['ndsi'].values), [-1, 1])
    data_dict['ndwi'] = calc_norm(torch.Tensor(ds['ndwi'].values), [-1, 1])
    data_dict['snowon_cr'] = calc_norm(torch.Tensor(ds['snowon_cr'].values), norm_dict['cr'])
    data_dict['snowoff_cr'] = calc_norm(torch.Tensor(ds['snowoff_cr'].values), norm_dict['cr'])
    data_dict['delta_cr'] = calc_norm(torch.Tensor(ds['delta_cr'].values), norm_dict['delta_cr'])
    
    data_dict['fcf'] = torch.Tensor(ds['fcf'].values)

    # clamp values, add dimensions
    data_dict = {key: torch.clamp(data_dict[key], 0, 1)[None, None, :, :] for key in data_dict.keys()}

    # define input channels for model
    input_channels = [
        'snowon_vv',
        'snowon_vh',
        'snowoff_vv',
        'snowoff_vh',
        'blue',
        'green',
        'red',
        'fcf',
        'elevation',
        'ndvi',
        'ndsi',
        'ndwi',
        'snowon_cr',
        'snowoff_cr']

    #load previous model
    print('loading model')
    model = crunchy_snow.models.ResDepth(n_input_channels=len(input_channels))
    model.load_state_dict(torch.load(model_path))
    model.to('cuda');

    tile_size = 1024

    xmin=0
    xmax=tile_size
    ymin=0
    ymax=tile_size
    
    inputs = torch.cat([data_dict[channel] for channel in input_channels], dim=1)
    inputs_pad = F.pad(inputs, (0, tile_size, 0, tile_size), 'constant', 0)
    pred_pad = torch.empty_like(inputs_pad[0, 0, :, :])
    
    for i in range(math.ceil((len(ds.x)/tile_size))):
        #print(f'column {i}')
        for j in range(math.ceil((len(ds.y)/tile_size))):
            #print(f'row {j}')
            ymin = j*tile_size
            ymax = (j+1)*tile_size
            xmin = i*tile_size
            xmax = (i+1)*tile_size
            
            # predict noise in tile
            with torch.no_grad():
                tile_pred_sd = model(inputs_pad[:, :, ymin:ymax, xmin:xmax].to('cuda'))
            pred_pad[ymin:ymax, xmin:xmax] = tile_pred_sd.detach().squeeze()
    
    # recover original dimensions
    pred_sd = pred_pad[0:(len(ds.y)), 0:(len(ds.x))]
    # undo normalization
    pred_sd = undo_norm(pred_sd, crunchy_snow.dataset.norm_dict['aso_sd'])
    # add to xarray dataset
    ds['predicted_sd'] = (('y', 'x'), pred_sd.to('cpu').numpy())

    ds = ds.rio.write_crs(crs)
    # write out geotif
    ds.predicted_sd.rio.to_raster(f'{out_dir}/{out_name}')

    if delete_inputs == True:
        os.remove(data_fn)
    
    return ds

def main():
    parser = get_parser()
    args = parser.parse_args()

    # download data
    crs = download_data(args.aoi, args.target_date, args.snowoff_date, args.cloud_cover, args.out_dir)
    # apply model
    ds = apply_model(args.out_dir, args.model_path, args.delete_inputs, crs=crs)

if __name__ == "__main__":
   main()
    
    
    
    

    
    
    
    
        
        
        



