#! /usr/bin/env python

import numpy as np
import xarray as xr
import pystac_client
import planetary_computer
import rasterio as rio
import rioxarray as rxr
from rioxarray.merge import merge_arrays
from urllib.request import urlretrieve
from tqdm import tqdm
from pyproj import Proj, transform
from os.path import basename, exists, expanduser, join
from pathlib import Path
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import odc.stac
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import torch
from glob import glob
import seaborn as sns
import torch.nn.functional as F
import math
import pickle
import xdem
import time
import subprocess

from deep_snow.utils import calc_norm, undo_norm, db_scale, calc_dowy
from deep_snow.dataset import norm_dict
import deep_snow.models

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
    parser.add_argument("out_crs", type=str, help="coordinate reference system for predicted snow depths: 'utm' or 'wgs84'")
    
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


def url_tqdm_hook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to

def url_download(url, out_fp, overwrite = False):
    # check if file already exists
    if not exists(out_fp) or overwrite == True:
        # this tqdm progress bar comes from: https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
        with tqdm(unit = 'B', unit_scale = True, unit_divisor = 1024, miniters = 1, desc = out_fp) as t:
            urlretrieve(url, out_fp, reporthook = url_tqdm_hook(t))
    # if already exists. skip download.
    else:
        print('file already exists, skipping')

def download_fcf(out_fp):
    # this is the url from Lievens et al. 2021 paper
    fcf_url = 'https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif'
    # download just forest cover fraction to out file
    url_download(fcf_url, out_fp)

def download_data(aoi, target_date, snowoff_date, buffer_period=6, out_dir, cloud_cover):

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

    snowon_date_range = date_range(target_date, buffer_period)
    snowoff_date_range = date_range(snowoff_date, buffer_period)

    stac = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace)

    max_retries = 100
    retry_delay = 5  # seconds

    # Search for snow-on Sentinel-1 data
    print('searching for Sentinel-1 snow-on data')
    for attempt in range(max_retries):
        try:
            search = stac.search(
                intersects=aoi,
                datetime=snowon_date_range,
                collections=["sentinel-1-rtc"])
            items = search.item_collection()
            break  # Exit the loop if successful
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # Wait before retrying
            else:
                raise  # Raise the last exception if max retries reached
    
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
    for attempt in range(max_retries):
        try:
            search = stac.search(
                intersects=aoi,
                datetime=snowoff_date_range,
                collections=["sentinel-1-rtc"])
            items = search.item_collection()
            break  # Exit the loop if successful
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # Wait before retrying
            else:
                raise  # Raise the last exception if max retries reached
    
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
    for attempt in range(max_retries):
        try:
            search = stac.search(
                intersects=aoi,
                datetime=snowon_date_range,
                collections=["sentinel-2-l2a"],
                query={"eo:cloud_cover": {"lt": cloud_cover}})
            items = search.item_collection()
            break  # Exit the loop if successful
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # Wait before retrying
            else:
                raise  # Raise the last exception if max retries reached
    
    s2_ds = odc.stac.load(items,chunks={"x": 2048, "y": 2048},
                          like=snowon_s1_ds,
                          groupby='solar_day').where(lambda x: x > 0, other=np.nan)
    print(f"Returned {len(s2_ds.time)} acquisitions")

    # remove clouds
    s2_ds = s2_ds.where(s2_ds['SCL'] != 9) # high probability cloud cover
    s2_ds = s2_ds.where(s2_ds['SCL'] != 8) # high probability cloud cover
    s2_ds = s2_ds.where(s2_ds['SCL'] != 0) # nodata
    
    # compute median
    s2_ds = s2_ds.median(dim='time').squeeze().compute()

    # download and unzip snodas snow depths
    print('Searching for snodas data')
    target_datetime = pd.to_datetime(target_date)
    snodas_url = f'https://noaadata.apps.nsidc.org/NOAA/G02158/masked/{target_datetime.year}/{target_datetime.strftime("%m")}_{target_datetime.strftime("%b")}/SNODAS_{target_date}.tar'
    os.makedirs('/tmp/snodas', exist_ok=True)
    # Download the file
    subprocess.run([
        "wget", "-P", "/tmp/snodas", "-nd", "--no-check-certificate",
        "--reject", "index.html*", "-np", "-e", "robots=off", snodas_url
    ], check=True)
    
    # Extract the tarball
    snodas_fn = f"/tmp/snodas/SNODAS_{target_date}.tar"
    subprocess.run(["tar", "-xf", snodas_fn, "-C", "/tmp/snodas"], check=True)
    
    # Decompress the .gz files
    subprocess.run(["gzip", "-d", "-f"] + [f for f in os.listdir('/tmp/snodas') if f.startswith("us_ssmv11036") and f.endswith(".gz")], cwd="/tmp/snodas", check=True)
    
    # crop, reproject, and mask snodas snow depths
    snodas_da = rxr.open_rasterio(glob('/tmp/snodas/us_ssmv11036*.txt')[0]).squeeze()
    #snodas_clipped_da = snodas_da.rio.clip_box(snowon_s1_ds, crs="EPSG:4326")
    snodas_resampled_da = snodas_da.rio.reproject_match(snowon_s1_ds, resampling=rio.enums.Resampling.bilinear)
    snodas_resampled_da = snodas_resampled_da.where(snodas_resampled_da != -9999)/1000
    snodas_ds = snodas_resampled_da.to_dataset(name="snodas_sd")

    # search for COP30 DEM 
    print('searching for COP30 dem data')
    for attempt in range(max_retries):
        try:
            search = stac.search(
                collections=["cop-dem-glo-30"],
                intersects=aoi
            )
            items = search.item_collection()
            break  # Exit the loop if successful
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # Wait before retrying
            else:
                raise  # Raise the last exception if max retries reached
        
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
    fcf_ds = fcf_ds.rio.clip_box(*aoi_gpd.total_bounds,crs="EPSG:4326") 
    # promote to dataset
    fcf_ds = fcf_ds.rename('fcf').squeeze().to_dataset()
    # reproject to match radar dataset
    fcf_ds = fcf_ds.rio.reproject_match(snowon_s1_ds, resampling=rio.enums.Resampling.bilinear)
    # set values above 100 to nodata
    fcf_ds['fcf'] = fcf_ds['fcf'].where(fcf_ds['fcf'] <= 100, np.nan)/100

    # combine datasets
    print('combining datasets')
    ds_list = [snowon_s1_ds, snowoff_s1_ds, s2_ds, snodas_ds, cop30_ds, fcf_ds]
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
    utm_proj = Proj(proj='utm', zone=crs.name[-3:-1], ellps='WGS84') 
    wgs84_proj = Proj(proj='latlong', datum='WGS84')
    
    x, y = np.meshgrid(ds['x'].values, ds['y'].values)
    lon, lat = transform(utm_proj, wgs84_proj, x, y)
    ds['latitude'] = (('y', 'x'), lat)
    ds['longitude'] = (('y', 'x'), lon)

    # dowy
    dowy_1d = calc_dowy(pd.to_datetime(target_date).dayofyear)
    ds['dowy'] = (('y', 'x'), np.full_like(ds['latitude'], dowy_1d))

    # add terrain variables
    dem_transform = (50, 0.0, ds.isel(x=0, y=0).x.item(), 0.0, 50, ds.isel(x=0, y=0).y.item())
    dem = xdem.DEM.from_array(ds.elevation.values, dem_transform, crs=ds.rio.crs)
    ds['aspect'] = (('y', 'x'), xdem.terrain.aspect(dem).data.data)
    ds['northness'] = np.cos(np.deg2rad(ds.aspect))
    ds['slope'] = (('y', 'x'), xdem.terrain.slope(dem).data.data)
    ds['curvature'] = (('y', 'x'), xdem.terrain.curvature(dem).data.data)
    ds['tpi'] = (('y', 'x'), xdem.terrain.topographic_position_index(dem).data.data)
    ds['tri'] = (('y', 'x'), xdem.terrain.terrain_ruggedness_index(dem).data.data)

    # make gap map
    ds['data_gaps'] = np.multiply(((np.isnan(ds.snowon_vv) +
                                    np.isnan(ds.snowon_vh) +
                                    np.isnan(ds.snowoff_vv) +
                                    np.isnan(ds.snowoff_vh)) > 0), 1)
    ## NOTE we're taking the median of all bands, need to fix before using scene class map
    ds['data_gaps'] = xr.where(ds['SCL'] == 9, 1, ds['data_gaps']) # high probability cloud cover
    ds['data_gaps'] = xr.where(ds['SCL'] == 0, 1, ds['data_gaps']) # nodata

    data_fn = f'{out_dir}/model_inputs.nc'
    print('writing input data')
    ds.to_netcdf(data_fn)
    print('finished preparing dataset!')

    return crs

def apply_model(crs, model_path, out_dir, out_name, write_tif, delete_inputs, out_crs, gpu=True):
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
    data_dict['aerosol_optical_thickness'] = calc_norm(torch.Tensor(ds['AOT'].values), norm_dict['aerosol_optical_thickness'])
    data_dict['coastal_aerosol'] = calc_norm(torch.Tensor(ds['B01'].values), norm_dict['coastal_aerosol'])
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
    data_dict['snodas_sd'] = calc_norm(torch.Tensor(ds['snodas_sd'].values), norm_dict['aso_sd'])
    data_dict['elevation'] = calc_norm(torch.Tensor(ds['elevation'].values), norm_dict['elevation'])
    data_dict['aspect'] = calc_norm(torch.Tensor(ds['aspect'].values), norm_dict['aspect'])
    data_dict['northness'] = calc_norm(torch.Tensor(ds['northness'].values), [0, 1])
    data_dict['slope'] = calc_norm(torch.Tensor(ds['slope'].values), norm_dict['slope'])
    data_dict['curvature'] = calc_norm(torch.Tensor(ds['curvature'].values), norm_dict['curvature'])
    data_dict['tpi'] = calc_norm(torch.Tensor(ds['tpi'].values), norm_dict['tpi'])
    data_dict['tri'] = calc_norm(torch.Tensor(ds['tri'].values), norm_dict['tri'])
    data_dict['latitude'] = calc_norm(torch.Tensor(ds['latitude'].values), norm_dict['latitude'])
    data_dict['longitude'] = calc_norm(torch.Tensor(ds['longitude'].values), norm_dict['longitude'])
    data_dict['dowy'] = calc_norm(torch.Tensor(ds['dowy'].values), [0, 365])
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
    input_channels = ['snodas_sd',
                  'blue',
                  'swir1',
                  'ndsi',
                  'elevation',
                  'northness',
                  'slope',
                  'curvature',
                  'dowy',
                  'delta_cr',
                  'fcf'
                 ]

    #load previous model
    print('loading model')
    model = deep_snow.models.ResDepth(n_input_channels=len(input_channels), depth=5)
    if gpu == True:
        model.load_state_dict(torch.load(model_path))
        model.to('cuda')
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    tile_size = 1024
    padding = 50

    xmin=0
    xmax=tile_size
    ymin=0
    ymax=tile_size
    
    inputs = torch.cat([data_dict[channel] for channel in input_channels], dim=1)
    inputs_pad = F.pad(inputs, (0, tile_size, 0, tile_size), 'constant', 0)
    pred_pad = torch.empty_like(inputs_pad[0, 0, :, :])
    
    for i in range(math.ceil((len(ds.x)/(tile_size-2*padding)))):
        for j in range(math.ceil((len(ds.y)/(tile_size-2*padding)))):
            ymin = j*(tile_size-2*padding)
            ymax = ymin + tile_size
            xmin = i*(tile_size-2*padding)
            xmax = xmin + tile_size

            # predict noise in tile
            with torch.no_grad():
                if gpu == True:
                    tile_pred_sd = model(inputs_pad[:, :, ymin:ymax, xmin:xmax].to('cuda'))
                else:
                    tile_pred_sd = model(inputs_pad[:, :, ymin:ymax, xmin:xmax])
            xmax = xmax - padding
            ymax = ymax - padding
            
            if ymin == 0 and xmin == 0:
                tile_pred_sd = tile_pred_sd.detach().squeeze()[:-padding, :-padding]
            elif ymin == 0:
                xmin = xmin + padding
                tile_pred_sd = tile_pred_sd.detach().squeeze()[:-padding, padding:-padding]
            elif xmin == 0: 
                ymin = ymin + padding
                tile_pred_sd = tile_pred_sd.detach().squeeze()[padding:-padding, :-padding]
            else:
                xmin = xmin + padding
                ymin = ymin + padding
                tile_pred_sd = tile_pred_sd.detach().squeeze()[padding:-padding, padding:-padding]
            
            pred_pad[ymin:ymax, xmin:xmax] = tile_pred_sd
    
    # recover original dimensions
    pred_sd = pred_pad[0:(len(ds.y)), 0:(len(ds.x))]
    # undo normalization
    pred_sd = undo_norm(pred_sd, deep_snow.dataset.norm_dict['aso_sd'])
    # add to xarray dataset
    if gpu == True:
        ds['predicted_sd'] = (('y', 'x'), pred_sd.to('cpu').numpy())
    else:
        ds['predicted_sd'] = (('y', 'x'), pred_sd.numpy())
   
    # set negatives to 0
    ds['predicted_sd'] = ds['predicted_sd'].where(ds['predicted_sd'] > 0, 0)

    # ds = calculate_uncertainty(ds, model_path)
    # mask areas with missing data
    #ds['predicted_sd_corrected'] = ds['predicted_sd_corrected'].where(ds['data_gaps'] == 0)
    ds = ds.rio.write_crs(crs)

    if out_crs == 'wgs84':
        crs = 'EPSG:4326'
        ds = ds.rio.reproject(crs)
        ds = ds.rio.write_crs(crs)
    
    if write_tif == True:
        # write out geotif
        ds.predicted_sd.rio.to_raster(f'{out_dir}/{out_name}_sd.tif', compress='lzw')
        #ds.precision_map.rio.to_raster(f'{out_dir}/{out_name}_precision.tif', compress='lzw')

    if delete_inputs == True:
        os.remove(data_fn)

    print('prediction finished!')
    
    return ds

def calculate_uncertainty(ds, model_path):
    module_dir = os.path.dirname(os.path.abspath(__file__))
    bias_path = os.path.join(module_dir, "data", f"{model_path.split('/')[-1]}_bias_interpolator.pkl")
    
    # Load the bias interpolator
    with open(bias_path, 'rb') as f:
        predicted_interpolator = pickle.load(f)
    
    # # Load the interpolated NMAD array
    # interpolated_nmad = np.load(f'{dirname}/data/interpolated_nmad.npy')
    
    # # Load the bin edges
    # with open(f'{dirname}/data/bin_edges.pkl', 'rb') as f:
    #     bins = pickle.load(f)
    
    # predicted_sd_bins = bins['predicted_sd_bins']
    # elevation_bins = bins['elevation_bins']
    # slope_bins = bins['slope_bins']
    # fcf_bins = bins['fcf_bins']

    # apply bias correction
    # Flatten the predicted_sd values to 1D
    predicted_sd_flat = ds['predicted_sd'].values.flatten()
    # Predict the bias for the 1D array
    predicted_bias_flat = predicted_interpolator(predicted_sd_flat)
    # Reshape the predicted bias back to the original shape
    predicted_bias_reshaped = predicted_bias_flat.reshape(ds['predicted_sd'].shape)
    # Add the bias-corrected prediction to the dataset
    ds['bias_predicted'] = (('y', 'x'), predicted_bias_reshaped)
    ds['predicted_sd_corrected'] = ds['predicted_sd'] + ds['bias_predicted']

    # def bin_data(data, bins):
    #     binned_data = np.digitize(data, bins) - 1
    #     # Handle NaN values separately
    #     binned_data = np.where(binned_data > 9, np.nan, binned_data)
    #     return binned_data

    # # Apply the binning function
    # ds['predicted_sd_corrected_bins'] = xr.apply_ufunc(bin_data, ds['predicted_sd_corrected'], kwargs={'bins': predicted_sd_bins}, dask='allowed')
    # ds['elevation_bins'] = xr.apply_ufunc(bin_data, ds['elevation'], kwargs={'bins': elevation_bins}, dask='allowed')
    # ds['slope_bins'] = xr.apply_ufunc(bin_data, ds['slope'], kwargs={'bins': slope_bins}, dask='allowed')
    # ds['fcf_bins'] = xr.apply_ufunc(bin_data, ds['fcf'], kwargs={'bins': fcf_bins}, dask='allowed')

    # def map_nmad(psd_bins, elev_bins, slope_bins, fcf_bins, nmad_array):
    #     valid = ~np.isnan(psd_bins) & ~np.isnan(elev_bins) & ~np.isnan(slope_bins) & ~np.isnan(fcf_bins)
    #     result = np.full(psd_bins.shape, np.nan)
    #     result[valid] = nmad_array[psd_bins[valid].astype(int), elev_bins[valid].astype(int), slope_bins[valid].astype(int), fcf_bins[valid].astype(int)]
    #     return result

    # # Apply the NMAD mapping function to the dataset
    # ds['precision_map'] = xr.apply_ufunc(map_nmad, 
    #                                      ds['predicted_sd_corrected_bins'], 
    #                                      ds['elevation_bins'], 
    #                                      ds['slope_bins'], 
    #                                      ds['fcf_bins'], 
    #                                      kwargs={'nmad_array': interpolated_nmad}, 
    #                                      dask='parallelized', 
    #                                      output_dtypes=[float])
    return ds

    

def predict_sd(aoi, target_date, snowoff_date, model_path, out_dir, out_crs='utm', out_name='deep-snow_sd.tif', write_tif=True, delete_inputs=False, cloud_cover=25):
    # download data
    crs = download_data(aoi, target_date, snowoff_date, out_dir, cloud_cover)
    # apply model
    ds = apply_model(crs, model_path, out_dir, out_name, write_tif, delete_inputs, out_crs='utm')

    return ds

def generate_dates(target_date_str, start_date_str):
    target_date = datetime.strptime(target_date_str, "%Y%m%d")
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    date_list = []

    while target_date >= start_date:
        date_list.append(target_date.strftime("%Y%m%d"))
        target_date -= timedelta(days=12)

    return date_list

def predict_sd_ts(aoi, target_date, snowoff_date, model_path, out_dir, out_crs='utm', out_name='deep-snow_sd.tif', delete_inputs=False, cloud_cover=25):
    ds_list = []
    date_list = generate_dates(target_date, snowoff_date)
    for i, date in enumerate(date_list):
        print('--------------------------------------')
        print(f'working on {date}, {i+1}/{len(date_list)}')
        ds = predict_sd(aoi, date, snowoff_date, model_path, out_dir, out_crs, out_name='deep-snow_sd.tif', write_tif=False, delete_inputs=True, cloud_cover=25)
        ds = ds.expand_dims(time=[pd.to_datetime(date, format="%Y%m%d")])
        if i > 0:
            ds = ds.rio.reproject_match(ds_list[0])
        ds_list.append(ds)
    ds = xr.concat(ds_list, dim='time')
    
    return ds
        
def main():
    parser = get_parser()
    args = parser.parse_args()

    # if no cuda raise an error
    if not torch.cuda.is_available():
        raise RuntimeError("No cuda enabled GPU found on this platform.")

    # make sure out_dir exists or create it
    Path(args.out_dir).mkdir(exist_ok = True)
    # download data
    ds = deep_snow(args.aoi, args.target_date, args.snowoff_date, args.model_path, args.out_dir, args.delete_inputs, args.cloud_cover)

if __name__ == "__main__":
   main()
    
    
    
    

    
    
    
    
        
        
        



