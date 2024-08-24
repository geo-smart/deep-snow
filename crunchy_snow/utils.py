import numpy as np
import torch
import geopandas as gpd
from shapely.geometry import box
import math

def calc_dowy(doy):
    'calculate day of water year from day of year'
    if doy < 274:
        dowy = doy + (365-274)
    elif doy >= 274:
        dowy = doy-274
    return dowy

def calc_norm(tensor, minmax_list):
    '''
    normalize a tensor between 0 and 1 using a min and max value stored in a list
    '''
    normalized = (tensor-minmax_list[0])/(minmax_list[1]-minmax_list[0])
    normalized = torch.nan_to_num(normalized, 0)
    return normalized

def undo_norm(tensor, minmax_list):
    '''
    undo tensor normalization
    '''
    original = (tensor*(minmax_list[1]-minmax_list[0]))+minmax_list[0]
    return original

def db_scale(x, epsilon=1e-10):
    # Add epsilon only where x is zero
    x_with_epsilon = np.where(x==0, epsilon, x)
    # Calculate the logarithm
    log_x = 10 * np.log10(x_with_epsilon)
    # Set the areas where x was originally zero back to zero
    log_x[x==0] = 0
    return log_x

def create_grid(aoi, grid_size_km=100, output_shapefile='aoi_grid.shp'):
    """
    Creates a grid of bounding boxes within the specified AOI and saves it as a shapefile.

    Parameters:
    aoi (dict): A dictionary with 'minlon', 'minlat', 'maxlon', 'maxlat' as keys defining the AOI.
    grid_size_km (float): The desired grid cell size in kilometers. Default is 100 km.
    output_shapefile (str): The name of the output shapefile. Default is 'aoi_grid.shp'.
    """
    # Calculate the width and height of the AOI in degrees
    width = aoi['maxlon'] - aoi['minlon']
    height = aoi['minlat'] - aoi['maxlat']

    # Define the grid size in degrees (100 km converted to degrees, rough approximation)
    grid_size_deg = grid_size_km / 111.0  # Approximate conversion

    # Calculate the number of grid cells in each dimension
    n_cols = math.ceil(abs(width) / grid_size_deg)
    n_rows = math.ceil(abs(height) / grid_size_deg)

    # Generate the grid cells
    grid_cells = []
    for i in range(n_cols):
        for j in range(n_rows):
            minlon = aoi['minlon'] + i * grid_size_deg
            maxlon = min(minlon + grid_size_deg, aoi['maxlon'])
            minlat = aoi['minlat'] - j * grid_size_deg
            maxlat = max(minlat - grid_size_deg, aoi['maxlat'])
            
            grid_cells.append(box(minlon, maxlat, maxlon, minlat))

    # Create a GeoDataFrame with the grid cells
    grid_gdf = gpd.GeoDataFrame({'geometry': grid_cells}, crs='EPSG:4326')
    return grid_gdf


