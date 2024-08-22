import geopandas as gpd
from shapely.geometry import box
import math

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
