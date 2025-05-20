import geopandas as gpd
import shapely.geometry
import math
import json
import argparse
import os

def parse_bounding_box(value):
    try:
        minlon, minlat, maxlon, maxlat = map(float, value.split())
        return {'minlon':minlon, 'minlat':minlat, 'maxlon':maxlon, 'maxlat':maxlat}
    except ValueError:
        raise argparse.ArgumentTypeError("Bounding box must be in format 'minlon minlat maxlon maxlat' with float values.")

def get_parser():
    parser = argparse.ArgumentParser(description="generate tiles for snow depth prediction")
    parser.add_argument("target_date", type=str, help="target date for snow depths with format YYYYmmdd")
    parser.add_argument("aoi", type=parse_bounding_box, help="area of interest in format 'minlon minlat maxlon maxlat'")
    return parser

def create_tiles(min_lat, max_lat, min_lon, max_lon, tile_size=1.5, padding=0.05):
    """
    Create a GeoDataFrame of tiles over the area of interest with an overlap on the top and bottom edges.
    
    Parameters:
        min_lat (float): Minimum latitude of the area of interest.
        max_lat (float): Maximum latitude of the area of interest.
        min_lon (float): Minimum longitude of the area of interest.
        max_lon (float): Maximum longitude of the area of interest.
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the tile polygons.
    """
    tiles = []
    for i in range(math.ceil((max_lon - min_lon)/(tile_size - 2*padding))):
        for j in range(math.ceil((max_lat - min_lat)/(tile_size - 2*padding))):
            ymin = min_lat + j*(tile_size-2*padding)
            ymax = ymin + tile_size
            xmin = min_lon + i*(tile_size-2*padding)
            xmax = xmin + tile_size
            tile = shapely.geometry.Polygon([
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin)
            ])
            tiles.append(tile)
    
    return gpd.GeoDataFrame(geometry=tiles, crs="EPSG:4326")


def check_tiles(tiles_gdf, land_path):
    """
    Filters tiles to retain only those that overlap with land areas.

    Parameters:
        tiles_gdf (gpd.GeoDataFrame): A GeoDataFrame containing the tile polygons.
        land_path (str): File path to the Natural Earth land polygons dataset (https://www.naturalearthdata.com/downloads/50m-physical-vectors/50m-land/)

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing only the tiles that intersect land.
    """
    land_gdf = gpd.read_file(land_path)
    land_merged = land_gdf.dissolve().geometry.unary_union
    filtered_tiles = tiles_gdf[tiles_gdf.intersects(land_merged)]
    if filtered_tiles.empty:
        raise ValueError("No tiles overlap with the land polygons.")
    
    return filtered_tiles

def main():
    parser = get_parser()
    args = parser.parse_args()

    # create tiles geodataframe
    tiles_gdf = create_tiles(args.aoi['minlat'], args.aoi['maxlat'], args.aoi['minlon'], args.aoi['maxlon'])
    land_path = 'data/polygons/ne_50m_land.shp'
    tiles_gdf = check_tiles(tiles_gdf, land_path)

    # set up matrix job
    tiles = []
    for i, tile in tiles_gdf.iterrows():
        minlon, minlat, maxlon, maxlat = tile.geometry.bounds
        shortname = f'{args.target_date}_{minlon:.{2}f}_{minlat:.{2}f}_{maxlon:.{2}f}_{maxlat:.{2}f}'
        tiles.append({'tile_date':args.target_date, 'aoi':f'{minlon} {minlat} {maxlon} {maxlat}', 'name':shortname})
    matrixJSON = f'{{"include":{json.dumps(tiles)}}}'
    print(f'number of tiles: {len(tiles)}')
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        print(f'MATRIX_PARAMS_COMBINATIONS={matrixJSON}', file=f)

if __name__ == "__main__":
   main()
