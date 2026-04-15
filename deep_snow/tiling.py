import json
import math

from deep_snow.inputs import format_bounding_box


def create_tiles(min_lat, max_lat, min_lon, max_lon, tile_size=1.5, padding=0.05):
    import geopandas as gpd
    import shapely.geometry

    if tile_size <= 2 * padding:
        raise ValueError("tile_size must be larger than twice the padding.")

    tiles = []
    step = tile_size - 2 * padding

    for i in range(math.ceil((max_lon - min_lon) / step)):
        for j in range(math.ceil((max_lat - min_lat) / step)):
            core_minlat = min_lat + j * step
            core_maxlat = min(core_minlat + step, max_lat)
            core_minlon = min_lon + i * step
            core_maxlon = min(core_minlon + step, max_lon)
            tile = shapely.geometry.box(
                core_minlon,
                core_minlat,
                core_maxlon,
                core_maxlat,
            )
            tiles.append(tile)

    return gpd.GeoDataFrame(geometry=tiles, crs="EPSG:4326")


def check_tiles(tiles_gdf, land_path):
    import geopandas as gpd

    land_gdf = gpd.read_file(land_path)
    land_merged = land_gdf.dissolve().geometry.unary_union
    filtered_tiles = tiles_gdf[tiles_gdf.intersects(land_merged)]
    if filtered_tiles.empty:
        raise ValueError("No tiles overlap with the land polygons.")

    return filtered_tiles


def build_tile_jobs(target_date, aoi, land_path, tile_size=1.5, padding=0.05):
    core_tiles_gdf = create_tiles(
        aoi["minlat"],
        aoi["maxlat"],
        aoi["minlon"],
        aoi["maxlon"],
        tile_size=tile_size,
        padding=padding,
    )
    core_tiles_gdf = check_tiles(core_tiles_gdf, land_path)

    jobs = []
    for _, tile in core_tiles_gdf.iterrows():
        core_minlon, core_minlat, core_maxlon, core_maxlat = tile.geometry.bounds
        processing_aoi = {
            "minlon": core_minlon - padding,
            "minlat": core_minlat - padding,
            "maxlon": core_maxlon + padding,
            "maxlat": core_maxlat + padding,
        }
        clip_aoi = {
            "minlon": core_minlon,
            "minlat": core_minlat,
            "maxlon": core_maxlon,
            "maxlat": core_maxlat,
        }
        jobs.append(
            {
                "tile_date": target_date,
                "aoi": format_bounding_box(processing_aoi),
                "clip_aoi": format_bounding_box(clip_aoi),
                "name": (
                    f"{target_date}_{core_minlon:.2f}_{core_minlat:.2f}_{core_maxlon:.2f}_{core_maxlat:.2f}"
                ),
            }
        )

    return jobs


def build_matrix_json(items):
    return json.dumps({"include": items})
