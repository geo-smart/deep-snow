import argparse
import os

from deep_snow.inputs import parse_bounding_box
from deep_snow.resources import get_default_land_path
from deep_snow.tiling import build_matrix_json, build_tile_jobs

def get_parser():
    parser = argparse.ArgumentParser(description="generate tiles for snow depth prediction")
    parser.add_argument("target_date", type=str, help="target date for snow depths with format YYYYmmdd")
    parser.add_argument("aoi", type=parse_bounding_box, help="area of interest in format 'minlon minlat maxlon maxlat'")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    tiles = build_tile_jobs(
        target_date=args.target_date,
        aoi=args.aoi,
        land_path=get_default_land_path(),
    )
    matrixJSON = build_matrix_json(tiles)
    print(f"Prepared {len(tiles)} tile job(s) for {args.target_date}.")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, 'a') as f:
            print(f'MATRIX_PARAMS_COMBINATIONS={matrixJSON}', file=f)
            print(f'TILE_COUNT={len(tiles)}', file=f)
    else:
        print(matrixJSON)

if __name__ == "__main__":
   main()
