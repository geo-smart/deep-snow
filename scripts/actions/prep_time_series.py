import argparse
import os

from deep_snow.tiling import build_matrix_json
from deep_snow.workflows import build_time_series_jobs

def get_parser():
    parser = argparse.ArgumentParser(description="find dates for snow depth prediction")
    parser.add_argument("begin_date", type=str, help="earliest date to predict snow depths with format YYYYmmdd")
    parser.add_argument("end_date", type=str, help="most recent date to predict snow depths with format YYYYmmdd")
    parser.add_argument("snow_off_day", type=str, help="snow-off month and day (perhaps late summer) with format mmdd")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    date_list_matrix = build_time_series_jobs(args.begin_date, args.end_date, args.snow_off_day)
    matrixJSON = build_matrix_json(date_list_matrix)
    print(f"Prepared {len(date_list_matrix)} time-series date job(s).")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, 'a') as f:
            print(f'MATRIX_PARAMS_COMBINATIONS={matrixJSON}', file=f)
            print(f'DATE_COUNT={len(date_list_matrix)}', file=f)
    else:
        print(matrixJSON)

if __name__ == "__main__":
   main()
