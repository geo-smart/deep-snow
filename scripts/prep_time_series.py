from datetime import datetime, timedelta
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
    parser = argparse.ArgumentParser(description="find dates for snow depth prediction")
    parser.add_argument("end_date", type=str, help="most recent date to predict snow depths with format YYYYmmdd")
    parser.add_argument("begin_date", type=str, help="earliest date to predict snow depths with format YYYYmmdd")
    parser.add_argument("snow_off_day", type=str, help="snow-off month and day (perhaps late summer) with format mmdd")
    parser.add_argument("aoi", type=parse_bounding_box, help="area of interest in format 'minlon minlat maxlon maxlat'")
    return parser

def generate_dates(target_date_str, start_date_str):
    target_date = datetime.strptime(target_date_str, "%Y%m%d")
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    date_list = []

    while target_date >= start_date:
        date_list.append(target_date.strftime("%Y%m%d"))
        target_date -= timedelta(days=12)

    return date_list

def most_recent_occurrence(date_str: str, mmdd: str) -> str:
    """
    Given a reference date and a month-day string (MMDD),
    returns the most recent occurrence of that month-day before the given date.
    
    :param date_str: The reference date in YYYYMMDD format.
    :param mmdd: The target month and day in MMDD format.
    :return: The most recent occurrence of MMDD before the reference date in YYYYMMDD format.
    """
    ref_date = datetime.strptime(date_str, "%Y%m%d")
    target_date = datetime(ref_date.year, int(mmdd[:2]), int(mmdd[2:]))
    
    if target_date >= ref_date:
        target_date = target_date.replace(year=ref_date.year - 1)
    
    return target_date.strftime("%Y%m%d")

def main():
    parser = get_parser()
    args = parser.parse_args()

    date_list = generate_dates(args.end_date, args.begin_date)

    # set up matrix job
    date_list_matrix = []
    for date in date_list:
        snow_off_date = most_recent_occurrence(date, args.snow_off_day)
        shortname = f'{date}_{args.aoi["minlon"]:.{2}f}_{args.aoi["minlat"]:.{2}f}_{args.aoi["maxlon"]:.{2}f}_{args.aoi["maxlat"]:.{2}f}'
        date_list_matrix.append({'target_date':date, 'snow_off_date':snow_off_date, 'aoi':args.aoi, 'name':shortname})
    matrixJSON = f'{{"include":{json.dumps(date_list_matrix)}}}'
    print(f'number of dates: {len(date_list_matrix)}')
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        print(f'MATRIX_PARAMS_COMBINATIONS={matrixJSON}', file=f)

if __name__ == "__main__":
   main()