import argparse
from datetime import datetime, timedelta
from pathlib import Path


def parse_bool(value):
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError("Boolean value must be one of: True, False, 1, 0, yes, no.")


def parse_bounding_box(value):
    try:
        minlon, minlat, maxlon, maxlat = map(float, value.split())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Bounding box must be in format 'minlon minlat maxlon maxlat' with float values."
        ) from exc

    if minlon >= maxlon or minlat >= maxlat:
        raise argparse.ArgumentTypeError(
            "Bounding box must satisfy minlon < maxlon and minlat < maxlat."
        )

    return {
        "minlon": minlon,
        "minlat": minlat,
        "maxlon": maxlon,
        "maxlat": maxlat,
    }


def format_bounding_box(aoi):
    return f'{aoi["minlon"]} {aoi["minlat"]} {aoi["maxlon"]} {aoi["maxlat"]}'


def build_output_name(target_date, aoi):
    return (
        f'{target_date}_deep-snow_'
        f'{aoi["minlon"]:.2f}_{aoi["minlat"]:.2f}_{aoi["maxlon"]:.2f}_{aoi["maxlat"]:.2f}'
    )


def get_default_fcf_cache_path():
    return str(Path("data") / "cache" / "fcf" / "wus_fcf.tif")


def get_default_snodas_cache_dir():
    return str(Path("data") / "cache" / "snodas")


def generate_dates(target_date_str, start_date_str, step_days=12):
    target_date = datetime.strptime(target_date_str, "%Y%m%d")
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    date_list = []

    while target_date >= start_date:
        date_list.append(target_date.strftime("%Y%m%d"))
        target_date -= timedelta(days=step_days)

    return date_list


def most_recent_occurrence(date_str, mmdd):
    ref_date = datetime.strptime(date_str, "%Y%m%d")
    target_date = datetime(ref_date.year, int(mmdd[:2]), int(mmdd[2:]))

    if target_date >= ref_date:
        target_date = target_date.replace(year=ref_date.year - 1)

    return target_date.strftime("%Y%m%d")
