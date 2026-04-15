import gzip
import os
import shutil
import tarfile
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
import rasterio as rio
import rioxarray as rxr
from rasterio.errors import NotGeoreferencedWarning
from rioxarray.merge import merge_arrays
from shapely.geometry import shape
from deep_snow.errors import EmptyAcquisitionError, TransientAcquisitionError
from deep_snow.inputs import (
    get_default_fcf_cache_path,
    get_default_hill_pptwt_cache_path,
    get_default_hill_td_cache_path,
    get_default_snodas_cache_dir,
)

SENTINEL1_DESCENDING_PASS_HOUR_UTC_GT = 11
DEFAULT_HILL_PPTWT_URL = os.environ.get(
    "DEEP_SNOW_HILL_PPTWT_URL",
    "https://github.com/geo-smart/deep-snow-data/releases/download/v0.1.0/ppt_wt_final.txt",
)
DEFAULT_HILL_TD_URL = os.environ.get(
    "DEEP_SNOW_HILL_TD_URL",
    "https://github.com/geo-smart/deep-snow-data/releases/download/v0.1.0/td_final.txt",
)


def date_range(date_str, padding):
    date = datetime.strptime(date_str, "%Y%m%d")
    start_date = date - timedelta(days=padding)
    end_date = date + timedelta(days=padding)
    return f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"


def _log_stage(title):
    print(f"\n[acquire] {title}")


def _log_detail(message):
    print(f"  - {message}")


def _display_path(path):
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path_obj.as_posix()


def _reproject_match_quietly(dataset, match_ds, *, resampling):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        return dataset.rio.reproject_match(match_ds, resampling=resampling)


def build_aoi_geometry(aoi):
    return {
        "type": "Polygon",
        "coordinates": [[
            [aoi["minlon"], aoi["maxlat"]],
            [aoi["minlon"], aoi["minlat"]],
            [aoi["maxlon"], aoi["minlat"]],
            [aoi["maxlon"], aoi["maxlat"]],
            [aoi["minlon"], aoi["maxlat"]],
        ]],
    }


def build_aoi_geodataframe(aoi):
    return gpd.GeoDataFrame({"geometry": [shape(build_aoi_geometry(aoi))]}).set_crs(crs="EPSG:4326")


def create_stac_client():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def _is_likely_transient_error(exc):
    message = str(exc).lower()
    transient_markers = [
        "timeout",
        "timed out",
        "temporarily unavailable",
        "temporary failure",
        "connection reset",
        "connection aborted",
        "connection refused",
        "remote disconnected",
        "service unavailable",
        "too many requests",
        "502",
        "503",
        "504",
    ]
    return isinstance(exc, (TimeoutError, ConnectionError, HTTPError, URLError)) or any(
        marker in message for marker in transient_markers
    )


def search_item_collection(stac, max_retries=5, retry_delay=5, **search_kwargs):
    for attempt in range(max_retries):
        try:
            search = stac.search(**search_kwargs)
            return search.item_collection()
        except Exception as exc:
            if not _is_likely_transient_error(exc):
                raise
            _log_detail(f"transient search failure on attempt {attempt + 1}: {exc}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise TransientAcquisitionError(
                    "STAC search failed after retrying transient errors."
                ) from exc


def _serialize_time_values(time_values):
    return [pd.Timestamp(value).isoformat() for value in time_values]


def _format_time_summary(times):
    if not times:
        return "none"
    if len(times) <= 3:
        return ", ".join(times)
    return f"{times[0]} ... {times[-1]} ({len(times)} total)"


def _get_valid_pixel_fraction(ds):
    reference_var = next(iter(ds.data_vars))
    return float(ds[reference_var].notnull().mean().item())


def _select_acquisitions(ds, *, target_date, selection_strategy):
    time_values = pd.to_datetime(ds.time.values)

    if selection_strategy == "composite":
        selected_ds = ds.median(dim="time").squeeze().compute()
        return selected_ds, _serialize_time_values(ds.time.values), _get_valid_pixel_fraction(selected_ds)

    if selection_strategy != "nearest_usable":
        raise ValueError(
            "Acquisition selection_strategy must be one of: 'composite', 'nearest_usable'."
        )

    target_timestamp = pd.Timestamp(target_date)
    sorted_indexes = np.argsort(np.abs(time_values - target_timestamp))
    selected_ds = None
    selected_times = []

    for idx in sorted_indexes:
        candidate = ds.isel(time=int(idx)).squeeze().compute()
        candidate_valid_fraction = _get_valid_pixel_fraction(candidate)
        if candidate_valid_fraction == 0:
            continue
        selected_times.append(pd.Timestamp(time_values[int(idx)]).isoformat())
        if selected_ds is None:
            selected_ds = candidate
        else:
            selected_ds = selected_ds.combine_first(candidate)
        if _get_valid_pixel_fraction(selected_ds) >= 0.999999:
            break

    if selected_ds is None:
        raise EmptyAcquisitionError(
            "Acquisitions were found, but none contained usable pixels after masking."
        )

    return selected_ds, selected_times, _get_valid_pixel_fraction(selected_ds)


def url_download(url, out_fp, overwrite=False):
    if not os.path.exists(out_fp) or overwrite:
        _log_detail(f"downloading {_display_path(out_fp)}")
        urlretrieve(url, out_fp)
    else:
        _log_detail(f"using cached file: {_display_path(out_fp)}")


def url_download_with_retries(url, out_fp, overwrite=False, max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        try:
            url_download(url, out_fp, overwrite=overwrite)
            return
        except (HTTPError, URLError) as exc:
            if attempt == max_retries - 1:
                raise
            _log_detail(f"download attempt {attempt + 1} failed: {exc}; retrying")
            time.sleep(retry_delay)


def download_fcf(out_fp):
    url_download("https://github.com/geo-smart/deep-snow-data/releases/download/v0.1.0/wus_fcf.tif", out_fp)


def ensure_fcf_file(fcf_path=None):
    resolved_path = Path(fcf_path or get_default_fcf_cache_path())
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    download_fcf(str(resolved_path))
    return str(resolved_path)


def download_hill_pptwt(out_fp):
    url_download(DEFAULT_HILL_PPTWT_URL, out_fp)


def download_hill_td(out_fp):
    url_download(DEFAULT_HILL_TD_URL, out_fp)


def ensure_hill_file(path, downloader):
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    downloader(str(resolved_path))
    return str(resolved_path)


def ensure_hill_inputs(pptwt_path=None, td_path=None):
    _log_stage("Hill SWE support rasters")
    resolved_pptwt_path = ensure_hill_file(
        pptwt_path or get_default_hill_pptwt_cache_path(),
        download_hill_pptwt,
    )
    resolved_td_path = ensure_hill_file(
        td_path or get_default_hill_td_cache_path(),
        download_hill_td,
    )
    return resolved_pptwt_path, resolved_td_path


def get_snodas_cache_dir():
    cache_dir = Path(get_default_snodas_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def extract_tar_archive(archive_path, destination_dir):
    with tarfile.open(archive_path) as archive:
        archive.extractall(destination_dir, filter="data")


def decompress_gzip_files(directory, pattern="us_ssmv11036*.gz"):
    decompressed_paths = []
    for gz_path in sorted(Path(directory).glob(pattern)):
        output_path = gz_path.with_suffix("")
        with gzip.open(gz_path, "rb") as source, open(output_path, "wb") as destination:
            shutil.copyfileobj(source, destination)
        decompressed_paths.append(output_path)
    return decompressed_paths


def load_sentinel1_dataset(
    stac,
    aoi_geometry,
    date_window,
    *,
    target_date,
    crs=None,
    bbox=None,
    like=None,
    rename_map=None,
    orbit_selection="descending",
    selection_strategy="composite",
):
    items = search_item_collection(
        stac,
        intersects=aoi_geometry,
        datetime=date_window,
        collections=["sentinel-1-rtc"],
    )
    if len(items) == 0:
        raise EmptyAcquisitionError(
            f"No Sentinel-1 RTC items found in requested window {date_window}."
        )
    load_kwargs = {"chunks": {"x": 2048, "y": 2048}, "groupby": "sat:absolute_orbit"}
    if like is not None:
        load_kwargs["like"] = like
    else:
        load_kwargs["crs"] = crs
        load_kwargs["resolution"] = 50
        load_kwargs["bbox"] = bbox

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        ds = odc.stac.load(items, **load_kwargs)
    if "time" not in ds.coords or len(ds.time) == 0:
        raise EmptyAcquisitionError(
            f"Sentinel-1 RTC load returned no acquisitions for requested window {date_window}."
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        if orbit_selection == "descending":
            ds = ds.where(ds.time.dt.hour > SENTINEL1_DESCENDING_PASS_HOUR_UTC_GT, drop=True)
        elif orbit_selection != "all":
            raise ValueError("Sentinel-1 orbit_selection must be one of: 'descending', 'all'.")
        if "time" not in ds.coords or len(ds.time) == 0:
            raise EmptyAcquisitionError(
                "Sentinel-1 RTC acquisitions were found, but none remained after orbit selection "
                f"for window {date_window}."
            )
        filtered_times = _serialize_time_values(ds.time.values)
        _log_detail(f"acquisitions found after orbit selection: {len(filtered_times)}")
        ds, selected_times, valid_pixel_fraction = _select_acquisitions(
            ds,
            target_date=target_date,
            selection_strategy=selection_strategy,
        )
        _log_detail(f"selected acquisitions: {_format_time_summary(selected_times)}")
    if rename_map:
        ds = ds.rename(rename_map)
    return ds, {
        "source": "sentinel-1-rtc",
        "requested_window": date_window,
        "available_acquisition_count": len(filtered_times),
        "available_acquisition_times": filtered_times,
        "selected_acquisition_count": len(selected_times),
        "selected_acquisition_times": selected_times,
        "valid_pixel_fraction": valid_pixel_fraction,
        "pass_selection": orbit_selection,
        "selection_strategy": selection_strategy,
        "groupby": "sat:absolute_orbit",
    }


def load_sentinel2_dataset(
    stac,
    aoi_geometry,
    date_window,
    cloud_cover,
    like,
    *,
    target_date,
    selection_strategy="composite",
):
    items = search_item_collection(
        stac,
        intersects=aoi_geometry,
        datetime=date_window,
        collections=["sentinel-2-l2a"],
        query={"eo:cloud_cover": {"lt": cloud_cover}},
    )
    if len(items) == 0:
        raise EmptyAcquisitionError(
            f"No Sentinel-2 L2A items found in requested window {date_window} with cloud cover < {cloud_cover}."
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        ds = odc.stac.load(
            items,
            chunks={"x": 2048, "y": 2048},
            like=like,
            groupby="solar_day",
        ).where(lambda x: x > 0, other=np.nan)
    if "time" not in ds.coords or len(ds.time) == 0:
        raise EmptyAcquisitionError(
            f"Sentinel-2 load returned no acquisitions for requested window {date_window}."
        )
    available_times = _serialize_time_values(ds.time.values)
    _log_detail(f"acquisitions found: {len(available_times)}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        ds = ds.where(ds["SCL"] != 9)
        ds = ds.where(ds["SCL"] != 8)
        ds = ds.where(ds["SCL"] != 0)
        selected_ds, selected_times, valid_pixel_fraction = _select_acquisitions(
            ds,
            target_date=target_date,
            selection_strategy=selection_strategy,
        )
        _log_detail(f"selected acquisitions: {_format_time_summary(selected_times)}")

        return selected_ds, {
            "source": "sentinel-2-l2a",
            "requested_window": date_window,
            "cloud_cover_lt": cloud_cover,
            "selected_acquisition_count": len(selected_times),
            "selected_acquisition_times": selected_times,
            "valid_pixel_fraction": valid_pixel_fraction,
            "available_acquisition_count": len(available_times),
            "available_acquisition_times": available_times,
            "masked_scl_classes_before_composite": [0, 8, 9],
            "selection_strategy": selection_strategy,
            "groupby": "solar_day",
        }


def load_snodas_dataset(target_date, match_ds):
    _log_stage("SNODAS snow depth")
    target_datetime = pd.to_datetime(target_date)
    snodas_url = (
        "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/"
        f"{target_datetime.year}/{target_datetime.strftime('%m')}_{target_datetime.strftime('%b')}/"
        f"SNODAS_{target_date}.tar"
    )
    snodas_cache_dir = get_snodas_cache_dir()
    snodas_fn = snodas_cache_dir / f"SNODAS_{target_date}.tar"
    if snodas_fn.exists() and snodas_fn.stat().st_size > 0:
        _log_detail(f"using cached SNODAS archive for {target_date}")
    else:
        url_download_with_retries(snodas_url, str(snodas_fn), overwrite=True)
        _log_detail(f"downloaded SNODAS archive for {target_date}")
    extract_tar_archive(snodas_fn, snodas_cache_dir)
    decompress_gzip_files(snodas_cache_dir)
    snodas_txt_paths = sorted(snodas_cache_dir.glob("us_ssmv11036*.txt"))
    if not snodas_txt_paths:
        raise FileNotFoundError(
            f"SNODAS archive {snodas_fn} did not produce an extracted us_ssmv11036*.txt file."
        )
    snodas_da = rxr.open_rasterio(snodas_txt_paths[0]).squeeze()
    snodas_resampled = _reproject_match_quietly(snodas_da, match_ds, resampling=rio.enums.Resampling.bilinear)
    snodas_resampled = snodas_resampled.where(snodas_resampled != -9999) / 1000
    return snodas_resampled.to_dataset(name="snodas_sd"), {
        "source": "snodas",
        "target_date": target_date,
        "archive_url": snodas_url,
    }


def load_cop30_dataset(stac, aoi_geometry, match_ds):
    _log_stage("COP30 elevation")
    items = search_item_collection(stac, collections=["cop-dem-glo-30"], intersects=aoi_geometry)
    data = []
    for item in items:
        dem_path = planetary_computer.sign(item.assets["data"]).href
        data.append(rxr.open_rasterio(dem_path))
    cop30_da = merge_arrays(data)
    cop30_ds = cop30_da.rename("elevation").squeeze().to_dataset()
    return _reproject_match_quietly(cop30_ds, match_ds, resampling=rio.enums.Resampling.bilinear).compute(), {
        "source": "cop-dem-glo-30",
        "item_count": len(items),
        "item_ids": [item.id for item in items],
    }


def load_fcf_dataset(fcf_path, aoi_bounds, match_ds):
    _log_stage("Forest cover fraction")
    resolved_fcf_path = ensure_fcf_file(fcf_path)
    fcf_ds = rxr.open_rasterio(resolved_fcf_path)
    fcf_ds = fcf_ds.rio.clip_box(*aoi_bounds, crs="EPSG:4326")
    fcf_ds = fcf_ds.rename("fcf").squeeze().to_dataset()
    fcf_ds = _reproject_match_quietly(fcf_ds, match_ds, resampling=rio.enums.Resampling.bilinear)
    fcf_ds["fcf"] = fcf_ds["fcf"].where(fcf_ds["fcf"] <= 100, np.nan) / 100
    return fcf_ds, {
        "source": "forest-cover-fraction",
        "path": resolved_fcf_path,
    }


def acquire_prediction_inputs(
    aoi,
    target_date,
    snowoff_date,
    buffer_period,
    cloud_cover,
    fcf_path=None,
    predict_swe=False,
    hill_pptwt_path=None,
    hill_td_path=None,
    sentinel1_orbit_selection="descending",
    selection_strategy="composite",
):
    aoi_geometry = build_aoi_geometry(aoi)
    aoi_gdf = build_aoi_geodataframe(aoi)
    crs = aoi_gdf.estimate_utm_crs()
    stac = create_stac_client()

    _log_stage("Sentinel-1 snow-on")
    snowon_s1_ds, snowon_s1_metadata = load_sentinel1_dataset(
        stac,
        aoi_geometry,
        date_range(target_date, buffer_period),
        target_date=target_date,
        crs=crs,
        bbox=aoi_gdf.total_bounds,
        rename_map={"vv": "snowon_vv", "vh": "snowon_vh"},
        orbit_selection=sentinel1_orbit_selection,
        selection_strategy=selection_strategy,
    )
    _log_stage("Sentinel-1 snow-off")
    snowoff_s1_ds, snowoff_s1_metadata = load_sentinel1_dataset(
        stac,
        aoi_geometry,
        date_range(snowoff_date, buffer_period),
        target_date=snowoff_date,
        like=snowon_s1_ds,
        rename_map={"vv": "snowoff_vv", "vh": "snowoff_vh"},
        orbit_selection=sentinel1_orbit_selection,
        selection_strategy=selection_strategy,
    )
    _log_stage("Sentinel-2 snow-on")
    s2_ds, s2_metadata = load_sentinel2_dataset(
        stac,
        aoi_geometry,
        date_range(target_date, buffer_period),
        cloud_cover=cloud_cover,
        like=snowon_s1_ds,
        target_date=target_date,
        selection_strategy=selection_strategy,
    )
    snodas_ds, snodas_metadata = load_snodas_dataset(target_date, snowon_s1_ds)
    cop30_ds, cop30_metadata = load_cop30_dataset(stac, aoi_geometry, snowon_s1_ds)
    fcf_ds, fcf_metadata = load_fcf_dataset(fcf_path, aoi_gdf.total_bounds, snowon_s1_ds)
    if predict_swe:
        hill_pptwt_path, hill_td_path = ensure_hill_inputs(hill_pptwt_path, hill_td_path)

    return {
        "snowon_s1": snowon_s1_ds,
        "snowoff_s1": snowoff_s1_ds,
        "s2": s2_ds,
        "snodas": snodas_ds,
        "cop30": cop30_ds,
        "fcf": fcf_ds,
    }, crs, {
        "sentinel1_snowon": snowon_s1_metadata,
        "sentinel1_snowoff": snowoff_s1_metadata,
        "sentinel2_snowon": s2_metadata,
        "snodas": snodas_metadata,
        "cop30": cop30_metadata,
        "fcf": fcf_metadata,
        "buffer_period_days": buffer_period,
        "sentinel1_pass_selection": sentinel1_orbit_selection,
        "selection_strategy": selection_strategy,
        "target_date": target_date,
        "snowoff_date": snowoff_date,
        "hill_pptwt_path": hill_pptwt_path,
        "hill_td_path": hill_td_path,
    }
