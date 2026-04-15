from pathlib import Path
import json

import numpy as np
import pandas as pd
import xarray as xr
import xdem
from pyproj import Proj, Transformer

from deep_snow.utils import calc_dowy, db_scale


def merge_prediction_sources(raw_inputs):
    print("\n[prepare] combining source datasets")
    return xr.merge(
        [
            raw_inputs["snowon_s1"],
            raw_inputs["snowoff_s1"],
            raw_inputs["s2"],
            raw_inputs["snodas"],
            raw_inputs["cop30"],
            raw_inputs["fcf"],
        ],
        compat="override",
        join="override",
    ).squeeze()


def add_radar_features(ds):
    ds["snowon_vv"] = (("y", "x"), db_scale(ds["snowon_vv"]))
    ds["snowon_vh"] = (("y", "x"), db_scale(ds["snowon_vh"]))
    ds["snowoff_vv"] = (("y", "x"), db_scale(ds["snowoff_vv"]))
    ds["snowoff_vh"] = (("y", "x"), db_scale(ds["snowoff_vh"]))
    ds["snowon_cr"] = ds["snowon_vh"] - ds["snowon_vv"]
    ds["snowoff_cr"] = ds["snowoff_vh"] - ds["snowoff_vv"]
    ds["delta_cr"] = ds["snowon_cr"] - ds["snowoff_cr"]
    return ds


def add_optical_features(ds):
    ds["ndvi"] = (ds["B08"] - ds["B04"]) / (ds["B08"] + ds["B04"])
    ds["ndsi"] = (ds["B03"] - ds["B11"]) / (ds["B03"] + ds["B11"])
    ds["ndwi"] = (ds["B03"] - ds["B08"]) / (ds["B03"] + ds["B08"])
    return ds


def add_coordinate_features(ds, crs):
    utm_proj = Proj(proj="utm", zone=crs.name[-3:-1], ellps="WGS84")
    wgs84_proj = Proj(proj="latlong", datum="WGS84")
    transformer = Transformer.from_proj(utm_proj, wgs84_proj, always_xy=True)
    x_coords, y_coords = np.meshgrid(ds["x"].values, ds["y"].values)
    lon, lat = transformer.transform(x_coords, y_coords)
    ds["latitude"] = (("y", "x"), lat)
    ds["longitude"] = (("y", "x"), lon)
    return ds


def add_dowy_feature(ds, target_date):
    dowy_value = calc_dowy(pd.to_datetime(target_date).dayofyear)
    ds["dowy"] = (("y", "x"), np.full_like(ds["latitude"], dowy_value))
    return ds


def add_terrain_features(ds):
    elevation = ds["elevation"].squeeze()
    dem = xdem.DEM.from_array(
        elevation.values,
        transform=elevation.rio.transform(),
        crs=elevation.rio.crs,
    )
    ds["aspect"] = (("y", "x"), xdem.terrain.aspect(dem).data.data)
    ds["northness"] = np.cos(np.deg2rad(ds.aspect))
    ds["slope"] = (("y", "x"), xdem.terrain.slope(dem).data.data)
    ds["curvature"] = (("y", "x"), xdem.terrain.curvature(dem).data.data)
    ds["tpi"] = (("y", "x"), xdem.terrain.topographic_position_index(dem).data.data)
    ds["tri"] = (("y", "x"), xdem.terrain.terrain_ruggedness_index(dem).data.data)
    return ds


def add_gap_mask(ds):
    ds["gap_s1_snowon"] = (
        ("y", "x"),
        (
            np.isnan(ds.snowon_vv.values)
            | np.isnan(ds.snowon_vh.values)
        ).astype(np.uint8),
    )
    ds["gap_s1_snowoff"] = (
        ("y", "x"),
        (
            np.isnan(ds.snowoff_vv.values)
            | np.isnan(ds.snowoff_vh.values)
        ).astype(np.uint8),
    )
    ds["gap_s2"] = (
        ("y", "x"),
        (
            np.isnan(ds.B02.values)
            | np.isnan(ds.B11.values)
            | (ds.SCL.values == 9)
            | (ds.SCL.values == 0)
        ).astype(np.uint8),
    )
    ds["data_gaps"] = (
        ("y", "x"),
        (
            ds["gap_s1_snowon"].values.astype(bool)
            | ds["gap_s1_snowoff"].values.astype(bool)
            | ds["gap_s2"].values.astype(bool)
        ).astype(np.uint8),
    )
    return ds


def add_input_provenance(ds, input_provenance):
    if not input_provenance:
        return ds

    ds.attrs["deep_snow_input_provenance_json"] = json.dumps(input_provenance, sort_keys=True)
    return ds


def build_prediction_dataset(raw_inputs, target_date, crs, input_provenance=None):
    ds = merge_prediction_sources(raw_inputs)
    ds = ds.rio.write_crs(crs)
    ds = add_radar_features(ds)
    ds = add_optical_features(ds)
    ds = add_coordinate_features(ds, crs)
    ds = add_dowy_feature(ds, target_date)
    ds = add_terrain_features(ds)
    ds = add_gap_mask(ds)
    ds = add_input_provenance(ds, input_provenance)
    return ds


def write_model_inputs(ds, out_dir):
    data_fn = Path(out_dir) / "model_inputs.nc"
    print(f"[prepare] writing model inputs to {data_fn.as_posix()}")
    ds.to_netcdf(data_fn)
    print("[prepare] model inputs ready")
    return str(data_fn)
