import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

import deep_snow.models
from deep_snow.dataset import norm_dict
from deep_snow.utils import calc_norm, undo_norm


MODEL_TILE_SIZE = 1024
MODEL_TILE_PADDING = 50
HILL_ACCUMULATION_PARAMETERS = [0.0533, 0.948, 0.1701, -0.1314, 0.2922]
HILL_ABLATION_PARAMETERS = [0.0481, 1.0395, 0.1699, -0.0461, 0.1804]
PREDICTION_INPUT_CHANNELS = [
    "snodas_sd",
    "blue",
    "swir1",
    "ndsi",
    "elevation",
    "northness",
    "slope",
    "curvature",
    "dowy",
    "delta_cr",
    "fcf",
]
FEATURE_SPECS = {
    "snowon_vv": ("snowon_vv", norm_dict["vv"]),
    "snowon_vh": ("snowon_vh", norm_dict["vh"]),
    "snowoff_vv": ("snowoff_vv", norm_dict["vv"]),
    "snowoff_vh": ("snowoff_vh", norm_dict["vh"]),
    "aerosol_optical_thickness": ("AOT", norm_dict["aerosol_optical_thickness"]),
    "coastal_aerosol": ("B01", norm_dict["coastal_aerosol"]),
    "blue": ("B02", norm_dict["blue"]),
    "green": ("B03", norm_dict["green"]),
    "red": ("B04", norm_dict["red"]),
    "red_edge1": ("B05", norm_dict["red_edge1"]),
    "red_edge2": ("B06", norm_dict["red_edge2"]),
    "red_edge3": ("B07", norm_dict["red_edge3"]),
    "nir": ("B08", norm_dict["nir"]),
    "water_vapor": ("B09", norm_dict["water_vapor"]),
    "swir1": ("B11", norm_dict["swir1"]),
    "swir2": ("B12", norm_dict["swir2"]),
    "scene_class_map": ("SCL", norm_dict["scene_class_map"]),
    "water_vapor_product": ("WVP", norm_dict["water_vapor_product"]),
    "snodas_sd": ("snodas_sd", norm_dict["aso_sd"]),
    "elevation": ("elevation", norm_dict["elevation"]),
    "aspect": ("aspect", norm_dict["aspect"]),
    "northness": ("northness", [0, 1]),
    "slope": ("slope", norm_dict["slope"]),
    "curvature": ("curvature", norm_dict["curvature"]),
    "tpi": ("tpi", norm_dict["tpi"]),
    "tri": ("tri", norm_dict["tri"]),
    "latitude": ("latitude", norm_dict["latitude"]),
    "longitude": ("longitude", norm_dict["longitude"]),
    "dowy": ("dowy", [0, 365]),
    "ndvi": ("ndvi", [-1, 1]),
    "ndsi": ("ndsi", [-1, 1]),
    "ndwi": ("ndwi", [-1, 1]),
    "snowon_cr": ("snowon_cr", norm_dict["cr"]),
    "snowoff_cr": ("snowoff_cr", norm_dict["cr"]),
    "delta_cr": ("delta_cr", norm_dict["delta_cr"]),
    "fcf": ("fcf", None),
}


def get_prediction_input_channels():
    return list(PREDICTION_INPUT_CHANNELS)


def get_prediction_tile_boundaries(height, width, tile_size=MODEL_TILE_SIZE, padding=MODEL_TILE_PADDING):
    stride = tile_size - 2 * padding
    row_boundaries = []
    col_boundaries = []

    for y_index in range(1, math.ceil(height / stride)):
        boundary = y_index * stride + padding
        if boundary < height:
            row_boundaries.append(boundary)

    for x_index in range(1, math.ceil(width / stride)):
        boundary = x_index * stride + padding
        if boundary < width:
            col_boundaries.append(boundary)

    return {
        "tile_size": tile_size,
        "padding": padding,
        "stride": stride,
        "row_boundaries": row_boundaries,
        "col_boundaries": col_boundaries,
    }


def load_prediction_dataset(out_dir):
    print(f"\n[predict] reading model inputs from {(Path(out_dir) / 'model_inputs.nc').as_posix()}")
    ds = xr.open_dataset(Path(out_dir) / "model_inputs.nc", decode_times=False)
    if "data_gaps" in ds:
        data_gap_count = int(ds["data_gaps"].fillna(0).sum().item())
        data_gap_fraction = float(ds["data_gaps"].fillna(0).mean().item())
        ds.attrs["deep_snow_input_gap_pixel_count"] = data_gap_count
        ds.attrs["deep_snow_input_gap_fraction"] = data_gap_fraction
        for gap_name in ("gap_s1_snowon", "gap_s1_snowoff", "gap_s2"):
            if gap_name in ds:
                ds.attrs[f"deep_snow_{gap_name}_pixel_count"] = int(ds[gap_name].fillna(0).sum().item())
                ds.attrs[f"deep_snow_{gap_name}_fraction"] = float(ds[gap_name].fillna(0).mean().item())
        if data_gap_count > 0:
            print(
                "[predict] WARNING: input gap mask indicates "
                f"{data_gap_count} pixel(s) with missing inputs "
                f"({data_gap_fraction:.2%} of the prediction grid). "
                "Model inputs will still be zero-filled for inference compatibility."
            )
    return ds.fillna(0)


def build_normalized_feature_dict(ds):
    feature_dict = {}
    for feature_name, (source_name, norm) in FEATURE_SPECS.items():
        tensor = torch.as_tensor(ds[source_name].values, dtype=torch.float32)
        if norm is not None:
            tensor = calc_norm(tensor, norm)
        feature_dict[feature_name] = torch.clamp(tensor, 0, 1)[None, None, :, :]
    return feature_dict


def build_model_inputs(ds, input_channels=None):
    channels = input_channels or get_prediction_input_channels()
    feature_dict = build_normalized_feature_dict(ds)
    return torch.cat([feature_dict[channel] for channel in channels], dim=1)


def log_model_input_summary(ds, inputs, input_channels=None):
    channels = input_channels or get_prediction_input_channels()
    height = int(inputs.shape[-2])
    width = int(inputs.shape[-1])
    print(
        "[predict] model inputs ready: "
        f"channels={len(channels)} | grid={height}x{width} | pixels={height * width}"
    )
    if "deep_snow_input_gap_fraction" in ds.attrs:
        print(
            "[predict] input gap summary: "
            f"combined={ds.attrs.get('deep_snow_input_gap_fraction', 0.0):.2%} | "
            f"S1 snow-on={ds.attrs.get('deep_snow_gap_s1_snowon_fraction', 0.0):.2%} | "
            f"S1 snow-off={ds.attrs.get('deep_snow_gap_s1_snowoff_fraction', 0.0):.2%} | "
            f"S2={ds.attrs.get('deep_snow_gap_s2_fraction', 0.0):.2%}"
        )


def validate_model_inputs(inputs, input_channels=None):
    channels = input_channels or get_prediction_input_channels()
    nonfinite_mask = ~torch.isfinite(inputs)
    if not bool(nonfinite_mask.any()):
        return

    bad_channel_indexes = torch.nonzero(nonfinite_mask.any(dim=(0, 2, 3)), as_tuple=False).flatten().tolist()
    bad_channels = [channels[index] for index in bad_channel_indexes]
    raise ValueError(
        "Model inputs contain non-finite values after preprocessing for channel(s): "
        f"{', '.join(bad_channels)}."
    )


def load_resdepth_models(model_paths, gpu, checkpoint_loader, input_channels=None):
    channels = input_channels or get_prediction_input_channels()
    models = []
    for model_path in model_paths:
        model = deep_snow.models.ResDepth(n_input_channels=len(channels), depth=5)
        checkpoint_loader(model, model_path, gpu)
        model.eval()
        models.append(model)
    return models


def _predict_tile(models, tile_inputs, gpu):
    with torch.no_grad():
        if gpu:
            tile_inputs = tile_inputs.to("cuda")
        tile_predictions = [model(tile_inputs) for model in models]
        if len(tile_predictions) == 1:
            return tile_predictions[0]
        return torch.median(torch.stack(tile_predictions, dim=0), dim=0).values


def _write_tile_prediction(pred_pad, tile_prediction, ymin, xmin, tile_size, padding):
    xmax = xmin + tile_size - padding
    ymax = ymin + tile_size - padding
    tile_prediction = tile_prediction.detach().cpu().squeeze()

    if ymin == 0 and xmin == 0:
        tile_prediction = tile_prediction[:-padding, :-padding]
    elif ymin == 0:
        xmin += padding
        tile_prediction = tile_prediction[:-padding, padding:-padding]
    elif xmin == 0:
        ymin += padding
        tile_prediction = tile_prediction[padding:-padding, :-padding]
    else:
        xmin += padding
        ymin += padding
        tile_prediction = tile_prediction[padding:-padding, padding:-padding]

    pred_pad[ymin:ymax, xmin:xmax] = tile_prediction


def predict_in_tiles(inputs, models, tile_size=MODEL_TILE_SIZE, padding=MODEL_TILE_PADDING, gpu=True):
    stride = tile_size - 2 * padding
    inputs_pad = F.pad(inputs, (0, tile_size, 0, tile_size), "constant", 0)
    pred_pad = torch.empty_like(inputs_pad[0, 0, :, :])
    tile_count_x = math.ceil(inputs.shape[-1] / stride)
    tile_count_y = math.ceil(inputs.shape[-2] / stride)
    total_tiles = tile_count_x * tile_count_y

    print(
        "[predict] running tiled inference: "
        f"{total_tiles} model tile(s) | tile_size={tile_size} | padding={padding} | stride={stride}"
    )

    tile_number = 0
    for x_index in range(math.ceil(inputs.shape[-1] / stride)):
        for y_index in range(math.ceil(inputs.shape[-2] / stride)):
            tile_number += 1
            ymin = y_index * stride
            xmin = x_index * stride
            tile_start = time.perf_counter()
            print(
                "[predict] inference tile "
                f"{tile_number}/{total_tiles}: x={x_index + 1}/{tile_count_x}, "
                f"y={y_index + 1}/{tile_count_y}, window=rows {ymin}:{ymin + tile_size}, "
                f"cols {xmin}:{xmin + tile_size}"
            )
            tile_prediction = _predict_tile(
                models,
                inputs_pad[:, :, ymin : ymin + tile_size, xmin : xmin + tile_size],
                gpu=gpu,
            )
            _write_tile_prediction(pred_pad, tile_prediction, ymin, xmin, tile_size, padding)
            print(
                f"[predict] finished inference tile {tile_number}/{total_tiles} "
                f"in {time.perf_counter() - tile_start:.2f}s"
            )

    return pred_pad[: inputs.shape[-2], : inputs.shape[-1]]


def _hill_swe(depth_m, pptwt_mm, td_c, dowy):
    depth_mm = depth_m * 1000.0
    acc = HILL_ACCUMULATION_PARAMETERS
    abl = HILL_ABLATION_PARAMETERS

    accumulation_term = (
        acc[0]
        * depth_mm**acc[1]
        * pptwt_mm**acc[2]
        * td_c**acc[3]
        * dowy**acc[4]
        * ((-np.tanh(0.01 * (dowy - 180))) + 1)
        / 2
    )
    ablation_term = (
        abl[0]
        * depth_mm**abl[1]
        * pptwt_mm**abl[2]
        * td_c**abl[3]
        * dowy**abl[4]
        * (np.tanh(0.01 * (dowy - 180)) + 1)
        / 2
    )
    return accumulation_term + ablation_term


def _write_float_nodata(da):
    return da.rio.write_nodata(np.nan, encoded=True)


def add_density_and_swe(
    ds,
    *,
    pptwt_path,
    td_path,
):
    import rioxarray

    if "predicted_sd" not in ds:
        raise ValueError("predicted_sd must be present before deriving SWE and density.")
    if "dowy" not in ds:
        raise ValueError("dowy must be present before deriving SWE and density.")

    depth = ds["predicted_sd"]
    pptwt = rioxarray.open_rasterio(pptwt_path).squeeze("band", drop=True).rio.write_crs("EPSG:4326")
    td = rioxarray.open_rasterio(td_path).squeeze("band", drop=True).rio.write_crs("EPSG:4326")

    pptwt = pptwt.rio.reproject_match(depth)
    td = td.rio.reproject_match(depth)

    valid_observation = depth.notnull()
    snow_present = valid_observation & (depth > 0)
    swe_mm = _hill_swe(
        depth.where(snow_present),
        pptwt.where(snow_present),
        td.where(snow_present),
        ds["dowy"].where(snow_present),
    )
    swe_m = (swe_mm / 1000.0).where(snow_present, 0).where(valid_observation)
    density = xr.where(snow_present, (swe_m / depth) * 1000.0, 0).where(valid_observation)

    ds["predicted_swe"] = _write_float_nodata(swe_m.astype(np.float32))
    ds["predicted_density"] = _write_float_nodata(density.astype(np.float32))
    ds["predicted_swe"].attrs.update(
        {
            "long_name": "Predicted snow water equivalent",
            "units": "m",
            "source": "Hill et al. depth-to-SWE model applied to predicted snow depth",
        }
    )
    ds["predicted_density"].attrs.update(
        {
            "long_name": "Predicted bulk snow density",
            "units": "kg m-3",
            "source": "Hill et al. depth-to-SWE model applied to predicted snow depth",
        }
    )
    ds.attrs["deep_snow_hill_pptwt_path"] = str(Path(pptwt_path).as_posix())
    ds.attrs["deep_snow_hill_td_path"] = str(Path(td_path).as_posix())
    return ds


def finalize_prediction_dataset(
    ds,
    pred_sd,
    crs,
    out_dir,
    out_name,
    write_tif,
    delete_inputs,
    out_crs,
    data_fn,
    predict_swe=False,
    hill_pptwt_path=None,
    hill_td_path=None,
    crop_bounds=None,
    crop_crs=None,
):
    pred_sd = undo_norm(pred_sd, norm_dict["aso_sd"])
    ds["predicted_sd"] = (("y", "x"), pred_sd.cpu().numpy().astype(np.float32))
    ds["predicted_sd"] = ds["predicted_sd"].where(ds["predicted_sd"] > 0, 0)
    ds["predicted_sd"] = _write_float_nodata(ds["predicted_sd"])
    ds = ds.rio.write_crs(crs)

    if out_crs == "wgs84":
        ds = ds.rio.reproject("EPSG:4326")
        ds = ds.rio.write_crs("EPSG:4326")

    if crop_bounds is not None:
        ds = ds.rio.clip_box(*crop_bounds, crs=crop_crs)

    if predict_swe:
        ds = add_density_and_swe(
            ds,
            pptwt_path=hill_pptwt_path,
            td_path=hill_td_path,
        )

    if write_tif:
        output_path = f"{out_dir}/{out_name}_sd.tif"
        print(f"[predict] writing snow depth GeoTIFF to {output_path}")
        ds.predicted_sd.rio.to_raster(output_path, compress="lzw")
        ds.attrs["deep_snow_predicted_tif_path"] = output_path
        if predict_swe:
            swe_output_path = f"{out_dir}/{out_name}_swe.tif"
            density_output_path = f"{out_dir}/{out_name}_density.tif"
            print(f"[predict] writing SWE GeoTIFF to {swe_output_path}")
            ds.predicted_swe.rio.to_raster(swe_output_path, compress="lzw")
            print(f"[predict] writing density GeoTIFF to {density_output_path}")
            ds.predicted_density.rio.to_raster(density_output_path, compress="lzw")
            ds.attrs["deep_snow_predicted_swe_tif_path"] = swe_output_path
            ds.attrs["deep_snow_predicted_density_tif_path"] = density_output_path
        if "data_gaps" in ds:
            gap_output_path = f"{out_dir}/{out_name}_input_gaps.tif"
            print(f"[predict] writing input-gap GeoTIFF to {gap_output_path}")
            ds.data_gaps.rio.to_raster(gap_output_path, compress="lzw")
            ds.attrs["deep_snow_input_gaps_tif_path"] = gap_output_path
            detailed_gap_output_path = f"{out_dir}/{out_name}_input_gaps.nc"
            detailed_gap_vars = [
                gap_name
                for gap_name in ("gap_s1_snowon", "gap_s1_snowoff", "gap_s2", "data_gaps")
                if gap_name in ds
            ]
            if detailed_gap_vars:
                print(f"[predict] writing detailed input-gap NetCDF to {detailed_gap_output_path}")
                ds[detailed_gap_vars].to_netcdf(detailed_gap_output_path)
                ds.attrs["deep_snow_input_gaps_netcdf_path"] = detailed_gap_output_path

    if delete_inputs:
        print(f"[predict] removing intermediate inputs: {data_fn}")
        os.remove(data_fn)

    print("[predict] prediction finished")
    return ds


def apply_models(
    crs,
    model_paths,
    out_dir,
    out_name,
    write_tif,
    delete_inputs,
    out_crs,
    checkpoint_loader,
    gpu=True,
    predict_swe=False,
    hill_pptwt_path=None,
    hill_td_path=None,
    crop_bounds=None,
    crop_crs=None,
):
    data_fn = str(Path(out_dir) / "model_inputs.nc")
    ds = load_prediction_dataset(out_dir)
    inputs = build_model_inputs(ds)
    validate_model_inputs(inputs, input_channels=get_prediction_input_channels())
    log_model_input_summary(ds, inputs, input_channels=get_prediction_input_channels())
    print(f"[predict] preparing {len(model_paths)} model(s) for inference")
    models = load_resdepth_models(
        model_paths,
        gpu=gpu,
        checkpoint_loader=checkpoint_loader,
        input_channels=get_prediction_input_channels(),
    )
    pred_sd = predict_in_tiles(inputs, models, gpu=gpu)
    return finalize_prediction_dataset(
        ds=ds,
        pred_sd=pred_sd,
        crs=crs,
        out_dir=out_dir,
        out_name=out_name,
        write_tif=write_tif,
        delete_inputs=delete_inputs,
        out_crs=out_crs,
        data_fn=data_fn,
        predict_swe=predict_swe,
        hill_pptwt_path=hill_pptwt_path,
        hill_td_path=hill_td_path,
        crop_bounds=crop_bounds,
        crop_crs=crop_crs,
    )
