#! /usr/bin/env python

import argparse
from pathlib import Path

from deep_snow.api import (
    apply_model as api_apply_model,
    apply_model_ensemble as api_apply_model_ensemble,
    download_data as api_download_data,
    predict_sd as api_predict_sd,
    predict_sd_ts as api_predict_sd_ts,
)
from deep_snow.errors import ModelCompatibilityError
from deep_snow.inputs import parse_bool, parse_bounding_box
from deep_snow.model_loading import load_resdepth_checkpoint

def download_data(
    aoi,
    target_date,
    snowoff_date,
    buffer_period,
    out_dir,
    cloud_cover,
    fcf_path=None,
    sentinel1_orbit_selection="descending",
    selection_strategy="composite",
):
    return api_download_data(
        aoi,
        target_date,
        snowoff_date,
        buffer_period,
        out_dir,
        cloud_cover,
        fcf_path=fcf_path,
        sentinel1_orbit_selection=sentinel1_orbit_selection,
        selection_strategy=selection_strategy,
    )


def apply_model(crs, model_path, out_dir, out_name, write_tif, delete_inputs, out_crs, gpu=True):
    return api_apply_model(crs, model_path, out_dir, out_name, write_tif, delete_inputs, out_crs, gpu=gpu)


def apply_model_ensemble(crs, model_paths_list, out_dir, out_name, write_tif, delete_inputs, out_crs, gpu=True):
    return api_apply_model_ensemble(crs, model_paths_list, out_dir, out_name, write_tif, delete_inputs, out_crs, gpu=gpu)


def predict_sd(
    aoi,
    target_date,
    snowoff_date,
    model_path=None,
    out_dir="data",
    out_crs="utm",
    out_name=None,
    write_tif=True,
    delete_inputs=False,
    cloud_cover=25,
    buffer_period=6,
    fcf_path=None,
    gpu=None,
    use_ensemble=False,
    model_paths_list=None,
    sentinel1_orbit_selection="descending",
    selection_strategy="composite",
):
    return api_predict_sd(
        aoi=aoi,
        target_date=target_date,
        snowoff_date=snowoff_date,
        model_path=model_path,
        out_dir=out_dir,
        out_crs=out_crs,
        out_name=out_name,
        write_tif=write_tif,
        delete_inputs=delete_inputs,
        cloud_cover=cloud_cover,
        buffer_period=buffer_period,
        fcf_path=fcf_path,
        gpu=gpu,
        use_ensemble=use_ensemble,
        model_paths_list=model_paths_list,
        sentinel1_orbit_selection=sentinel1_orbit_selection,
        selection_strategy=selection_strategy,
    )


def predict_sd_ts(
    aoi,
    target_date,
    snowoff_date,
    model_path=None,
    out_dir="data",
    out_crs="utm",
    out_name="deep-snow_sd.tif",
    delete_inputs=False,
    cloud_cover=25,
    buffer_period=6,
    fcf_path=None,
    gpu=None,
    use_ensemble=False,
    model_paths_list=None,
    sentinel1_orbit_selection="descending",
    selection_strategy="composite",
):
    return api_predict_sd_ts(
        aoi=aoi,
        target_date=target_date,
        snowoff_date=snowoff_date,
        model_path=model_path,
        out_dir=out_dir,
        out_crs=out_crs,
        out_name=out_name,
        delete_inputs=delete_inputs,
        cloud_cover=cloud_cover,
        buffer_period=buffer_period,
        fcf_path=fcf_path,
        gpu=gpu,
        use_ensemble=use_ensemble,
        model_paths_list=model_paths_list,
        sentinel1_orbit_selection=sentinel1_orbit_selection,
        selection_strategy=selection_strategy,
    )


def get_parser():
    parser = argparse.ArgumentParser(description="CNN predictions of snow depth from remote sensing data")
    parser.add_argument("target_date", type=str, help="target date for snow depths with format YYYYmmdd")
    parser.add_argument("snow_off_date", type=str, help="snow off date (perhaps previous late summer) with format YYYYmmdd")
    parser.add_argument("aoi", type=parse_bounding_box, help="area of interest in format 'minlon minlat maxlon maxlat'")
    parser.add_argument("model_path", type=str, help="path to model weights to use")
    parser.add_argument("out_dir", type=str, help="directory to write inputs and outputs to")
    parser.add_argument("--cloud-cover", type=float, default=25, help="percent cloud cover allowed in Sentinel-2 images (0-100)")
    parser.add_argument("--buffer-period", type=int, default=6, help="days before and after each date to search for imagery")
    parser.add_argument("--delete-inputs", type=parse_bool, default=False, help="if True, delete input dataset from disk after processing")
    parser.add_argument("--write-tif", type=parse_bool, default=True, help="if True, write the predicted snow depth GeoTIFF")
    parser.add_argument("--out-name", type=str, default=None, help="name prefix for predicted snow depth output")
    parser.add_argument("--out-crs", type=str, default="utm", help="coordinate reference system for predicted snow depths: 'utm' or 'wgs84'")
    parser.add_argument("--fcf-path", type=str, default=None, help="path to a cached regional forest cover fraction raster")
    parser.add_argument("--gpu", type=parse_bool, default=None, help="if True, run inference on GPU. Defaults to using GPU when available.")
    parser.add_argument(
        "--s1-orbit-selection",
        choices=["descending", "all"],
        default="descending",
        help="Sentinel-1 orbit selection",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["composite", "nearest_usable"],
        default="composite",
        help="Acquisition selection strategy for both Sentinel-1 and Sentinel-2",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    predict_sd(
        aoi=args.aoi,
        target_date=args.target_date,
        snowoff_date=args.snow_off_date,
        model_path=args.model_path,
        out_dir=args.out_dir,
        out_crs=args.out_crs,
        out_name=args.out_name,
        write_tif=args.write_tif,
        delete_inputs=args.delete_inputs,
        cloud_cover=args.cloud_cover,
        buffer_period=args.buffer_period,
        fcf_path=args.fcf_path,
        gpu=args.gpu,
        sentinel1_orbit_selection=args.s1_orbit_selection,
        selection_strategy=args.selection_strategy,
    )


if __name__ == "__main__":
    main()
    
    
    
    

    
    
    
    
        
        
        
