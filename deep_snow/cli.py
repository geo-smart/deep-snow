import argparse

from deep_snow.inputs import parse_bool, parse_bounding_box
from deep_snow.resources import get_default_land_path


def add_predict_sd_arguments(parser):
    parser.add_argument("target_date", type=str, help="Target date in YYYYmmdd format.")
    parser.add_argument("snow_off_date", type=str, help="Snow-off reference date in YYYYmmdd format.")
    parser.add_argument("aoi", type=parse_bounding_box, help="Bounding box: 'minlon minlat maxlon maxlat'.")
    parser.add_argument("cloud_cover", type=float, help="Maximum allowed Sentinel-2 cloud cover percentage.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a single model checkpoint. Defaults to the packaged single-model choice.",
    )
    parser.add_argument(
        "--use-ensemble",
        action="store_true",
        help="Use the configured ensemble instead of the default single model.",
    )
    parser.add_argument(
        "--out-name",
        default=None,
        help="Optional output name prefix. Defaults to a name derived from the date and AOI.",
    )
    parser.add_argument("--out-dir", default="data", help="Directory for model inputs and outputs.")
    parser.add_argument("--out-crs", default="wgs84", help="Output CRS for written rasters.")
    parser.add_argument("--buffer-period", type=int, default=6, help="Days before and after each target date to search for imagery.")
    parser.add_argument("--fcf-path", default=None, help="Optional path to a cached forest cover fraction raster.")
    parser.add_argument(
        "--tile-size-degrees",
        type=float,
        default=None,
        help="Optional geographic tile size in degrees for automatic large-AOI local tiling. Defaults to the batch tile size.",
    )
    parser.add_argument(
        "--tile-large-aoi",
        type=parse_bool,
        default=True,
        help="When True, automatically split AOIs larger than the batch tile size into local tiles and mosaic them.",
    )
    parser.add_argument("--delete-inputs", type=parse_bool, default=True, help="Delete model_inputs.nc after prediction finishes.")
    parser.add_argument("--write-tif", type=parse_bool, default=True, help="Write the predicted snow depth GeoTIFF.")
    parser.add_argument("--predict-swe", type=parse_bool, default=False, help="Also derive snow density and SWE from predicted depth using the Hill et al. model.")
    parser.add_argument("--gpu", type=parse_bool, default=False, help="Run inference on GPU when available.")


def add_predict_sd_timeseries_arguments(parser):
    parser.add_argument("begin_date", type=str, help="Earliest prediction date in YYYYmmdd format.")
    parser.add_argument("end_date", type=str, help="Latest prediction date in YYYYmmdd format.")
    parser.add_argument("snow_off_day", type=str, help="Snow-off month/day in mmdd format.")
    parser.add_argument("aoi", type=parse_bounding_box, help="Bounding box: 'minlon minlat maxlon maxlat'.")
    parser.add_argument("cloud_cover", type=float, help="Maximum allowed Sentinel-2 cloud cover percentage.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a single model checkpoint. Defaults to the packaged single-model choice.",
    )
    parser.add_argument(
        "--use-ensemble",
        action="store_true",
        help="Use the configured ensemble instead of the default single model.",
    )
    parser.add_argument(
        "--out-name",
        default=None,
        help="Optional output name prefix. When omitted, each date gets its own derived name.",
    )
    parser.add_argument("--out-dir", default="data", help="Directory for model inputs and outputs.")
    parser.add_argument("--out-crs", default="wgs84", help="Output CRS for written rasters.")
    parser.add_argument("--buffer-period", type=int, default=6, help="Days before and after each target date to search for imagery.")
    parser.add_argument("--fcf-path", default=None, help="Optional path to a cached forest cover fraction raster.")
    parser.add_argument(
        "--tile-size-degrees",
        type=float,
        default=None,
        help="Optional geographic tile size in degrees for automatic large-AOI local tiling.",
    )
    parser.add_argument(
        "--tile-large-aoi",
        type=parse_bool,
        default=True,
        help="When True, automatically split AOIs larger than the batch tile size into local tiles and mosaic them.",
    )
    parser.add_argument("--delete-inputs", type=parse_bool, default=True, help="Delete model_inputs.nc after each prediction finishes.")
    parser.add_argument("--write-tif", type=parse_bool, default=True, help="Write GeoTIFF outputs for each prediction in the time series.")
    parser.add_argument("--predict-swe", type=parse_bool, default=False, help="Also derive snow density and SWE from predicted depth for each time step.")
    parser.add_argument("--gpu", type=parse_bool, default=False, help="Run inference on GPU when available.")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Local helpers for deep-snow prediction workflows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_sd_parser = subparsers.add_parser(
        "predict-sd",
        help="Predict snow depth locally for one AOI/date, automatically tiling large AOIs when needed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_predict_sd_arguments(predict_sd_parser)

    predict_sd_timeseries_parser = subparsers.add_parser(
        "predict-sd-timeseries",
        help="Predict a local snow-depth time series over one AOI by repeatedly running predict-sd",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_predict_sd_timeseries_arguments(predict_sd_timeseries_parser)

    predict_tile_parser = subparsers.add_parser(
        "predict-tile",
        help=argparse.SUPPRESS,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Hidden compatibility alias for advanced users who want to force a single tile.
    add_predict_sd_arguments(predict_tile_parser)

    predict_batch_parser = subparsers.add_parser(
        "predict-batch",
        help=argparse.SUPPRESS,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Hidden compatibility alias for the older local command naming.
    add_predict_sd_arguments(predict_batch_parser)

    predict_time_series_parser = subparsers.add_parser(
        "predict-time-series",
        help=argparse.SUPPRESS,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Hidden compatibility alias for the older local command naming.
    add_predict_sd_timeseries_arguments(predict_time_series_parser)

    tiles_parser = subparsers.add_parser(
        "prep-tiles",
        help="Generate tile jobs for a bounding box",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    tiles_parser.add_argument("target_date", type=str, help="Target date in YYYYmmdd format.")
    tiles_parser.add_argument("aoi", type=parse_bounding_box, help="Bounding box: 'minlon minlat maxlon maxlat'.")
    tiles_parser.add_argument(
        "--land-path",
        default=get_default_land_path(),
        help="Land polygons used to filter empty ocean tiles.",
    )

    timeseries_parser = subparsers.add_parser(
        "prep-time-series",
        help="Generate target dates and snow-off dates for a time series",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    timeseries_parser.add_argument("begin_date", type=str, help="Earliest prediction date in YYYYmmdd format.")
    timeseries_parser.add_argument("end_date", type=str, help="Latest prediction date in YYYYmmdd format.")
    timeseries_parser.add_argument("snow_off_day", type=str, help="Snow-off month/day in mmdd format.")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.command in {"predict-sd", "predict-batch"}:
        from deep_snow.workflows import predict_batch

        predict_batch(
            target_date=args.target_date,
            snow_off_date=args.snow_off_date,
            aoi=args.aoi,
            cloud_cover=args.cloud_cover,
            use_ensemble=args.use_ensemble,
            model_path=args.model_path,
            out_name=args.out_name,
            out_dir=args.out_dir,
            out_crs=args.out_crs,
            buffer_period=args.buffer_period,
            fcf_path=args.fcf_path,
            tile_large_aoi=args.tile_large_aoi,
            tile_size_degrees=args.tile_size_degrees,
            delete_inputs=args.delete_inputs,
            write_tif=args.write_tif,
            predict_swe=args.predict_swe,
            gpu=args.gpu,
        )
        return

    if args.command == "predict-tile":
        from deep_snow.workflows import predict_tile

        predict_tile(
            target_date=args.target_date,
            snow_off_date=args.snow_off_date,
            aoi=args.aoi,
            cloud_cover=args.cloud_cover,
            use_ensemble=args.use_ensemble,
            model_path=args.model_path,
            out_name=args.out_name,
            out_dir=args.out_dir,
            out_crs=args.out_crs,
            buffer_period=args.buffer_period,
            fcf_path=args.fcf_path,
            delete_inputs=args.delete_inputs,
            write_tif=args.write_tif,
            predict_swe=args.predict_swe,
            gpu=args.gpu,
        )
        return

    if args.command in {"predict-sd-timeseries", "predict-time-series"}:
        from deep_snow.api import predict_sd_timeseries

        predict_sd_timeseries(
            begin_date=args.begin_date,
            end_date=args.end_date,
            snow_off_day=args.snow_off_day,
            aoi=args.aoi,
            cloud_cover=args.cloud_cover,
            use_ensemble=args.use_ensemble,
            model_path=args.model_path,
            out_name=args.out_name,
            out_dir=args.out_dir,
            out_crs=args.out_crs,
            buffer_period=args.buffer_period,
            fcf_path=args.fcf_path,
            tile_large_aoi=args.tile_large_aoi,
            tile_size_degrees=args.tile_size_degrees,
            delete_inputs=args.delete_inputs,
            write_tif=args.write_tif,
            predict_swe=args.predict_swe,
            gpu=args.gpu,
        )
        return

    if args.command == "prep-tiles":
        from deep_snow.tiling import build_matrix_json, build_tile_jobs

        print(
            build_matrix_json(
                build_tile_jobs(
                    target_date=args.target_date,
                    aoi=args.aoi,
                    land_path=args.land_path,
                )
            )
        )
        return

    if args.command == "prep-time-series":
        from deep_snow.tiling import build_matrix_json
        from deep_snow.workflows import build_time_series_jobs

        print(build_matrix_json(build_time_series_jobs(args.begin_date, args.end_date, args.snow_off_day)))


if __name__ == "__main__":
    main()
