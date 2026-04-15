import argparse
from deep_snow.inputs import parse_bounding_box
from deep_snow.workflows import predict_tile


def get_parser():
    parser = argparse.ArgumentParser(description="CNN predictions of snow depth from remote sensing data")
    parser.add_argument("target_date", type=str, help="target date for snow depths with format YYYYmmdd")
    parser.add_argument("snow_off_date", type=str, help="snow-off date (perhaps previous late summer) with format YYYYmmdd")
    parser.add_argument("aoi", type=parse_bounding_box, help="area of interest in format 'minlon minlat maxlon maxlat'")
    parser.add_argument("cloud_cover", type=str, help="percent cloud cover allowed in Sentinel-2 images (0-100)")
    parser.add_argument("use_ensemble", type=str, help="whether to use model ensemble, True or False")
    parser.add_argument(
        "--s1-orbit-selection",
        choices=["descending", "all"],
        default="descending",
        help="Sentinel-1 orbit selection",
    )
    parser.add_argument(
        "--clip-aoi",
        type=parse_bounding_box,
        default=None,
        help="optional output crop area in format 'minlon minlat maxlon maxlat'",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["composite", "nearest_usable"],
        default="composite",
        help="Acquisition selection strategy for both Sentinel-1 and Sentinel-2",
    )
    parser.add_argument(
        "--predict-swe",
        choices=["True", "False"],
        default="False",
        help="Also derive snow density and SWE from predicted depth",
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    predict_tile(
        target_date=args.target_date,
        snow_off_date=args.snow_off_date,
        aoi=args.aoi,
        cloud_cover=args.cloud_cover,
        use_ensemble=args.use_ensemble == "True",
        crop_aoi=args.clip_aoi,
        sentinel1_orbit_selection=args.s1_orbit_selection,
        selection_strategy=args.selection_strategy,
        predict_swe=args.predict_swe == "True",
    )

if __name__ == "__main__":
   main()
