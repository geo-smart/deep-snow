from deep_snow.application import download_data, apply_model
import argparse
import time

def parse_bounding_box(value):
    try:
        minlon, minlat, maxlon, maxlat = map(float, value.split())
        return {'minlon':minlon, 'minlat':minlat, 'maxlon':maxlon, 'maxlat':maxlat}
    except ValueError:
        raise argparse.ArgumentTypeError("Bounding box must be in format 'minlon minlat maxlon maxlat' with float values.")

def get_parser():
    parser = argparse.ArgumentParser(description="CNN predictions of snow depth from remote sensing data")
    parser.add_argument("target_date", type=str, help="target date for snow depths with format YYYYmmdd")
    parser.add_argument("snow_off_date", type=str, help="snow-off date (perhaps previous late summer) with format YYYYmmdd")
    parser.add_argument("aoi", type=parse_bounding_box, help="area of interest in format 'minlon minlat maxlon maxlat'")
    parser.add_argument("cloud_cover", type=str, help="percent cloud cover allowed in Sentinel-2 images (0-100)")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    out_dir = '../data'
    out_name = f'{args.target_date}_deep-snow_{args.aoi["minlon"]:.{2}f}_{args.aoi["minlat"]:.{2}f}_{args.aoi["maxlon"]:.{2}f}_{args.aoi["maxlat"]:.{2}f}'
    model_path = '../weights/quinn_ResDepth_v10_256epochs'

    max_retries = 100
    retry_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            crs = download_data(aoi=args.aoi, target_date=args.target_date, snowoff_date=args.snow_off_date, out_dir=out_dir, cloud_cover=float(args.cloud_cover))
            ds = apply_model(out_dir=out_dir, out_name=out_name, crs=crs, write_tif=True, model_path=model_path, delete_inputs=True, out_crs='wgs84')
            break  # Exit the loop if successful
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # Wait before retrying
            else:
                raise  # Raise the last exception if max retries reached

if __name__ == "__main__":
   main()