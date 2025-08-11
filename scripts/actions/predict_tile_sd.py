from deep_snow.application import download_data, apply_model, apply_model_ensemble
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
    parser.add_argument("use_ensemble", type=str, help="whether to use model ensemble, True or False")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    out_dir = 'data'
    out_name = f'{args.target_date}_deep-snow_{args.aoi["minlon"]:.{2}f}_{args.aoi["minlat"]:.{2}f}_{args.aoi["maxlon"]:.{2}f}_{args.aoi["maxlat"]:.{2}f}'
    model_1_path = 'weights/ResDepth_lr0.000457131171011064_weightdecay0.00010523970398286011_epochs62_mintestloss0.00091'
    model_2_path = 'weights/ResDepth_lr0.00025036613931876504_weightdecay0.00020109428801183744_epochs63_mintestloss0.00091'
    model_3_path = 'weights/ResDepth_lr0.0001572907262097884_weightdecay0.00013101368652881237_epochs98_mintestloss0.00090'
    model_4_path = 'weights/ResDepth_lr0.00011563677025564128_weightdecay0.0003567649258551211_epochs63_mintestloss0.00092'
    model_5_path = 'weights/ResDepth_lr0.00026633575524604445_weightdecay8.770493085204089e-05_epochs85_mintestloss0.00090'
    model_paths_list = [model_1_path, model_2_path, model_3_path, model_4_path, model_5_path]
    buffer_period = 6

    max_retries = 100
    retry_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            crs = download_data(aoi=args.aoi, target_date=args.target_date, buffer_period=buffer_period, snowoff_date=args.snow_off_date, out_dir=out_dir, cloud_cover=float(args.cloud_cover))
            if args.use_ensemble == "True":
                ds = apply_model(out_dir=out_dir, out_name=out_name, crs=crs, write_tif=True, model_path_list=model_path_list, delete_inputs=True, out_crs='wgs84', gpu=False)
            else:
                ds = apply_model(out_dir=out_dir, out_name=out_name, crs=crs, write_tif=True, model_path=model_3_path, delete_inputs=True, out_crs='wgs84', gpu=False)
            break  # Exit the loop if successful
        except ValueError as e:
            if str(e) == "Can't load empty sequence":
                buffer_period += 2
                print(f"ValueError encountered: {e}. Increasing buffer_period to {buffer_period} and retrying...")
            elif str(e) == "TypeError: 'list' object cannot be interpreted as an integer":
                buffer_period += 2
                print(f"TypeError encountered: {e}. Increasing buffer_period to {buffer_period} and retrying...")
            else:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

if __name__ == "__main__":
   main()
