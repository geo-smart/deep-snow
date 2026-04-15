import contextlib
import io
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError

from deep_snow.errors import EmptyAcquisitionError, TransientAcquisitionError
from deep_snow.inputs import build_output_name, generate_dates, most_recent_occurrence, parse_bounding_box
from deep_snow.resources import (
    get_default_hill_pptwt_path,
    get_default_hill_td_path,
    get_default_land_path,
    get_default_model_paths,
    get_default_single_model_path,
)

DEFAULT_MODEL_PATHS = get_default_model_paths()

DEFAULT_SINGLE_MODEL_PATH = get_default_single_model_path()
DEFAULT_LOCAL_TILE_SIZE_DEGREES = 1.5
DEFAULT_LOCAL_TILE_PADDING_DEGREES = 0.05
DEFAULT_TILE_LAND_PATH = get_default_land_path()
DEFAULT_HILL_PPTWT_PATH = get_default_hill_pptwt_path()
DEFAULT_HILL_TD_PATH = get_default_hill_td_path()
DEFAULT_MAX_BUFFER_EXPANSIONS = 3
DEFAULT_BUFFER_EXPANSION_STEP_DAYS = 2
SENTINEL2_START_DATE = "20150623"
SNOW_OFF_WARNING_MAX_OFFSET_DAYS = 548
CONUS_BOUNDS = {
    "minlon": -125.0,
    "minlat": 24.0,
    "maxlon": -66.0,
    "maxlat": 50.0,
}
WESTERN_US_MAX_LON = -100.0


def resolve_prediction_models(use_ensemble, model_path=None, model_paths_list=None):
    if use_ensemble:
        return None, model_paths_list or DEFAULT_MODEL_PATHS

    return model_path or DEFAULT_SINGLE_MODEL_PATH, None


def aoi_within_bounds(aoi, bounds):
    return (
        aoi["minlon"] >= bounds["minlon"]
        and aoi["minlat"] >= bounds["minlat"]
        and aoi["maxlon"] <= bounds["maxlon"]
        and aoi["maxlat"] <= bounds["maxlat"]
    )


def validate_prediction_aoi(aoi):
    if not aoi_within_bounds(aoi, CONUS_BOUNDS):
        raise ValueError(
            "AOI is outside CONUS. The current deep-snow model is not expected to provide good "
            "results outside CONUS."
        )

    if aoi["maxlon"] > WESTERN_US_MAX_LON:
        print(
            "WARNING: AOI is outside the Western U.S. training domain. "
            "The current deep-snow model was trained in the Western U.S. and has not been validated "
            "for the eastern U.S."
        )


def validate_prediction_dates(target_date, snow_off_date, *, current_date=None):
    target_dt = datetime.strptime(target_date, "%Y%m%d")
    snow_off_dt = datetime.strptime(snow_off_date, "%Y%m%d")
    today_dt = current_date or datetime.now()

    if target_dt.date() > today_dt.date():
        raise ValueError(
            f"target_date {target_date} is in the future relative to the current date "
            f"{today_dt.strftime('%Y%m%d')}."
        )

    if target_dt < datetime.strptime(SENTINEL2_START_DATE, "%Y%m%d"):
        raise ValueError(
            f"target_date {target_date} is before Sentinel-2 availability "
            f"({SENTINEL2_START_DATE})."
        )

    if snow_off_dt >= target_dt:
        raise ValueError(
            f"snow_off_date {snow_off_date} must be earlier than target_date {target_date}."
        )

    if (target_dt - snow_off_dt).days > SNOW_OFF_WARNING_MAX_OFFSET_DAYS:
        print(
            "WARNING: snow_off_date is far earlier than target_date. "
            "Please confirm the reference snow-off date is intentional."
        )


def format_default_model_guidance(use_ensemble):
    if use_ensemble:
        return (
            "Use the current ensemble defaults from deep_snow.workflows.DEFAULT_MODEL_PATHS "
            "or rerun with --use-ensemble."
        )

    return (
        "Use the current default single model "
        f"('{Path(DEFAULT_SINGLE_MODEL_PATH).name}') or rerun with --use-ensemble."
    )


def aoi_requires_local_tiling(aoi, tile_size_degrees=DEFAULT_LOCAL_TILE_SIZE_DEGREES):
    width = aoi["maxlon"] - aoi["minlon"]
    height = aoi["maxlat"] - aoi["minlat"]
    return width > tile_size_degrees or height > tile_size_degrees


def resolve_local_tile_size_degrees(tile_size_degrees):
    if tile_size_degrees is None:
        return DEFAULT_LOCAL_TILE_SIZE_DEGREES
    if tile_size_degrees <= 2 * DEFAULT_LOCAL_TILE_PADDING_DEGREES:
        raise ValueError(
            "tile_size_degrees must be larger than twice the fixed tile padding "
            f"({2 * DEFAULT_LOCAL_TILE_PADDING_DEGREES})."
        )
    return tile_size_degrees


def resolve_tiled_output_crs(out_crs):
    if str(out_crs).lower() == "wgs84":
        return "wgs84"

    print(
        "Large-AOI tiled local prediction requires a shared mosaic CRS. "
        f"Requested out_crs='{out_crs}', so the final mosaic will be written in 'wgs84'."
    )
    return "wgs84"


def _augment_input_provenance_with_retry_history(
    input_provenance,
    *,
    initial_buffer_period,
    attempted_buffer_periods,
    final_buffer_period,
):
    if input_provenance is None:
        return None

    input_provenance = dict(input_provenance)
    input_provenance["initial_buffer_period_days"] = initial_buffer_period
    input_provenance["attempted_buffer_period_days"] = list(attempted_buffer_periods)
    input_provenance["final_buffer_period_days"] = final_buffer_period
    input_provenance["buffer_expansion_count"] = max(0, len(attempted_buffer_periods) - 1)
    return input_provenance


def _extract_gap_summary_fields(ds):
    attrs = getattr(ds, "attrs", {})
    return {
        "input_gap_fraction": attrs.get("deep_snow_input_gap_fraction"),
        "input_gap_pixel_count": attrs.get("deep_snow_input_gap_pixel_count"),
        "input_gaps_tif_path": attrs.get("deep_snow_input_gaps_tif_path"),
        "input_gaps_netcdf_path": attrs.get("deep_snow_input_gaps_netcdf_path"),
        "gap_s1_snowon_fraction": attrs.get("deep_snow_gap_s1_snowon_fraction"),
        "gap_s1_snowoff_fraction": attrs.get("deep_snow_gap_s1_snowoff_fraction"),
        "gap_s2_fraction": attrs.get("deep_snow_gap_s2_fraction"),
    }


class _TeeStream(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def _prediction_log_capture(*, enabled, log_path):
    if not enabled:
        yield
        return

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = _TeeStream(log_file, sys.stdout)
        with contextlib.redirect_stdout(tee):
            yield


def _get_prediction_log_path(out_dir, out_name):
    return Path(out_dir) / f"{out_name}_log.txt"


def _mosaic_tiled_predictions(tile_datasets, *, out_dir, out_name, write_tif):
    from rioxarray.merge import merge_arrays

    pred_sd = merge_arrays([tile_ds.predicted_sd for tile_ds in tile_datasets])
    ds = pred_sd.to_dataset(name="predicted_sd").rio.write_crs("EPSG:4326")
    output_path = f"{out_dir}/{out_name}_sd.tif"

    if write_tif:
        print(f"[predict] writing tiled GeoTIFF mosaic to {output_path}")
        ds.predicted_sd.rio.to_raster(output_path, compress="lzw")
    else:
        print(
            "[predict] tiled mosaic assembled in memory; "
            f"GeoTIFF not written because write_tif=False. Output path would be {output_path}"
        )

    if all("predicted_swe" in tile_ds for tile_ds in tile_datasets):
        pred_swe = merge_arrays([tile_ds.predicted_swe for tile_ds in tile_datasets])
        pred_density = merge_arrays([tile_ds.predicted_density for tile_ds in tile_datasets])
        ds["predicted_swe"] = pred_swe
        ds["predicted_density"] = pred_density
        if write_tif:
            swe_output_path = f"{out_dir}/{out_name}_swe.tif"
            density_output_path = f"{out_dir}/{out_name}_density.tif"
            print(f"[predict] writing tiled SWE GeoTIFF mosaic to {swe_output_path}")
            ds.predicted_swe.rio.to_raster(swe_output_path, compress="lzw")
            print(f"[predict] writing tiled density GeoTIFF mosaic to {density_output_path}")
            ds.predicted_density.rio.to_raster(density_output_path, compress="lzw")
            ds.attrs["deep_snow_predicted_swe_tif_path"] = swe_output_path
            ds.attrs["deep_snow_predicted_density_tif_path"] = density_output_path

    return ds


def _predict_large_aoi_in_tiles(
    *,
    target_date,
    snow_off_date,
    aoi,
    cloud_cover,
    use_ensemble,
    crop_aoi,
    sentinel1_orbit_selection,
    selection_strategy,
    out_dir,
    requested_out_crs,
    buffer_period,
    delete_inputs,
    write_tif,
    gpu,
    out_name,
    model_path,
    model_paths_list,
    fcf_path,
    predict_swe,
    hill_pptwt_path,
    hill_td_path,
    max_retries,
    retry_delay,
    tile_size_degrees,
    tile_padding_degrees,
    land_path,
    max_buffer_expansions,
    buffer_expansion_step_days,
):
    from deep_snow.tiling import build_tile_jobs

    tile_jobs = build_tile_jobs(
        target_date=target_date,
        aoi=aoi,
        land_path=land_path,
        tile_size=tile_size_degrees,
        padding=tile_padding_degrees,
    )
    effective_out_crs = resolve_tiled_output_crs(requested_out_crs)
    tile_root = Path(out_dir) / "tiles"
    tile_datasets = []

    print(
        f"[predict] AOI exceeds {tile_size_degrees} degrees in at least one dimension; "
        f"running {len(tile_jobs)} local tile job(s) and mosaicing the cropped tile cores."
    )

    for index, job in enumerate(tile_jobs, start=1):
        tile_name = job["name"]
        print(f"[predict] tile {index}/{len(tile_jobs)}: {tile_name}")
        tile_ds = predict_tile(
            target_date=target_date,
            snow_off_date=snow_off_date,
            aoi=parse_bounding_box(job["aoi"]),
            cloud_cover=cloud_cover,
            use_ensemble=use_ensemble,
            crop_aoi=parse_bounding_box(job["clip_aoi"]),
            sentinel1_orbit_selection=sentinel1_orbit_selection,
            selection_strategy=selection_strategy,
            out_dir=str(tile_root / tile_name),
            out_crs=effective_out_crs,
            buffer_period=buffer_period,
            delete_inputs=delete_inputs,
            write_tif=False,
            gpu=gpu,
            out_name=tile_name,
            model_path=model_path,
            model_paths_list=model_paths_list,
            fcf_path=fcf_path,
            predict_swe=predict_swe,
            hill_pptwt_path=hill_pptwt_path,
            hill_td_path=hill_td_path,
            max_retries=max_retries,
            retry_delay=retry_delay,
            tile_large_aoi=False,
            emit_summary=False,
            max_buffer_expansions=max_buffer_expansions,
            buffer_expansion_step_days=buffer_expansion_step_days,
        )
        tile_datasets.append(tile_ds)

    ds = _mosaic_tiled_predictions(
        tile_datasets,
        out_dir=out_dir,
        out_name=out_name,
        write_tif=write_tif,
    )
    ds.attrs.update(
        {
            "deep_snow_tiled_aoi": True,
            "deep_snow_requested_out_crs": requested_out_crs,
            "deep_snow_effective_out_crs": effective_out_crs,
            "deep_snow_tile_count": len(tile_jobs),
            "deep_snow_tile_size_degrees": tile_size_degrees,
            "deep_snow_tile_padding_degrees": tile_padding_degrees,
            "deep_snow_tile_names": ",".join(job["name"] for job in tile_jobs),
        }
    )
    return ds, effective_out_crs


def _predict_single_tile(
    target_date,
    snow_off_date,
    aoi,
    cloud_cover,
    use_ensemble,
    crop_aoi=None,
    sentinel1_orbit_selection="descending",
    selection_strategy="composite",
    out_dir="data",
    out_crs="wgs84",
    buffer_period=6,
    delete_inputs=True,
    write_tif=True,
    gpu=False,
    out_name=None,
    model_path=None,
    model_paths_list=None,
    fcf_path=None,
    predict_swe=False,
    hill_pptwt_path=DEFAULT_HILL_PPTWT_PATH,
    hill_td_path=DEFAULT_HILL_TD_PATH,
    max_retries=100,
    retry_delay=5,
    emit_summary=True,
    validate_inputs=True,
    max_buffer_expansions=DEFAULT_MAX_BUFFER_EXPANSIONS,
    buffer_expansion_step_days=DEFAULT_BUFFER_EXPANSION_STEP_DAYS,
):
    from deep_snow.api import (
        apply_model,
        apply_model_ensemble,
        attach_prediction_metadata,
        build_prediction_summary,
        download_data,
        print_prediction_summary,
        read_prediction_input_provenance,
    )
    from deep_snow.acquisition import ensure_hill_inputs

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if validate_inputs:
        validate_prediction_dates(target_date, snow_off_date)
        validate_prediction_aoi(aoi)
    summary_aoi = crop_aoi or aoi
    out_name = out_name or build_output_name(target_date, summary_aoi)
    if predict_swe:
        hill_pptwt_path, hill_td_path = ensure_hill_inputs(hill_pptwt_path, hill_td_path)
    model_path, model_paths_list = resolve_prediction_models(
        use_ensemble=use_ensemble,
        model_path=model_path,
        model_paths_list=model_paths_list,
    )
    if use_ensemble:
        print(f"Using ensemble prediction with {len(model_paths_list)} models.")
    else:
        print(f"Using model: {Path(model_path).name}")

    initial_buffer_period = buffer_period
    current_buffer_period = buffer_period
    attempted_buffer_periods = [buffer_period]
    transient_attempt = 0
    buffer_expansions = 0

    while True:
        try:
            crs = download_data(
                aoi=aoi,
                target_date=target_date,
                buffer_period=current_buffer_period,
                snowoff_date=snow_off_date,
                out_dir=out_dir,
                cloud_cover=float(cloud_cover),
                fcf_path=fcf_path,
                sentinel1_orbit_selection=sentinel1_orbit_selection,
                selection_strategy=selection_strategy,
            )
            input_provenance = _augment_input_provenance_with_retry_history(
                read_prediction_input_provenance(out_dir),
                initial_buffer_period=initial_buffer_period,
                attempted_buffer_periods=attempted_buffer_periods,
                final_buffer_period=current_buffer_period,
            )
            if use_ensemble:
                ds = apply_model_ensemble(
                    out_dir=out_dir,
                    out_name=out_name,
                    crs=crs,
                    write_tif=write_tif,
                    model_paths_list=model_paths_list,
                    delete_inputs=delete_inputs,
                    out_crs=out_crs,
                    gpu=gpu,
                    predict_swe=predict_swe,
                    hill_pptwt_path=hill_pptwt_path,
                    hill_td_path=hill_td_path,
                    crop_bounds=(
                        crop_aoi["minlon"],
                        crop_aoi["minlat"],
                        crop_aoi["maxlon"],
                        crop_aoi["maxlat"],
                    ) if crop_aoi is not None else None,
                    crop_crs="EPSG:4326" if crop_aoi is not None else None,
                )
                summary = build_prediction_summary(
                    target_date=target_date,
                    snowoff_date=snow_off_date,
                    aoi=summary_aoi,
                    out_dir=out_dir,
                    out_name=out_name,
                    out_crs=out_crs,
                    cloud_cover=float(cloud_cover),
                    buffer_period=current_buffer_period,
                    gpu=gpu,
                    use_ensemble=True,
                    model_paths_list=model_paths_list,
                    fcf_path=fcf_path,
                    write_tif=write_tif,
                    delete_inputs=delete_inputs,
                    input_provenance=input_provenance,
                    sentinel1_orbit_selection=sentinel1_orbit_selection,
                    selection_strategy=selection_strategy,
                    initial_buffer_period=initial_buffer_period,
                    attempted_buffer_periods=attempted_buffer_periods,
                    predict_swe=predict_swe,
                    predicted_swe_tif_path=ds.attrs.get("deep_snow_predicted_swe_tif_path"),
                    predicted_density_tif_path=ds.attrs.get("deep_snow_predicted_density_tif_path"),
                    **_extract_gap_summary_fields(ds),
                )
                ds = attach_prediction_metadata(ds, summary)
                if emit_summary:
                    print_prediction_summary(summary)
                return ds

            ds = apply_model(
                out_dir=out_dir,
                out_name=out_name,
                crs=crs,
                write_tif=write_tif,
                model_path=model_path,
                delete_inputs=delete_inputs,
                out_crs=out_crs,
                gpu=gpu,
                predict_swe=predict_swe,
                hill_pptwt_path=hill_pptwt_path,
                hill_td_path=hill_td_path,
                crop_bounds=(
                    crop_aoi["minlon"],
                    crop_aoi["minlat"],
                    crop_aoi["maxlon"],
                    crop_aoi["maxlat"],
                ) if crop_aoi is not None else None,
                crop_crs="EPSG:4326" if crop_aoi is not None else None,
            )
            summary = build_prediction_summary(
                target_date=target_date,
                snowoff_date=snow_off_date,
                aoi=summary_aoi,
                out_dir=out_dir,
                out_name=out_name,
                out_crs=out_crs,
                cloud_cover=float(cloud_cover),
                buffer_period=current_buffer_period,
                gpu=gpu,
                use_ensemble=False,
                model_path=model_path,
                fcf_path=fcf_path,
                write_tif=write_tif,
                delete_inputs=delete_inputs,
                input_provenance=input_provenance,
                sentinel1_orbit_selection=sentinel1_orbit_selection,
                selection_strategy=selection_strategy,
                initial_buffer_period=initial_buffer_period,
                attempted_buffer_periods=attempted_buffer_periods,
                predict_swe=predict_swe,
                predicted_swe_tif_path=ds.attrs.get("deep_snow_predicted_swe_tif_path"),
                predicted_density_tif_path=ds.attrs.get("deep_snow_predicted_density_tif_path"),
                **_extract_gap_summary_fields(ds),
            )
            ds = attach_prediction_metadata(ds, summary)
            if emit_summary:
                print_prediction_summary(summary)
            return ds
        except EmptyAcquisitionError as exc:
            if buffer_expansions < max_buffer_expansions:
                next_buffer_period = current_buffer_period + buffer_expansion_step_days
                print(
                    "WARNING: "
                    f"{type(exc).__name__} encountered with buffer_period={current_buffer_period}: {exc}. "
                    f"Expanding buffer_period to {next_buffer_period} days "
                    f"({buffer_expansions + 1}/{max_buffer_expansions}) and retrying..."
                )
                current_buffer_period = next_buffer_period
                attempted_buffer_periods.append(current_buffer_period)
                buffer_expansions += 1
                transient_attempt = 0
                continue
            raise EmptyAcquisitionError(
                f"{exc} Failed after attempting buffer periods {attempted_buffer_periods}."
            ) from exc
        except (TransientAcquisitionError, HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            transient_attempt += 1
            print(f"Transient failure on attempt {transient_attempt}/{max_retries}: {exc}")
            if transient_attempt < max_retries:
                time.sleep(retry_delay)
                continue
            raise
        except Exception:
            raise


def predict_tile(
    target_date,
    snow_off_date,
    aoi,
    cloud_cover,
    use_ensemble,
    crop_aoi=None,
    sentinel1_orbit_selection="descending",
    selection_strategy="composite",
    out_dir="data",
    out_crs="wgs84",
    buffer_period=6,
    delete_inputs=True,
    write_tif=True,
    gpu=False,
    out_name=None,
    model_path=None,
    model_paths_list=None,
    fcf_path=None,
    predict_swe=False,
    hill_pptwt_path=DEFAULT_HILL_PPTWT_PATH,
    hill_td_path=DEFAULT_HILL_TD_PATH,
    max_retries=100,
    retry_delay=5,
    emit_summary=True,
    max_buffer_expansions=DEFAULT_MAX_BUFFER_EXPANSIONS,
    buffer_expansion_step_days=DEFAULT_BUFFER_EXPANSION_STEP_DAYS,
):
    summary_aoi = crop_aoi or aoi
    resolved_out_name = out_name or build_output_name(target_date, summary_aoi)
    with _prediction_log_capture(
        enabled=write_tif,
        log_path=_get_prediction_log_path(out_dir, resolved_out_name),
    ):
        return _predict_single_tile(
            target_date=target_date,
            snow_off_date=snow_off_date,
            aoi=aoi,
            cloud_cover=cloud_cover,
            use_ensemble=use_ensemble,
            crop_aoi=crop_aoi,
            sentinel1_orbit_selection=sentinel1_orbit_selection,
            selection_strategy=selection_strategy,
            out_dir=out_dir,
            out_crs=out_crs,
            buffer_period=buffer_period,
            delete_inputs=delete_inputs,
            write_tif=write_tif,
            gpu=gpu,
            out_name=resolved_out_name,
            model_path=model_path,
            model_paths_list=model_paths_list,
            fcf_path=fcf_path,
            predict_swe=predict_swe,
            hill_pptwt_path=hill_pptwt_path,
            hill_td_path=hill_td_path,
            max_retries=max_retries,
            retry_delay=retry_delay,
            emit_summary=emit_summary,
            validate_inputs=True,
            max_buffer_expansions=max_buffer_expansions,
            buffer_expansion_step_days=buffer_expansion_step_days,
        )


def predict_batch(
    target_date,
    snow_off_date,
    aoi,
    cloud_cover,
    use_ensemble,
    crop_aoi=None,
    sentinel1_orbit_selection="descending",
    selection_strategy="composite",
    out_dir="data",
    out_crs="wgs84",
    buffer_period=6,
    delete_inputs=True,
    write_tif=True,
    gpu=False,
    out_name=None,
    model_path=None,
    model_paths_list=None,
    fcf_path=None,
    predict_swe=False,
    hill_pptwt_path=DEFAULT_HILL_PPTWT_PATH,
    hill_td_path=DEFAULT_HILL_TD_PATH,
    max_retries=100,
    retry_delay=5,
    emit_summary=True,
    tile_large_aoi=True,
    tile_size_degrees=None,
    tile_padding_degrees=DEFAULT_LOCAL_TILE_PADDING_DEGREES,
    land_path=DEFAULT_TILE_LAND_PATH,
    max_buffer_expansions=DEFAULT_MAX_BUFFER_EXPANSIONS,
    buffer_expansion_step_days=DEFAULT_BUFFER_EXPANSION_STEP_DAYS,
):
    from deep_snow.api import (
        attach_prediction_metadata,
        build_prediction_summary,
        print_prediction_summary,
    )

    validate_prediction_dates(target_date, snow_off_date)
    validate_prediction_aoi(aoi)
    tile_size_degrees = resolve_local_tile_size_degrees(tile_size_degrees)
    summary_aoi = crop_aoi or aoi
    out_name = out_name or build_output_name(target_date, summary_aoi)
    model_path, model_paths_list = resolve_prediction_models(
        use_ensemble=use_ensemble,
        model_path=model_path,
        model_paths_list=model_paths_list,
    )

    with _prediction_log_capture(
        enabled=write_tif,
        log_path=_get_prediction_log_path(out_dir, out_name),
    ):
        if (
            tile_large_aoi
            and crop_aoi is None
            and aoi_requires_local_tiling(aoi, tile_size_degrees=tile_size_degrees)
        ):
            ds, effective_out_crs = _predict_large_aoi_in_tiles(
                target_date=target_date,
                snow_off_date=snow_off_date,
                aoi=aoi,
                cloud_cover=cloud_cover,
                use_ensemble=use_ensemble,
                crop_aoi=crop_aoi,
                sentinel1_orbit_selection=sentinel1_orbit_selection,
                selection_strategy=selection_strategy,
                out_dir=out_dir,
                requested_out_crs=out_crs,
                buffer_period=buffer_period,
                delete_inputs=delete_inputs,
                write_tif=write_tif,
                gpu=gpu,
                out_name=out_name,
                model_path=model_path,
                model_paths_list=model_paths_list,
                fcf_path=fcf_path,
                predict_swe=predict_swe,
                hill_pptwt_path=hill_pptwt_path,
                hill_td_path=hill_td_path,
                max_retries=max_retries,
                retry_delay=retry_delay,
                tile_size_degrees=tile_size_degrees,
                tile_padding_degrees=tile_padding_degrees,
                land_path=land_path,
                max_buffer_expansions=max_buffer_expansions,
                buffer_expansion_step_days=buffer_expansion_step_days,
            )
            summary = build_prediction_summary(
                target_date=target_date,
                snowoff_date=snow_off_date,
                aoi=summary_aoi,
                out_dir=out_dir,
                out_name=out_name,
                out_crs=effective_out_crs,
                cloud_cover=float(cloud_cover),
                buffer_period=buffer_period,
                gpu=gpu,
                use_ensemble=use_ensemble,
                model_path=model_path,
                model_paths_list=model_paths_list,
                fcf_path=fcf_path,
                write_tif=write_tif,
                delete_inputs=delete_inputs,
                input_provenance=None,
                sentinel1_orbit_selection=sentinel1_orbit_selection,
                selection_strategy=selection_strategy,
                initial_buffer_period=buffer_period,
                attempted_buffer_periods=[buffer_period],
                predict_swe=predict_swe,
                predicted_swe_tif_path=ds.attrs.get("deep_snow_predicted_swe_tif_path"),
                predicted_density_tif_path=ds.attrs.get("deep_snow_predicted_density_tif_path"),
                **_extract_gap_summary_fields(ds),
            )
            summary["tiled_aoi"] = True
            summary["requested_out_crs"] = out_crs
            summary["tile_count"] = int(ds.attrs.get("deep_snow_tile_count", 0))
            ds = attach_prediction_metadata(ds, summary)
            if emit_summary:
                print_prediction_summary(summary)
            return ds

        return _predict_single_tile(
            target_date=target_date,
            snow_off_date=snow_off_date,
            aoi=aoi,
            cloud_cover=cloud_cover,
            use_ensemble=use_ensemble,
            crop_aoi=crop_aoi,
            sentinel1_orbit_selection=sentinel1_orbit_selection,
            selection_strategy=selection_strategy,
            out_dir=out_dir,
            out_crs=out_crs,
            buffer_period=buffer_period,
            delete_inputs=delete_inputs,
            write_tif=write_tif,
            gpu=gpu,
            out_name=out_name,
            model_path=model_path,
            model_paths_list=model_paths_list,
            fcf_path=fcf_path,
            predict_swe=predict_swe,
            hill_pptwt_path=hill_pptwt_path,
            hill_td_path=hill_td_path,
            max_retries=max_retries,
            retry_delay=retry_delay,
            emit_summary=emit_summary,
            validate_inputs=False,
            max_buffer_expansions=max_buffer_expansions,
            buffer_expansion_step_days=buffer_expansion_step_days,
        )


def predict_time_series(
    begin_date,
    end_date,
    snow_off_day,
    aoi,
    cloud_cover,
    use_ensemble=False,
    out_dir="data",
    out_crs="wgs84",
    out_name=None,
    delete_inputs=False,
    write_tif=True,
    buffer_period=6,
    fcf_path=None,
    gpu=False,
    model_path=None,
    model_paths_list=None,
    sentinel1_orbit_selection="descending",
    selection_strategy="composite",
    predict_swe=False,
    hill_pptwt_path=DEFAULT_HILL_PPTWT_PATH,
    hill_td_path=DEFAULT_HILL_TD_PATH,
    tile_large_aoi=True,
    tile_size_degrees=None,
    tile_padding_degrees=DEFAULT_LOCAL_TILE_PADDING_DEGREES,
    land_path=DEFAULT_TILE_LAND_PATH,
    max_retries=100,
    retry_delay=5,
    max_buffer_expansions=DEFAULT_MAX_BUFFER_EXPANSIONS,
    buffer_expansion_step_days=DEFAULT_BUFFER_EXPANSION_STEP_DAYS,
):
    import pandas as pd
    import xarray as xr

    jobs = build_time_series_jobs(begin_date, end_date, snow_off_day)
    ds_list = []

    for index, job in enumerate(jobs, start=1):
        print("--------------------------------------")
        print(f"working on {job['target_date']}, {index}/{len(jobs)}")
        current_out_name = out_name or build_output_name(job["target_date"], aoi)
        ds = predict_batch(
            target_date=job["target_date"],
            snow_off_date=job["snow_off_date"],
            aoi=aoi,
            cloud_cover=cloud_cover,
            use_ensemble=use_ensemble,
            out_dir=out_dir,
            out_crs=out_crs,
            out_name=current_out_name,
            delete_inputs=delete_inputs,
            write_tif=write_tif,
            buffer_period=buffer_period,
            fcf_path=fcf_path,
            gpu=gpu,
            model_path=model_path,
            model_paths_list=model_paths_list,
            sentinel1_orbit_selection=sentinel1_orbit_selection,
            selection_strategy=selection_strategy,
            predict_swe=predict_swe,
            hill_pptwt_path=hill_pptwt_path,
            hill_td_path=hill_td_path,
            tile_large_aoi=tile_large_aoi,
            tile_size_degrees=tile_size_degrees,
            tile_padding_degrees=tile_padding_degrees,
            land_path=land_path,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_buffer_expansions=max_buffer_expansions,
            buffer_expansion_step_days=buffer_expansion_step_days,
        )
        ds = ds.expand_dims(time=[pd.to_datetime(job["target_date"], format="%Y%m%d")])
        if index > 1:
            ds = ds.rio.reproject_match(ds_list[0])
        ds_list.append(ds)

    return xr.concat(ds_list, dim="time")


def build_time_series_jobs(begin_date, end_date, snow_off_day):
    jobs = []
    for target_date in generate_dates(end_date, begin_date):
        jobs.append(
            {
                "target_date": target_date,
                "snow_off_date": most_recent_occurrence(target_date, snow_off_day),
            }
        )
    return jobs
