import pandas as pd
import xarray as xr
from pathlib import Path
import json

from deep_snow.inputs import build_output_name, generate_dates

# Public API plus compatibility aliases that remain documented for existing users.
__all__ = [
    "apply_model",
    "apply_model_ensemble",
    "attach_prediction_metadata",
    "build_prediction_summary",
    "download_data",
    "predict_sd",
    "predict_sd_timeseries",
    "predict_sd_ts",
    "predict_snow_depth",
    "predict_snow_depth_time_series",
    "print_prediction_summary",
    "read_prediction_input_provenance",
]


def build_prediction_summary(
    *,
    target_date,
    snowoff_date,
    aoi,
    out_dir,
    out_name,
    out_crs,
    cloud_cover,
    buffer_period,
    gpu,
    use_ensemble,
    model_path=None,
    model_paths_list=None,
    fcf_path=None,
    write_tif=True,
    delete_inputs=False,
    input_provenance=None,
    sentinel1_orbit_selection="descending",
    selection_strategy="composite",
    initial_buffer_period=None,
    attempted_buffer_periods=None,
    input_gap_fraction=None,
    input_gap_pixel_count=None,
    input_gaps_tif_path=None,
    input_gaps_netcdf_path=None,
    gap_s1_snowon_fraction=None,
    gap_s1_snowoff_fraction=None,
    gap_s2_fraction=None,
    predict_swe=False,
    predicted_swe_tif_path=None,
    predicted_density_tif_path=None,
):
    return {
        "target_date": target_date,
        "snowoff_date": snowoff_date,
        "aoi": aoi,
        "out_dir": out_dir,
        "out_name": out_name,
        "out_crs": out_crs,
        "cloud_cover": cloud_cover,
        "buffer_period": buffer_period,
        "gpu": gpu,
        "use_ensemble": use_ensemble,
        "fcf_path": fcf_path,
        "write_tif": write_tif,
        "delete_inputs": delete_inputs,
        "sentinel1_pass_selection": sentinel1_orbit_selection,
        "selection_strategy": selection_strategy,
        "model_path": model_path,
        "model_paths_list": model_paths_list,
        "model_count": len(model_paths_list) if model_paths_list is not None else 1,
        "model_inputs_path": str((Path(out_dir) / "model_inputs.nc").as_posix()),
        "predicted_tif_path": str((Path(out_dir) / f"{out_name}_sd.tif").as_posix()) if write_tif else None,
        "input_provenance": input_provenance,
        "initial_buffer_period": buffer_period if initial_buffer_period is None else initial_buffer_period,
        "attempted_buffer_periods": attempted_buffer_periods,
        "input_gap_fraction": input_gap_fraction,
        "input_gap_pixel_count": input_gap_pixel_count,
        "input_gaps_tif_path": input_gaps_tif_path,
        "input_gaps_netcdf_path": input_gaps_netcdf_path,
        "gap_s1_snowon_fraction": gap_s1_snowon_fraction,
        "gap_s1_snowoff_fraction": gap_s1_snowoff_fraction,
        "gap_s2_fraction": gap_s2_fraction,
        "predict_swe": predict_swe,
        "predicted_swe_tif_path": predicted_swe_tif_path,
        "predicted_density_tif_path": predicted_density_tif_path,
    }


def attach_prediction_metadata(ds, summary):
    ds.attrs.update(
        {
            "deep_snow_target_date": summary["target_date"],
            "deep_snow_snowoff_date": summary["snowoff_date"],
            "deep_snow_out_dir": summary["out_dir"],
            "deep_snow_out_name": summary["out_name"],
            "deep_snow_out_crs": summary["out_crs"],
            "deep_snow_cloud_cover": summary["cloud_cover"],
            "deep_snow_buffer_period": summary["buffer_period"],
            "deep_snow_gpu": summary["gpu"],
            "deep_snow_use_ensemble": summary["use_ensemble"],
            "deep_snow_model_count": summary["model_count"],
            "deep_snow_model_inputs_path": summary["model_inputs_path"],
            "deep_snow_sentinel1_pass_selection": summary["sentinel1_pass_selection"],
            "deep_snow_selection_strategy": summary["selection_strategy"],
            "deep_snow_predict_swe": summary["predict_swe"],
        }
    )
    if summary["model_path"] is not None:
        ds.attrs["deep_snow_model_path"] = summary["model_path"]
    if summary["predicted_tif_path"] is not None:
        ds.attrs["deep_snow_predicted_tif_path"] = summary["predicted_tif_path"]
    if summary["predicted_swe_tif_path"] is not None:
        ds.attrs["deep_snow_predicted_swe_tif_path"] = summary["predicted_swe_tif_path"]
    if summary["predicted_density_tif_path"] is not None:
        ds.attrs["deep_snow_predicted_density_tif_path"] = summary["predicted_density_tif_path"]
    if summary["input_gap_fraction"] is not None:
        ds.attrs["deep_snow_input_gap_fraction"] = summary["input_gap_fraction"]
    if summary["input_gap_pixel_count"] is not None:
        ds.attrs["deep_snow_input_gap_pixel_count"] = summary["input_gap_pixel_count"]
    if summary["input_gaps_tif_path"] is not None:
        ds.attrs["deep_snow_input_gaps_tif_path"] = summary["input_gaps_tif_path"]
    if summary["input_gaps_netcdf_path"] is not None:
        ds.attrs["deep_snow_input_gaps_netcdf_path"] = summary["input_gaps_netcdf_path"]
    if summary["gap_s1_snowon_fraction"] is not None:
        ds.attrs["deep_snow_gap_s1_snowon_fraction"] = summary["gap_s1_snowon_fraction"]
    if summary["gap_s1_snowoff_fraction"] is not None:
        ds.attrs["deep_snow_gap_s1_snowoff_fraction"] = summary["gap_s1_snowoff_fraction"]
    if summary["gap_s2_fraction"] is not None:
        ds.attrs["deep_snow_gap_s2_fraction"] = summary["gap_s2_fraction"]
    if summary["input_provenance"] is not None:
        ds.attrs["deep_snow_input_provenance_json"] = json.dumps(summary["input_provenance"], sort_keys=True)
    return ds


def read_prediction_input_provenance(out_dir):
    model_inputs_path = Path(out_dir) / "model_inputs.nc"
    if not model_inputs_path.exists():
        return None

    with xr.open_dataset(model_inputs_path, decode_times=False) as ds:
        provenance_json = ds.attrs.get("deep_snow_input_provenance_json")

    if provenance_json is None:
        return None

    return json.loads(provenance_json)


def print_prediction_summary(summary):
    print("\nprediction summary")
    print(
        "  request: "
        f"target={summary['target_date']} | snow_off={summary['snowoff_date']} | "
        f"aoi={summary['aoi']['minlon']} {summary['aoi']['minlat']} "
        f"{summary['aoi']['maxlon']} {summary['aoi']['maxlat']}"
    )
    print()
    if summary["model_path"] is not None:
        print(f"  model: {Path(summary['model_path']).name}")
    else:
        print(f"  model: ensemble ({summary['model_count']} models)")
    print()
    print(
        "  output: "
        f"crs={summary['out_crs']} | "
        f"selection={summary['selection_strategy']} | s1_orbit={summary['sentinel1_pass_selection']}"
    )
    if summary["predicted_tif_path"] is not None:
        print(f"  files: predicted={summary['predicted_tif_path']}")
    else:
        print("  files: predicted=not written")
    if summary.get("predicted_swe_tif_path") is not None:
        print(f"         swe={summary['predicted_swe_tif_path']}")
    if summary.get("predicted_density_tif_path") is not None:
        print(f"         density={summary['predicted_density_tif_path']}")
    print(f"         inputs={summary['model_inputs_path']}")
    print()
    attempted_buffer_periods = summary.get("attempted_buffer_periods") or []
    if attempted_buffer_periods:
        if len(attempted_buffer_periods) == 1:
            print(f"  acquisition window: {attempted_buffer_periods[0]} days")
        else:
            print(
                "  acquisition window: "
                f"expanded from {attempted_buffer_periods[0]} to {attempted_buffer_periods[-1]} days "
                f"after trying {attempted_buffer_periods}"
            )
        print()
    if summary.get("input_gap_pixel_count") is not None:
        print(
            "  gaps: "
            f"{summary['input_gap_pixel_count']} pixel(s) "
            f"({summary.get('input_gap_fraction', 0):.2%} of grid)"
        )
        if summary.get("input_gaps_tif_path") is not None:
            print(f"        combined={summary['input_gaps_tif_path']}")
        if summary.get("input_gaps_netcdf_path") is not None:
            print(f"        detailed={summary['input_gaps_netcdf_path']}")
        detailed_gap_parts = []
        if summary.get("gap_s1_snowon_fraction") is not None:
            detailed_gap_parts.append(f"S1 snow-on {summary['gap_s1_snowon_fraction']:.2%}")
        if summary.get("gap_s1_snowoff_fraction") is not None:
            detailed_gap_parts.append(f"S1 snow-off {summary['gap_s1_snowoff_fraction']:.2%}")
        if summary.get("gap_s2_fraction") is not None:
            detailed_gap_parts.append(f"S2 {summary['gap_s2_fraction']:.2%}")
        if detailed_gap_parts:
            print(f"        breakdown={', '.join(detailed_gap_parts)}")
        print()
    if summary.get("tiled_aoi"):
        print(
            "  tiling: "
            f"local mosaic from {summary.get('tile_count', 'unknown')} tile(s)"
        )
        requested_out_crs = summary.get("requested_out_crs")
        if requested_out_crs is not None and requested_out_crs != summary["out_crs"]:
            print(
                "          "
                f"requested crs overridden {requested_out_crs} -> {summary['out_crs']}"
            )
        print()
    if summary["input_provenance"] is not None:
        snowon = summary["input_provenance"].get("sentinel1_snowon", {})
        snowoff = summary["input_provenance"].get("sentinel1_snowoff", {})
        s2 = summary["input_provenance"].get("sentinel2_snowon", {})
        if snowon:
            print(
                "  acquisitions: "
                f"S1 snow-on={snowon.get('selected_acquisition_count')} "
                f"({', '.join(snowon.get('selected_acquisition_times', [])[:3])}"
                f"{'...' if snowon.get('selected_acquisition_count', 0) > 3 else ''})"
            )
        if snowoff:
            print(
                "               "
                f"S1 snow-off={snowoff.get('selected_acquisition_count')} "
                f"({', '.join(snowoff.get('selected_acquisition_times', [])[:3])}"
                f"{'...' if snowoff.get('selected_acquisition_count', 0) > 3 else ''})"
            )
        if s2:
            print(
                "               "
                f"S2={s2.get('selected_acquisition_count')} "
                f"({', '.join(s2.get('selected_acquisition_times', [])[:3])}"
                f"{'...' if s2.get('selected_acquisition_count', 0) > 3 else ''})"
            )
def download_data(
    aoi,
    target_date,
    snowoff_date,
    buffer_period,
    out_dir,
    cloud_cover,
    fcf_path=None,
    predict_swe=False,
    hill_pptwt_path=None,
    hill_td_path=None,
    sentinel1_orbit_selection="descending",
    selection_strategy="composite",
):
    from deep_snow.acquisition import acquire_prediction_inputs
    from deep_snow.preprocessing import build_prediction_dataset, write_model_inputs

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    raw_inputs, crs, input_provenance = acquire_prediction_inputs(
        aoi=aoi,
        target_date=target_date,
        snowoff_date=snowoff_date,
        buffer_period=buffer_period,
        cloud_cover=cloud_cover,
        fcf_path=fcf_path,
        predict_swe=predict_swe,
        hill_pptwt_path=hill_pptwt_path,
        hill_td_path=hill_td_path,
        sentinel1_orbit_selection=sentinel1_orbit_selection,
        selection_strategy=selection_strategy,
    )
    ds = build_prediction_dataset(
        raw_inputs,
        target_date=target_date,
        crs=crs,
        input_provenance=input_provenance,
    )
    write_model_inputs(ds, out_dir)
    return crs


def apply_model(
    crs,
    model_path,
    out_dir,
    out_name,
    write_tif,
    delete_inputs,
    out_crs,
    gpu=True,
    predict_swe=False,
    hill_pptwt_path=None,
    hill_td_path=None,
    crop_bounds=None,
    crop_crs=None,
):
    from deep_snow.model_loading import load_resdepth_checkpoint
    from deep_snow.prediction import apply_models

    return apply_models(
        crs=crs,
        model_paths=[model_path],
        out_dir=out_dir,
        out_name=out_name,
        write_tif=write_tif,
        delete_inputs=delete_inputs,
        out_crs=out_crs,
        checkpoint_loader=load_resdepth_checkpoint,
        gpu=gpu,
        predict_swe=predict_swe,
        hill_pptwt_path=hill_pptwt_path,
        hill_td_path=hill_td_path,
        crop_bounds=crop_bounds,
        crop_crs=crop_crs,
    )


def apply_model_ensemble(
    crs,
    model_paths_list,
    out_dir,
    out_name,
    write_tif,
    delete_inputs,
    out_crs,
    gpu=True,
    predict_swe=False,
    hill_pptwt_path=None,
    hill_td_path=None,
    crop_bounds=None,
    crop_crs=None,
):
    from deep_snow.model_loading import load_resdepth_checkpoint
    from deep_snow.prediction import apply_models

    return apply_models(
        crs=crs,
        model_paths=model_paths_list,
        out_dir=out_dir,
        out_name=out_name,
        write_tif=write_tif,
        delete_inputs=delete_inputs,
        out_crs=out_crs,
        checkpoint_loader=load_resdepth_checkpoint,
        gpu=gpu,
        predict_swe=predict_swe,
        hill_pptwt_path=hill_pptwt_path,
        hill_td_path=hill_td_path,
        crop_bounds=crop_bounds,
        crop_crs=crop_crs,
    )


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
    predict_swe=False,
):
    """Predict snow depth for one AOI/date.

    This is the main local Python entry point. Large AOIs are tiled and
    mosaiced automatically through the underlying workflow when needed.
    """
    return predict_batch(
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
        predict_swe=predict_swe,
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
    predict_swe=False,
):
    """Legacy time-series alias retained for compatibility.

    This older helper steps backward from ``target_date`` toward
    ``snowoff_date`` using the package's fixed acquisition cadence.
    New code should prefer ``predict_sd_timeseries``.
    """
    jobs = [{"target_date": date, "snow_off_date": snowoff_date} for date in generate_dates(target_date, snowoff_date)]
    return _predict_time_series_jobs(
        jobs=jobs,
        aoi=aoi,
        model_path=model_path,
        out_dir=out_dir,
        out_crs=out_crs,
        out_name=out_name,
        write_tif=False,
        delete_inputs=delete_inputs,
        cloud_cover=cloud_cover,
        buffer_period=buffer_period,
        fcf_path=fcf_path,
        gpu=gpu,
        use_ensemble=use_ensemble,
        model_paths_list=model_paths_list,
        sentinel1_orbit_selection=sentinel1_orbit_selection,
        selection_strategy=selection_strategy,
        tile_large_aoi=True,
        predict_swe=predict_swe,
    )


def predict_sd_timeseries(
    *,
    aoi,
    begin_date,
    end_date,
    snow_off_day,
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
    tile_large_aoi=True,
    tile_size_degrees=None,
    predict_swe=False,
):
    """Predict a snow-depth time series for one AOI across an explicit date range."""
    return predict_time_series(
        aoi=aoi,
        begin_date=begin_date,
        end_date=end_date,
        snow_off_day=snow_off_day,
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
        tile_large_aoi=tile_large_aoi,
        tile_size_degrees=tile_size_degrees,
        predict_swe=predict_swe,
    )


def _predict_time_series_jobs(
    *,
    jobs,
    aoi,
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
    tile_large_aoi=True,
    tile_size_degrees=None,
    predict_swe=False,
):
    """Internal helper that runs a prepared list of time-series jobs."""
    from deep_snow.workflows import predict_batch as workflow_predict_batch

    ds_list = []
    for index, job in enumerate(jobs, start=1):
        print("--------------------------------------")
        print(f"working on {job['target_date']}, {index}/{len(jobs)}")
        current_out_name = out_name or build_output_name(job["target_date"], aoi)
        ds = workflow_predict_batch(
            target_date=job["target_date"],
            snow_off_date=job["snow_off_date"],
            aoi=aoi,
            cloud_cover=cloud_cover,
            use_ensemble=use_ensemble,
            out_dir=out_dir,
            out_crs=out_crs,
            out_name=current_out_name,
            write_tif=write_tif,
            delete_inputs=delete_inputs,
            buffer_period=buffer_period,
            fcf_path=fcf_path,
            gpu=gpu,
            model_path=model_path,
            model_paths_list=model_paths_list,
            sentinel1_orbit_selection=sentinel1_orbit_selection,
            selection_strategy=selection_strategy,
            tile_large_aoi=tile_large_aoi,
            tile_size_degrees=tile_size_degrees,
            predict_swe=predict_swe,
        )
        ds = ds.expand_dims(time=[pd.to_datetime(job["target_date"], format="%Y%m%d")])
        if index > 1:
            ds = ds.rio.reproject_match(ds_list[0])
        ds_list.append(ds)
    return xr.concat(ds_list, dim="time")


def predict_tile(
    *,
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
    predict_swe=False,
):
    """Advanced single-tile helper.

    This bypasses automatic large-AOI tiling and is mainly useful for
    debugging or lower-level workflow composition.
    """
    from deep_snow.workflows import predict_tile as workflow_predict_tile

    return workflow_predict_tile(
        target_date=target_date,
        snow_off_date=snowoff_date,
        aoi=aoi,
        cloud_cover=cloud_cover,
        use_ensemble=use_ensemble,
        out_dir=out_dir,
        out_crs=out_crs,
        out_name=out_name,
        write_tif=write_tif,
        delete_inputs=delete_inputs,
        buffer_period=buffer_period,
        fcf_path=fcf_path,
        gpu=gpu,
        model_path=model_path,
        model_paths_list=model_paths_list,
        sentinel1_orbit_selection=sentinel1_orbit_selection,
        selection_strategy=selection_strategy,
        predict_swe=predict_swe,
    )


def predict_batch(
    *,
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
    tile_large_aoi=True,
    tile_size_degrees=None,
    predict_swe=False,
):
    """Lower-level batch helper used by ``predict_sd``.

    It exposes automatic AOI tiling explicitly, which is useful internally
    and for advanced users, but most local callers should prefer ``predict_sd``.
    """
    from deep_snow.workflows import predict_batch as workflow_predict_batch

    return workflow_predict_batch(
        target_date=target_date,
        snow_off_date=snowoff_date,
        aoi=aoi,
        cloud_cover=cloud_cover,
        use_ensemble=use_ensemble,
        out_dir=out_dir,
        out_crs=out_crs,
        out_name=out_name,
        write_tif=write_tif,
        delete_inputs=delete_inputs,
        buffer_period=buffer_period,
        fcf_path=fcf_path,
        gpu=gpu,
        model_path=model_path,
        model_paths_list=model_paths_list,
        sentinel1_orbit_selection=sentinel1_orbit_selection,
        selection_strategy=selection_strategy,
        tile_large_aoi=tile_large_aoi,
        tile_size_degrees=tile_size_degrees,
        predict_swe=predict_swe,
    )


def predict_time_series(
    *,
    aoi,
    begin_date,
    end_date,
    snow_off_day,
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
    tile_large_aoi=True,
    tile_size_degrees=None,
    predict_swe=False,
):
    """Lower-level time-series helper used by ``predict_sd_timeseries``."""
    from deep_snow.workflows import build_time_series_jobs

    return _predict_time_series_jobs(
        jobs=build_time_series_jobs(begin_date, end_date, snow_off_day),
        aoi=aoi,
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
        tile_large_aoi=tile_large_aoi,
        tile_size_degrees=tile_size_degrees,
        predict_swe=predict_swe,
    )


def predict_snow_depth(**kwargs):
    """Compatibility alias for older code paths."""
    return predict_batch(**kwargs)


def predict_snow_depth_time_series(**kwargs):
    """Compatibility alias for older code paths."""
    return predict_time_series(**kwargs)
