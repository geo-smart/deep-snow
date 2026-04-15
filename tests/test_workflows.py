import unittest
from datetime import datetime
from pathlib import Path
import tempfile
from unittest.mock import patch

from deep_snow.inputs import get_default_fcf_cache_path
from deep_snow.resources import get_default_hill_pptwt_path, get_default_hill_td_path, get_default_land_path
from deep_snow.workflows import (
    DEFAULT_MODEL_PATHS,
    DEFAULT_SINGLE_MODEL_PATH,
    format_default_model_guidance,
    resolve_prediction_models,
)
from deep_snow import workflows


class WorkflowTests(unittest.TestCase):
    def test_resolve_prediction_models_uses_default_single_model(self):
        model_path, model_paths_list = resolve_prediction_models(use_ensemble=False)

        self.assertEqual(model_path, DEFAULT_SINGLE_MODEL_PATH)
        self.assertTrue(Path(model_path).exists())
        self.assertIsNone(model_paths_list)

    def test_resolve_prediction_models_uses_default_ensemble_models(self):
        model_path, model_paths_list = resolve_prediction_models(use_ensemble=True)

        self.assertIsNone(model_path)
        self.assertEqual(model_paths_list, DEFAULT_MODEL_PATHS)
        self.assertTrue(all(Path(path).exists() for path in model_paths_list))

    def test_resolve_prediction_models_respects_explicit_model_path(self):
        model_path, model_paths_list = resolve_prediction_models(
            use_ensemble=False,
            model_path="weights/custom-model",
        )

        self.assertEqual(model_path, "weights/custom-model")
        self.assertIsNone(model_paths_list)

    def test_format_default_model_guidance_mentions_single_model_default(self):
        guidance = format_default_model_guidance(use_ensemble=False)

        self.assertIn(Path(DEFAULT_SINGLE_MODEL_PATH).name, guidance)

    def test_format_default_model_guidance_mentions_ensemble_defaults(self):
        guidance = format_default_model_guidance(use_ensemble=True)

        self.assertIn("DEFAULT_MODEL_PATHS", guidance)

    def test_default_fcf_cache_path_points_to_wus_cache_file(self):
        cache_path = get_default_fcf_cache_path().replace("\\", "/")

        self.assertTrue(cache_path.endswith("data/cache/fcf/wus_fcf.tif"))

    def test_default_tile_land_path_points_to_packaged_land_shapefile(self):
        land_path = Path(get_default_land_path())

        self.assertTrue(land_path.exists())
        self.assertEqual(land_path.name, "ne_50m_land.shp")

    def test_default_hill_density_paths_point_to_cache_files(self):
        self.assertTrue(get_default_hill_pptwt_path().replace("\\", "/").endswith("data/cache/hill/ppt_wt_final.txt"))
        self.assertTrue(get_default_hill_td_path().replace("\\", "/").endswith("data/cache/hill/td_final.txt"))


class WorkflowRoutingTests(unittest.TestCase):
    def test_validate_prediction_dates_rejects_future_target_date(self):
        with self.assertRaises(ValueError):
            workflows.validate_prediction_dates(
                "20990101",
                "20230910",
                current_date=datetime(2026, 4, 15),
            )

    def test_validate_prediction_dates_rejects_pre_sentinel2_target_date(self):
        with self.assertRaises(ValueError):
            workflows.validate_prediction_dates(
                "20150101",
                "20140101",
                current_date=datetime(2026, 4, 15),
            )

    def test_validate_prediction_dates_rejects_snow_off_not_earlier_than_target(self):
        with self.assertRaises(ValueError):
            workflows.validate_prediction_dates(
                "20240320",
                "20240320",
                current_date=datetime(2026, 4, 15),
            )

    def test_validate_prediction_dates_warns_for_very_old_snow_off_date(self):
        with patch("builtins.print") as mock_print:
            workflows.validate_prediction_dates(
                "20240320",
                "20220101",
                current_date=datetime(2026, 4, 15),
            )

        mock_print.assert_called_once()
        self.assertIn("far earlier than target_date", mock_print.call_args.args[0])

    def test_validate_prediction_aoi_rejects_aoi_outside_conus(self):
        with self.assertRaises(ValueError):
            workflows.validate_prediction_aoi(
                {"minlon": -150.0, "minlat": 60.0, "maxlon": -149.0, "maxlat": 61.0}
            )

    def test_validate_prediction_aoi_warns_for_conus_aoi_outside_western_us(self):
        with patch("builtins.print") as mock_print:
            workflows.validate_prediction_aoi(
                {"minlon": -90.0, "minlat": 35.0, "maxlon": -89.0, "maxlat": 36.0}
            )

        mock_print.assert_called_once()
        self.assertIn("not been validated", mock_print.call_args.args[0])

    def test_resolve_local_tile_size_degrees_defaults_to_batch_tile_size(self):
        self.assertEqual(
            workflows.resolve_local_tile_size_degrees(None),
            workflows.DEFAULT_LOCAL_TILE_SIZE_DEGREES,
        )

    def test_resolve_local_tile_size_degrees_rejects_too_small_tiles(self):
        with self.assertRaises(ValueError):
            workflows.resolve_local_tile_size_degrees(0.1)

    def test_aoi_requires_local_tiling_uses_batch_tile_size_threshold(self):
        self.assertFalse(
            workflows.aoi_requires_local_tiling(
                {"minlon": -108.2, "minlat": 37.55, "maxlon": -107.61, "maxlat": 38.09}
            )
        )
        self.assertTrue(
            workflows.aoi_requires_local_tiling(
                {"minlon": -120.0, "minlat": 35.0, "maxlon": -117.5, "maxlat": 38.0}
            )
        )

    def test_resolve_tiled_output_crs_forces_wgs84_for_tiled_mosaic(self):
        with patch("builtins.print") as mock_print:
            resolved = workflows.resolve_tiled_output_crs("utm")

        self.assertEqual(resolved, "wgs84")
        mock_print.assert_called_once()

    def test_predict_batch_routes_large_aoi_to_tiled_workflow(self):
        fake_ds = type("FakeDataset", (), {"attrs": {"deep_snow_tile_count": 2}})()

        with patch("deep_snow.workflows.validate_prediction_dates") as mock_validate_dates:
            with patch("deep_snow.workflows.validate_prediction_aoi") as mock_validate_aoi:
                with patch("deep_snow.workflows._predict_large_aoi_in_tiles", return_value=(fake_ds, "wgs84")) as mock_tiled:
                    with patch("deep_snow.api.attach_prediction_metadata", side_effect=lambda ds, summary: ds) as mock_attach:
                        with patch("deep_snow.api.build_prediction_summary", side_effect=lambda **kwargs: kwargs) as mock_summary:
                            with patch("deep_snow.api.print_prediction_summary") as mock_print_summary:
                                result = workflows.predict_batch(
                                    target_date="20240320",
                                    snow_off_date="20230910",
                                    aoi={"minlon": -120.0, "minlat": 35.0, "maxlon": -117.5, "maxlat": 38.0},
                                    cloud_cover=25,
                                    use_ensemble=False,
                                    out_dir="tmp-output",
                                    out_crs="utm",
                                    emit_summary=False,
                                )

        self.assertIs(result, fake_ds)
        mock_validate_dates.assert_called_once()
        mock_validate_aoi.assert_called_once()
        mock_tiled.assert_called_once()
        mock_summary.assert_called_once()
        mock_attach.assert_called_once()
        mock_print_summary.assert_not_called()

    def test_predict_tile_does_not_route_large_aoi_to_tiled_workflow(self):
        fake_ds = type("FakeDataset", (), {"attrs": {}})()

        with patch("deep_snow.workflows.validate_prediction_dates"):
            with patch("deep_snow.workflows.validate_prediction_aoi"):
                with patch("deep_snow.workflows._predict_large_aoi_in_tiles") as mock_tiled:
                    with patch("deep_snow.api.download_data", return_value="EPSG:32612"):
                        with patch("deep_snow.api.read_prediction_input_provenance", return_value={}):
                            with patch("deep_snow.api.apply_model", return_value=fake_ds):
                                with patch("deep_snow.api.attach_prediction_metadata", side_effect=lambda ds, summary: ds):
                                    with patch("deep_snow.api.build_prediction_summary", side_effect=lambda **kwargs: kwargs):
                                        result = workflows.predict_tile(
                                            target_date="20240320",
                                            snow_off_date="20230910",
                                            aoi={"minlon": -120.0, "minlat": 35.0, "maxlon": -117.5, "maxlat": 38.0},
                                            cloud_cover=25,
                                            use_ensemble=False,
                                            emit_summary=False,
                                        )

        self.assertIs(result, fake_ds)
        mock_tiled.assert_not_called()

    def test_predict_tile_warns_and_expands_buffer_period_for_empty_acquisition(self):
        fake_ds = type("FakeDataset", (), {"attrs": {}})()

        with patch("deep_snow.workflows.validate_prediction_dates"):
            with patch("deep_snow.workflows.validate_prediction_aoi"):
                with patch(
                    "deep_snow.api.download_data",
                    side_effect=[
                        workflows.EmptyAcquisitionError("No Sentinel-2 items found."),
                        "EPSG:32612",
                    ],
                ) as mock_download:
                    with patch("deep_snow.api.read_prediction_input_provenance", return_value={}) as mock_read_provenance:
                        with patch("deep_snow.api.apply_model", return_value=fake_ds):
                            with patch("deep_snow.api.attach_prediction_metadata", side_effect=lambda ds, summary: ds):
                                with patch("deep_snow.api.build_prediction_summary", side_effect=lambda **kwargs: kwargs) as mock_summary:
                                    with patch("builtins.print") as mock_print:
                                        result = workflows.predict_tile(
                                            target_date="20240320",
                                            snow_off_date="20230910",
                                            aoi={"minlon": -108.2, "minlat": 37.55, "maxlon": -108.0, "maxlat": 37.75},
                                            cloud_cover=25,
                                            use_ensemble=False,
                                            emit_summary=False,
                                            max_buffer_expansions=2,
                                            buffer_expansion_step_days=2,
                                        )

        self.assertIs(result, fake_ds)
        self.assertEqual(mock_download.call_count, 2)
        self.assertEqual(mock_download.call_args_list[0].kwargs["buffer_period"], 6)
        self.assertEqual(mock_download.call_args_list[1].kwargs["buffer_period"], 8)
        self.assertEqual(
            mock_summary.call_args.kwargs["attempted_buffer_periods"],
            [6, 8],
        )
        self.assertEqual(mock_summary.call_args.kwargs["initial_buffer_period"], 6)
        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn("WARNING:", printed)
        self.assertIn("Expanding buffer_period to 8 days", printed)
        mock_read_provenance.assert_called_once()

    def test_predict_tile_stops_after_transient_retry_cap(self):
        with patch("deep_snow.workflows.validate_prediction_dates"):
            with patch("deep_snow.workflows.validate_prediction_aoi"):
                with patch(
                    "deep_snow.api.download_data",
                    side_effect=workflows.TransientAcquisitionError("temporary service unavailable"),
                ) as mock_download:
                    with patch("time.sleep") as mock_sleep:
                        with self.assertRaises(workflows.TransientAcquisitionError):
                            workflows.predict_tile(
                                target_date="20240320",
                                snow_off_date="20230910",
                                aoi={"minlon": -108.2, "minlat": 37.55, "maxlon": -108.0, "maxlat": 37.75},
                                cloud_cover=25,
                                use_ensemble=False,
                                emit_summary=False,
                                max_retries=2,
                            )

        self.assertEqual(mock_download.call_count, 2)
        mock_sleep.assert_called_once()

    def test_predict_time_series_uses_batch_predictions_for_each_date(self):
        class FakeRio:
            def __init__(self, result):
                self._result = result

            def reproject_match(self, other):
                return self._result

        class FakeDataset:
            def __init__(self):
                self.attrs = {}
                self.rio = FakeRio(self)

            def expand_dims(self, **kwargs):
                return self

        first_ds = FakeDataset()
        second_ds = FakeDataset()

        with patch(
            "deep_snow.workflows.build_time_series_jobs",
            return_value=[
                {"target_date": "20240320", "snow_off_date": "20230910"},
                {"target_date": "20240308", "snow_off_date": "20230910"},
            ],
        ):
            with patch("deep_snow.workflows.predict_batch", side_effect=[first_ds, second_ds]) as mock_predict_batch:
                with patch("xarray.concat", return_value="combined-dataset") as mock_concat:
                    result = workflows.predict_time_series(
                        begin_date="20240301",
                        end_date="20240320",
                        snow_off_day="0901",
                        aoi={"minlon": -108.2, "minlat": 37.55, "maxlon": -108.0, "maxlat": 37.75},
                        cloud_cover=25,
                    )

        self.assertEqual(result, "combined-dataset")
        self.assertEqual(mock_predict_batch.call_count, 2)
        self.assertEqual(mock_predict_batch.call_args_list[0].kwargs["target_date"], "20240320")
        self.assertEqual(mock_predict_batch.call_args_list[1].kwargs["target_date"], "20240308")
        mock_concat.assert_called_once()

    def test_predict_tile_writes_log_file_when_writing_tifs(self):
        fake_ds = type("FakeDataset", (), {"attrs": {}})()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("deep_snow.workflows._predict_single_tile", return_value=fake_ds) as mock_predict:
                with patch("builtins.print") as mock_print:
                    result = workflows.predict_tile(
                        target_date="20240320",
                        snow_off_date="20230910",
                        aoi={"minlon": -108.2, "minlat": 37.55, "maxlon": -108.0, "maxlat": 37.75},
                        cloud_cover=25,
                        use_ensemble=False,
                        out_dir=tmpdir,
                        write_tif=True,
                    )

            self.assertIs(result, fake_ds)
            mock_predict.assert_called_once()
            log_path = Path(tmpdir) / "20240320_deep-snow_-108.20_37.55_-108.00_37.75_log.txt"
            self.assertTrue(log_path.exists())
            self.assertIn("[predict] writing run log", log_path.read_text(encoding="utf-8"))
