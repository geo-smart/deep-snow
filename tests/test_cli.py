import unittest
from unittest.mock import patch

from deep_snow import cli


class CliTests(unittest.TestCase):
    def test_predict_sd_cli_passes_main_local_options_to_workflow(self):
        argv = [
            "deep-snow",
            "predict-sd",
            "20240320",
            "20230910",
            "-108.2 37.55 -107.61 38.09",
            "25",
            "--model-path",
            "weights/custom-model",
            "--out-name",
            "custom-output",
            "--out-dir",
            "tmp-output",
            "--out-crs",
            "utm",
            "--buffer-period",
            "10",
            "--fcf-path",
            "data/cache/fcf/custom_wus_fcf.tif",
            "--tile-size-degrees",
            "2.0",
            "--delete-inputs",
            "False",
            "--write-tif",
            "False",
            "--predict-swe",
            "True",
            "--gpu",
            "True",
        ]

        with patch("deep_snow.workflows.predict_batch") as mock_predict:
            with patch("sys.argv", argv):
                cli.main()

        mock_predict.assert_called_once_with(
            target_date="20240320",
            snow_off_date="20230910",
            aoi={
                "minlon": -108.2,
                "minlat": 37.55,
                "maxlon": -107.61,
                "maxlat": 38.09,
            },
            cloud_cover=25.0,
            use_ensemble=False,
            model_path="weights/custom-model",
            out_name="custom-output",
            out_dir="tmp-output",
            out_crs="utm",
            buffer_period=10,
            fcf_path="data/cache/fcf/custom_wus_fcf.tif",
            tile_large_aoi=True,
            tile_size_degrees=2.0,
            delete_inputs=False,
            write_tif=False,
            predict_swe=True,
            gpu=True,
        )

    def test_predict_tile_cli_remains_available_for_advanced_single_tile_runs(self):
        argv = [
            "deep-snow",
            "predict-tile",
            "20240320",
            "20230910",
            "-108.2 37.55 -107.61 38.09",
            "25",
            "--model-path",
            "weights/custom-model",
            "--out-name",
            "custom-output",
            "--out-dir",
            "tmp-output",
            "--out-crs",
            "utm",
            "--buffer-period",
            "10",
            "--fcf-path",
            "data/cache/fcf/custom_wus_fcf.tif",
            "--delete-inputs",
            "False",
            "--write-tif",
            "False",
            "--predict-swe",
            "True",
            "--gpu",
            "True",
        ]

        with patch("deep_snow.workflows.predict_tile") as mock_predict:
            with patch("sys.argv", argv):
                cli.main()

        mock_predict.assert_called_once_with(
            target_date="20240320",
            snow_off_date="20230910",
            aoi={
                "minlon": -108.2,
                "minlat": 37.55,
                "maxlon": -107.61,
                "maxlat": 38.09,
            },
            cloud_cover=25.0,
            use_ensemble=False,
            model_path="weights/custom-model",
            out_name="custom-output",
            out_dir="tmp-output",
            out_crs="utm",
            buffer_period=10,
            fcf_path="data/cache/fcf/custom_wus_fcf.tif",
            delete_inputs=False,
            write_tif=False,
            predict_swe=True,
            gpu=True,
        )

    def test_predict_sd_timeseries_cli_passes_expected_arguments(self):
        argv = [
            "deep-snow",
            "predict-sd-timeseries",
            "20240101",
            "20240320",
            "0901",
            "-108.2 37.55 -107.61 38.09",
            "25",
            "--use-ensemble",
            "--out-dir",
            "tmp-output",
            "--tile-large-aoi",
            "False",
        ]

        with patch("deep_snow.api.predict_sd_timeseries") as mock_predict:
            with patch("sys.argv", argv):
                cli.main()

        mock_predict.assert_called_once_with(
            begin_date="20240101",
            end_date="20240320",
            snow_off_day="0901",
            aoi={
                "minlon": -108.2,
                "minlat": 37.55,
                "maxlon": -107.61,
                "maxlat": 38.09,
            },
            cloud_cover=25.0,
            use_ensemble=True,
            model_path=None,
            out_name=None,
            out_dir="tmp-output",
            out_crs="wgs84",
            buffer_period=6,
            fcf_path=None,
            tile_large_aoi=False,
            tile_size_degrees=None,
            delete_inputs=True,
            write_tif=True,
            predict_swe=False,
            gpu=False,
        )
