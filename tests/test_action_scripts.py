import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deep_snow.resources import get_default_land_path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ActionScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predict_tile_script = load_module(
            "predict_tile_sd_script",
            Path("scripts/actions/predict_tile_sd.py"),
        )
        cls.prep_tiles_script = load_module(
            "prep_tiles_script",
            Path("scripts/actions/prep_tiles.py"),
        )
        cls.prep_time_series_script = load_module(
            "prep_time_series_script",
            Path("scripts/actions/prep_time_series.py"),
        )

    def test_predict_tile_wrapper_passes_expected_arguments(self):
        argv = [
            "predict_tile_sd.py",
            "20240320",
            "20230910",
            "-108.2 37.55 -107.61 38.09",
            "25",
            "True",
        ]

        with patch.object(self.predict_tile_script, "predict_tile") as mock_predict:
            with patch("sys.argv", argv):
                self.predict_tile_script.main()

        mock_predict.assert_called_once_with(
            target_date="20240320",
            snow_off_date="20230910",
            aoi={
                "minlon": -108.2,
                "minlat": 37.55,
                "maxlon": -107.61,
                "maxlat": 38.09,
            },
            cloud_cover="25",
            use_ensemble=True,
            crop_aoi=None,
            sentinel1_orbit_selection="descending",
            selection_strategy="composite",
            predict_swe=False,
        )

    def test_predict_tile_wrapper_passes_clip_aoi_when_provided(self):
        argv = [
            "predict_tile_sd.py",
            "20240320",
            "20230910",
            "-108.2 37.55 -107.61 38.09",
            "25",
            "False",
            "--clip-aoi",
            "-108.2 37.55 -108.0 37.75",
        ]

        with patch.object(self.predict_tile_script, "predict_tile") as mock_predict:
            with patch("sys.argv", argv):
                self.predict_tile_script.main()

        mock_predict.assert_called_once_with(
            target_date="20240320",
            snow_off_date="20230910",
            aoi={
                "minlon": -108.2,
                "minlat": 37.55,
                "maxlon": -107.61,
                "maxlat": 38.09,
            },
            cloud_cover="25",
            use_ensemble=False,
            crop_aoi={
                "minlon": -108.2,
                "minlat": 37.55,
                "maxlon": -108.0,
                "maxlat": 37.75,
            },
            sentinel1_orbit_selection="descending",
            selection_strategy="composite",
            predict_swe=False,
        )

    def test_predict_tile_wrapper_passes_sentinel_selection_options(self):
        argv = [
            "predict_tile_sd.py",
            "20240320",
            "20230910",
            "-108.2 37.55 -107.61 38.09",
            "25",
            "False",
            "--s1-orbit-selection",
            "all",
            "--selection-strategy",
            "nearest_usable",
        ]

        with patch.object(self.predict_tile_script, "predict_tile") as mock_predict:
            with patch("sys.argv", argv):
                self.predict_tile_script.main()

        mock_predict.assert_called_once_with(
            target_date="20240320",
            snow_off_date="20230910",
            aoi={
                "minlon": -108.2,
                "minlat": 37.55,
                "maxlon": -107.61,
                "maxlat": 38.09,
            },
            cloud_cover="25",
            use_ensemble=False,
            crop_aoi=None,
            sentinel1_orbit_selection="all",
            selection_strategy="nearest_usable",
            predict_swe=False,
        )

    def test_predict_tile_wrapper_passes_predict_swe_option(self):
        argv = [
            "predict_tile_sd.py",
            "20240320",
            "20230910",
            "-108.2 37.55 -107.61 38.09",
            "25",
            "False",
            "--predict-swe",
            "True",
        ]

        with patch.object(self.predict_tile_script, "predict_tile") as mock_predict:
            with patch("sys.argv", argv):
                self.predict_tile_script.main()

        self.assertTrue(mock_predict.call_args.kwargs["predict_swe"])

    def test_prep_tiles_wrapper_writes_matrix_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            github_output = Path(tmpdir) / "github_output.txt"
            argv = [
                "prep_tiles.py",
                "20240320",
                "-108.2 37.55 -107.61 38.09",
            ]

            with patch.object(
                self.prep_tiles_script,
                "build_tile_jobs",
                return_value=[{"name": "tile-a"}],
            ) as mock_build_jobs:
                with patch.object(
                    self.prep_tiles_script,
                    "build_matrix_json",
                    return_value='{"include":[{"name":"tile-a"}]}',
                ) as mock_matrix:
                    with patch("sys.argv", argv):
                        with patch.dict(os.environ, {"GITHUB_OUTPUT": str(github_output)}, clear=False):
                            stdout = io.StringIO()
                            with contextlib.redirect_stdout(stdout):
                                self.prep_tiles_script.main()

            mock_build_jobs.assert_called_once_with(
                target_date="20240320",
                aoi={
                    "minlon": -108.2,
                    "minlat": 37.55,
                    "maxlon": -107.61,
                    "maxlat": 38.09,
                },
                land_path=get_default_land_path(),
            )
            mock_matrix.assert_called_once_with([{"name": "tile-a"}])
            self.assertIn("Prepared 1 tile job(s) for 20240320.", stdout.getvalue())
            self.assertIn(
                'MATRIX_PARAMS_COMBINATIONS={"include":[{"name":"tile-a"}]}',
                github_output.read_text(),
            )

    def test_prep_time_series_wrapper_writes_matrix_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            github_output = Path(tmpdir) / "github_output.txt"
            argv = [
                "prep_time_series.py",
                "20220101",
                "20220301",
                "0901",
            ]

            with patch.object(
                self.prep_time_series_script,
                "build_time_series_jobs",
                return_value=[{"target_date": "20220301", "snow_off_date": "20210901"}],
            ) as mock_jobs:
                with patch.object(
                    self.prep_time_series_script,
                    "build_matrix_json",
                    return_value='{"include":[{"target_date":"20220301","snow_off_date":"20210901"}]}',
                ) as mock_matrix:
                    with patch("sys.argv", argv):
                        with patch.dict(os.environ, {"GITHUB_OUTPUT": str(github_output)}, clear=False):
                            stdout = io.StringIO()
                            with contextlib.redirect_stdout(stdout):
                                self.prep_time_series_script.main()

            mock_jobs.assert_called_once_with("20220101", "20220301", "0901")
            mock_matrix.assert_called_once_with(
                [{"target_date": "20220301", "snow_off_date": "20210901"}]
            )
            self.assertIn("Prepared 1 time-series date job(s).", stdout.getvalue())
            self.assertIn(
                'MATRIX_PARAMS_COMBINATIONS={"include":[{"target_date":"20220301","snow_off_date":"20210901"}]}',
                github_output.read_text(),
            )

    def test_prep_time_series_wrapper_prints_matrix_locally_without_github_output(self):
        argv = [
            "prep_time_series.py",
            "20220101",
            "20220301",
            "0901",
        ]

        with patch.object(
            self.prep_time_series_script,
            "build_time_series_jobs",
            return_value=[{"target_date": "20220301", "snow_off_date": "20210901"}],
        ):
            with patch.object(
                self.prep_time_series_script,
                "build_matrix_json",
                return_value='{"include":[{"target_date":"20220301","snow_off_date":"20210901"}]}',
            ):
                with patch("sys.argv", argv):
                    with patch.dict(os.environ, {}, clear=True):
                        stdout = io.StringIO()
                        with contextlib.redirect_stdout(stdout):
                            self.prep_time_series_script.main()

        output = stdout.getvalue()
        self.assertIn("Prepared 1 time-series date job(s).", output)
        self.assertIn('{"include":[{"target_date":"20220301","snow_off_date":"20210901"}]}', output)
