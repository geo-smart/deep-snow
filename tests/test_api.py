import unittest
from unittest.mock import patch

from deep_snow import api
from deep_snow.workflows import DEFAULT_MODEL_PATHS, DEFAULT_SINGLE_MODEL_PATH


class FakeDataset:
    def __init__(self):
        self.attrs = {}


class ApiTests(unittest.TestCase):
    def test_resolve_api_models_uses_default_single_model(self):
        model_path, model_paths_list = api.resolve_api_models(use_ensemble=False)

        self.assertEqual(model_path, DEFAULT_SINGLE_MODEL_PATH)
        self.assertIsNone(model_paths_list)

    def test_resolve_api_models_uses_default_ensemble_models(self):
        model_path, model_paths_list = api.resolve_api_models(use_ensemble=True)

        self.assertIsNone(model_path)
        self.assertEqual(model_paths_list, DEFAULT_MODEL_PATHS)

    def test_predict_sd_uses_default_single_model_when_none_provided(self):
        fake_ds = FakeDataset()

        with patch("deep_snow.api.download_data", return_value="EPSG:32612"):
            with patch("deep_snow.api.apply_model", return_value=fake_ds) as mock_apply_model:
                with patch("deep_snow.api.attach_prediction_metadata", side_effect=lambda ds, summary: ds):
                    with patch("deep_snow.api.print_prediction_summary"):
                        result = api.predict_sd(
                            aoi={"minlon": -108.2, "minlat": 37.55, "maxlon": -108.0, "maxlat": 37.75},
                            target_date="20240320",
                            snowoff_date="20230910",
                        )

        self.assertIs(result, fake_ds)
        mock_apply_model.assert_called_once()
        self.assertEqual(mock_apply_model.call_args.args[1], DEFAULT_SINGLE_MODEL_PATH)

    def test_predict_sd_can_use_default_ensemble_models(self):
        fake_ds = FakeDataset()

        with patch("deep_snow.api.download_data", return_value="EPSG:32612"):
            with patch("deep_snow.api.apply_model_ensemble", return_value=fake_ds) as mock_apply_ensemble:
                with patch("deep_snow.api.attach_prediction_metadata", side_effect=lambda ds, summary: ds):
                    with patch("deep_snow.api.print_prediction_summary"):
                        result = api.predict_sd(
                            aoi={"minlon": -108.2, "minlat": 37.55, "maxlon": -108.0, "maxlat": 37.75},
                            target_date="20240320",
                            snowoff_date="20230910",
                            use_ensemble=True,
                        )

        self.assertIs(result, fake_ds)
        mock_apply_ensemble.assert_called_once()
        self.assertEqual(mock_apply_ensemble.call_args.args[1], DEFAULT_MODEL_PATHS)
