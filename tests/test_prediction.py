import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from deep_snow import api as prediction_api
    from deep_snow import prediction
    from deep_snow.application import apply_model, apply_model_ensemble
except ImportError:  # pragma: no cover - depends on optional runtime deps
    torch = None
    prediction_api = None
    prediction = None
    apply_model = None
    apply_model_ensemble = None


@unittest.skipIf(torch is None or prediction is None or prediction_api is None, "torch-backed prediction runtime not available")
class PredictionHelpersTests(unittest.TestCase):
    def test_build_model_inputs_uses_requested_channel_order(self):
        fake_features = {
            "blue": torch.full((1, 1, 2, 2), 2.0),
            "snodas_sd": torch.full((1, 1, 2, 2), 1.0),
            "fcf": torch.full((1, 1, 2, 2), 3.0),
        }

        with patch.object(prediction, "build_normalized_feature_dict", return_value=fake_features):
            stacked = prediction.build_model_inputs(
                ds=object(),
                input_channels=["fcf", "snodas_sd", "blue"],
            )

        self.assertEqual(stacked.shape, (1, 3, 2, 2))
        self.assertTrue(torch.equal(stacked[:, 0], fake_features["fcf"][:, 0]))
        self.assertTrue(torch.equal(stacked[:, 1], fake_features["snodas_sd"][:, 0]))
        self.assertTrue(torch.equal(stacked[:, 2], fake_features["blue"][:, 0]))

    def test_build_normalized_feature_dict_uses_model_feature_names_for_raw_dataset_bands(self):
        ds = {
            "B02": type("FakeVar", (), {"values": [[1.0, 2.0], [3.0, 4.0]]})(),
            "fcf": type("FakeVar", (), {"values": [[0.2, 0.3], [0.4, 0.5]]})(),
        }

        with patch.dict(
            prediction.FEATURE_SPECS,
            {
                "blue": ("B02", None),
                "fcf": ("fcf", None),
            },
            clear=True,
        ):
            feature_dict = prediction.build_normalized_feature_dict(ds)

        self.assertIn("blue", feature_dict)
        self.assertIn("fcf", feature_dict)
        self.assertNotIn("B02", feature_dict)

    def test_attach_prediction_metadata_records_output_paths(self):
        ds = type("FakeDataset", (), {"attrs": {}})()
        summary = {
            "target_date": "20240320",
            "snowoff_date": "20230910",
            "out_dir": "tmp-output",
            "out_name": "tile-a",
            "out_crs": "utm",
            "cloud_cover": 25,
            "buffer_period": 6,
            "gpu": False,
            "use_ensemble": False,
            "sentinel1_pass_selection": "descending",
            "selection_strategy": "composite",
            "model_count": 1,
            "model_inputs_path": "tmp-output/model_inputs.nc",
            "model_path": "weights/model-a",
            "predicted_tif_path": "tmp-output/tile-a_sd.tif",
            "input_gap_fraction": 0.125,
            "input_gap_pixel_count": 42,
            "input_gaps_tif_path": "tmp-output/tile-a_input_gaps.tif",
            "input_gaps_netcdf_path": "tmp-output/tile-a_input_gaps.nc",
            "gap_s1_snowon_fraction": 0.05,
            "gap_s1_snowoff_fraction": 0.03,
            "gap_s2_fraction": 0.09,
            "predict_swe": True,
            "predicted_swe_tif_path": "tmp-output/tile-a_swe.tif",
            "predicted_density_tif_path": "tmp-output/tile-a_density.tif",
            "input_provenance": None,
        }

        prediction_api.attach_prediction_metadata(ds, summary)

        self.assertEqual(ds.attrs["deep_snow_target_date"], "20240320")
        self.assertEqual(ds.attrs["deep_snow_model_path"], "weights/model-a")
        self.assertEqual(ds.attrs["deep_snow_predicted_tif_path"], "tmp-output/tile-a_sd.tif")
        self.assertEqual(ds.attrs["deep_snow_predicted_swe_tif_path"], "tmp-output/tile-a_swe.tif")
        self.assertEqual(ds.attrs["deep_snow_predicted_density_tif_path"], "tmp-output/tile-a_density.tif")
        self.assertEqual(ds.attrs["deep_snow_input_gap_fraction"], 0.125)
        self.assertEqual(ds.attrs["deep_snow_input_gap_pixel_count"], 42)
        self.assertEqual(ds.attrs["deep_snow_input_gaps_tif_path"], "tmp-output/tile-a_input_gaps.tif")
        self.assertEqual(ds.attrs["deep_snow_input_gaps_netcdf_path"], "tmp-output/tile-a_input_gaps.nc")
        self.assertEqual(ds.attrs["deep_snow_gap_s1_snowon_fraction"], 0.05)
        self.assertEqual(ds.attrs["deep_snow_gap_s1_snowoff_fraction"], 0.03)
        self.assertEqual(ds.attrs["deep_snow_gap_s2_fraction"], 0.09)

    def test_print_prediction_summary_shows_model_basename(self):
        summary = {
            "target_date": "20240320",
            "snowoff_date": "20230910",
            "aoi": {"minlon": -108.2, "minlat": 37.55, "maxlon": -108.0, "maxlat": 37.75},
            "model_path": "/tmp/models/example-model.ckpt",
            "model_count": 1,
            "out_crs": "wgs84",
            "gpu": False,
            "selection_strategy": "composite",
            "sentinel1_pass_selection": "descending",
            "predicted_tif_path": "tmp-output/tile-a_sd.tif",
            "predicted_swe_tif_path": None,
            "predicted_density_tif_path": None,
            "model_inputs_path": "tmp-output/model_inputs.nc",
            "buffer_period": 6,
            "initial_buffer_period": 6,
            "attempted_buffer_periods": [6],
            "input_gap_pixel_count": None,
            "input_provenance": None,
        }

        with patch("builtins.print") as mock_print:
            prediction_api.print_prediction_summary(summary)

        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list if call.args)
        self.assertIn(Path(summary["model_path"]).name, printed)
        self.assertNotIn(summary["model_path"], printed)

    def test_predict_in_tiles_restores_original_extent(self):
        class EchoModel:
            def __call__(self, tile_inputs):
                return tile_inputs[:, :1]

        inputs = torch.arange(1, 1 + (5 * 7), dtype=torch.float32).reshape(1, 1, 5, 7)
        pred = prediction.predict_in_tiles(
            inputs=inputs,
            models=[EchoModel()],
            tile_size=4,
            padding=1,
            gpu=False,
        )

        self.assertTrue(torch.equal(pred, inputs[0, 0]))


@unittest.skipIf(apply_model is None or apply_model_ensemble is None, "application runtime not available")
class ApplicationWrapperTests(unittest.TestCase):
    def test_apply_model_delegates_to_api_layer(self):
        with patch("deep_snow.application.api_apply_model", return_value="single-result") as mock_apply:
            result = apply_model(
                crs="EPSG:32612",
                model_path="weights/model-a",
                out_dir="tmp-output",
                out_name="tile-a",
                write_tif=False,
                delete_inputs=True,
                out_crs="utm",
                gpu=False,
            )

        self.assertEqual(result, "single-result")
        mock_apply.assert_called_once_with(
            "EPSG:32612",
            "weights/model-a",
            "tmp-output",
            "tile-a",
            False,
            True,
            "utm",
            gpu=False,
        )

    def test_apply_model_ensemble_delegates_to_api_layer(self):
        with patch("deep_snow.application.api_apply_model_ensemble", return_value="ensemble-result") as mock_apply:
            result = apply_model_ensemble(
                crs="EPSG:32612",
                model_paths_list=["weights/model-a", "weights/model-b"],
                out_dir="tmp-output",
                out_name="tile-a",
                write_tif=False,
                delete_inputs=True,
                out_crs="utm",
                gpu=False,
            )

        self.assertEqual(result, "ensemble-result")
        mock_apply.assert_called_once_with(
            "EPSG:32612",
            ["weights/model-a", "weights/model-b"],
            "tmp-output",
            "tile-a",
            False,
            True,
            "utm",
            gpu=False,
        )
