import unittest
from unittest.mock import patch

from deep_snow import application


class ApplicationTests(unittest.TestCase):
    def test_predict_sd_delegates_to_api_layer(self):
        with patch("deep_snow.application.api_predict_sd", return_value="api-result") as mock_predict:
            result = application.predict_sd(
                aoi={"minlon": -108.2, "minlat": 37.55, "maxlon": -108.0, "maxlat": 37.75},
                target_date="20240320",
                snowoff_date="20230910",
                model_path="weights/custom-model",
                out_dir="tmp-output",
                out_crs="utm",
                out_name="custom-output",
                write_tif=False,
                delete_inputs=True,
                cloud_cover=20,
                buffer_period=10,
                fcf_path="data/cache/fcf/custom_wus_fcf.tif",
                gpu=True,
                use_ensemble=True,
                model_paths_list=["weights/model-a", "weights/model-b"],
                sentinel1_orbit_selection="all",
                selection_strategy="nearest_usable",
            )

        self.assertEqual(result, "api-result")
        mock_predict.assert_called_once_with(
            aoi={"minlon": -108.2, "minlat": 37.55, "maxlon": -108.0, "maxlat": 37.75},
            target_date="20240320",
            snowoff_date="20230910",
            model_path="weights/custom-model",
            out_dir="tmp-output",
            out_crs="utm",
            out_name="custom-output",
            write_tif=False,
            delete_inputs=True,
            cloud_cover=20,
            use_ensemble=True,
            buffer_period=10,
            fcf_path="data/cache/fcf/custom_wus_fcf.tif",
            gpu=True,
            model_paths_list=["weights/model-a", "weights/model-b"],
            sentinel1_orbit_selection="all",
            selection_strategy="nearest_usable",
        )
