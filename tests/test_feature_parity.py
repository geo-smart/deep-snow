"""Train/serve feature-parity test: the two production feature paths must agree.

The ensemble is trained on features from deep_snow.dataset.Datasetv2 and served
features from deep_snow.preprocessing + deep_snow.prediction. Any divergence between
the two paths silently degrades production predictions (this caught a real bug: the
serve path normalized northness with [0, 1] instead of [-1, 1], zeroing south-facing
slopes — found 2026-07-27, experiments/05_feature_parity/).

Feeds the SAME raw variables from the committed regression fixture through both paths
and asserts the 11 normalized input channels agree pixel-for-pixel. Terrain rasters
come from the fixture on both sides, so this pins feature assembly + normalization
(terrain-derivative provenance is xdem in both the tiler and serve — see
experiments/05_feature_parity/README.md).
"""
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import torch
    import xarray as xr

    from deep_snow.dataset import Datasetv2
    from deep_snow.prediction import PREDICTION_INPUT_CHANNELS, build_model_inputs
    from deep_snow.preprocessing import add_optical_features, add_radar_features
    from deep_snow.utils import calc_dowy
except ImportError:  # pragma: no cover - depends on optional runtime deps
    torch = None

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "regression"
ATOL = 5e-6  # normalized-unit scale is [0, 1]; observed cross-path diff is 0


def serve_features(path):
    """(11, H, W) serve-path features, mirroring preprocessing.build_prediction_dataset
    for everything the production channels need from already-gridded rasters."""
    with xr.open_dataset(path) as raw:
        ds = add_radar_features(raw.copy())
        ds = add_optical_features(ds)
        ds["northness"] = np.cos(np.deg2rad(ds["aspect"]))
        dowy = calc_dowy(pd.to_datetime(path.name.split("_")[4]).dayofyear)
        ds["dowy"] = xr.full_like(ds["elevation"], dowy, dtype=np.float64)
        return build_model_inputs(ds, input_channels=list(PREDICTION_INPUT_CHANNELS))[0]


@unittest.skipIf(torch is None, "torch-backed prediction runtime not available")
class FeatureParityTests(unittest.TestCase):
    def test_train_and_serve_features_identical(self):
        tile_paths = sorted(FIXTURE_DIR.glob("*.nc"))
        if not tile_paths:
            raise unittest.SkipTest(
                f"regression fixture missing at {FIXTURE_DIR} "
                "(build with experiments/04_regression_fixture/make_fixture.py)"
            )

        for path in tile_paths:
            train_ds = Datasetv2([str(path)], list(PREDICTION_INPUT_CHANNELS),
                                 norm=True, augment=False, cache_data=False)
            train_x = torch.cat(train_ds[0], 0).float()
            serve_x = serve_features(path).float()

            diff = (train_x - serve_x).abs()
            worst = {
                channel: float(diff[index].max())
                for index, channel in enumerate(PREDICTION_INPUT_CHANNELS)
                if float(diff[index].max()) > ATOL
            }
            self.assertFalse(
                worst,
                f"{path.name}: train/serve feature mismatch (max abs diff per channel: "
                f"{worst}). The ensemble was trained on Datasetv2 features — a serve-path "
                "divergence silently degrades production predictions.",
            )


if __name__ == "__main__":
    unittest.main()
