"""Numerical regression test: pins the 5-member production ensemble's predictions.

Runs the ensemble on a small committed fixture (4 real test tiles spanning shallow to
very deep snow) through the train-path feature pipeline (Datasetv2) and asserts the
median predictions match the stored reference to within 1 mm. Any refactor of
deep_snow that changes model outputs — feature prep, normalization, architecture,
weight loading, ensembling — fails this test instead of silently shifting predictions.

Fixture + reference are built by experiments/04_regression_fixture/make_fixture.py
(which imports predict_fixture_tiles from this module, so the reference is generated
by exactly the code under test). Regenerate only for an intended, documented change.
"""
import json
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    import xarray  # noqa: F401  (netCDF backend needed by Datasetv2)

    from deep_snow import resources
    from deep_snow.dataset import Datasetv2
    from deep_snow.model_loading import load_resdepth_checkpoint
    from deep_snow.models import ResDepth
    from deep_snow.prediction import PREDICTION_INPUT_CHANNELS
except ImportError:  # pragma: no cover - depends on optional runtime deps
    torch = None

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "regression"
DEPTH_SCALE = 12.0  # norm_dict['aso_sd'] = [0, 12] m
MAX_ABS_TOL_M = 1e-3
MEAN_ABS_TOL_M = 1e-4


def predict_fixture_tiles(paths):
    """Ensemble-median snow depth (meters) for tile paths: (N, 128, 128) float32.

    Single-threaded CPU so results are reproducible across machines; no randomness
    (augment=False, eval mode, inference only).
    """
    torch.set_num_threads(1)
    channels = list(PREDICTION_INPUT_CHANNELS)
    ds = Datasetv2(paths, channels, norm=True, augment=False, cache_data=False)
    x = torch.stack([torch.cat(ds[i], 0).float() for i in range(len(paths))], 0)

    models = []
    for weight_path in resources.get_default_model_paths():
        model = ResDepth(n_input_channels=len(channels), depth=5)
        load_resdepth_checkpoint(model, weight_path, gpu=False)
        model.eval()
        models.append(model)

    with torch.no_grad():
        preds = torch.stack([m(x)[:, 0] for m in models], 0)  # (M, N, H, W) in [0, 1]
        median_m = torch.median(preds, dim=0).values * DEPTH_SCALE
    return median_m.numpy().astype(np.float32)


@unittest.skipIf(torch is None, "torch-backed prediction runtime not available")
class NumericalRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference = FIXTURE_DIR / "expected_predictions.npz"
        if not reference.exists():
            raise unittest.SkipTest(
                f"regression fixture missing at {FIXTURE_DIR} "
                "(build with experiments/04_regression_fixture/make_fixture.py)"
            )
        cls.reference = np.load(reference)
        cls.provenance = json.loads((FIXTURE_DIR / "provenance.json").read_text())

    def test_channel_set_unchanged(self):
        self.assertEqual(
            list(PREDICTION_INPUT_CHANNELS),
            list(self.reference["channels"]),
            "PREDICTION_INPUT_CHANNELS changed; regenerate the fixture if intended",
        )

    def test_ensemble_has_five_members(self):
        self.assertEqual(len(resources.get_default_model_paths()), 5)

    def test_predictions_match_reference(self):
        files = list(self.reference["files"])
        paths = [str(FIXTURE_DIR / f) for f in files]
        for p in paths:
            self.assertTrue(Path(p).exists(), f"fixture tile missing: {p}")

        pred_m = predict_fixture_tiles(paths)
        expected_m = self.reference["pred_m"]
        self.assertEqual(pred_m.shape, expected_m.shape)

        diff = np.abs(pred_m - expected_m)
        worst = {f: float(diff[i].max()) for i, f in enumerate(files)}
        self.assertLessEqual(
            float(diff.max()), MAX_ABS_TOL_M,
            f"ensemble predictions moved > {MAX_ABS_TOL_M * 1000:.0f} mm vs pinned "
            f"reference (per-tile max abs diff, meters: {worst}). If this change is "
            "intended, regenerate the fixture and document it in WORKLOG.md.",
        )
        self.assertLessEqual(float(diff.mean()), MEAN_ABS_TOL_M)


if __name__ == "__main__":
    unittest.main()
