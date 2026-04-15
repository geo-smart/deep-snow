import unittest
from unittest.mock import patch

try:
    import torch
    from deep_snow.application import ModelCompatibilityError, load_resdepth_checkpoint
except ImportError:  # pragma: no cover - depends on heavy optional runtime deps
    torch = None
    ModelCompatibilityError = None
    load_resdepth_checkpoint = None


class DummyWeight:
    def __init__(self, in_channels):
        self.shape = (64, in_channels, 3, 3)


class DummyConv:
    def __init__(self, in_channels):
        self.weight = DummyWeight(in_channels)


class DummyModel:
    def __init__(self, in_channels):
        self.encoder = [[[DummyConv(in_channels)]]]

    def load_state_dict(self, state_dict):
        raise RuntimeError(
            "Error(s) in loading state_dict for ResDepth:\n"
            "size mismatch for encoder.0.0.0.weight: copying a param with shape "
            "torch.Size([64, 9, 3, 3]) from checkpoint, the shape in current model is "
            "torch.Size([64, 11, 3, 3])."
        )

    def to(self, device):
        return self


@unittest.skipIf(torch is None or load_resdepth_checkpoint is None, "torch-backed application runtime not available")
class ModelLoadingTests(unittest.TestCase):
    def test_load_resdepth_checkpoint_raises_friendly_compatibility_error(self):
        model = DummyModel(in_channels=11)

        with patch.object(torch, "load", return_value={"fake": "state"}):
            with self.assertRaises(ModelCompatibilityError) as ctx:
                load_resdepth_checkpoint(model, "weights/outdated-model", gpu=False)

        message = str(ctx.exception)
        self.assertIn("weights/outdated-model", message)
        self.assertIn("input feature set", message)
        self.assertIn("11 input channels", message)
