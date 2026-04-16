import unittest
from unittest.mock import PropertyMock, patch

import numpy as np
import xarray as xr

from deep_snow import preprocessing


class _FakeRioAccessor:
    def __init__(self, *, nodata=None, transform="fake-transform", crs="EPSG:32612"):
        self.nodata = nodata
        self._transform = transform
        self.crs = crs

    def transform(self):
        return self._transform


class _FakeTerrainResult:
    def __init__(self, data):
        self.data = type("DataHolder", (), {"data": data})()


class PreprocessingTests(unittest.TestCase):
    def test_add_terrain_features_passes_nan_nodata_when_elevation_has_nonfinite_values(self):
        elevation = xr.DataArray(
            np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32),
            dims=("y", "x"),
            name="elevation",
        )
        ds = xr.Dataset({"elevation": elevation})

        terrain_output = np.zeros((2, 2), dtype=np.float32)

        with patch.object(type(elevation), "rio", new_callable=PropertyMock, return_value=_FakeRioAccessor(nodata=None)):
            with patch.object(preprocessing.xdem.DEM, "from_array", return_value="fake-dem") as mock_from_array:
                with patch.object(preprocessing.xdem.terrain, "aspect", return_value=_FakeTerrainResult(terrain_output)):
                    with patch.object(preprocessing.xdem.terrain, "slope", return_value=_FakeTerrainResult(terrain_output)):
                        with patch.object(preprocessing.xdem.terrain, "curvature", return_value=_FakeTerrainResult(terrain_output)):
                            with patch.object(
                                preprocessing.xdem.terrain,
                                "topographic_position_index",
                                return_value=_FakeTerrainResult(terrain_output),
                            ):
                                with patch.object(
                                    preprocessing.xdem.terrain,
                                    "terrain_ruggedness_index",
                                    return_value=_FakeTerrainResult(terrain_output),
                                ):
                                    preprocessing.add_terrain_features(ds)

        self.assertTrue(np.isnan(mock_from_array.call_args.kwargs["nodata"]))


if __name__ == "__main__":
    unittest.main()
