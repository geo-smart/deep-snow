import gzip
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from deep_snow import acquisition
except ImportError:  # pragma: no cover - depends on optional runtime deps
    acquisition = None


@unittest.skipIf(acquisition is None, "acquisition runtime not available")
class AcquisitionHelpersTests(unittest.TestCase):
    def test_get_snodas_cache_dir_uses_project_cache_directory(self):
        cache_dir = acquisition.get_snodas_cache_dir()

        self.assertTrue(cache_dir.exists())
        self.assertEqual(cache_dir.name, "snodas")
        self.assertEqual(cache_dir.parent.name, "cache")
        self.assertEqual(cache_dir.parent.parent.name, "data")

    def test_extract_and_decompress_snodas_archive_with_stdlib_helpers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gz_path = tmp_path / "us_ssmv11036tS__T0001TTNATS2024032005HP001.dat.gz"
            txt_payload = b"example snodas contents\n"

            with gzip.open(gz_path, "wb") as gz_file:
                gz_file.write(txt_payload)

            archive_path = tmp_path / "SNODAS_20240320.tar"
            with tarfile.open(archive_path, "w") as archive:
                archive.add(gz_path, arcname=gz_path.name)

            extract_dir = tmp_path / "extract"
            extract_dir.mkdir()

            acquisition.extract_tar_archive(archive_path, extract_dir)
            decompressed_paths = acquisition.decompress_gzip_files(extract_dir)

            self.assertEqual(len(decompressed_paths), 1)
            self.assertEqual(decompressed_paths[0].name, gz_path.stem)
            self.assertEqual(decompressed_paths[0].read_bytes(), txt_payload)

    def test_select_acquisitions_composite_skips_unreadable_acquisition(self):
        class GoodCandidate:
            def __init__(self):
                self.data_vars = {"band": object()}

            def squeeze(self):
                return self

        class FailingMedianResult:
            def squeeze(self):
                return self

            def compute(self):
                raise AssertionError("broken asset")

        class FakeConcatResult:
            def __init__(self, candidate):
                self._candidate = candidate

            def median(self, dim):
                return self._candidate

        class FakeDataset:
            def __init__(self):
                self.time = type("FakeTime", (), {"values": ["2024-03-01", "2024-03-13"]})()

            def median(self, dim):
                self.median_dim = dim
                return FailingMedianResult()

        fake_ds = FakeDataset()
        good_candidate = GoodCandidate()
        concat_result = FakeConcatResult(good_candidate)

        with patch.object(
            acquisition,
            "_compute_usable_acquisitions",
            return_value=([good_candidate], ["2024-03-13T00:00:00"], ["2024-03-01T00:00:00"]),
        ):
            with patch.object(acquisition, "_get_valid_pixel_fraction", return_value=0.75):
                with patch("xarray.concat", return_value=concat_result) as mock_concat:
                    selected_ds, selected_times, valid_pixel_fraction = acquisition._select_acquisitions(
                        fake_ds,
                        target_date="20240320",
                        selection_strategy="composite",
                    )

        self.assertIs(selected_ds, good_candidate)
        self.assertEqual(selected_times, ["2024-03-13T00:00:00"])
        self.assertEqual(valid_pixel_fraction, 0.75)
        mock_concat.assert_called_once_with([good_candidate], dim="time")

    def test_select_acquisitions_composite_uses_lazy_median_when_it_succeeds(self):
        class GoodComposite:
            def __init__(self):
                self.data_vars = {"band": object()}

        class FakeMedianResult:
            def __init__(self, composite):
                self._composite = composite

            def squeeze(self):
                return self

            def compute(self):
                return self._composite

        class FakeDataset:
            def __init__(self, composite):
                self.time = type("FakeTime", (), {"values": ["2024-03-01", "2024-03-13"]})()
                self._composite = composite
                self.median_dim = None

            def median(self, dim):
                self.median_dim = dim
                return FakeMedianResult(self._composite)

        composite = GoodComposite()
        fake_ds = FakeDataset(composite)

        with patch.object(acquisition, "_compute_usable_acquisitions") as mock_compute_usable:
            with patch.object(acquisition, "_get_valid_pixel_fraction", return_value=0.6):
                selected_ds, selected_times, valid_pixel_fraction = acquisition._select_acquisitions(
                    fake_ds,
                    target_date="20240320",
                    selection_strategy="composite",
                )

        self.assertIs(selected_ds, composite)
        self.assertEqual(
            selected_times,
            ["2024-03-01T00:00:00", "2024-03-13T00:00:00"],
        )
        self.assertEqual(valid_pixel_fraction, 0.6)
        self.assertEqual(fake_ds.median_dim, "time")
        mock_compute_usable.assert_not_called()

    def test_acquire_prediction_inputs_expands_only_missing_source_buffer(self):
        class FakeAOIGDF:
            total_bounds = (-108.2, 37.55, -108.0, 37.75)

            @staticmethod
            def estimate_utm_crs():
                return "EPSG:32612"

        with patch.object(acquisition, "build_aoi_geometry", return_value={"type": "Polygon"}):
            with patch.object(acquisition, "build_aoi_geodataframe", return_value=FakeAOIGDF()):
                with patch.object(acquisition, "create_stac_client", return_value=object()):
                    with patch.object(
                        acquisition,
                        "load_sentinel1_dataset",
                        side_effect=[
                            ("snowon", {"source": "s1-on"}),
                            ("snowoff", {"source": "s1-off"}),
                        ],
                    ) as mock_s1:
                        with patch.object(
                            acquisition,
                            "load_sentinel2_dataset",
                            side_effect=[
                                acquisition.EmptyAcquisitionError("No Sentinel-2 items found."),
                                ("s2", {"source": "s2"}),
                            ],
                        ) as mock_s2:
                            with patch.object(acquisition, "load_snodas_dataset", return_value=("snodas", {"source": "snodas"})):
                                with patch.object(acquisition, "load_cop30_dataset", return_value=("cop30", {"source": "cop30"})):
                                    with patch.object(acquisition, "load_fcf_dataset", return_value=("fcf", {"source": "fcf"})):
                                        raw_inputs, crs, input_provenance = acquisition.acquire_prediction_inputs(
                                            aoi={"minlon": -108.2, "minlat": 37.55, "maxlon": -108.0, "maxlat": 37.75},
                                            target_date="20240320",
                                            snowoff_date="20230910",
                                            buffer_period=6,
                                            cloud_cover=25,
                                            fcf_path="fcf.tif",
                                            max_buffer_expansions=2,
                                            buffer_expansion_step_days=2,
                                        )

        self.assertEqual(crs, "EPSG:32612")
        self.assertEqual(raw_inputs["snowon_s1"], "snowon")
        self.assertEqual(raw_inputs["snowoff_s1"], "snowoff")
        self.assertEqual(raw_inputs["s2"], "s2")
        self.assertEqual(mock_s1.call_count, 2)
        self.assertEqual(mock_s2.call_count, 2)
        self.assertEqual(mock_s1.call_args_list[0].args[2], "2024-03-14/2024-03-26")
        self.assertEqual(mock_s1.call_args_list[1].args[2], "2023-09-04/2023-09-16")
        self.assertEqual(mock_s2.call_args_list[0].args[2], "2024-03-14/2024-03-26")
        self.assertEqual(mock_s2.call_args_list[1].args[2], "2024-03-12/2024-03-28")
        self.assertEqual(
            input_provenance["sentinel1_snowon"]["attempted_buffer_period_days"],
            [6],
        )
        self.assertEqual(
            input_provenance["sentinel1_snowoff"]["attempted_buffer_period_days"],
            [6],
        )
        self.assertEqual(
            input_provenance["sentinel2_snowon"]["attempted_buffer_period_days"],
            [6, 8],
        )
