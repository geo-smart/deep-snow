import gzip
import tarfile
import tempfile
import unittest
from pathlib import Path

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
