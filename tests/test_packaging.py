import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PackagingSmokeTests(unittest.TestCase):
    def test_non_editable_install_exposes_packaged_default_assets_outside_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            install_root = tmp_path / "install"
            install_root.mkdir()
            workdir = tmp_path / "workdir"
            workdir.mkdir()

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--target",
                    str(install_root),
                    str(REPO_ROOT),
                ],
                check=True,
                cwd=workdir,
            )

            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                str(install_root)
                if not existing_pythonpath
                else os.pathsep.join([str(install_root), existing_pythonpath])
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "import deep_snow; "
                        "from deep_snow.workflows import DEFAULT_SINGLE_MODEL_PATH, DEFAULT_TILE_LAND_PATH, DEFAULT_HILL_PPTWT_PATH, DEFAULT_HILL_TD_PATH; "
                        "print(Path(deep_snow.__file__).resolve()); "
                        "print(Path(DEFAULT_SINGLE_MODEL_PATH).exists()); "
                        "print(Path(DEFAULT_SINGLE_MODEL_PATH).resolve()); "
                        "print(Path(DEFAULT_TILE_LAND_PATH).exists()); "
                        "print(Path(DEFAULT_TILE_LAND_PATH).resolve()); "
                        "print(Path(DEFAULT_HILL_PPTWT_PATH).name); "
                        "print(Path(DEFAULT_HILL_PPTWT_PATH).resolve()); "
                        "print(Path(DEFAULT_HILL_TD_PATH).name); "
                        "print(Path(DEFAULT_HILL_TD_PATH).resolve())"
                    ),
                ],
                check=True,
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
            )

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        module_path = Path(lines[0])
        single_model_exists = lines[1] == "True"
        single_model_path = Path(lines[2])
        land_path_exists = lines[3] == "True"
        land_path = Path(lines[4])
        hill_pptwt_name = lines[5]
        hill_pptwt_path = Path(lines[6])
        hill_td_name = lines[7]
        hill_td_path = Path(lines[8])

        self.assertTrue(single_model_exists)
        self.assertTrue(land_path_exists)
        self.assertTrue(module_path.is_relative_to(install_root))
        self.assertTrue(single_model_path.is_relative_to(install_root))
        self.assertTrue(land_path.is_relative_to(install_root))
        self.assertEqual(land_path.name, "ne_50m_land.shp")
        self.assertEqual(hill_pptwt_name, "ppt_wt_final.txt")
        self.assertEqual(hill_td_name, "td_final.txt")
        self.assertTrue(hill_pptwt_path.name.endswith("ppt_wt_final.txt"))
        self.assertTrue(hill_td_path.name.endswith("td_final.txt"))
