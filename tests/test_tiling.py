import json
import unittest

from deep_snow.tiling import build_matrix_json
from deep_snow.workflows import build_time_series_jobs


class TilingTests(unittest.TestCase):
    def test_build_time_series_jobs_pairs_target_and_snow_off_dates(self):
        jobs = build_time_series_jobs("20240101", "20240125", "0901")

        self.assertEqual(
            jobs,
            [
                {"target_date": "20240125", "snow_off_date": "20230901"},
                {"target_date": "20240113", "snow_off_date": "20230901"},
                {"target_date": "20240101", "snow_off_date": "20230901"},
            ],
        )

    def test_build_matrix_json_wraps_items_for_github_actions(self):
        matrix_json = build_matrix_json([{"name": "tile-1"}])

        self.assertEqual(json.loads(matrix_json), {"include": [{"name": "tile-1"}]})
