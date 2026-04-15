import unittest

from deep_snow.inputs import (
    build_output_name,
    format_bounding_box,
    generate_dates,
    most_recent_occurrence,
    parse_bool,
    parse_bounding_box,
)


class InputsTests(unittest.TestCase):
    def test_parse_bounding_box_returns_expected_dict(self):
        aoi = parse_bounding_box("-108.2 37.55 -107.61 38.09")

        self.assertEqual(
            aoi,
            {
                "minlon": -108.2,
                "minlat": 37.55,
                "maxlon": -107.61,
                "maxlat": 38.09,
            },
        )

    def test_format_bounding_box_round_trips(self):
        aoi = {"minlon": -108.2, "minlat": 37.55, "maxlon": -107.61, "maxlat": 38.09}

        self.assertEqual(format_bounding_box(aoi), "-108.2 37.55 -107.61 38.09")

    def test_build_output_name_uses_compact_coordinates(self):
        aoi = {"minlon": -108.2, "minlat": 37.55, "maxlon": -107.61, "maxlat": 38.09}

        self.assertEqual(
            build_output_name("20240320", aoi),
            "20240320_deep-snow_-108.20_37.55_-107.61_38.09",
        )

    def test_generate_dates_uses_twelve_day_steps(self):
        self.assertEqual(
            generate_dates("20240125", "20240101"),
            ["20240125", "20240113", "20240101"],
        )

    def test_most_recent_occurrence_uses_previous_year_when_needed(self):
        self.assertEqual(most_recent_occurrence("20240101", "0901"), "20230901")

    def test_parse_bool_accepts_common_true_false_strings(self):
        self.assertTrue(parse_bool("True"))
        self.assertFalse(parse_bool("no"))
