import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class GitHubWorkflowTests(unittest.TestCase):
    def test_time_series_workflow_declares_and_forwards_selection_inputs(self):
        workflow_path = REPO_ROOT / ".github" / "workflows" / "batch_predict_ts.yml"
        workflow_text = workflow_path.read_text()

        self.assertIn("selection_strategy:", workflow_text)
        self.assertIn("s1_orbit_selection:", workflow_text)
        self.assertIn("selection_strategy: ${{ inputs.selection_strategy }}", workflow_text)
        self.assertIn("s1_orbit_selection: ${{ inputs.s1_orbit_selection }}", workflow_text)

    def test_workflows_declare_and_forward_predict_swe(self):
        time_series_text = (REPO_ROOT / ".github" / "workflows" / "batch_predict_ts.yml").read_text()
        batch_text = (REPO_ROOT / ".github" / "workflows" / "batch_predict_sd.yml").read_text()
        tile_text = (REPO_ROOT / ".github" / "workflows" / "predict_tile_sd.yml").read_text()

        self.assertIn("predict_swe:", time_series_text)
        self.assertIn("predict_swe: ${{ inputs.predict_swe }}", time_series_text)
        self.assertIn("predict_swe:", batch_text)
        self.assertIn("predict_swe: ${{ inputs.predict_swe }}", batch_text)
        self.assertIn("predict_swe:", tile_text)
        self.assertIn("--predict-swe ${{ inputs.predict_swe }}", tile_text)
        self.assertIn("data/*_swe.tif", tile_text)
        self.assertIn("data/*_density.tif", tile_text)
