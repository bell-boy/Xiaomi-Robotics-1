import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('comparison', Path(__file__).resolve().parents[1]/'scripts/summarize_robocasa365_comparison.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class TestComparisonCoverage(unittest.TestCase):
    def test_incomplete_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                comparison.load_run(Path(d), ['Task'], 2)

    def test_missing_or_duplicate_episode_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'DONE').touch()
            (p/'results').mkdir()
            (p/'results/summary.json').write_text(json.dumps({'tasks': {'Task': {
                'num_episodes': 2, 'episodes': [{'episode': 0}, {'episode': 0}]}}}))
            with self.assertRaisesRegex(ValueError, 'Duplicate/missing'):
                comparison.load_run(p, ['Task'], 2)

    def test_missing_video_prevents_complete_report(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'DONE').touch()
            (p/'results').mkdir()
            (p/'results/summary.json').write_text(json.dumps({'tasks': {'Task': {
                'num_episodes': 1, 'episodes': [{'episode': 0, 'seed': 7, 'success': True}]}}}))
            with self.assertRaisesRegex(ValueError, 'Missing video'):
                comparison.load_run(p, ['Task'], 1)


if __name__ == '__main__':
    unittest.main()
