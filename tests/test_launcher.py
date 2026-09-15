import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('launcher', Path(__file__).resolve().parents[1] / 'scripts/launch_robocasa_batched.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class TestShards(unittest.TestCase):
    def test_episodes_are_unique_complete_and_can_fill_32_workers(self):
        jobs = launcher.make_jobs(launcher.TASKS, 100, 5, 0)
        self.assertGreaterEqual(len(jobs), 32)
        for task in launcher.TASKS:
            seeds = [s for name, start, count in jobs if name == task for s in range(start, start + count)]
            self.assertEqual(seeds, list(range(100)))

    def test_partial_final_shard_and_seed_offset(self):
        self.assertEqual(launcher.make_jobs(['OpenDrawer'], 7, 3, 10),
                         [('OpenDrawer', 10, 3), ('OpenDrawer', 13, 3), ('OpenDrawer', 16, 1)])


if __name__ == '__main__':
    unittest.main()
