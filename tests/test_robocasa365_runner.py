import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'runner', Path(__file__).resolve().parents[1]/'scripts/run_robocasa365_comparison.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TestRunPlan(unittest.TestCase):
    def plan(self, argv):
        with tempfile.TemporaryDirectory() as d:
            args = runner.parse_args(argv+['--evaluations-root', d, '--dry-run'])
            args.run_tag = args.run_tag or f'{args.task_set}-{args.episodes_per_task}'
            root, stages = runner.plan(args)
            return args, root, {name: command for name, _, command in stages}

    def test_episode_count_is_shared_by_every_stage(self):
        args, root, stages = self.plan(['--episodes-per-task', '25'])
        self.assertEqual(args.episodes_per_task, 25)
        for name in ('trained', 'base'):
            index = stages[name].index('--num-trials')
            self.assertEqual(stages[name][index+1], '25')
        index = stages['summary'].index('--trials')
        self.assertEqual(stages['summary'][index+1], '25')
        self.assertEqual(root.name, 'atomic_seen-25')

    def test_task_subset_and_tag_flow_through(self):
        args, root, stages = self.plan(
            ['--episodes-per-task', '4', '--task-name', 'CloseBlenderLid', '--horizon', '32'])
        self.assertIn('--horizon', stages['trained'])
        self.assertEqual(stages['trained'][stages['trained'].index('--horizon')+1], '32')
        self.assertIn('CloseBlenderLid', stages['trained'])
        self.assertEqual(stages['summary'][stages['summary'].index('--groups')+1], 'atomic_seen')

    def test_single_policy_run_skips_the_comparison(self):
        _, _, stages = self.plan(['--policies', 'trained'])
        self.assertEqual(set(stages), {'trained'})


if __name__ == '__main__':
    unittest.main()
