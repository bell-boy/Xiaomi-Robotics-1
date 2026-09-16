import importlib.util
from pathlib import Path
import unittest

try:
    import torch  # noqa: F401  the conversion script imports torch at module level
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch and transformers are only installed on the GPU host')
class TestKeyPartition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            'conversion', Path(__file__).resolve().parents[1]/'scripts/convert_base_robocasa365.py')
        cls.conversion = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.conversion)

    def test_base_only_and_training_only_tensors_are_recorded_separately(self):
        base = ['vlm.model.language_model.embed_tokens.weight',
                'state_projector_choice.layers.0.weight',
                'action_projector_choice.1.layers.0.weight',
                'score_projector_choice.1.layers.0.weight',
                'vlm.model.action_embed.weight',
                'vlm.model.score_embed.weight']
        expected = ['vlm.model.language_model.embed_tokens.weight']
        groups = self.conversion.partition_keys(base, expected)
        self.assertEqual(groups['base_only'], ['vlm.model.action_embed.weight', 'vlm.model.score_embed.weight'])
        self.assertEqual(groups['training_only'], ['action_projector_choice.1.layers.0.weight',
                                                   'score_projector_choice.1.layers.0.weight',
                                                   'state_projector_choice.layers.0.weight'])
        self.assertEqual(groups['unexpected'], [])
        self.assertEqual(groups['missing'], [])

    def test_unknown_base_tensor_still_fails_the_audit(self):
        groups = self.conversion.partition_keys(['vlm.model.mystery.weight'], [])
        self.assertEqual(groups['unexpected'], ['vlm.model.mystery.weight'])

    def test_missing_inference_tensor_is_reported(self):
        groups = self.conversion.partition_keys([], ['vlm.model.language_model.embed_tokens.weight'])
        self.assertEqual(groups['missing'], ['vlm.model.language_model.embed_tokens.weight'])


if __name__ == '__main__':
    unittest.main()
