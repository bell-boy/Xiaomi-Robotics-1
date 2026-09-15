import unittest
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy'))
from batching import collate, seeded_noise


class TestCollation(unittest.TestCase):
    def test_mixed_lengths_and_flattened_images(self):
        rows = []
        for i, length in enumerate([3, 5]):
            rows.append(dict(input_ids=torch.full((1, length), i + 1),
                             attention_mask=torch.ones(1, length, dtype=torch.long),
                             pixel_values=torch.full((6, 4), i),
                             image_grid_thw=torch.full((3, 3), i),
                             state=torch.full((1, 1, 60), i),
                             action_mask=torch.ones(1, 10, 7)))
        batch = collate(rows, 99)
        self.assertEqual(batch['input_ids'].tolist(), [[99, 99, 1, 1, 1], [2, 2, 2, 2, 2]])
        self.assertEqual(batch['attention_mask'][0].tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(tuple(batch['pixel_values'].shape), (12, 4))
        self.assertTrue(torch.equal(batch['pixel_values'][6:], rows[1]['pixel_values']))
        self.assertEqual(batch['state'][:, 0, 0].tolist(), [0, 1])

    def test_noise_independent_of_batch_order_and_global_rng(self):
        mask = torch.ones(3, 10, 7)
        before = torch.get_rng_state()
        noise = seeded_noise(mask, [42, 7, 42])
        reverse = seeded_noise(mask, [7, 42, 42])
        self.assertTrue(torch.equal(noise[0], noise[2]))
        self.assertTrue(torch.equal(noise[0], reverse[1]))
        self.assertTrue(torch.equal(noise[:1], seeded_noise(mask[:1], [42])))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
