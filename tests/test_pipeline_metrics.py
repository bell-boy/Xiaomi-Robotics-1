import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'deploy'))
from server import Server
from test_batched_server import FakePolicy, request
spec = importlib.util.spec_from_file_location('pipeline', ROOT/'scripts/benchmark_robocasa_pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class TestPipelineMetrics(unittest.TestCase):
    def test_fingerprint_is_order_independent_and_detects_input_changes(self):
        tensor = torch.tensor([[1., 2.]], dtype=torch.bfloat16)
        a = dict(state=tensor, seed=42)
        b = dict(seed=42, state=tensor.clone())
        self.assertEqual(pipeline.fingerprint(a), pipeline.fingerprint(b))
        b['state'][0, 0] = 3
        self.assertNotEqual(pipeline.fingerprint(a), pipeline.fingerprint(b))

    def test_metrics_do_not_change_responses_and_timings_are_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'metrics.jsonl'
            server = Server(policy=FakePolicy(), host='127.0.0.1', port=0,
                            batch_wait_ms=10, metrics_jsonl=str(path))
            thread = threading.Thread(target=server.serve)
            thread.start()
            self.assertTrue(server.ready.wait(3))
            try:
                with socket.create_connection(('127.0.0.1', server.port), timeout=3) as conn:
                    self.assertEqual(request(conn, 7), 70)
                deadline = time.monotonic()+3
                while True:
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    if any(r['kind']=='queue_sample' for r in rows):
                        break
                    if time.monotonic()>deadline:
                        self.fail('No queue sampling event')
                    time.sleep(.01)
                batch = next(r for r in rows if r['kind']=='batch')
                req = next(r for r in rows if r['kind']=='request')
                self.assertEqual(batch['size'], 1)
                self.assertEqual(batch['request_ids'], [req['request_id']])
                self.assertEqual(req['batch_id'], batch['batch_id'])
                self.assertLessEqual(req['received_start'], req['enqueued_at'])
                self.assertLessEqual(req['enqueued_at'], batch['start'])
                self.assertLessEqual(batch['end'], req['sent_at'])
                self.assertGreaterEqual(batch['queue_wait_ms'][0], 0)
            finally:
                server.close()
                thread.join(3)
                server.metrics_file.close()


if __name__ == '__main__':
    unittest.main()
