"""Socket integration tests run without CUDA or model weights."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pickle
import socket
import struct
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy'))
from server import Server, recv_all


class FakePolicy:
    def __init__(self):
        self.batches = []
        self.thread_ids = set()

    def validate(self, data):
        if not isinstance(data, int):
            raise ValueError('Expected integer')

    def __call__(self, requests):
        self.thread_ids.add(threading.get_ident())
        self.batches.append(requests)
        if -1 in requests:
            raise RuntimeError('deliberate failure')
        return [r * 10 for r in requests]


def request(conn, value, fragment=False):
    payload = pickle.dumps(value)
    header = struct.pack('>I', len(payload))
    if fragment:
        for byte in header:
            conn.sendall(bytes([byte]))
    else:
        conn.sendall(header)
    conn.sendall(payload)
    size = struct.unpack('>I', recv_all(conn, 4))[0]
    return pickle.loads(recv_all(conn, size))


class TestServer(unittest.TestCase):
    def setUp(self):
        self.policy = FakePolicy()
        self.server = Server(policy=self.policy, host='127.0.0.1', port=0,
                             max_batch_size=32, batch_wait_ms=50)
        self.thread = threading.Thread(target=self.server.serve)
        self.thread.start()
        self.assertTrue(self.server.ready.wait(3))

    def tearDown(self):
        self.server.close()
        self.thread.join(3)
        self.assertFalse(self.thread.is_alive())

    def connect(self):
        return socket.create_connection(('127.0.0.1', self.server.port), timeout=5)

    def test_32_clients_correct_routing_and_persistent_connections(self):
        barrier = threading.Barrier(32)
        def client(i):
            with self.connect() as conn:
                barrier.wait(5)
                self.assertEqual(request(conn, i, fragment=True), i * 10)
                barrier.wait(5)
                self.assertEqual(request(conn, i + 100), (i + 100) * 10)
        with ThreadPoolExecutor(32) as pool:
            list(pool.map(client, range(32)))
        self.assertTrue(any(len(b) > 1 for b in self.policy.batches))
        self.assertTrue(all(len(b) <= 32 for b in self.policy.batches))
        self.assertEqual(len(self.policy.thread_ids), 1)

    def test_partial_batch_flush_and_error_recovery(self):
        with self.connect() as conn:
            start = time.monotonic()
            self.assertEqual(request(conn, 3), 30)
            self.assertLess(time.monotonic() - start, 1)
            self.assertIn('error', request(conn, 'invalid'))
            self.assertIn('error', request(conn, -1))
            self.assertEqual(request(conn, 4), 40)

    def test_disconnect_does_not_block_other_clients(self):
        with self.connect() as conn:
            conn.sendall(b'\x00\x01')
        with self.connect() as conn:
            self.assertEqual(request(conn, 7), 70)


if __name__ == '__main__':
    unittest.main()
