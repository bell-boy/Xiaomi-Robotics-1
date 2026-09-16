import hashlib
import http.server
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest

spec = importlib.util.spec_from_file_location(
    'download_parallel', Path(__file__).resolve().parents[1]/'scripts/download_parallel.py')
dp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp)

PAYLOAD = bytes(range(256))*4096*4  # 4 MiB, deterministic and incompressible enough


class RangeHandler(http.server.BaseHTTPRequestHandler):
    requests = 0
    payload = PAYLOAD

    def do_GET(self):
        type(self).requests += 1
        header = self.headers.get('Range')
        if not header:
            start, end = 0, len(self.payload)-1
        else:
            start, end = (int(part) for part in header.split('=')[1].split('-'))
        body = self.payload[start:end+1]
        self.send_response(206 if header else 200)
        self.send_header('Content-Range', f'bytes {start}-{end}/{len(self.payload)}')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ParallelDownloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), RangeHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/payload.bin'
        cls.digest = hashlib.sha256(PAYLOAD).hexdigest()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_blocks_cover_the_file_without_gaps_or_overlap(self):
        ranges = dp.plan_blocks(10, block=4)
        self.assertEqual(ranges, [(0, 3), (4, 7), (8, 9)])

    def test_thread_count_scales_with_file_size(self):
        self.assertEqual(dp.threads_for(1_000), 8)
        self.assertEqual(dp.threads_for(4_000_000_000), 32)
        self.assertEqual(dp.threads_for(4_000_000_000, requested=4), 4)

    def test_download_assembles_and_verifies(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'payload.bin'
            report = dp.download(self.url, output, size=len(PAYLOAD), sha256_hex=self.digest,
                                 threads=4, block=512*1024, progress=None)
            self.assertEqual(output.read_bytes(), PAYLOAD)
            self.assertEqual(report['sha256'], self.digest)
            self.assertFalse((Path(d)/'payload.bin.blocks').exists())

    def test_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, 'Checksum mismatch'):
                dp.download(self.url, Path(d)/'p.bin', sha256_hex='0'*64, threads=2,
                            block=1024*1024, progress=None)

    def test_complete_blocks_are_reused_without_a_request(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/'000000000000.part'
            target.write_bytes(PAYLOAD[:1024*1024])
            before = RangeHandler.requests
            dp.fetch_block(self.url, 0, 1024*1024-1, len(PAYLOAD), target)
            self.assertEqual(RangeHandler.requests, before)


if __name__ == '__main__':
    unittest.main()
