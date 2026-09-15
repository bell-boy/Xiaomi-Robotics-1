import pickle
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy'))
from client import Client


class FragmentedSocket:
    def __init__(self, payload):
        self.payload = bytearray(payload)

    def recv(self, count):
        # Both header and body arrive one byte at a time.
        result = bytes(self.payload[:1])
        del self.payload[:1]
        return result

    def sendall(self, payload):
        pass


class TestClientProtocol(unittest.TestCase):
    def client(self, value):
        payload = pickle.dumps(value)
        client = Client.__new__(Client)
        client.client_socket = FragmentedSocket(struct.pack('>I', len(payload)) + payload)
        return client

    def test_fragmented_header_and_body(self):
        self.assertEqual(self.client([1, 2, 3])._recv_with_length_prefix(), [1, 2, 3])

    def test_server_error_is_raised_before_action_decode(self):
        with self.assertRaisesRegex(RuntimeError, 'queue full'):
            self.client({'error': 'queue full'})(task_id='robocasa_mg')

    def test_truncated_header_fails(self):
        client = Client.__new__(Client)
        client.client_socket = FragmentedSocket(b'\x00\x00')
        with self.assertRaises(ConnectionError):
            client._recv_with_length_prefix()


if __name__ == '__main__':
    unittest.main()
