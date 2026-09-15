# Copyright (C) 2026 Xiaomi Corporation.
"""One model worker and concurrent clients. Pickle requires trusted clients.
Bind to loopback and use SSH forwarding; never expose this port publicly.
"""
import argparse
from concurrent.futures import Future
import logging
import pickle
import queue
import socket
import struct
import threading
import time

LOG = logging.getLogger(__name__)
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def recv_all(conn, length):
    data = bytearray()
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            raise EOFError("Connection closed")
        data.extend(packet)
    return bytes(data)


class Server:
    def __init__(self, model_path=None, host="localhost", port=10086,
                 max_batch_size=16, batch_wait_ms=10, max_queue_size=256,
                 max_clients=128, request_timeout=300, policy=None):
        if min(max_batch_size, max_queue_size, max_clients) < 1 or batch_wait_ms < 0 or request_timeout <= 0:
            raise ValueError("Invalid server limits")
        self.host, self.port = host, port
        self.max_batch_size = max_batch_size
        self.batch_wait = batch_wait_ms / 1000
        self.request_timeout = request_timeout
        if policy is None:
            from batching import XR1BatchPolicy
            policy = XR1BatchPolicy(model_path)
        self.policy = policy
        self.requests = queue.Queue(maxsize=max_queue_size)
        self.slots = threading.BoundedSemaphore(max_clients)
        self.stopped = threading.Event()
        self.ready = threading.Event()
        self.connections = set()
        self.connections_lock = threading.Lock()

    def _infer_loop(self):
        while not self.stopped.is_set():
            try:
                first = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + self.batch_wait
            while len(batch) < self.max_batch_size:
                try:
                    item = self.requests.get_nowait()
                except queue.Empty:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = self.requests.get(timeout=remaining)
                    except queue.Empty:
                        break
                batch.append(item)
            batch = [(data, future) for data, future in batch
                     if future.set_running_or_notify_cancel()]
            if not batch:
                continue
            start = time.monotonic()
            try:
                outputs = self.policy([data for data, _ in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError("Policy returned incorrect batch length")
                for (_, future), output in zip(batch, outputs):
                    future.set_result(output)
                LOG.info("batch_size=%d inference_ms=%.1f queue_depth=%d", len(batch),
                         (time.monotonic() - start) * 1000, self.requests.qsize())
            except Exception as exc:
                LOG.exception("Inference batch failed")
                for _, future in batch:
                    future.set_exception(exc)

    def _handle(self, conn):
        future = None
        try:
            conn.settimeout(self.request_timeout)
            while not self.stopped.is_set():
                length = struct.unpack(">I", recv_all(conn, 4))[0]
                if not 0 < length <= MAX_MESSAGE_BYTES:
                    raise ValueError("Invalid request size")
                data = pickle.loads(recv_all(conn, length))
                try:
                    self.policy.validate(data)
                    future = Future()
                    self.requests.put_nowait((data, future))
                    result = future.result(timeout=self.request_timeout)
                except Exception as exc:
                    if future is not None:
                        future.cancel()
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                response = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
                conn.sendall(struct.pack(">I", len(response)) + response)
        except (EOFError, OSError, ValueError, pickle.UnpicklingError):
            pass
        finally:
            if future is not None:
                future.cancel()
            conn.close()
            with self.connections_lock:
                self.connections.discard(conn)
            self.slots.release()

    def serve(self):
        threading.Thread(target=self._infer_loop, daemon=True).start()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((self.host, self.port))
                self.port = listener.getsockname()[1]
                listener.listen(128)
                listener.settimeout(0.2)
                self.ready.set()
                LOG.info("Ready on %s:%d; max_batch_size=%d wait_ms=%.1f",
                         self.host, self.port, self.max_batch_size, self.batch_wait * 1000)
                while not self.stopped.is_set():
                    try:
                        conn, _ = listener.accept()
                    except socket.timeout:
                        continue
                    if not self.slots.acquire(blocking=False):
                        conn.close()
                        continue
                    with self.connections_lock:
                        self.connections.add(conn)
                    threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
        finally:
            self.close()

    def close(self):
        self.stopped.set()
        with self.connections_lock:
            for conn in self.connections:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        while True:
            try:
                _, future = self.requests.get_nowait()
                future.cancel()
            except queue.Empty:
                break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--batch-wait-ms", type=float, default=10)
    parser.add_argument("--max-queue-size", type=int, default=256)
    parser.add_argument("--max-clients", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = vars(parse_args())
    args["model_path"] = args.pop("model")
    server = Server(**args)
    try:
        server.serve()
    except KeyboardInterrupt:
        server.close()
