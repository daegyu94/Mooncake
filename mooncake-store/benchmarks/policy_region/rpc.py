"""Local TCP control-plane transport with measured framed MessagePack bytes.

Persistent connections and the same codec are used by all architecture variants.
Wire counters exclude TCP/IP headers/retransmissions, hello and diagnostics.
"""

import multiprocessing as mp
import resource
import socket
import socketserver
import struct
import threading
import time

import msgpack
from coordinator import Coordinator, StoreError

LIMIT = 64 << 20


def receive(sock, size):
    output = bytearray()
    while len(output) < size:
        part = sock.recv(size - len(output))
        if not part:
            raise EOFError("connection closed")
        output.extend(part)
    return bytes(output)


def frame(sock):
    size = struct.unpack("!I", receive(sock, 4))[0]
    if size > LIMIT:
        raise ValueError("frame too large")
    return receive(sock, size)


def encode(value):
    data = msgpack.packb(value, use_bin_type=True)
    if len(data) > LIMIT:
        raise ValueError("frame too large")
    return struct.pack("!I", len(data)) + data


class RPC:
    def __init__(self, address):
        self.socket = socket.create_connection(address, timeout=30)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.stats = {"rpcs": 0, "tx_bytes": 0, "rx_bytes": 0}
        self.boot, self.mode = self.call("hello")

    def call(self, op, **args):
        if op != "hello":
            args["boot"] = self.boot
        request = encode([op, args])
        self.socket.sendall(request)
        raw = frame(self.socket)
        if op not in ("hello", "stats"):
            self.stats["rpcs"] += 1
            self.stats["tx_bytes"] += len(request)
            self.stats["rx_bytes"] += 4 + len(raw)
        ok, result = msgpack.unpackb(raw, raw=False)
        if not ok:
            raise StoreError(result)
        return result

    def close(self):
        self.socket.close()


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            try:
                raw = frame(self.request)
            except EOFError:
                return
            cpu = time.thread_time_ns()
            op, args = msgpack.unpackb(raw, raw=False)
            try:
                result = self.server.coordinator.dispatch(op, args)
                if op == "stats":
                    with self.server.counter_lock:
                        result["transport"] = self.server.counters.copy()
                    result["peak_rss_kib"] = resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                reply = encode([True, result])
            except StoreError as error:
                reply = encode([False, str(error)])
            if op not in ("hello", "stats"):
                with self.server.counter_lock:
                    self.server.counters["rpcs"] += 1
                    self.server.counters["rx_bytes"] += len(raw) + 4
                    self.server.counters["tx_bytes"] += len(reply)
                    self.server.counters["cpu_ns"] += time.thread_time_ns() - cpu
            self.request.sendall(reply)


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False


def serve(mode, pipe):
    with Server(("127.0.0.1", 0), Handler) as server:
        server.coordinator = Coordinator(mode)
        server.counter_lock = threading.Lock()
        server.counters = {"rpcs": 0, "rx_bytes": 0, "tx_bytes": 0, "cpu_ns": 0}
        pipe.send(server.server_address)
        pipe.close()
        server.serve_forever()


class Peer:
    def __init__(self, mode):
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=False)
        self.process = ctx.Process(target=serve, args=(mode, child))
        self.process.start()
        child.close()
        if not parent.poll(20):
            self.close()
            raise TimeoutError("control server startup")
        self.address = parent.recv()
        parent.close()

    def close(self):
        # The child owns only prototype metadata/sockets, never data-plane IO.
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(5)
        if self.process.is_alive():
            raise RuntimeError("control server did not stop")
