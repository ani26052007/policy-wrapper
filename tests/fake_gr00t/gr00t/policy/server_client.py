"""ZMQ REQ/REP + msgpack, numpy arrays as ``{"__ndarray_class__": True, "as_npy": <np.save bytes>}``.

Request: ``{"endpoint": name, "data": {...}}`` (no ``data`` for ping / kill). Reply: the handler's
return value, or ``{"error": message}``."""

import io

import msgpack
import numpy as np
import zmq


def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class MsgSerializer:
    @staticmethod
    def to_bytes(data):
        return msgpack.packb(data, default=_encode)

    @staticmethod
    def from_bytes(data):
        return msgpack.unpackb(data, object_hook=_decode)


class PolicyServer:
    def __init__(self, policy, host="*", port=5555, api_token=None):
        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self.requests = []
        self.handlers = {
            "ping": lambda: {"status": "ok"},
            "kill": self._kill,
            "get_action": lambda observation, options=None: list(policy.get_action(observation, options)),
            "reset": lambda options=None: policy.reset(options),
            "get_modality_config": lambda: {},
        }

    def _kill(self):
        self.running = False
        return {"status": "killed"}

    def run(self):
        try:
            while self.running:
                request = MsgSerializer.from_bytes(self.socket.recv())
                self.requests.append(request)
                try:
                    reply = self.handlers[request.get("endpoint", "get_action")](**request.get("data", {}) or {})
                except Exception as e:
                    reply = {"error": f"{type(e).__name__}: {e}"}
                self.socket.send(MsgSerializer.to_bytes(reply))
        finally:
            self.socket.close(linger=0)
            self.context.term()


class PolicyClient:
    def __init__(self, host="localhost", port=5555, timeout_ms=15000, api_token=None, strict=False):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{host}:{port}")

    def call_endpoint(self, endpoint, data=None, requires_input=True):
        request = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        self.socket.send(MsgSerializer.to_bytes(request))
        response = MsgSerializer.from_bytes(self.socket.recv())
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def ping(self):
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            return False

    def kill_server(self):
        self.call_endpoint("kill", requires_input=False)

    def get_action(self, observation, options=None):
        return tuple(self.call_endpoint("get_action", {"observation": observation, "options": options}))

    def reset(self, options=None):
        return self.call_endpoint("reset", {"options": options})
