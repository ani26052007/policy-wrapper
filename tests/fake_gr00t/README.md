Test stand-ins for the `gr00t` package (Isaac-GR00T is not installed in the test environment).
Written for these tests; no GR00T code is copied. Only the names the client and `serve.py`
import exist. `server_client` speaks the GR00T ZMQ/msgpack wire format, so the client under test
talks to a real REP socket.
