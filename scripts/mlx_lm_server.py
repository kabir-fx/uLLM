"""``mlx_lm.server`` with uvicorn's listen queue of 2048 (macOS caps it at ``kern.ipc.somaxconn``,
128), so it gets the same queue as ``ullm serve``. Takes ``mlx_lm.server``'s arguments.

``socketserver`` listens with a queue of 5, and ``mlx_lm.server`` has no flag for it. While the
server is busy, macOS refuses the connections past the queue: the first ``mlx-lm-batched-c8`` run
lost 4 requests that way (upstream candidate 3)."""

from collections.abc import Callable
from http.server import ThreadingHTTPServer

from mlx_lm import server

LISTEN_QUEUE = 2048
# mlx-lm 0.32.2 leaves server.main unannotated.
serve: Callable[[], None] = server.main


def main() -> None:
    # server.run and _run_http_server take this class as a default argument, bound at import.
    ThreadingHTTPServer.request_queue_size = LISTEN_QUEUE
    serve()


if __name__ == "__main__":
    main()
