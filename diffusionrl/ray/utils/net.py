"""Network helpers used to discover the local node IP and a free TCP port.

These are the only stateless helpers the Ray actors actually call at runtime
(``BaseTrainRayActor.get_master_info`` and ``RolloutActor._setup_distributed_env``).
"""
import socket

import ray


def get_node_ip() -> str:
    """Return the IP address of the current node.

    Prefers Ray's canonical address (``ray.util.get_node_ip_address``) and
    falls back to a UDP-connect trick when Ray is not reachable. The UDP
    trick avoids the multi-NIC ``127.0.1.1`` surprise that
    ``socket.gethostbyname(socket.gethostname())`` can return on some Linux
    hosts.
    """
    try:
        return ray.util.get_node_ip_address()
    except Exception:
        pass

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_free_port(start_port: int = 10000, max_tries: int = 1000) -> int:
    """Find a free TCP port by bind-and-release, starting from ``start_port``."""
    for port in range(start_port, start_port + max_tries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", port))
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError(
        f"Could not find free port in range {start_port}-{start_port + max_tries}"
    )
