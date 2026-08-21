"""Small shared helpers for the two P2P node types (SwarmNode in swarm.py,
InferenceNode in inference_swarm.py) that were independently, identically
duplicated in both files. Deliberately minimal: the two nodes' start/stop
lifecycles and request handlers differ enough (different HTTP endpoint
sets, different server attribute names, different print prefixes) that
forcing them through one base class would trade a little duplication for a
worse abstraction. This just removes the pieces that were genuinely
byte-for-byte the same.
"""

from __future__ import annotations

import socket


def detect_local_ip() -> str:
    """Fast, offline, LAN-scoped guess (which local interface the OS would
    route through) -- NOT the real internet-facing address behind NAT. Used
    as the immediate default so node construction stays cheap and
    network-free; start(attempt_nat_traversal=True) replaces this with the
    real public address via STUN when the caller actually wants the node to
    be internet-reachable."""
    for af, probe in [(socket.AF_INET, "8.8.8.8"), (socket.AF_INET6, "2001:4860:4860::8888")]:
        try:
            s = socket.socket(af, socket.SOCK_DGRAM)
            s.connect((probe, 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except OSError:
            continue
    return "127.0.0.1"
