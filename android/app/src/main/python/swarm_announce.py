import hashlib
import json
import time
import urllib.request
import urllib.error


def _bundle_dht_key(bundle_id):
    return hashlib.sha256(f"ickle:torickle:bundle:{bundle_id}".encode()).digest()[:20]


class BundleAnnouncement:
    def __init__(self, bundle_id, model_hash, piece_count, total_bytes,
                 payload_sha256, merkle_root, peer_id, host, port,
                 timestamp=None, signature="", use_tls=False):
        self.bundle_id = bundle_id
        self.model_hash = model_hash
        self.piece_count = piece_count
        self.total_bytes = total_bytes
        self.payload_sha256 = payload_sha256
        self.merkle_root = merkle_root
        self.peer_id = peer_id
        self.host = host
        self.port = port
        self.use_tls = use_tls
        self.timestamp = timestamp or time.time()
        self.signature = signature

    def to_dict(self):
        return {
            "bundle_id": self.bundle_id,
            "model_hash": self.model_hash,
            "piece_count": self.piece_count,
            "total_bytes": self.total_bytes,
            "payload_sha256": self.payload_sha256,
            "merkle_root": self.merkle_root,
            "peer_id": self.peer_id,
            "host": self.host,
            "port": self.port,
            "use_tls": self.use_tls,
            "timestamp": self.timestamp,
            "signature": self.signature,
        }

    def dht_key(self):
        return _bundle_dht_key(self.bundle_id)

    def sign(self, identity):
        payload = self.to_dict()
        payload.pop("signature", None)
        self.signature = identity.sign(payload)


def announce_to_swarm(announcement, swarm_host, swarm_port):
    url = f"http://{swarm_host}:{swarm_port}/torickle/v1/announce"
    body = json.dumps(announcement.to_dict()).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        exc.close()
        return {"accepted": False, "error": f"HTTP {exc.code}: {raw}"}
    except urllib.error.URLError as exc:
        return {"accepted": False, "error": str(exc)}
