import hashlib
import hmac
import json
import secrets


def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def generate_nonce():
    return secrets.token_urlsafe(12)


def generate_client_secret():
    return secrets.token_urlsafe(32)


def sign_payload(secret, payload):
    msg = canonical_json(payload).encode("utf-8")
    key = secret.encode("utf-8")
    return hmac.new(key=key, msg=msg, digestmod=hashlib.sha256).hexdigest()


def verify_signature(secret, payload, signature):
    expected = sign_payload(secret, payload)
    return hmac.compare_digest(expected, signature)


def sign_request(secret, method, path, client_id, nonce, timestamp):
    payload = canonical_json({
        "method": method,
        "path": path,
        "client_id": client_id,
        "nonce": nonce,
        "timestamp": timestamp,
    })
    msg = payload.encode("utf-8")
    key = secret.encode("utf-8")
    return hmac.new(key=key, msg=msg, digestmod=hashlib.sha256).hexdigest()


def verify_request_signature(secret, method, path, client_id, nonce, timestamp, signature):
    expected = sign_request(secret, method, path, client_id, nonce, timestamp)
    return hmac.compare_digest(expected, signature)
