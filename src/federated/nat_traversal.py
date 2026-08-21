"""Real NAT traversal for the swarm's TCP listener: STUN-based public-IP
discovery (RFC 5389) plus best-effort UPnP IGD automatic port forwarding.

Why this exists: SwarmNode._detect_local_ip() (swarm.py) opens a UDP
socket toward a public IP purely to see which *local* interface the OS
would route through -- on any machine behind a home router that returns
the private LAN address (192.168.x.x/10.x.x.x), which is useless as the
"external_host" announced to other peers; nobody outside the LAN can reach
it. Two real, protocol-standard mechanisms close that gap without any
Ickle-specific hosted infrastructure:

- STUN (RFC 5389) asks a public STUN server "what IP/port did my packet
  come from" -- that's the actual internet-facing address. Public STUN
  servers are generic internet infrastructure, not something Ickle-specific:
  the same well-known servers (e.g. Google's) are the default in virtually
  every WebRTC/VoIP stack.
- UPnP IGD (Internet Gateway Device) lets software ask a *home router* --
  not any external server -- to forward an external port to this machine,
  automatically, the same mechanism game consoles and torrent clients use.
  Many routers ship with this on by default; when it's off or unsupported
  this fails safe (logged, non-fatal) and the node just stays as reachable
  as it already was.

Neither mechanism solves peer *discovery* (finding strangers with no prior
contact) -- that still needs a known bootstrap peer address (see
bootstrap_node.py). What this module fixes is peers who *do* know each
other's address still failing to connect because one of them is behind NAT.
"""
from __future__ import annotations

import re
import secrets
import socket
import struct
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape as _xml_escape

STUN_MAGIC_COOKIE = 0x2112A442
DEFAULT_STUN_SERVERS = [
    ("stun.l.google.com", 19302),
    ("stun1.l.google.com", 19302),
    ("stun.cloudflare.com", 3478),
]

_ATTR_XOR_MAPPED_ADDRESS = 0x0020
_ATTR_MAPPED_ADDRESS = 0x0001

SSDP_ADDR = ("239.255.255.250", 1900)
_IGD_SEARCH_TARGETS = (
    "urn:schemas-upnp-org:device:InternetGatewayDevice:1",
    "urn:schemas-upnp-org:device:InternetGatewayDevice:2",
)
_PORT_MAPPING_SERVICE_TYPES = (
    "urn:schemas-upnp-org:service:WANIPConnection:1",
    "urn:schemas-upnp-org:service:WANIPConnection:2",
    "urn:schemas-upnp-org:service:WANPPPConnection:1",
)


def stun_get_external_address(
    servers: list[tuple[str, int]] | None = None, *, timeout: float = 2.5
) -> tuple[str, int] | None:
    """Ask a public STUN server what IP:port this machine's outbound UDP
    traffic actually appears as -- the real internet-facing address, unlike
    a local-interface guess. Tries each server in turn; returns None if
    none answered (offline, UDP blocked by a firewall, etc.)."""
    for host, port in servers or DEFAULT_STUN_SERVERS:
        result = _stun_query_one(host, port, timeout=timeout)
        if result:
            return result
    return None


def _stun_query_one(host: str, port: int, *, timeout: float) -> tuple[str, int] | None:
    transaction_id = secrets.token_bytes(12)
    # Binding Request: type(2)=0x0001, length(2)=0, magic cookie(4), transaction id(12)
    request = struct.pack("!HHI12s", 0x0001, 0, STUN_MAGIC_COOKIE, transaction_id)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            sock.sendto(request, (host, port))
            data, _ = sock.recvfrom(2048)
    except (OSError, socket.timeout):
        return None
    return _parse_stun_response(data, transaction_id)


def _parse_stun_response(data: bytes, expected_transaction_id: bytes) -> tuple[str, int] | None:
    if len(data) < 20:
        return None
    msg_type, msg_len, magic_cookie, transaction_id = struct.unpack("!HHI12s", data[:20])
    if msg_type != 0x0101:  # Binding Success Response
        return None
    if magic_cookie != STUN_MAGIC_COOKIE or transaction_id != expected_transaction_id:
        return None

    offset = 20
    end = min(len(data), 20 + msg_len)
    mapped_address = None
    xor_mapped_address = None
    while offset + 4 <= end:
        attr_type, attr_len = struct.unpack("!HH", data[offset : offset + 4])
        value = data[offset + 4 : offset + 4 + attr_len]
        if attr_type == _ATTR_XOR_MAPPED_ADDRESS and len(value) >= 8:
            xor_mapped_address = _parse_xor_mapped_address(value)
        elif attr_type == _ATTR_MAPPED_ADDRESS and len(value) >= 8:
            mapped_address = _parse_mapped_address(value)
        # Attributes are padded to a 4-byte boundary.
        offset += 4 + attr_len + ((4 - attr_len % 4) % 4)
    return xor_mapped_address or mapped_address


def _parse_mapped_address(value: bytes) -> tuple[str, int] | None:
    _reserved, family = value[0], value[1]
    if family != 0x01:  # IPv4 only -- swarm's TCP listener is IPv4
        return None
    port = struct.unpack("!H", value[2:4])[0]
    ip = socket.inet_ntoa(value[4:8])
    return ip, port


def _parse_xor_mapped_address(value: bytes) -> tuple[str, int] | None:
    _reserved, family = value[0], value[1]
    if family != 0x01:
        return None
    xport = struct.unpack("!H", value[2:4])[0]
    port = xport ^ (STUN_MAGIC_COOKIE >> 16)
    xaddr = struct.unpack("!I", value[4:8])[0]
    addr = xaddr ^ STUN_MAGIC_COOKIE
    ip = socket.inet_ntoa(struct.pack("!I", addr))
    return ip, port


def _local_ip_for_router() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def _ssdp_discover(*, timeout: float = 2.0) -> str | None:
    """Broadcast an SSDP M-SEARCH for an Internet Gateway Device and return
    the LOCATION URL of the first router that answers, or None."""
    for search_target in _IGD_SEARCH_TARGETS:
        request = (
            "M-SEARCH * HTTP/1.1\r\n"
            f"HOST: {SSDP_ADDR[0]}:{SSDP_ADDR[1]}\r\n"
            'MAN: "ssdp:discover"\r\n'
            "MX: 2\r\n"
            f"ST: {search_target}\r\n\r\n"
        ).encode("utf-8")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(timeout)
                sock.sendto(request, SSDP_ADDR)
                data, _ = sock.recvfrom(4096)
        except (OSError, socket.timeout):
            continue
        match = re.search(rb"LOCATION:\s*(\S+)", data, re.IGNORECASE)
        if match:
            return match.group(1).decode("utf-8", errors="ignore").strip()
    return None


def _find_port_mapping_control_url(description_url: str, *, timeout: float) -> tuple[str, str] | None:
    """Fetch a router's UPnP device description XML and return
    (control_url, service_type) for whichever WAN connection service it
    advertises, or None if it doesn't expose port mapping."""
    try:
        with urllib.request.urlopen(description_url, timeout=timeout) as resp:
            xml_bytes = resp.read()
    except (urllib.error.URLError, OSError):
        return None

    base_match = re.match(r"(https?://[^/]+)", description_url)
    base_url = base_match.group(1) if base_match else description_url

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    ns = {"u": "urn:schemas-upnp-org:device-1-0"}
    for service in root.iter("{urn:schemas-upnp-org:device-1-0}service"):
        service_type = (service.findtext("u:serviceType", namespaces=ns) or "").strip()
        if service_type not in _PORT_MAPPING_SERVICE_TYPES:
            continue
        control_url = (service.findtext("u:controlURL", namespaces=ns) or "").strip()
        if not control_url:
            continue
        if control_url.startswith("http://") or control_url.startswith("https://"):
            full_url = control_url
        else:
            full_url = base_url + ("" if control_url.startswith("/") else "/") + control_url
        return full_url, service_type
    return None


def _soap_request(control_url: str, service_type: str, action: str, body_xml: str, *, timeout: float) -> bool:
    envelope = (
        '<?xml version="1.0"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        f"<s:Body><u:{action} xmlns:u=\"{service_type}\">{body_xml}</u:{action}></s:Body>"
        "</s:Envelope>"
    ).encode("utf-8")
    req = urllib.request.Request(
        control_url,
        data=envelope,
        method="POST",
        headers={
            "Content-Type": 'text/xml; charset="utf-8"',
            "SOAPAction": f'"{service_type}#{action}"',
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError:
        return False
    except (urllib.error.URLError, OSError):
        return False


def upnp_add_port_mapping(
    port: int, *, protocol: str = "TCP", description: str = "Ickle Swarm", timeout: float = 3.0
) -> bool:
    """Best-effort: ask the local router (via UPnP IGD) to forward `port`
    straight through to this machine. Returns False (never raises) on any
    failure -- UPnP being off, unsupported, or blocked is common and not
    fatal; the node just isn't auto-forwarded and stays as reachable as
    before."""
    local_ip = _local_ip_for_router()
    if not local_ip:
        return False
    description_url = _ssdp_discover(timeout=timeout)
    if not description_url:
        return False
    found = _find_port_mapping_control_url(description_url, timeout=timeout)
    if not found:
        return False
    control_url, service_type = found
    body = (
        "<NewRemoteHost></NewRemoteHost>"
        f"<NewExternalPort>{port}</NewExternalPort>"
        f"<NewProtocol>{protocol.upper()}</NewProtocol>"
        f"<NewInternalPort>{port}</NewInternalPort>"
        f"<NewInternalClient>{local_ip}</NewInternalClient>"
        "<NewEnabled>1</NewEnabled>"
        f"<NewPortMappingDescription>{_xml_escape(description)}</NewPortMappingDescription>"
        "<NewLeaseDuration>0</NewLeaseDuration>"
    )
    return _soap_request(control_url, service_type, "AddPortMapping", body, timeout=timeout)


def upnp_remove_port_mapping(port: int, *, protocol: str = "TCP", timeout: float = 3.0) -> bool:
    """Best-effort cleanup counterpart to upnp_add_port_mapping()."""
    description_url = _ssdp_discover(timeout=timeout)
    if not description_url:
        return False
    found = _find_port_mapping_control_url(description_url, timeout=timeout)
    if not found:
        return False
    control_url, service_type = found
    body = f"<NewRemoteHost></NewRemoteHost><NewExternalPort>{port}</NewExternalPort><NewProtocol>{protocol.upper()}</NewProtocol>"
    return _soap_request(control_url, service_type, "DeletePortMapping", body, timeout=timeout)
