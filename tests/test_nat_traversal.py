import socket
import struct
import unittest
from unittest import mock

from src.federated import nat_traversal as nat


def _build_stun_response(txn: bytes, ip: str, port: int, *, xor: bool = True) -> bytes:
    ip_int = struct.unpack("!I", socket.inet_aton(ip))[0]
    if xor:
        xport = port ^ (nat.STUN_MAGIC_COOKIE >> 16)
        xaddr = ip_int ^ nat.STUN_MAGIC_COOKIE
        attr_value = struct.pack("!BBHI", 0, 1, xport, xaddr)
        attr_type = nat._ATTR_XOR_MAPPED_ADDRESS
    else:
        attr_value = struct.pack("!BBHI", 0, 1, port, ip_int)
        attr_type = nat._ATTR_MAPPED_ADDRESS
    attr = struct.pack("!HH", attr_type, len(attr_value)) + attr_value
    header = struct.pack("!HHI12s", 0x0101, len(attr), nat.STUN_MAGIC_COOKIE, txn)
    return header + attr


class StunParsingTests(unittest.TestCase):
    def test_parses_xor_mapped_address(self):
        txn = b"abcdefghijkl"
        response = _build_stun_response(txn, "203.0.113.7", 51820, xor=True)
        self.assertEqual(nat._parse_stun_response(response, txn), ("203.0.113.7", 51820))

    def test_parses_legacy_mapped_address(self):
        txn = b"abcdefghijkl"
        response = _build_stun_response(txn, "198.51.100.9", 4242, xor=False)
        self.assertEqual(nat._parse_stun_response(response, txn), ("198.51.100.9", 4242))

    def test_rejects_mismatched_transaction_id(self):
        txn = b"abcdefghijkl"
        other_txn = b"zzzzzzzzzzzz"
        response = _build_stun_response(txn, "203.0.113.7", 51820)
        self.assertIsNone(nat._parse_stun_response(response, other_txn))

    def test_rejects_non_success_response(self):
        txn = b"abcdefghijkl"
        header = struct.pack("!HHI12s", 0x0111, 0, nat.STUN_MAGIC_COOKIE, txn)  # error response type
        self.assertIsNone(nat._parse_stun_response(header, txn))

    def test_rejects_short_response(self):
        self.assertIsNone(nat._parse_stun_response(b"short", b"x" * 12))

    def test_ipv6_mapped_address_is_ignored(self):
        # family=0x02 (IPv6) isn't handled -- the swarm's listener is IPv4-only.
        attr_value = struct.pack("!BBH", 0, 2, 1234) + b"\x00" * 16
        attr = struct.pack("!HH", nat._ATTR_XOR_MAPPED_ADDRESS, len(attr_value)) + attr_value
        txn = b"abcdefghijkl"
        header = struct.pack("!HHI12s", 0x0101, len(attr), nat.STUN_MAGIC_COOKIE, txn)
        self.assertIsNone(nat._parse_stun_response(header + attr, txn))


class StunQueryFailureTests(unittest.TestCase):
    def test_get_external_address_returns_none_when_all_servers_unreachable(self):
        with mock.patch.object(nat, "_stun_query_one", return_value=None):
            self.assertIsNone(nat.stun_get_external_address(servers=[("example.invalid", 3478)]))

    def test_query_one_swallows_socket_errors(self):
        with mock.patch("socket.socket") as mock_socket_cls:
            mock_socket = mock_socket_cls.return_value.__enter__.return_value
            mock_socket.recvfrom.side_effect = OSError("network unreachable")
            self.assertIsNone(nat._stun_query_one("example.invalid", 3478, timeout=0.1))


class UpnpFailureTests(unittest.TestCase):
    def test_add_port_mapping_returns_false_without_local_ip(self):
        with mock.patch.object(nat, "_local_ip_for_router", return_value=None):
            self.assertFalse(nat.upnp_add_port_mapping(8790))

    def test_add_port_mapping_returns_false_without_router_response(self):
        with (
            mock.patch.object(nat, "_local_ip_for_router", return_value="192.168.1.50"),
            mock.patch.object(nat, "_ssdp_discover", return_value=None),
        ):
            self.assertFalse(nat.upnp_add_port_mapping(8790))

    def test_add_port_mapping_returns_false_without_control_url(self):
        with (
            mock.patch.object(nat, "_local_ip_for_router", return_value="192.168.1.50"),
            mock.patch.object(nat, "_ssdp_discover", return_value="http://192.168.1.1/desc.xml"),
            mock.patch.object(nat, "_find_port_mapping_control_url", return_value=None),
        ):
            self.assertFalse(nat.upnp_add_port_mapping(8790))

    def test_add_port_mapping_sends_expected_soap_action(self):
        captured = {}

        def fake_soap(control_url, service_type, action, body_xml, *, timeout):
            captured["control_url"] = control_url
            captured["service_type"] = service_type
            captured["action"] = action
            captured["body_xml"] = body_xml
            return True

        with (
            mock.patch.object(nat, "_local_ip_for_router", return_value="192.168.1.50"),
            mock.patch.object(nat, "_ssdp_discover", return_value="http://192.168.1.1/desc.xml"),
            mock.patch.object(
                nat, "_find_port_mapping_control_url",
                return_value=("http://192.168.1.1/ctl", "urn:schemas-upnp-org:service:WANIPConnection:1"),
            ),
            mock.patch.object(nat, "_soap_request", side_effect=fake_soap),
        ):
            self.assertTrue(nat.upnp_add_port_mapping(8790, description="Ickle Swarm"))
        self.assertEqual(captured["action"], "AddPortMapping")
        self.assertIn("192.168.1.50", captured["body_xml"])
        self.assertIn("8790", captured["body_xml"])


class DeviceDescriptionParsingTests(unittest.TestCase):
    def test_finds_wan_ip_connection_control_url(self):
        xml_bytes = b"""<?xml version="1.0"?>
        <root xmlns="urn:schemas-upnp-org:device-1-0">
          <device>
            <deviceList>
              <device>
                <deviceList>
                  <device>
                    <serviceList>
                      <service>
                        <serviceType>urn:schemas-upnp-org:service:WANIPConnection:1</serviceType>
                        <controlURL>/ctl/IPConn</controlURL>
                      </service>
                    </serviceList>
                  </device>
                </deviceList>
              </device>
            </deviceList>
          </device>
        </root>"""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.read.return_value = xml_bytes
            result = nat._find_port_mapping_control_url("http://192.168.1.1:5000/rootDesc.xml", timeout=1.0)
        self.assertEqual(result, ("http://192.168.1.1:5000/ctl/IPConn", "urn:schemas-upnp-org:service:WANIPConnection:1"))

    def test_returns_none_when_no_matching_service(self):
        xml_bytes = b"""<?xml version="1.0"?>
        <root xmlns="urn:schemas-upnp-org:device-1-0">
          <device><serviceList>
            <service>
              <serviceType>urn:schemas-upnp-org:service:Layer3Forwarding:1</serviceType>
              <controlURL>/ctl/L3F</controlURL>
            </service>
          </serviceList></device>
        </root>"""
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value.read.return_value = xml_bytes
            result = nat._find_port_mapping_control_url("http://192.168.1.1:5000/rootDesc.xml", timeout=1.0)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
