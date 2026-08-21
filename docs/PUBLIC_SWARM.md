# Public Ickle swarm

Ickle has trackerless, internet-scale peer discovery. A fresh installation can
find other participating Ickle nodes without exchanging addresses and without
an Ickle-operated account, tracker, or peer directory.

Joining is deliberately opt-in. Use **Control room → Network → Join swarm** in
the app, or start a CLI swarm node with:

```bash
python -m src.federated.swarm start --public-discovery --nat-traversal --daemon
```

## How a fresh node finds the swarm

There is no way for two internet devices with no shared information to find
one another literally from nothing. Trackerless torrent clients solve this
with a distributed hash table (DHT) plus a short list of interchangeable DNS
bootstrap routers. Ickle now follows that design:

1. Ickle starts its normal HTTP swarm listener.
2. STUN discovers the public address and UPnP asks the local router for a TCP
   port mapping. Both are best-effort.
3. The node asks several public Mainline DHT routers for nodes near Ickle's
   stable, versioned network key:
   `034e41f8a47f1701e233a56493fe2d409bcef030`.
4. It performs a bounded iterative BEP 5 `get_peers` lookup and announces its
   swarm port using the write tokens returned by nearby DHT nodes.
5. Every returned `IP:port` is treated as hostile candidate data. Ickle only
   keeps an endpoint if it answers `/torickle/v1` as an Ickle swarm node.
6. Verified Ickle peers exchange signed bundle announcements and explicitly
   shared Epistemic Commons events directly over the existing protocol.
7. The DHT lookup and announcement refresh in the background every 15 minutes.

The implementation is in `src/federated/public_dht.py`. It implements the
small required bencode/KRPC subset directly, follows
[BEP 5](https://www.bittorrent.org/beps/bep_0005.html), and creates
IP-bound node IDs following [BEP 42](https://www.bittorrent.org/beps/bep_0042.html).

## What is and is not centralised

The bootstrap router hostnames are an initial introduction mechanism, not an
Ickle directory. Ickle tries routers operated by separate BitTorrent
implementations. No router receives special trust, and after the first replies
the lookup walks the distributed DHT itself. A router cannot forge a valid
Ickle bundle or Commons event.

This still has dependencies: DNS, some reachable DHT nodes, and an internet
route. “Decentralised” does not mean “has no infrastructure”; it means no one
Ickle service owns the membership list or sits between peers.

## Privacy boundary

Joining the public swarm reveals:

- the node's public IP address;
- its Ickle swarm TCP port;
- interest in Ickle's public network key.

The public DHT does **not** receive prompts, chats, model bytes, peer identity
keys, private reviews, or training data. Direct peers can naturally see the IP
that connects to them. Only reviews individually marked as shared leave the
device. Leaving the swarm stops refreshes and rebinds Ickle to loopback, though
old DHT records can remain cached until their normal expiry.

Peer traffic is currently plain HTTP. Ed25519 signatures give provenance and
tamper detection for shared artifacts, not transport confidentiality. Do not
send secrets to untrusted inference peers. Internet-facing deployments that
need confidentiality should place the listener behind TLS.

## Connectivity states

The Network view reports independent facts rather than one vague “connected”
badge:

- **DHT health** — how many queried DHT nodes answered.
- **Verified peers** — candidates that actually spoke the Ickle protocol.
- **Incoming ready** — UPnP succeeded or an external address was explicitly
  configured.
- **Outbound only** — discovery works, but automatic incoming port mapping did
  not. The node can still connect to reachable peers and contribute as a
  downloader/client.
- **Limited** — UDP is blocked, DNS failed, or the device is offline. A direct
  peer address remains available as a fallback.

Carrier-grade NAT, symmetric NAT, restrictive enterprise firewalls, and
routers with UPnP disabled can prevent unsolicited incoming connections. A
relay protocol would improve that last mile, but it would require someone to
donate relay bandwidth and introduces a different abuse/privacy boundary. The
current UI reports the limitation instead of pretending STUN alone made the
node reachable.

## Security limits

- DHT endpoints are untrusted and public-IP-only. Private, loopback, multicast,
  malformed, oversized, and non-Ickle candidates are rejected.
- Public discovery does not transitively trust a stranger's peer list, avoiding
  private-network request injection. Direct/manual bootstraps remain explicit.
- Bundle announcements and Commons events still require their existing
  Ed25519 validation. Discovery is not authorization.
- There is no global Sybil resistance. An attacker can create many endpoints
  or withhold responses around a network key. Local review-based trust and
  artifact signatures limit what discovery alone grants, but do not solve
  denial of service.
- Lookups, packet decoding, endpoint probes, event batches, and payload sizes
  are bounded.

The public DHT is therefore a rendezvous substrate—not a source of truth and
not a replacement for Ickle's application-level verification.
