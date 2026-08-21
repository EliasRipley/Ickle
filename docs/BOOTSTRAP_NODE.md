# Running a direct/private bootstrap peer

Ickle's swarm (`src/federated/swarm.py`) and inference-sharing node
(`src/federated/inference_swarm.py`) are genuinely serverless peer-to-peer:
two Ickle instances that know each other's address can connect and share
training deltas or answer prompts for each other directly, no company or
central server involved. The main model/training swarm now solves fresh-node
discovery through the trackerless BitTorrent Mainline DHT; see
[`PUBLIC_SWARM.md`](PUBLIC_SWARM.md). A manually configured bootstrap remains
useful for private communities, LANs, blocked-UDP environments, testing, and
the separate inference-serving CLI.

This page covers that explicit path: any friend group, team, or community can
run a long-lived peer whose address is shared directly. It does not depend on
the public DHT and can stay entirely inside a private network.

## What "bootstrap peer" means here

Any Ickle instance that:

1. stays running continuously (or close to it), and
2. is reachable from the internet at a stable address,

can serve as a bootstrap peer for others. It's not a special mode — it's an
ordinary swarm node that happens to be up long enough and reachable enough
for other peers to reliably find each other through it.

## Making a node internet-reachable

Two real, standard mechanisms now do this automatically instead of requiring
manual router configuration:

- **STUN** (RFC 5389) discovers this machine's actual public IP, rather than
  guessing from a local network interface (which is wrong the moment you're
  behind a home router / NAT — the old behavior announced a private
  `192.168.x.x` address that nobody outside the LAN could ever reach).
- **UPnP IGD** asks the local router to forward the swarm's port
  automatically, the same mechanism game consoles and torrent clients use.

Both are best-effort and non-fatal (`src/federated/nat_traversal.py`): if
UPnP is disabled on your router or STUN can't reach the internet, the node
falls back to its previous behavior instead of failing to start. They're
opt-in (`--nat-traversal` / the Control room → Network "Join swarm" switch)
because they make outbound network calls and (for UPnP) a change to your
router's port-forwarding table — not something to do silently.

If you're deploying on a VPS with a public IP already (the common case for a
dedicated bootstrap peer), you generally don't need either: your cloud
provider's firewall/security-group console is the actual place to open the
port, and `--external-host` lets you state the public IP directly instead of
relying on STUN to discover it.

## Running one

```bash
python -m src.federated.swarm start \
  --host 0.0.0.0 --port 8790 \
  --external-host <this machine's public IP or hostname> \
  --daemon
```

Add `--nat-traversal` instead of `--external-host` if you're behind a home
router rather than a VPS with a public IP directly attached.

Add `--public-discovery` if the same node should also announce itself through
the public trackerless swarm. Omit it for a direct/private bootstrap.

To also serve as an inference-sharing bootstrap peer (see
[`INFERENCE_SHARING.md`](INFERENCE_SHARING.md)):

```bash
python -m src.app infer serve --model models/your_model.pt --capacity 4 \
  --host 0.0.0.0 --port 8791
```

### Keeping it running (systemd)

```ini
# /etc/systemd/system/ickle-swarm.service
[Unit]
Description=Ickle swarm bootstrap peer
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/ickle
ExecStart=/opt/ickle/.venv/bin/python -m src.federated.swarm start \
  --host 0.0.0.0 --port 8790 --external-host YOUR.PUBLIC.IP --daemon
Restart=on-failure
RestartSec=5
User=ickle

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ickle-swarm
```

## Sharing it

Give the address (`host:port`) to whoever you want on the network. From the
web UI: **Control room → Network → Direct peer connection**. From the CLI:

```bash
python -m src.app swarm start --bootstrap your-bootstrap-host:8790 ...
```

Direct bootstraps can introduce peers to the local application-level store.
Public-DHT candidates use a stricter path and do not transitively trust a
stranger's returned peer list.

## What this is not

- Not required for the main public model/training swarm. The Mainline DHT is
  now the default discovery path after a user explicitly joins.
- Not an Ickle-operated public service. You (or whoever you trust) run this
  direct bootstrap and decide who receives its address.
- Not Sybil- or abuse-resistant. A bootstrap peer's address, once shared,
  can be added by anyone who has it. Share it the way you'd share access to
  any other resource you control — with people you trust, or behind
  whatever access control you're comfortable operating.
- Not a guarantee of uptime. If your bootstrap peer goes down, peers who
  only ever bootstrapped through it (and never learned about anyone else)
  lose their way in until it's back or they add another one.
