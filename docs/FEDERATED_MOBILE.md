# Federated and Android contribution

Ickle currently implements a coordinator-mediated federated LoRA path plus signed peer announcements and Torickle bundle exchange. It is a practical prototype, not yet a coordinator-free global training fabric.

## Trust and privacy model

Clients register, fetch the active round, train locally, and submit a signed adapter delta. Raw corpus text is not uploaded by this protocol. However, model updates can leak properties of training data; contributors should never use passwords, private messages, health records, or other secrets as a local corpus.

The coordinator validates signatures, timestamps, nonces, tensor names/shapes/finite values, update norm bounds, and reported sample-count bounds. A client may contribute only once per round. Robust aggregation reduces the impact of outliers, but open registration is still vulnerable to Sybil attacks. For a controlled community deployment, enable registration admission control.

## Safe coordinator setup

Local development:

```bash
python -m src.app federated-server --host 127.0.0.1 --port 8788 \
  --base-model models/tiny.pt --min-clients 3
```

Internet-facing deployment requires HTTPS. Supply a certificate and key directly, or bind to loopback behind a TLS reverse proxy. An admission secret can be read from `ICKLE_REGISTRATION_SECRET` or passed with `--registration-secret`. Do not put secrets directly in shared shell history.

```bash
python -m src.app federated-server --host 0.0.0.0 --port 8788 \
  --cert path/to/fullchain.pem --key path/to/private-key.pem \
  --base-model models/tiny.pt --min-clients 5
```

Ickle intentionally refuses a non-loopback plaintext listener unless `--allow-insecure-public` is supplied.

## Desktop client

```bash
python -m src.app federated-client \
  --server https://coordinator.example \
  --local-data IckleTraining/corpuses/local_contribution.txt \
  --base-model models/tiny.pt
```

## Android status

The Android project embeds a NumPy training client with Chaquopy and a foreground training service. It can register, train a small local adapter, submit a signed update, and optionally announce a Torickle bundle. Before broad release it still needs production battery/thermal constraints, unmetered-network controls, secure enrollment-secret input, background scheduling through WorkManager, and end-to-end device testing.

Contributors being able to see whether training is active and stop it immediately is delivered today (`MainActivity`'s status text and start/stop button). Choosing resource or network limits is **not** built yet -- there is no in-app control for it. Both are product requirements, not optional polish; the second is one of the gaps called out above, not a shipped guarantee.

## Inference sharing (donating compute to answer prompts, not just train)

Alongside training contribution ("seed"), peers can now also donate spare
compute to answer other users' prompts directly ("peer" usage of the network,
in the torrent-ratio sense) — `python -m src.app infer serve/find/ask/report`.
This is a separate, signed P2P protocol layered on the same peer-discovery
DHT. See [docs/INFERENCE_SHARING.md](INFERENCE_SHARING.md) for the protocol,
trust model, and CLI reference. The Android inference-serving path is not yet
built — see Future work in that doc.
