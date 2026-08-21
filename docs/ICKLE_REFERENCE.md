# Ickle command reference

## First run

```bash
python -m src.app preflight
python -m src.app reality-check
python -m src.app show-profile --cpu-pct 70 --ram-pct 70 --gpu-pct 0
python -m src.app serve-web --port 8787
```

Open `http://127.0.0.1:8787`. The status control reports the selected model and whether a trainer heartbeat is genuinely live. During generation, the send button becomes a Stop button.

## Train and evaluate

Use a new output path for experiments. Do not overwrite the active model directly.

```bash
python -m src.app train --data IckleTraining/corpuses/my_corpus.txt \
  --init-model models/tiny.pt \
  --out models/candidates/ickle_candidate.pt \
  --best-model-path models/candidates/ickle_candidate.best.pt \
  --checkpoint-path models/candidates/ickle_candidate.checkpoint.pt \
  --status-file IckleTraining/training_live.json \
  --steps 200

python -m src.app chat-benchmark \
  --model models/candidates/ickle_candidate.best.pt \
  --baseline-model models/tiny.pt
```

The default benchmark is `data/maintenance/user_chat_benchmark.json`. It combines general response checks with required facts and forbidden claims. A candidate should be promoted only when it improves on the baseline and passes the relevant specialist evaluations.

## Main command families

- Local use: `chat`, `hub`, `serve-web`, `app` (desktop window).
- Training: `train`, `lora-train`, `dpo-train`, `train-autopilot`, `continual-guard`.
- Data: `open-dataset-ingest`, `sanitize-training-data`, and the `build-*-corpus` commands.
- Evaluation: `chat-benchmark`, `honesty-context-eval`, `reality-check`.
- Community compute: `federated-server`, `federated-client`, `swarm`, `torickle`.
- Model operations: `model-library`, `model-maintain`, `quantize`, `export-onnx`, `knowledge-modules`.

## Operational rules

- Keep the active model unchanged until a candidate passes evaluation.
- Treat a stale heartbeat as stopped, even if an old file says `running`.
- Failed worker tasks are not silently restarted; retry them explicitly.
- Keep raw personal data local. Federated updates can still leak information in some threat models, so do not train on secrets.
- Use TLS for any coordinator reachable beyond the local machine.
