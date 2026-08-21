# Additive knowledge modules

Knowledge modules are topic-specific LoRA adapters. They let the system add or remove specialist behavior without repeatedly overwriting the core model.

List and resolve modules:

```bash
python -m src.app knowledge-modules list --json
python -m src.app knowledge-modules resolve --help
```

Register a trained adapter:

```bash
python -m src.app knowledge-modules register \
  --module-id acoustics_v1 \
  --module-path models/modules/acoustics_v1.pt \
  --topics "acoustics,sound,vibration" \
  --base-model models/tiny.pt \
  --enabled
```

Record the exact base-model hash, training-data provenance, evaluation result, and licence alongside every published module. Do not compose adapters trained against incompatible base weights merely because their tensor shapes match.
