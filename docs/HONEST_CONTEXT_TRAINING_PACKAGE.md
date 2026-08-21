# Honest and context-aware training package

This workflow builds supervised examples and preference pairs that reward direct answers, explicit uncertainty, evidence use, and continuity across follow-up questions.

```bash
python -m src.app build-honest-context-package \
  --training-root IckleTraining \
  --out-dir IckleTraining/corpuses/honest_context \
  --json
```

Inspect generated pairs before training. The package improves behavior but cannot make unsupported knowledge true. Evaluate factual accuracy separately and keep web-derived claims tied to provenance and collection time.
