# Continual-learning guard

Continual training can improve a new topic while damaging earlier capabilities. `continual-guard` mixes core examples, a replay buffer, and new examples, then evaluates both old and new behavior before promotion.

Typical flow:

```bash
python -m src.app continual-guard update-replay \
  --source-corpus IckleTraining/corpuses/core.txt

python -m src.app continual-guard run-step \
  --core-corpus IckleTraining/corpuses/core.txt \
  --new-corpus IckleTraining/corpuses/new_topic.txt \
  --baseline-model models/tiny.pt \
  --out-model models/candidates/ickle_candidate.pt \
  --promotion-gate --resume-if-possible --json
```

Keep the default user benchmark present, set explicit limits for acceptable core-score loss and required new-topic gain, and review the JSON report. Promotion must be a separate decision from training completion.

Every guarded step also auto-builds and mixes in a smart conversation corpus, a benchmark-focused corpus, and — by default (`auto_include_verified_corrections`) — an oversampled corpus of the owner's own adopted Epistemic Commons corrections, so a human correction can become part of the model, not just a prompt-time patch. See [Epistemic Commons: from correction to model](EPISTEMIC_COMMONS.md#from-correction-to-model).
