# Ickle documentation

Start with [the project README](../README.md) for the vision and current status, or [the command reference](ICKLE_REFERENCE.md) for commands. The remaining guides cover the parts of Ickle where operational choices affect model quality, privacy, or recoverability.

- [Federated and Android contribution](FEDERATED_MOBILE.md)
- [Inference sharing (P2P answer serving)](INFERENCE_SHARING.md)
- [Epistemic Commons (inspectable claims and P2P human review)](EPISTEMIC_COMMONS.md)
- [Running a bootstrap peer](BOOTSTRAP_NODE.md)
- [Continual-learning guard](CONTINUAL_LEARNING_GUARD.md)
- [Additive knowledge modules](ADDITIVE_KNOWLEDGE_MODULES.md)
- [Honest/context training package](HONEST_CONTEXT_TRAINING_PACKAGE.md)
- [Runtime and training workspace separation](WORKSPACE_SEPARATION.md)
- [Heavy-training blueprint](HEAVY_TRAINING_BLUEPRINT_2026.md)

The source of truth for available commands is always:

```bash
python -m src.app --help
python -m src.app <command> --help
```
