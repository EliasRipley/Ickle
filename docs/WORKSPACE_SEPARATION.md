# Workspace separation

Application code and runtime state belong in the Ickle project. Large corpora, dataset caches, and generated training mixes belong in a separate IckleTraining workspace.

Set an explicit location:

```powershell
$env:ICKLE_TRAINING_ROOT = "D:\IckleTraining"
python -m src.app workspace-check
```

The current checkout falls back to `IckleTraining/` inside the project when no environment variable is set. That is convenient for development but receives a warning because cleanup, packaging, or source-control operations can accidentally mix code and training data.

Recommended layout:

```text
C:\Projects\Ickle\          application, tests, runtime state
D:\IckleTraining\           corpora, dataset cache, generated mixes
C:\Models\Ickle\            optional large model archive
```
