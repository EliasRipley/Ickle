param(
    [int]$Steps = 12000,
    [string]$ModelOut = "models/ickle_spm_longrun.pt",
    [string]$CheckpointPath = "models/ickle_spm_longrun.pt.checkpoint.pt",
    [string]$BaselineModel = "models/ickle_clean.pt",
    [string]$CoreCorpus = "data/ickle_curated_only.txt",
    [string]$TrainingRoot = "C:\Projects\IckleTraining",
    [switch]$UseOpenDatasets,
    [int]$OasstRecords = 5000,
    [int]$OpenHermesRecords = 5000,
    [int]$CorpusMaxLines = 50000,
    [switch]$Resume,
    [switch]$LegacyDirectTrain
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Resolve-Path "$PSScriptRoot\..")

Write-Host "[Ickle] Repo root:" (Get-Location).Path
Write-Host "[Ickle] Steps:" $Steps
Write-Host "[Ickle] ModelOut:" $ModelOut
Write-Host "[Ickle] Checkpoint:" $CheckpointPath

if ($UseOpenDatasets) {
    Write-Host "[Ickle] Refreshing bounded open datasets..."
    python -m src.app open-dataset-ingest --preset oasst1 --max-records $OasstRecords --max-chars-per-record 220 --cleanup-temp-cache
    python -m src.app open-dataset-ingest --preset openhermes_2_5 --max-records $OpenHermesRecords --max-chars-per-record 200 --cleanup-temp-cache
}

Write-Host "[Ickle] Building cleaned corpus..."
python -m src.app build-clean-corpus --training-root $TrainingRoot --out data/ickle_clean_corpus.txt --max-lines $CorpusMaxLines --dictionary-items 0

if (-not $LegacyDirectTrain) {
    Write-Host "[Ickle] Running guarded continual step (anti-regression gates + replay)..."
    $guardArgs = @(
        "-m", "src.app", "continual-guard", "run-step",
        "--core-corpus", $CoreCorpus,
        "--new-corpus", "data/ickle_clean_corpus.txt",
        "--baseline-model", $BaselineModel,
        "--out-model", $ModelOut,
        "--checkpoint-path", $CheckpointPath,
        "--promote-to", $BaselineModel,
        "--steps", "$Steps",
        "--profile", "laptop",
        "--json"
    )
    if ($Resume) {
        $guardArgs += "--resume-if-possible"
    } else {
        $guardArgs += "--no-resume-if-possible"
    }
    python @guardArgs
    Write-Host "[Ickle] Guarded continual step finished."
    return
}

$args = @(
    "-u",
    "-m", "src.train",
    "--data", "data/ickle_clean_corpus.txt",
    "--out", $ModelOut,
    "--steps", "$Steps",
    "--profile", "laptop",
    "--tokenizer", "sentencepiece",
    "--spm-vocab-size", "2048",
    "--spm-model-type", "bpe",
    "--checkpoint-every", "200",
    "--checkpoint-path", $CheckpointPath
)

if ($Resume -and (Test-Path -LiteralPath $CheckpointPath)) {
    Write-Host "[Ickle] Resuming from checkpoint..."
    $args += @("--resume-from-checkpoint", $CheckpointPath)
}

Write-Host "[Ickle] Starting legacy direct training run..."
python @args

Write-Host "[Ickle] Legacy training command finished."
