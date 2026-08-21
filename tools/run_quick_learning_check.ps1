param(
    [string]$Instruction = "Ickle, learn as much as you can about Byzantine architecture and Hagia Sophia from the internet and train yourself.",
    [string]$BaseUrl = "http://127.0.0.1:8787",
    [int]$Steps = 100,
    [int]$MaxUrls = 6,
    [int]$MaxWikiPages = 5,
    [int]$MaxNewsResults = 4,
    [int]$TotalPairs = 3000,
    [int]$EvalPrompts = 8,
    [int]$WarmupSteps = 20,
    [double]$LearningRate = 3e-6,
    [string]$Profile = "laptop",
    [int]$PollSeconds = 8,
    [int]$MaxWaitSeconds = 360,
    [switch]$NoNews
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Resolve-Path "$PSScriptRoot\..")

function Invoke-IckleJsonPost {
    param(
        [string]$Url,
        [hashtable]$Body
    )
    $json = $Body | ConvertTo-Json -Depth 20
    return Invoke-RestMethod -Uri $Url -Method Post -ContentType "application/json" -Body $json
}

function Get-IckleTaskById {
    param(
        [string]$Url,
        [string]$TaskId
    )
    $all = (Invoke-RestMethod -Uri "$Url/api/tasks" -Method Get).tasks
    return @($all | Where-Object { [string]$_.task_id -eq $TaskId } | Select-Object -First 1)
}

function Get-TaskSafe {
    param(
        [string]$Url,
        [string]$TaskId
    )
    $row = Get-IckleTaskById -Url $Url -TaskId $TaskId
    if ($row.Count -eq 0) {
        return $null
    }
    return $row[0]
}

function Read-JsonFileIfExists {
    param([string]$PathValue)
    if (-not $PathValue) {
        return $null
    }
    $candidate = [string]$PathValue
    if (-not (Test-Path -LiteralPath $candidate)) {
        $candidate = Join-Path (Get-Location).Path $candidate
    }
    if (-not (Test-Path -LiteralPath $candidate)) {
        return $null
    }
    return Get-Content -LiteralPath $candidate -Raw | ConvertFrom-Json
}

# Quick status check so failures are obvious up front.
$null = Invoke-RestMethod -Uri "$BaseUrl/api/status" -Method Get

$infer = Invoke-IckleJsonPost -Url "$BaseUrl/api/tasks/infer" -Body @{
    instruction = $Instruction
    queue = $false
}
if (-not $infer.inferred) {
    throw "No task inference returned for instruction: $Instruction"
}

$inferredType = [string]$infer.inferred.task_type
$topic = [string]$infer.inferred.payload.topic
if (-not $topic) {
    throw "Inference did not include a topic."
}

$taskPayload = @{
    topic = $topic
    max_urls = $MaxUrls
    include_wikipedia = $true
    include_news = (-not $NoNews)
    auto_pipeline = $true
    unrestricted = $false
    use_continual_guard = $true
    profile = $Profile
    steps = $Steps
    total_pairs = $TotalPairs
    eval_core_prompts = $EvalPrompts
    eval_new_prompts = $EvalPrompts
    warmup_steps = $WarmupSteps
    lr = $LearningRate
    core_ratio = 0.50
    replay_ratio = 0.30
    new_ratio = 0.20
    max_wiki_pages = $MaxWikiPages
    max_news_results = $MaxNewsResults
}

$queueReq = @{
    task_type = "learn_web_topic"
    payload = $taskPayload
    idempotency_key = "quickcheck:$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')):$([Guid]::NewGuid().ToString('N').Substring(0,8))"
    max_attempts = 2
}

$queued = Invoke-IckleJsonPost -Url "$BaseUrl/api/tasks" -Body $queueReq
$rootId = [string]$queued.task_id
$guardTask = @($queued.auto_pipeline_followups | Where-Object { [string]$_.task_type -eq "continual_guard_step" } | Select-Object -First 1)
$guardId = if ($guardTask.Count -gt 0) { [string]$guardTask[0].task_id } else { "" }

Write-Host "[Ickle quick-check] Inferred type:" $inferredType
Write-Host "[Ickle quick-check] Topic:" $topic
Write-Host "[Ickle quick-check] Root task:" $rootId
if ($guardId) {
    Write-Host "[Ickle quick-check] Guard task:" $guardId
}

$deadline = (Get-Date).AddSeconds([Math]::Max(30, $MaxWaitSeconds))
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds ([Math]::Max(3, $PollSeconds))
    $rootTask = Get-TaskSafe -Url $BaseUrl -TaskId $rootId
    $guardTaskRow = if ($guardId) { Get-TaskSafe -Url $BaseUrl -TaskId $guardId } else { $null }
    $rootActive = $rootTask -and ([string]$rootTask.status -in @("queued", "running"))
    $guardActive = $guardTaskRow -and ([string]$guardTaskRow.status -in @("queued", "running"))
    if (-not $rootActive -and -not $guardActive) {
        break
    }
}

$root = Get-TaskSafe -Url $BaseUrl -TaskId $rootId
$guard = if ($guardId) { Get-TaskSafe -Url $BaseUrl -TaskId $guardId } else { $null }

if (-not $root) {
    throw "Root task missing from queue history: $rootId"
}

$reportPath = ""
if ($root.result -and $root.result.report_path) {
    $reportPath = [string]$root.result.report_path
}
$report = Read-JsonFileIfExists -PathValue $reportPath

$keptSources = @()
$lowRelevanceCount = 0
if ($report -and $report.sources) {
    foreach ($src in @($report.sources)) {
        $ratio = [double]$src.topic_match.ratio
        $overlap = [int]$src.topic_match.overlap
        $anchorOverlap = [int]$src.topic_match.anchor_overlap
        if ($ratio -lt 0.40 -and $overlap -lt 2 -and $anchorOverlap -lt 1) {
            $lowRelevanceCount += 1
        }
        $keptSources += [pscustomobject]@{
            title = [string]$src.title
            url = [string]$src.url
            source = [string]$src.source
            ratio = $ratio
            overlap = $overlap
            anchor_overlap = $anchorOverlap
        }
    }
}

$sourceCount = if ($report) { [int]$report.source_count } else { 0 }
$candidateCount = if ($report) { [int]$report.candidate_source_count } else { 0 }
$droppedCount = if ($report -and $report.dropped_sources) { @($report.dropped_sources).Count } else { 0 }

$inferencePass = ($inferredType -eq "learn_web_topic")
$sourcePass = ($sourceCount -gt 0 -and $lowRelevanceCount -eq 0)
$overallPass = ($inferencePass -and $sourcePass)

$summary = [pscustomobject]@{
    instruction = $Instruction
    inferred = $infer.inferred
    quick_settings = @{
        steps = $Steps
        max_urls = $MaxUrls
        max_wiki_pages = $MaxWikiPages
        max_news_results = $MaxNewsResults
        total_pairs = $TotalPairs
        eval_prompts = $EvalPrompts
        poll_seconds = $PollSeconds
        max_wait_seconds = $MaxWaitSeconds
    }
    tasks = @{
        root_id = $rootId
        root_status = [string]$root.status
        guard_id = $guardId
        guard_status = if ($guard) { [string]$guard.status } else { "" }
    }
    root_result = $root.result
    guard_result = if ($guard) { $guard.result } else { $null }
    source_audit = @{
        report_path = $reportPath
        candidate_source_count = $candidateCount
        source_count = $sourceCount
        dropped_source_count = $droppedCount
        low_relevance_count = $lowRelevanceCount
        kept_sources = $keptSources
    }
    gates = @{
        inference_pass = $inferencePass
        source_targeting_pass = $sourcePass
        overall_pass = $overallPass
    }
}

$summary | ConvertTo-Json -Depth 20
