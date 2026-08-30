$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$baseRoot = Join-Path $repoRoot "data\merged_v3"
$validationRoot = Join-Path $repoRoot "data\eval_merged_v3_fake_disjoint_seed42"
$scaledRoot = Join-Path $repoRoot "data\merged_v4_scaled_sid"
$checkpoint = Join-Path $repoRoot "checkpoints\merged\run_007\best.pt"
$outputRoot = Join-Path $repoRoot "checkpoints\merged\run_008_scaled_sid"

foreach ($required in @($baseRoot, $validationRoot, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required input is missing: $required"
    }
}
if ((Test-Path -LiteralPath (Join-Path $outputRoot "best.pt")) -or
    (Test-Path -LiteralPath (Join-Path $outputRoot "history.json"))) {
    throw "Run output already exists at $outputRoot; refusing to overwrite it."
}

$realRoot = New-Item -ItemType Directory -Force -Path (Join-Path $scaledRoot "real")
$fakeRoot = New-Item -ItemType Directory -Force -Path (Join-Path $scaledRoot "fake")

function Add-HardLinkIfMissing {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path -LiteralPath $Destination)) {
        New-Item -ItemType HardLink -Path $Destination -Target $Source | Out-Null
    }
}

# WildFake files keep their existing generator-aware names.
foreach ($label in @("real", "fake")) {
    $sourceDirectory = Join-Path $baseRoot $label
    $destinationDirectory = Join-Path $scaledRoot $label
    Get-ChildItem -LiteralPath $sourceDirectory -File -Filter "wildfake__*" | ForEach-Object {
        Add-HardLinkIfMissing $_.FullName (Join-Path $destinationDirectory $_.Name)
    }
}

# Reuse the old SID sample while upgrading its filenames to subtype-aware balancing groups.
Get-ChildItem -LiteralPath (Join-Path $baseRoot "real") -File -Filter "hf__SID_Set__*" |
    ForEach-Object {
        $identifier = $_.Name.Substring("hf__SID_Set__".Length)
        $name = "hf__SID_Set__label_0__${identifier}"
        Add-HardLinkIfMissing $_.FullName (Join-Path $realRoot.FullName $name)
    }
Get-ChildItem -LiteralPath (Join-Path $baseRoot "fake") -File -Filter "hf__SID_Set__*" |
    ForEach-Object {
        $identifier = $_.Name.Substring("hf__SID_Set__".Length)
        if ($identifier.StartsWith("full_synthetic_")) {
            $sourceLabel = 1
        } elseif ($identifier.StartsWith("tampered_")) {
            $sourceLabel = 2
        } else {
            throw "Cannot infer the original SID label from $($_.Name)"
        }
        $name = "hf__SID_Set__label_${sourceLabel}__${identifier}"
        Add-HardLinkIfMissing $_.FullName (Join-Path $fakeRoot.FullName $name)
    }

$env:PYTHONPATH = Join-Path $repoRoot "src"
python -m traceguard.materialize `
    --hf-dataset saberzl/SID_Set `
    --hf-samples-per-label 0=25000 1=25000 2=25000 `
    --hf-group-by-label `
    --hf-shuffle-buffer 200 `
    --seed 43 `
    --output-dir $scaledRoot
if ($LASTEXITCODE -ne 0) { throw "SID_Set materialization failed with exit code $LASTEXITCODE" }

python -m traceguard.train $scaledRoot `
    --validation-dir $validationRoot `
    --balance-groups `
    --init-checkpoint $checkpoint `
    --evaluate-initial `
    --robustness-profile low_resolution `
    --freeze-backbone `
    --cache-frozen-features `
    --feature-cache-views 2 `
    --head-batch-size 2048 `
    --positive-weight 1.0 `
    --output-dir $outputRoot `
    --epochs 15 `
    --early-stopping-patience 5 `
    --batch-size 128 `
    --workers 4 `
    --lr 1e-5 `
    --weight-decay 1e-2 `
    --seed 42 `
    --device cuda
if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
