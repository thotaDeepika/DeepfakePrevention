# SafeUpload Model Weights Downloader
# Run this script if models fail to load due to network issues.
# Uses Windows Invoke-WebRequest (respects system proxy — works even when GitHub is blocked in Python).

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# facenet_pytorch checks ~/.cache/torch/checkpoints (NOT hub/checkpoints)
$CheckpointDir   = "$env:USERPROFILE\.cache\torch\checkpoints"
$HubCheckpointDir = "$env:USERPROFILE\.cache\torch\hub\checkpoints"
New-Item -ItemType Directory -Force $CheckpointDir    | Out-Null
New-Item -ItemType Directory -Force $HubCheckpointDir | Out-Null

function Download-If-Missing {
    param($Url, $Name, $Filename)
    $dst1 = "$CheckpointDir\$Filename"
    $dst2 = "$HubCheckpointDir\$Filename"

    if ((Test-Path $dst1) -and (Test-Path $dst2)) {
        $size = [math]::Round((Get-Item $dst1).Length / 1MB, 1)
        Write-Host "[SKIP] $Name already cached ($size MB)" -ForegroundColor Green
        return
    }

    Write-Host "[DOWN] Downloading $Name..." -ForegroundColor Yellow
    try {
        # Download to the primary location facenet_pytorch looks in
        if (-not (Test-Path $dst1)) {
            Invoke-WebRequest -Uri $Url -OutFile $dst1 -TimeoutSec 300 -UseBasicParsing
        }
        # Copy to hub/checkpoints as well (for torch.hub compatibility)
        if (-not (Test-Path $dst2)) {
            Copy-Item $dst1 $dst2 -Force
        }
        $size = [math]::Round((Get-Item $dst1).Length / 1MB, 1)
        Write-Host "[OK]   $Name ready ($size MB)" -ForegroundColor Green
    } catch {
        Write-Host "[FAIL] $($Name): $_" -ForegroundColor Red
    }
}

Write-Host "=== SafeUpload Model Downloader ===" -ForegroundColor Cyan
Write-Host ""

# FaceNet vggface2 (106 MB) — used by facenet.py and arcface.py fallback
Download-If-Missing `
    "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt" `
    "FaceNet vggface2" `
    "20180402-114759-vggface2.pt"

Write-Host ""
Write-Host "All done! Start the server with:" -ForegroundColor Cyan
Write-Host "  cd app && python app.py" -ForegroundColor White
