# One-command setup for voice-agent-evals on Windows.
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1            # evals + voice demo
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -NoVoice   # evals only (smallest install)
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -NoPiper   # voice demo with Windows SAPI voice only
# Uses the OS certificate store (truststore) so it works behind corporate TLS
# inspection. Certificate verification is never disabled.
param(
    [switch]$NoVoice,
    [switch]$NoPiper
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Find-Python {
    foreach ($cand in @(@("py", "-3.12"), @("py", "-3.11"), @("python"))) {
        try {
            $exe = $cand[0]; $rest = @($cand | Select-Object -Skip 1)
            $v = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and [version]$v -ge [version]"3.11") { return ,$cand }
        } catch { }
    }
    throw "Python 3.11+ not found. Install Python 3.12 from python.org and re-run."
}

$py = Find-Python
Write-Host "Using Python: $($py -join ' ')"
if (-not (Test-Path .venv)) {
    $exe = $py[0]; $rest = @($py | Select-Object -Skip 1)
    & $exe @rest -m venv .venv
}
$vpy = ".\.venv\Scripts\python.exe"
$env:PIP_USE_FEATURE = "truststore"

& $vpy -m pip install --quiet truststore
& $vpy -m pip install --quiet --upgrade pip
if ($NoVoice) { $extras = "dev" } else { $extras = "voice,dev" }
if (-not $NoVoice -and -not $NoPiper) { $extras = "$extras,piper" }
& $vpy -m pip install --quiet -e ".[$extras]"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "Created .env from .env.example" }

if (-not $NoVoice -and -not $NoPiper) {
    New-Item -ItemType Directory -Force models | Out-Null
    if (-not (Test-Path models\en_US-lessac-medium.onnx)) {
        Write-Host "Downloading Piper voice en_US-lessac-medium (~63 MB) into .\models ..."
        & $vpy -c @"
import shutil, truststore
truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download
for f in ['en/en_US/lessac/medium/en_US-lessac-medium.onnx', 'en/en_US/lessac/medium/en_US-lessac-medium.onnx.json']:
    shutil.copy(hf_hub_download('rhasspy/piper-voices', f), 'models/' + f.split('/')[-1])
print('voice ok')
"@
    }
}
if (-not $NoVoice) {
    Write-Host "Pre-downloading faster-whisper tiny.en ..."
    & $vpy -c "import truststore; truststore.inject_into_ssl(); from faster_whisper import WhisperModel; WhisperModel('tiny.en', device='cpu', compute_type='int8'); print('stt ok')"
}

& .\.venv\Scripts\vae.exe validate
Write-Host ""
Write-Host "Done. Next:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  vae run            # run the eval suite (scripted backend, no LLM needed)"
Write-Host "  vae chat           # text chat with the agent"
if (-not $NoVoice) { Write-Host "  vae serve          # voice demo at http://127.0.0.1:8000" }
