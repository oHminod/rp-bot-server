# Windows PowerShell 5.1 is supplied by Windows; no Python, uv, Git or compiler required.
$ErrorActionPreference = 'Stop'
try {
    $projectRoot = Split-Path -Parent $PSScriptRoot
    $modelsRoot = $env:PULID_MODELS_ROOT
    if (-not $modelsRoot) { throw 'PULID_MODELS_ROOT absent.' }
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { throw 'Windows x64 requis; utilisez un terminal 64 bits.' }
    $uvVersion = (Get-Content -Raw -LiteralPath (Join-Path $projectRoot '.uv-version')).Trim()
    $pythonVersion = (Get-Content -Raw -LiteralPath (Join-Path $projectRoot '.python-version')).Trim()
    # Ignore global/user uv choices, active venvs and Python startup customizations.
    Get-ChildItem Env:UV_* | Remove-Item
    'PYTHONHOME','PYTHONPATH','VIRTUAL_ENV','CONDA_PREFIX' | ForEach-Object { Remove-Item "Env:$_" -ErrorAction SilentlyContinue }
    $env:UV_CACHE_DIR = Join-Path $modelsRoot 'other\uv-windows'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $modelsRoot 'other\uv-python-windows'
    $env:UV_UNMANAGED_INSTALL = Join-Path $modelsRoot 'other\uv-windows-bin'
    $env:UV_PYTHON_PREFERENCE = 'only-managed'
    $uv = Join-Path $env:UV_UNMANAGED_INSTALL 'uv.exe'
    $installed = ''
    if (Test-Path -LiteralPath $uv -PathType Leaf) { $installed = (& $uv --version) }
    if ($installed -notmatch ('^uv ' + [regex]::Escape($uvVersion) + '(\s|$)')) {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-RestMethod "https://astral.sh/uv/$uvVersion/install.ps1" | Invoke-Expression
        if ($LASTEXITCODE -ne 0) { throw "Installation de uv $uvVersion impossible : $uv" }
    }
    $installed = (& $uv --version)
    if ($LASTEXITCODE -ne 0 -or $installed -notmatch ('^uv ' + [regex]::Escape($uvVersion) + '(\s|$)')) { throw "uv $uvVersion requis : $uv" }
    & $uv python install "cpython-$pythonVersion-windows-x86_64-none" --no-bin --no-registry --no-config
    if ($LASTEXITCODE -ne 0) { throw "Python gere indisponible : $env:UV_PYTHON_INSTALL_DIR" }
    # Full patch-version path: never depend on uv's movable 3.11 junction.
    $python = Join-Path $env:UV_PYTHON_INSTALL_DIR "cpython-$pythonVersion-windows-x86_64-none\python.exe"
    & $python (Join-Path $PSScriptRoot 'install_environment.py') --uv $uv --models-root $modelsRoot --profile $env:PULID_INSTALL_PROFILE
    if ($LASTEXITCODE -ne 0) { throw 'Installation verrouillee interrompue; voir le diagnostic ci-dessus.' }
    exit 0
} catch {
    Write-Host "[ERREUR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
