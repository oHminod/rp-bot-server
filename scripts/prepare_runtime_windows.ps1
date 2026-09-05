param([Parameter(Mandatory=$true)][string]$ProjectRoot)
$ErrorActionPreference = 'Stop'
try {
    $python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    $pointer = Join-Path $ProjectRoot '.venv\pulid-python-path'
    if (Test-Path -LiteralPath $pointer -PathType Leaf) {
        $python = (Get-Content -LiteralPath $pointer -Raw -Encoding UTF8).TrimEnd("`r", "`n")
        if (-not [System.IO.Path]::IsPathRooted($python)) {
            $python = Join-Path $ProjectRoot $python
        }
    }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Python gere absent : $python. Deplacez le dossier complet, ou relancez install_windows.bat."
    }
    & $python -I (Join-Path $ProjectRoot 'scripts\check_environment.py') --prepare
    exit $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine("[ERREUR] $($_.Exception.Message)")
    exit 1
}
