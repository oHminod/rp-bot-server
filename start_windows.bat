@echo off
setlocal EnableExtensions

set "PROJECT_DIR=%~dp0"
set "PULID_PROJECT_ROOT=%PROJECT_DIR%"
set "TORCH_DLL_DIR=%PROJECT_DIR%.venv\Lib\site-packages\torch\lib"
set "PATH=%TORCH_DLL_DIR%;%PATH%"
set "SERVER_HOST=127.0.0.1"
set "SERVER_CORS="

if /I "%~1"=="--network" (
    set "SERVER_HOST=0.0.0.0"
    set "SERVER_CORS=--cors-origin *"
)

cd /d "%PROJECT_DIR%"

set "SERVER_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
if not exist "%SERVER_PYTHON%" (
    echo [ERREUR] Serveur PuLID non installe.
    echo Executez d'abord : install_windows.bat
    exit /b 1
)

set "PYTHONHOME="
set "PYTHONPATH="
"%SERVER_PYTHON%" "%PROJECT_DIR%scripts\check_environment.py"
if errorlevel 1 (
    echo [ERREUR] Python inutilisable ou dossier deplace. Relancez install_windows.bat.
    exit /b 1
)

if "%SERVER_HOST%"=="127.0.0.1" (
    echo Serveur PuLID local : http://127.0.0.1:12693
) else (
    echo [AVERTISSEMENT] Mode reseau avance : ecoute sur toutes les interfaces et CORS ouvert.
    powershell.exe -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.AddressState -eq 'Preferred' } | ForEach-Object { '  http://' + $_.IPAddress + ':12693' }"
)
echo.
echo Arret du serveur : Ctrl+C
echo.

"%SERVER_PYTHON%" -m pulid_app.server --host %SERVER_HOST% --port 12693 --device cuda --dtype float16 --offload none %SERVER_CORS% %*
exit /b %ERRORLEVEL%
