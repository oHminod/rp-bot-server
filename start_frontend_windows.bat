@echo off
setlocal EnableExtensions
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0"
rem Ignore global CUDA toolkits and Python/DLL search customizations.
for /f "tokens=1 delims==" %%V in ('set CUDA_PATH 2^>nul') do set "%%V="
set "CUDA_HOME="
set "CUDA_ROOT="
set "NVTOOLSEXT_PATH="
set "PROJECT_DIR=%~dp0"
set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
cd /d "%PROJECT_DIR%"
set "PYTHONHOME="
set "PYTHONPATH="
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%scripts\prepare_runtime_windows.ps1" -ProjectRoot "%PROJECT_DIR%."
if errorlevel 1 exit /b 1

"%VENV_PYTHON%" -I "%PROJECT_DIR%scripts\check_environment.py"
if errorlevel 1 exit /b 1
"%VENV_PYTHON%" -I "%PROJECT_DIR%frontend\server.py" --host 127.0.0.1 --port 8888 %*
exit /b %ERRORLEVEL%
