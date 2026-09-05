@echo off
setlocal EnableExtensions
rem Ignore global CUDA toolkits and Python/DLL search customizations.
for /f "tokens=1 delims==" %%V in ('set CUDA_PATH 2^>nul') do set "%%V="
set "CUDA_HOME="
set "CUDA_ROOT="
set "NVTOOLSEXT_PATH="
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0"

set "PROJECT_DIR=%~dp0"
set "PULID_PROJECT_ROOT=%PROJECT_DIR%"
set "PYTHONHOME="
set "PYTHONPATH="
set "VIRTUAL_ENV="
set "PULID_INSTALL_PROFILE=%PULID_INSTALL_PROFILE%"
if not defined PULID_INSTALL_PROFILE set "PULID_INSTALL_PROFILE=development"
set "PULID_CONFIGURE_NETWORK=0"

:parse_arguments
if "%~1"=="" goto :arguments_ready
if /I "%~1"=="--production" (
    set "PULID_INSTALL_PROFILE=production"
    shift
    goto :parse_arguments
)
if /I "%~1"=="--development" (
    set "PULID_INSTALL_PROFILE=development"
    shift
    goto :parse_arguments
)
if /I "%~1"=="--network" (
    set "PULID_CONFIGURE_NETWORK=1"
    shift
    goto :parse_arguments
)
if /I "%~1"=="--help" goto :usage
if /I "%~1"=="-h" goto :usage
echo [ERREUR] Option d'installation inconnue : %~1
exit /b 2

:arguments_ready
if /I "%PULID_INSTALL_PROFILE%"=="production" goto :profile_ready
if /I "%PULID_INSTALL_PROFILE%"=="development" goto :profile_ready
echo [ERREUR] Profil PULID_INSTALL_PROFILE inconnu : %PULID_INSTALL_PROFILE%
exit /b 2

:profile_ready
set "DEFAULT_MODELS_ROOT=%PROJECT_DIR%PuLID_models"
set "REQUESTED_MODELS_ROOT=%PULID_MODELS_ROOT%"
set "PULID_MODELS_ROOT="

if not defined REQUESTED_MODELS_ROOT goto :check_configured_models_root
rem An explicit root is final, even when it does not exist yet.
cd /d "%PROJECT_DIR%"
if errorlevel 1 goto :error_exit
for %%I in ("%REQUESTED_MODELS_ROOT%") do set "PULID_MODELS_ROOT=%%~fI"
goto :models_root_ready

:check_configured_models_root
set "CONFIG_MODELS_ROOT="
if not exist "%PROJECT_DIR%config\local.yaml" goto :check_default_models_root
for /f "delims=" %%I in ('powershell.exe -NoProfile -Command "$value = $null; foreach ($line in [IO.File]::ReadLines((Join-Path $env:PROJECT_DIR 'config\local.yaml'))) { if ($line.StartsWith('models_root:')) { $value = $line.Substring(12).Trim().Trim([char]39).Trim([char]34); break } }; if ($value) { if (-not [IO.Path]::IsPathRooted($value)) { $value = Join-Path $env:PROJECT_DIR $value }; [IO.Path]::GetFullPath($value) }"') do if not defined CONFIG_MODELS_ROOT set "CONFIG_MODELS_ROOT=%%I"
if not defined CONFIG_MODELS_ROOT goto :check_default_models_root
if exist "%CONFIG_MODELS_ROOT%\" (
    set "PULID_MODELS_ROOT=%CONFIG_MODELS_ROOT%"
    goto :existing_models_root
)

:check_default_models_root
if exist "%DEFAULT_MODELS_ROOT%\" (
    set "PULID_MODELS_ROOT=%DEFAULT_MODELS_ROOT%"
    goto :existing_models_root
)

:prompt_models_root
set "USE_DEFAULT="
set /p "USE_DEFAULT=Utiliser l'emplacement par defaut %DEFAULT_MODELS_ROOT% ? [O/n] "
if /I "%USE_DEFAULT%"=="N" goto :custom_models_root
if /I "%USE_DEFAULT%"=="NON" goto :custom_models_root
if /I "%USE_DEFAULT%"=="NO" goto :custom_models_root
set "PULID_MODELS_ROOT=%DEFAULT_MODELS_ROOT%"
goto :models_root_ready

:custom_models_root
set "CUSTOM_MODELS_ROOT="
set /p "CUSTOM_MODELS_ROOT=Chemin du dossier parent ou d'un dossier PuLID_models : "
if not defined CUSTOM_MODELS_ROOT goto :custom_models_root
set "CUSTOM_MODELS_ROOT=%CUSTOM_MODELS_ROOT:"=%"
for %%I in ("%CUSTOM_MODELS_ROOT%") do set "CUSTOM_MODELS_FULL=%%~fI"
for %%I in ("%CUSTOM_MODELS_FULL%") do set "CUSTOM_MODELS_NAME=%%~nxI"
if /I "%CUSTOM_MODELS_NAME%"=="PuLID_models" (
    set "PULID_MODELS_ROOT=%CUSTOM_MODELS_FULL%"
) else (
    set "PULID_MODELS_ROOT=%CUSTOM_MODELS_FULL%\PuLID_models"
)
goto :models_root_ready

:existing_models_root
echo Installation existante detectee : %PULID_MODELS_ROOT%

:models_root_ready
if exist "%PULID_MODELS_ROOT%\" goto :models_root_available
mkdir "%PULID_MODELS_ROOT%"
if errorlevel 1 (
    echo [ERREUR] Impossible de creer le dossier de modeles :
    echo   %PULID_MODELS_ROOT%
    goto :error_exit
)

:models_root_available

set "HF_HOME=%PULID_MODELS_ROOT%\huggingface"
set "HUGGINGFACE_HUB_CACHE=%PULID_MODELS_ROOT%\huggingface\hub"
set "TRANSFORMERS_CACHE=%PULID_MODELS_ROOT%\huggingface\transformers"
set "TORCH_HOME=%PULID_MODELS_ROOT%\torch"
set "XDG_CACHE_HOME=%PULID_MODELS_ROOT%\other"
set "MPLCONFIGDIR=%PULID_MODELS_ROOT%\other\matplotlib"
set "UV_CACHE_DIR=%PULID_MODELS_ROOT%\other\uv-windows"
set "UV_PYTHON_INSTALL_DIR=%PULID_MODELS_ROOT%\other\uv-python-windows"
set "NO_ALBUMENTATIONS_UPDATE=1"
set "LLAMA_CPP_VERSION=0.3.35"
set "LLAMA_CPP_PORTABLE_CPU_DLL_SHA256=cd91f4ed375998da4da57fedaab1b0638fba8b2af88e74a2632bc046e7fa4850"
set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
set "TORCH_DLL_DIR=%PROJECT_DIR%.venv\Lib\site-packages\torch\lib"
set "LLAMA_CPP_LIB_DIR=%PROJECT_DIR%.venv\Lib\site-packages\llama_cpp\lib"
set "PATH=%TORCH_DLL_DIR%;%PATH%"
cd /d "%PROJECT_DIR%"

echo Installation de Python gere et recreation de .venv depuis uv.lock...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%scripts\bootstrap_windows.ps1"
if errorlevel 1 goto :dependency_error

echo.
echo Installation ou reparation des modeles et configurations...
"%VENV_PYTHON%" -I -m pulid_app.installer --models-root "%PULID_MODELS_ROOT%" --sdxl ask
if errorlevel 1 goto :model_install_error

echo Verification de CUDA...
"%VENV_PYTHON%" -I -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponible : mettez a jour le pilote NVIDIA'; print('CUDA OK :', torch.cuda.get_device_name(0), '- PyTorch', torch.__version__)"
if errorlevel 1 goto :cuda_error

echo Verification du runtime GGUF CUDA...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $env:LLAMA_CPP_LIB_DIR 'ggml-cpu.dll')).Hash.ToLowerInvariant(); if ($actual -ne $env:LLAMA_CPP_PORTABLE_CPU_DLL_SHA256) { Write-Error ('Le backend CPU portable a ete remplace : ' + $actual); exit 1 }"
if errorlevel 1 goto :llama_portable_error

"%VENV_PYTHON%" -I -c "from pulid_app.models.text_embedding import _cuda_dll_search_path; dll_context = _cuda_dll_search_path('cuda'); dll_context.__enter__(); import llama_cpp; info = llama_cpp.llama_print_system_info().decode(); assert 'CUDA' in info, 'Backend CUDA absent de llama-cpp-python'; assert 'AVX512 = 1' not in info, 'Backend CPU AVX-512 incompatible encore installe'; print(info); print('llama-cpp-python CUDA OK :', llama_cpp.__version__); dll_context.__exit__(None, None, None)"
if errorlevel 1 goto :llama_cuda_error

echo Verification du chargement et du calcul BGE-M3 sur CUDA...
"%VENV_PYTHON%" -I "%PROJECT_DIR%scripts\verify_text_embedding.py" --device cuda
if errorlevel 1 goto :llama_context_error

echo Verification de l'installation et des modeles...
"%VENV_PYTHON%" -I -m pulid_app.cli doctor --allow-missing-sdxl
if errorlevel 1 goto :validation_error

"%VENV_PYTHON%" -I "%PROJECT_DIR%scripts\inspect_models.py" --show-cache-env --fail-on-internal-cache --allow-missing-sdxl
if errorlevel 1 goto :validation_error

if "%PULID_CONFIGURE_NETWORK%"=="1" goto :ask_firewall
echo.
echo Pare-feu Windows non modifie. Le serveur reste local par defaut.
echo Mode reseau avance : relancez install_windows.bat --network.
goto :firewall_done

:ask_firewall
echo.
set "OPEN_FIREWALL="
set /p "OPEN_FIREWALL=Autoriser l'acces a PuLID depuis d'autres appareils du reseau local ? [o/N] "
if not defined OPEN_FIREWALL goto :skip_firewall
if /I "%OPEN_FIREWALL%"=="O" goto :configure_firewall
if /I "%OPEN_FIREWALL%"=="OUI" goto :configure_firewall
if /I "%OPEN_FIREWALL%"=="Y" goto :configure_firewall
if /I "%OPEN_FIREWALL%"=="YES" goto :configure_firewall
if /I "%OPEN_FIREWALL%"=="N" goto :skip_firewall
if /I "%OPEN_FIREWALL%"=="NON" goto :skip_firewall
if /I "%OPEN_FIREWALL%"=="NO" goto :skip_firewall
echo Repondez oui ou non.
goto :ask_firewall

:configure_firewall
netsh advfirewall firewall show rule name="PuLID_API_12693" >nul 2>&1
if errorlevel 1 (
    echo Autorisation du port TCP 12693 sur les reseaux prives...
    echo Windows va demander une confirmation administrateur.
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$process = Start-Process -FilePath netsh.exe -ArgumentList @('advfirewall','firewall','add','rule','name=PuLID_API_12693','dir=in','action=allow','protocol=TCP','localport=12693','profile=private') -Verb RunAs -Wait -PassThru; exit $process.ExitCode"
    if errorlevel 1 (
        echo [AVERTISSEMENT] La regle de pare-feu n'a pas ete creee.
        echo Relancez install_windows.bat ou ajoutez manuellement le port TCP 12693 au profil prive.
    )
) else (
    echo La regle de pare-feu PuLID_API_12693 existe deja.
)
goto :firewall_done

:skip_firewall
echo Pare-feu Windows non modifie. PuLID reste utilisable sur cette machine.

:firewall_done

echo.
echo Installation terminee.
echo Lancez ensuite : start_windows.bat
echo.
echo Appuyez sur une touche pour fermer cette fenetre.
pause >nul
exit /b 0

:dependency_error
echo [ERREUR] Installation des dependances impossible.
echo InsightFace doit provenir de sa wheel officielle, sans Microsoft C++ Build Tools.
echo Le runtime GGUF doit provenir de la wheel CUDA 13.0 indiquee par le script.
goto :error_exit

:llama_portable_error
echo [ERREUR] Impossible de preparer le backend CPU portable de llama-cpp-python.
echo Ce backend evite les instructions AVX-512 incompatibles avec certains Core i9,
echo sans desactiver CUDA pour les embeddings.
echo Relancez install_windows.bat afin de retablir les DLL de la version %LLAMA_CPP_VERSION%.
goto :error_exit

:cuda_error
echo [ERREUR] PyTorch ne detecte pas la carte NVIDIA.
echo Installez le dernier pilote NVIDIA compatible puis relancez ce script.
goto :error_exit

:llama_cuda_error
echo [ERREUR] llama-cpp-python ne parvient pas a charger ses DLL CUDA.
echo Le script a recherche les DLL CUDA dans PyTorch et dans le pilote NVIDIA.
echo Dossier PyTorch :
echo   %TORCH_DLL_DIR%
echo Mettez a jour le pilote NVIDIA afin d'installer nvcudart_hybrid64.dll,
echo puis relancez install_windows.bat.
goto :error_exit

:llama_context_error
echo [ERREUR] Le runtime CUDA est present mais BGE-M3 ne peut pas charger ou calculer.
echo Le test utilise la fenetre complete de 8192 tokens et le meme chemin que le serveur.
echo Consultez la trace affichee ci-dessus puis relancez install_windows.bat.
goto :error_exit

:validation_error
echo [ERREUR] L'installation ou le dossier PuLID_models est incomplet.
echo Consultez les erreurs affichees ci-dessus, corrigez-les puis relancez ce script.
goto :error_exit

:model_install_error
echo [ERREUR] Les modeles ou configurations n'ont pas pu etre installes.
echo Verifiez la connexion reseau et l'espace libre, puis relancez install_windows.bat.
goto :error_exit

:error_exit
echo.
echo Appuyez sur une touche pour fermer cette fenetre.
pause >nul
exit /b 1

:usage
echo Usage : install_windows.bat [--production^|--development] [--network]
echo   --production  installe le runtime sans dependances de test
echo   --development conserve l'installation editable avec les dependances dev
echo   --network     propose explicitement la regle de pare-feu privee
exit /b 0
