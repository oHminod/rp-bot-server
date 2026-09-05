from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (PROJECT_ROOT / name).read_text(encoding="utf-8")


def test_historical_install_and_start_entrypoints_remain_supported() -> None:
    for name in (
        "install_macos.sh",
        "install_windows.bat",
        "start_pulid_server.sh",
        "start_windows.bat",
    ):
        assert (PROJECT_ROOT / name).is_file(), name

    assert (PROJECT_ROOT / "install_macos.sh").stat().st_mode & 0o111
    assert (PROJECT_ROOT / "start_pulid_server.sh").stat().st_mode & 0o111


def test_macos_production_profile_is_additive_and_excludes_dev_dependencies() -> None:
    installer = _read("install_macos.sh")
    wrapper = _read("install_production_macos.sh")

    assert (
        'PULID_INSTALL_PROFILE="${PULID_INSTALL_PROFILE:-development}"' in installer
    )
    assert 'scripts/install_environment.py' in installer
    assert '--profile "${PULID_INSTALL_PROFILE}"' in installer
    assert 'command -v uv' not in installer
    assert 'install_macos.sh" --production' in wrapper
    assert 'export PULID_PROJECT_ROOT="${PROJECT_DIR}"' in installer


def test_windows_production_profile_is_additive_and_excludes_dev_dependencies() -> None:
    installer = _read("install_windows.bat")
    wrapper = _read("install_production_windows.bat")

    assert "PULID_INSTALL_PROFILE=development" in installer
    assert 'scripts\\bootstrap_windows.ps1' in installer
    assert 'where uv.exe' not in installer
    assert "install_windows.bat\" --production" in wrapper
    assert 'set "PULID_PROJECT_ROOT=%PROJECT_DIR%"' in installer


def test_windows_defaults_to_loopback_with_explicit_advanced_network_mode() -> None:
    launcher = _read("start_windows.bat")
    installer = _read("install_windows.bat")

    assert 'set "SERVER_HOST=127.0.0.1"' in launcher
    assert 'if /I "%~1"=="--network"' in launcher
    assert 'set "SERVER_HOST=0.0.0.0"' in launcher
    assert 'set "SERVER_CORS=--cors-origin *"' in launcher
    assert "shift" not in launcher
    assert 'if "%PULID_CONFIGURE_NETWORK%"=="1" goto :ask_firewall' in installer
    assert "Mode reseau avance : relancez install_windows.bat --network." in installer
