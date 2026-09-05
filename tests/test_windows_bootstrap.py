"""Exercise the actual uv bootstrap block with a local, network-free installer."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _powershell() -> str:
    command = (
        os.environ.get('PULID_TEST_POWERSHELL')
        or shutil.which('powershell.exe')
        or shutil.which('pwsh')
    )
    if command is None:
        pytest.skip('PowerShell required; set PULID_TEST_POWERSHELL to a portable runtime')
    return command


def _quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _fake_uv(path: Path, version: str, exit_code: int = 0) -> None:
    if os.name == 'nt':
        content = f'@echo off\necho uv {version}\nexit /b {exit_code}\n'
    else:
        content = f'#!/bin/sh\necho "uv {version}"\nexit {exit_code}\n'
    path.write_text(content, encoding='ascii')
    path.chmod(0o755)


@pytest.mark.parametrize(
    'previous_exit,existing_version,installer_result,expected_success,expected_download',
    [
        ('$null', None, 'valid', True, True),
        ('17', None, 'valid', True, True),
        ('$null', None, 'missing', False, True),
        ('$null', None, 'wrong-version', False, True),
        ('$null', None, 'native-failure', False, True),
        ('$null', None, 'exception', False, True),
        ('17', '0.12.10', 'exception', True, False),
        ('0', '0.7.8', 'valid', True, True),
    ],
)
def test_uv_bootstrap_checks_installed_binary_not_installer_exit_code(
    tmp_path: Path,
    previous_exit: str,
    existing_version: str | None,
    installer_result: str,
    expected_success: bool,
    expected_download: bool,
) -> None:
    powershell = _powershell()
    suffix = '.cmd' if os.name == 'nt' else ''
    installation = tmp_path / 'PuLID with spaces'
    installation.mkdir()
    uv = installation / f'uv{suffix}'
    fixture = tmp_path / f'downloaded-uv{suffix}'
    _fake_uv(fixture, '0.7.8' if installer_result == 'wrong-version' else '0.12.10',
             8 if installer_result == 'native-failure' else 0)
    if existing_version is not None:
        _fake_uv(uv, existing_version)

    installer_script = f'Copy-Item -LiteralPath {_quote(fixture)} -Destination $uv -Force'
    if installer_result == 'missing':
        installer_script = '# Installer returned without creating uv'
    downloader = f'return {_quote(installer_script)}'
    if installer_result == 'exception':
        downloader = "throw 'Download failed'"

    source = (ROOT / 'scripts/bootstrap_windows.ps1').read_text(encoding='utf-8')
    # Run the production selection/install/verification code unchanged. Only the
    # network endpoint and native uv program are replaced by local fixtures.
    block = source[source.index("    $installed = ''"):source.index('    & $uv python install')]
    harness = tmp_path / 'verify-bootstrap.ps1'
    harness.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f'$uv = {_quote(uv)}\n'
        "$uvVersion = '0.12.10'\n"
        f'$global:LASTEXITCODE = {previous_exit}\n'
        'function Invoke-RestMethod {\n'
        '    param([string]$Uri)\n'
        "    Write-Host 'DOWNLOAD_CALLED'\n"
        f'    {downloader}\n'
        '}\ntry {\n' + block + "\nWrite-Host 'BOOTSTRAP_OK'\nexit 0\n"
        '} catch {\nWrite-Host $_.Exception.Message\nexit 1\n}\n',
        encoding='utf-8',
    )
    result = subprocess.run(
        [powershell, '-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy',
         'Bypass', '-File', str(harness)], cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    output = result.stdout + result.stderr
    assert (result.returncode == 0) == expected_success, output
    assert ('DOWNLOAD_CALLED' in output) == expected_download, output
    assert ('BOOTSTRAP_OK' in output) == expected_success, output
