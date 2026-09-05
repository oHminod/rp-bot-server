"""Offline relocation tests; native Windows runs also exercise the venv redirector."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import venv

import pytest

from scripts import check_environment

ROOT = Path(__file__).resolve().parents[1]


def state_fixture(root, python, profile='production'):
    (root / '.venv').mkdir(parents=True, exist_ok=True)
    (root / '.python-version').write_text(platform.python_version())
    (root / '.uv-version').write_text('0.12.10')
    (root / 'uv.lock').write_text('locked')
    state = dict(project_root=str(root), managed_python=str(python),
                 python=platform.python_version(), uv='0.12.10', profile=profile,
                 lock_sha256=hashlib.sha256(b'locked').hexdigest())
    (root / '.venv/pulid-runtime.json').write_text(json.dumps(state))
    return state


@pytest.mark.parametrize('windows', [False, True])
@pytest.mark.parametrize('external', [False, True])
def test_repair_rebases_internal_python_and_preserves_external_python(tmp_path, monkeypatch, windows, external):
    root = tmp_path / 'old'
    python = (tmp_path / 'external' if external else root / 'models') / ('python.exe' if windows else 'bin/python3.11')
    python.parent.mkdir(parents=True)
    python.touch()
    state_fixture(root, python, 'development')
    config = root / '.venv/pyvenv.cfg'
    config.write_text(f'home = {python.parent}\ninclude-system-site-packages = false\n')
    packages_relative = 'Lib/site-packages' if windows else 'lib/python3.11/site-packages'
    packages = root / '.venv' / packages_relative
    packages.mkdir(parents=True)
    (packages / '_pulid_app.pth').write_text(str(root / 'src'))
    (root / '.venv/bin').mkdir()
    moved = tmp_path / 'déplacé avec espaces'
    root.rename(moved)
    managed = python if external else moved / python.relative_to(root)
    monkeypatch.setattr(sys, 'platform', 'win32' if windows else 'darwin')
    monkeypatch.setattr(sys, '_base_executable', str(managed))
    check_environment.prepare_environment(moved)
    state = json.loads((moved / '.venv/pulid-runtime.json').read_text())
    assert state['managed_python'] == str(managed)
    assert state['project_root'] == str(moved)
    pointer = (moved / '.venv/pulid-python-path').read_text().strip()
    assert Path(pointer).is_absolute() == external
    assert f'home = {managed.parent}' in (moved / '.venv/pyvenv.cfg').read_text()
    packages = moved / '.venv' / packages_relative
    assert (packages / (packages / '_pulid_app.pth').read_text().strip()).resolve() == moved / 'src'
    if not windows:
        assert (moved / '.venv/bin/python').resolve() == managed
    paths = [moved / '.venv' / name for name in ('pyvenv.cfg', 'pulid-runtime.json', 'pulid-python-path')]
    mtimes = [p.stat().st_mtime_ns for p in paths]
    check_environment.prepare_environment(moved)
    assert [p.stat().st_mtime_ns for p in paths] == mtimes


@pytest.mark.parametrize('problem', ['lock', 'version', 'missing-python', 'foreign-python'])
def test_repair_refuses_incompatible_environment_before_mutation(tmp_path, monkeypatch, problem):
    python = tmp_path / 'python'
    python.touch()
    state_fixture(tmp_path, python)
    config = tmp_path / '.venv/pyvenv.cfg'
    config.write_text('sentinel')
    monkeypatch.setattr(sys, '_base_executable', str(python))
    if problem == 'lock':
        (tmp_path / 'uv.lock').write_text('changed')
    elif problem == 'version':
        (tmp_path / '.python-version').write_text('0.0.0')
    elif problem == 'missing-python':
        python.unlink()
    else:
        monkeypatch.setattr(sys, '_base_executable', str(tmp_path / 'global/python'))
    with pytest.raises(ValueError):
        check_environment.prepare_environment(tmp_path)
    assert config.read_text() == 'sentinel'


@pytest.mark.parametrize('profile', ['production', 'development'])
@pytest.mark.parametrize('frontend', [False, True])
def test_real_launcher_after_two_moves_without_installer(tmp_path, profile, frontend):
    root = tmp_path / 'initial'
    venv.EnvBuilder(with_pip=False).create(root / '.venv')
    state_fixture(root, Path(sys._base_executable), profile)
    (root / 'scripts').mkdir()
    for name in ('check_environment.py', 'prepare_runtime_macos.sh', 'prepare_runtime_windows.ps1'):
        shutil.copy2(ROOT / 'scripts' / name, root / 'scripts' / name)
    packages = root / '.venv' / ('Lib/site-packages' if os.name == 'nt' else 'lib/python3.11/site-packages')
    module_root = root / 'src' if profile == 'development' else packages
    (module_root / 'pulid_app').mkdir(parents=True)
    (module_root / 'pulid_app/__init__.py').write_text('')
    code = ('import sys, site, pulid_app\nfrom pathlib import Path\n'
            'assert sys.flags.isolated and not site.ENABLE_USER_SITE\n'
            'assert Path(pulid_app.__file__).is_relative_to(Path.cwd())\n'
            'assert Path(sys.prefix).resolve() == Path.cwd() / ".venv"\n'
            'print("MOVED_RUNTIME_OK")\n')
    (module_root / 'pulid_app/server.py').write_text(code)
    (root / 'frontend').mkdir()
    (root / 'frontend/server.py').write_text(code)
    if os.name == 'nt':
        launcher = 'start_frontend_windows.bat' if frontend else 'start_windows.bat'
        command = lambda path: [os.environ['COMSPEC'], '/d', '/c', str(path / launcher)]
    else:
        launcher = 'start_frontend_macos.sh' if frontend else 'start_pulid_server.sh'
        command = lambda path: ['/bin/bash', str(path / launcher)]
    shutil.copy2(ROOT / launcher, root / launcher)
    environment = dict(os.environ, PYTHONHOME='/missing-python', PYTHONPATH='/global-packages')
    # First startup also migrates installations made before portable metadata.
    for name in (None, 'PuLID déplacé avec espaces', 'second move'):
        if name:
            moved = tmp_path / name
            root.rename(moved)
            root = moved
        result = subprocess.run(command(root), cwd=root, env=environment, text=True,
                                capture_output=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'MOVED_RUNTIME_OK' in result.stdout
    assert not (root / 'install_environment.py').exists()


@pytest.mark.parametrize('relative,exit_code', [(True, 0), (False, 0), (True, 7)])
def test_powershell_runtime_pointer_and_native_exit(tmp_path, relative, exit_code):
    powershell = os.environ.get('PULID_TEST_POWERSHELL') or shutil.which('powershell.exe') or shutil.which('pwsh')
    if not powershell:
        pytest.skip('PowerShell unavailable')
    root = tmp_path / 'déplacé avec espaces'
    (root / '.venv').mkdir(parents=True)
    (root / 'scripts').mkdir()
    # An interpreter inside the fixture is needed only for testing path selection.
    # Use a native shell executable fixture, no installation or network involved.
    executable = root / ('python.cmd' if os.name == 'nt' else 'python')
    if os.name == 'nt':
        executable.write_text(f'@echo off\necho %*\nexit /b {exit_code}\n')
    else:
        executable.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@"\nexit {exit_code}\n')
        executable.chmod(0o755)
    pointer = executable.name if relative else str(executable)
    (root / '.venv/pulid-python-path').write_text(pointer + '\n', encoding='utf-8')
    result = subprocess.run([powershell, '-NoProfile', '-File', str(ROOT / 'scripts/prepare_runtime_windows.ps1'),
                             '-ProjectRoot', str(root)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == exit_code, result.stderr
    assert '-I' in result.stdout and '--prepare' in result.stdout
    assert str(root / 'scripts/check_environment.py') in result.stdout
