"""Optional native regression using the pinned uv and a private managed Python copy."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform != 'win32', reason='Requires native Windows executables and junctions')
def test_uv_trampoline_is_repaired_after_managed_python_moves(tmp_path):
    uv_raw = os.environ.get('PULID_TEST_UV')
    if not uv_raw:
        pytest.skip('Set PULID_TEST_UV to the pinned PuLID uv.exe to run the native relocation regression')
    uv = Path(uv_raw).resolve()
    version = (ROOT / '.python-version').read_text().strip()
    uv_version = (ROOT / '.uv-version').read_text().strip()
    assert platform.python_version() == version
    assert subprocess.check_output([str(uv), '--version'], text=True).split()[1] == uv_version

    root = tmp_path / 'original'
    installs = root / 'models/other/uv-python-windows'
    managed = installs / f'cpython-{version}-windows-x86_64-none'
    # Copy only the runtime, without packages/models, never mutate the installed
    # Python used by pytest. This optional test needs a few hundred MB in TEMP.
    shutil.copytree(Path(sys.base_prefix), managed,
                    ignore=shutil.ignore_patterns('site-packages', '__pycache__'))
    python = managed / 'python.exe'
    minor = installs / 'cpython-3.11-windows-x86_64-none'
    subprocess.run([os.environ['COMSPEC'], '/d', '/c', 'mklink', '/J', str(minor), str(managed)],
                   check=True, capture_output=True, text=True)
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(('UV_', 'PYTHON', 'PIP_'))
                   and key.upper() not in {'VIRTUAL_ENV', 'CONDA_PREFIX'}}
    environment.update(UV_PYTHON_INSTALL_DIR=str(installs), UV_CACHE_DIR=str(root / 'uv-cache'))
    try:
        subprocess.run([str(uv), 'venv', '--relocatable', '--python', '3.11', '--managed-python',
                        '--offline', '--no-config', str(root / '.venv')],
                       env=environment, cwd=root, check=True, capture_output=True, text=True)
    finally:
        # Removing the junction itself does not remove the managed runtime. It
        # reproduces a broken minor target without leaving a junction in pytest's
        # cleanup tree (including on a failed assertion).
        os.rmdir(minor)
    scripts = root / '.venv/Scripts'
    assert b'uv trampoline' in (scripts / 'python.exe').read_bytes()
    (root / 'scripts').mkdir()
    for name in ('check_environment.py', 'prepare_runtime_windows.ps1'):
        shutil.copy2(ROOT / 'scripts' / name, root / 'scripts' / name)
    for name in ('.python-version', '.uv-version', 'uv.lock'):
        shutil.copy2(ROOT / name, root / name)
    state = dict(project_root=str(root), managed_python=str(python),
                 managed_python_relative=str(python.relative_to(root)),
                 python=version, uv=uv_version, profile='production',
                 lock_sha256=hashlib.sha256((root / 'uv.lock').read_text(encoding='utf-8').encode()).hexdigest())
    (root / '.venv/pulid-runtime.json').write_text(json.dumps(state), encoding='utf-8')
    (root / '.venv/pulid-python-path').write_text(state['managed_python_relative'] + '\n', encoding='utf-8')
    (root / '.venv/keep.txt').write_text('dependencies are preserved')
    moved = tmp_path / 'PuLID déplacé avec espaces'
    root.rename(moved)
    venv_python = moved / '.venv/Scripts/python.exe'
    result = subprocess.run([str(venv_python), '-I', '-c', 'pass'], capture_output=True, text=True)
    assert result.returncode != 0 and 'uv trampoline failed to spawn' in result.stderr
    environment.update(PYTHONHOME=str(tmp_path / 'foreign'), PYTHONPATH=str(tmp_path / 'global-packages'))
    for name in (None, 'second move'):
        if name:
            target = tmp_path / name
            moved.rename(target)
            moved = target
        result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                 str(moved / 'scripts/prepare_runtime_windows.ps1'), '-ProjectRoot', str(moved)],
                                env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        venv_python = moved / '.venv/Scripts/python.exe'
        subprocess.run([str(venv_python), '-I', str(moved / 'scripts/check_environment.py')],
                       env=environment, cwd=tmp_path, check=True, capture_output=True, text=True)
        result = subprocess.run([str(venv_python), '-I', '-c',
                                 'import sys, site; assert sys.flags.isolated; assert not site.ENABLE_USER_SITE'],
                                env=environment, check=True, capture_output=True, text=True)
        expected = moved / python.relative_to(root)
        assert venv_python.read_bytes() == (expected.parent / 'Lib/venv/scripts/nt/python.exe').read_bytes()
        assert (moved / '.venv/keep.txt').read_text() == 'dependencies are preserved'
