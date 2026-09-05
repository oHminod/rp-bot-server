from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tomllib

import pytest

from scripts import check_environment, install_environment
from scripts.lock_environment import write_lock_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_lock_pins_cuda_and_every_distributed_package():
    install_environment.verify_lock(ROOT)
    lock = tomllib.loads((ROOT / 'uv.lock').read_text())
    assert lock['requires-python'] == '==3.11.16'
    for package in lock['package']:
        assert package['version']
        if package['name'] == 'pulid-app':
            continue
        artifacts = package.get('wheels', []) + ([package['sdist']] if 'sdist' in package else [])
        assert artifacts, package['name']
        assert all(a.get('hash', '').startswith('sha256:') for a in artifacts), package['name']
    torch = [p for p in lock['package'] if p['name'] == 'torch']
    assert {p['version'] for p in torch} == {'2.13.0', '2.13.0+cu130'}
    cuda = next(p for p in torch if p['version'].endswith('+cu130'))
    assert cuda['source']['registry'] == 'https://download.pytorch.org/whl/cu130'


def test_lock_manifest_rejects_changed_metadata(tmp_path):
    from scripts.lock_environment import LOCK_INPUTS
    for name in LOCK_INPUTS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_lock_manifest(tmp_path)
    (tmp_path / 'pyproject.toml').write_text('changed')
    with pytest.raises(RuntimeError, match='Verrou incohérent'):
        install_environment.verify_lock(tmp_path)


def test_managed_python_uses_exact_windows_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(install_environment.platform, 'machine', lambda: 'AMD64')
    assert install_environment.managed_python(ROOT, tmp_path) == (
        tmp_path / 'other/uv-python-windows/cpython-3.11.16-windows-x86_64-none/python.exe'
    )


def test_system_python_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(install_environment, 'managed_python', lambda *args: tmp_path / 'managed/python')
    with pytest.raises(RuntimeError, match='Python géré'):
        install_environment.verify_bootstrap(ROOT, tmp_path)


def test_uv_environment_ignores_system_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv('UV_PYTHON', '/system/python')
    monkeypatch.setenv('UV_INDEX_URL', 'https://unwanted.invalid')
    monkeypatch.setenv('UV_NO_MANAGED_PYTHON', 'true')
    monkeypatch.setenv('PYTHONHOME', '/system')
    monkeypatch.setenv('PYTHONUSERBASE', '/global/packages')
    monkeypatch.setenv('PIP_CONFIG_FILE', '/global/pip.conf')
    monkeypatch.setenv('PATH', '/global/bin')
    monkeypatch.setenv('CUDA_PATH', '/global/cuda')
    result = install_environment.clean_environment(ROOT, tmp_path)
    assert not {'UV_PYTHON', 'UV_INDEX_URL', 'UV_NO_MANAGED_PYTHON', 'PYTHONHOME'} & result.keys()
    assert result['UV_PYTHON_PREFERENCE'] == 'only-managed'
    assert result['UV_PYTHON_DOWNLOADS'] == 'never'
    assert not {'PYTHONUSERBASE', 'PIP_CONFIG_FILE', 'CUDA_PATH'} & result.keys()
    assert '/global/bin' not in result['PATH']
    assert result['PYTHONNOUSERSITE'] == '1'


def test_move_is_diagnosed_before_startup(tmp_path):
    (tmp_path / '.venv').mkdir()
    (tmp_path / '.python-version').write_text('3.11.16')
    (tmp_path / '.venv/pulid-runtime.json').write_text(json.dumps({'project_root': '/old/PuLID'}))
    with pytest.raises(RuntimeError, match='déplacé.*Relancez'):
        check_environment.check_environment(tmp_path)


def test_managed_environment_matches_current_location(tmp_path, monkeypatch):
    venv = tmp_path / '.venv'
    venv.mkdir()
    python = tmp_path / 'python'
    python.touch()
    (tmp_path / '.python-version').write_text('3.11.16')
    (tmp_path / '.uv-version').write_text('0.12.10')
    (tmp_path / 'uv.lock').write_text('locked')
    state = {'project_root': str(tmp_path), 'managed_python': str(python),
             'python': '3.11.16', 'uv': '0.12.10',
             'lock_sha256': hashlib.sha256(b'locked').hexdigest()}
    (venv / 'pulid-runtime.json').write_text(json.dumps(state))
    monkeypatch.setattr(sys, 'prefix', str(venv))
    monkeypatch.setattr(sys, '_base_executable', str(python))
    monkeypatch.setattr(check_environment.platform, 'python_version', lambda: '3.11.16')
    assert check_environment.check_environment(tmp_path) == state


@pytest.mark.parametrize('profile', ['production', 'development'])
def test_install_recreates_venv_and_uses_only_frozen_dependencies(tmp_path, monkeypatch, profile):
    monkeypatch.setattr(install_environment, 'verify_lock', lambda root: None)
    monkeypatch.setattr(install_environment, 'verify_bootstrap', lambda *args: tmp_path / 'managed/python')
    # Command orchestration is platform-independent; native GPU patch is tested separately.
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(install_environment.subprocess, 'check_output', lambda *args, **kwargs: 'uv 0.12.10\n')
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if 'venv' in command:
            (tmp_path / '.venv').mkdir()
    monkeypatch.setattr(install_environment.subprocess, 'run', run)
    (tmp_path / '.uv-version').write_text('0.12.10')
    (tmp_path / 'uv.lock').write_text('locked')
    (tmp_path / 'pyproject.toml').write_text((ROOT / 'pyproject.toml').read_text())
    install_environment.install(tmp_path, tmp_path / 'models', tmp_path / 'uv', profile)
    assert commands[0][1:3] == ['venv', '--clear']
    assert '--only-group' in commands[1] and 'build' in commands[1]
    assert '--no-build-isolation' in commands[2]
    assert all('--frozen' in command for command in commands if 'sync' in command)
    assert ('dev' in commands[2]) == (profile == 'development')
    assert ('--no-editable' in commands[2]) == (profile == 'production')
    assert commands[3][1:3] == ['pip', 'check']
    assert all('--no-config' in command for command in commands)
    assert '--no-build-package' in commands[2]
    assert (tmp_path / '.venv/pulid-runtime.json').is_file()


def test_install_refuses_venv_symlink_before_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(install_environment, 'verify_lock', lambda root: None)
    monkeypatch.setattr(install_environment, 'verify_bootstrap', lambda *args: tmp_path / 'managed/python')
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(install_environment.subprocess, 'check_output', lambda *args, **kwargs: 'uv 0.12.10\n')
    (tmp_path / '.uv-version').write_text('0.12.10')
    target = tmp_path / 'unrelated'
    target.mkdir()
    try:
        (tmp_path / '.venv').symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip('Symlink creation not allowed')
    with pytest.raises(RuntimeError, match='lien/jonction'):
        install_environment.install(tmp_path, tmp_path / 'models', tmp_path / 'uv', 'production')
    assert target.is_dir()


@pytest.mark.parametrize('corrupt', [False, True])
def test_windows_cpu_patch_checks_whole_wheel_and_preserves_cuda(tmp_path, monkeypatch, corrupt):
    from io import BytesIO
    import zipfile
    output = BytesIO()
    dll = b'portable CPU DLL fixture'
    with zipfile.ZipFile(output, 'w') as archive:
        archive.writestr('llama_cpp/lib/ggml-cpu.dll', dll)
    content = output.getvalue()
    library = tmp_path / 'Lib/site-packages/llama_cpp/lib'
    library.mkdir(parents=True)
    (library / 'ggml-cuda.dll').write_bytes(b'CUDA unchanged')
    monkeypatch.setattr(install_environment.urllib.request, 'urlopen', lambda *args, **kwargs: BytesIO(content))
    monkeypatch.setattr(install_environment, 'CPU_WHEEL_SHA256', 'bad' if corrupt else hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(install_environment, 'CPU_DLL_SHA256', hashlib.sha256(dll).hexdigest())
    if corrupt:
        with pytest.raises(RuntimeError, match='Empreinte wheel CPU'):
            install_environment.patch_windows_cpu_backend(tmp_path)
        assert not (library / 'ggml-cpu.dll').exists()
    else:
        install_environment.patch_windows_cpu_backend(tmp_path)
        assert (library / 'ggml-cpu.dll').read_bytes() == dll
    assert (library / 'ggml-cuda.dll').read_bytes() == b'CUDA unchanged'


def test_lock_manifest_accepts_windows_checkout_line_endings(tmp_path):
    from scripts.lock_environment import LOCK_INPUTS
    for name in LOCK_INPUTS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_lock_manifest(tmp_path)
    for name in LOCK_INPUTS:
        target = tmp_path / name
        target.write_bytes(target.read_bytes().replace(b'\n', b'\r\n'))
    install_environment.verify_lock(tmp_path)


@pytest.mark.parametrize('frontend', [False, True])
def test_launchers_ignore_python_environment_and_user_packages(tmp_path, frontend):
    import os
    import shutil
    import subprocess
    import venv

    project = tmp_path / 'PuLID with spaces'
    project.mkdir()
    venv.EnvBuilder(with_pip=False).create(project / '.venv')
    marker = tmp_path / 'global-python-loaded'
    unwanted = tmp_path / 'global-packages'
    unwanted.mkdir()
    (unwanted / 'sitecustomize.py').write_text(
        f'from pathlib import Path; Path({str(marker)!r}).touch()'
    )
    verification = (
        'import sys, site\n'
        'assert sys.flags.isolated == 1\n'
        'assert not site.ENABLE_USER_SITE\n'
        f'assert {str(unwanted)!r} not in sys.path\n'
        'print("ISOLATED_PULID_OK")\n'
    )
    (project / 'scripts').mkdir()
    (project / 'scripts/check_environment.py').write_text(verification)
    if frontend:
        (project / 'frontend').mkdir()
        (project / 'frontend/server.py').write_text(verification)
    else:
        packages = project / '.venv' / (
            'Lib/site-packages' if os.name == 'nt'
            else f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
        )
        (packages / 'pulid_app').mkdir(parents=True)
        (packages / 'pulid_app/__init__.py').write_text('')
        (packages / 'pulid_app/server.py').write_text(verification)
        (project / 'pulid_app.py').write_text('raise RuntimeError("CWD package used")')
    if os.name == 'nt':
        launcher = 'start_frontend_windows.bat' if frontend else 'start_windows.bat'
        command = [os.environ['COMSPEC'], '/d', '/c', str(project / launcher)]
    else:
        launcher = 'start_frontend_macos.sh' if frontend else 'start_pulid_server.sh'
        command = ['/bin/bash', str(project / launcher)]
    shutil.copy2(ROOT / launcher, project / launcher)
    environment = dict(os.environ, PYTHONHOME=str(tmp_path / 'wrong-python'),
                       PYTHONPATH=str(unwanted), PYTHONUSERBASE=str(unwanted))
    result = subprocess.run(command, cwd=project, env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('ISOLATED_PULID_OK') == 2
    assert not marker.exists()
