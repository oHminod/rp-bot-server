"""Mises à jour incrémentales : conserver les installations et refuser les runtimes incompatibles."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import pytest

from scripts import check_environment, install_environment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def installation(tmp_path, monkeypatch):
    venv = tmp_path / '.venv'
    venv.mkdir()
    (venv / 'keep.txt').write_text('existing package')
    python = tmp_path / 'managed/python'
    python.parent.mkdir()
    python.touch()
    (tmp_path / '.python-version').write_text('3.11.16')
    (tmp_path / '.uv-version').write_text('0.12.10')
    (tmp_path / 'uv.lock').write_text('updated locked packages')
    (tmp_path / 'pyproject.toml').write_text((ROOT / 'pyproject.toml').read_text())
    state = dict(project_root=str(tmp_path), managed_python=str(python),
                 python='3.11.16', uv='0.12.10', profile='production', lock_sha256='old-lock')
    state_path = venv / 'pulid-runtime.json'
    state_path.write_text(json.dumps(state))
    runtime = dict(prefix=str(venv), base=str(python), version='3.11.16')
    monkeypatch.setattr(install_environment, 'verify_lock', lambda root: None)
    monkeypatch.setattr(install_environment, 'verify_bootstrap', lambda *args: python)
    monkeypatch.setattr(install_environment.platform, 'python_version', lambda: '3.11.16')
    monkeypatch.setattr(sys, 'platform', 'linux')  # Native DLL patch is tested separately.
    monkeypatch.setattr(install_environment.subprocess, 'check_output',
                        lambda command, **kwargs: 'uv 0.12.10\n' if '--version' in command else json.dumps(runtime))
    calls = []
    monkeypatch.setattr(install_environment.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    return tmp_path, state_path, runtime, calls


@pytest.mark.parametrize('profile', ['production', 'development'])
def test_update_preserves_venv_and_profile_without_removing_runtime_packages(installation, profile):
    root, state_path, _, calls = installation
    state = json.loads(state_path.read_text())
    state['profile'] = profile
    state_path.write_text(json.dumps(state))
    inode = (root / '.venv/keep.txt').stat().st_ino
    install_environment.install(root, root / 'models', root / 'uv', update=True)
    assert not any('venv' in command or '--clear' in command or '--reinstall' in command for command in calls)
    syncs = [command for command in calls if 'sync' in command]
    assert len(syncs) == 2
    assert all('--frozen' in command and '--inexact' in command for command in syncs)
    assert '--only-group' in syncs[0] and '--no-install-project' in syncs[0]
    assert syncs[1][syncs[1].index('--reinstall-package') + 1] == 'pulid-app'
    assert ('dev' in syncs[1]) == (profile == 'development')
    assert ('--no-editable' in syncs[1]) == (profile == 'production')
    assert (root / '.venv/keep.txt').stat().st_ino == inode
    assert (root / '.venv/keep.txt').read_text() == 'existing package'
    updated = json.loads(state_path.read_text())
    assert updated['profile'] == profile
    assert updated['lock_sha256'] == hashlib.sha256((root / 'uv.lock').read_bytes()).hexdigest()
    assert calls[-2][-1] == '--prepare'
    assert calls[-1] == [str(root / '.venv/bin/python'), '-I', '-m', 'pulid_app.installer',
                         '--models-root', str(root / 'models'), '--qwen3vl-config-only']


@pytest.mark.parametrize('failure', ['missing_marker', 'bad_marker', 'python_pin', 'managed_path', 'project_path', 'profile', 'actual_python', 'actual_prefix', 'actual_base'])
def test_update_refuses_incompatible_environment_before_sync(installation, failure):
    root, state_path, runtime, calls = installation
    state = json.loads(state_path.read_text())
    if failure == 'missing_marker':
        state_path.unlink()
    elif failure == 'bad_marker':
        state_path.write_text('not json')
    elif failure.startswith('actual_'):
        runtime[{'actual_python': 'version', 'actual_prefix': 'prefix', 'actual_base': 'base'}[failure]] = 'wrong'
    else:
        state[{'python_pin': 'python', 'managed_path': 'managed_python', 'project_path': 'project_root', 'profile': 'profile'}[failure]] = 'wrong'
        state_path.write_text(json.dumps(state))
    before = state_path.read_bytes() if state_path.exists() else None
    with pytest.raises(RuntimeError, match='Aucune recréation automatique'):
        install_environment.install(root, root / 'models', root / 'uv', update=True)
    assert not calls
    assert (state_path.read_bytes() if state_path.exists() else None) == before
    assert (root / '.venv/keep.txt').read_text() == 'existing package'


def test_failed_sync_does_not_mark_new_lock_as_installed(installation, monkeypatch):
    root, state_path, _, _ = installation
    before = state_path.read_bytes()
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(install_environment.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        install_environment.install(root, root / 'models', root / 'uv', update=True)
    assert state_path.read_bytes() == before


def test_qwen_configuration_failure_is_not_reported_as_successful_update(installation, monkeypatch):
    root, _, _, calls = installation
    def run(command, **kwargs):
        calls.append(command)
        if '--qwen3vl-config-only' in command:
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(install_environment.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        install_environment.install(root, root / 'models', root / 'uv', update=True)
    assert (root / '.venv/keep.txt').read_text() == 'existing package'
    assert not any('--clear' in command for command in calls)


def test_update_rejects_profile_override(installation):
    root, _, _, calls = installation
    with pytest.raises(ValueError, match='conserve le profil'):
        install_environment.install(root, root / 'models', root / 'uv', 'development', update=True)
    assert not calls


def test_changed_lock_recommends_update_without_full_reinstallation(installation, monkeypatch, capsys):
    root, state_path, _, _ = installation
    monkeypatch.setattr(sys, 'platform', 'win32')
    state = json.loads(state_path.read_text())
    with pytest.raises(check_environment.DependencyLockChangedError, match='install_windows.bat --update'):
        check_environment.validate_state(root, state)
    monkeypatch.setattr(check_environment, 'check_environment', lambda root: check_environment.validate_state(root, state))
    # The main entrypoint normally derives the real project root; use our fixture.
    monkeypatch.setattr(check_environment, '__file__', str(root / 'scripts/check_environment.py'))
    monkeypatch.setattr(sys, 'argv', ['check_environment.py'])
    assert check_environment.main() == 1
    assert capsys.readouterr().err.strip() == '[ERREUR] Les dépendances verrouillées ont changé. Exécutez install_windows.bat --update.'


def test_correct_cpu_dll_is_reused_without_download(tmp_path, monkeypatch):
    library = tmp_path / 'Lib/site-packages/llama_cpp/lib'
    library.mkdir(parents=True)
    (library / 'ggml-cuda.dll').write_bytes(b'cuda')
    dll = library / 'ggml-cpu.dll'
    dll.write_bytes(b'verified portable CPU DLL')
    monkeypatch.setattr(install_environment, 'CPU_DLL_SHA256', hashlib.sha256(dll.read_bytes()).hexdigest())
    monkeypatch.setattr(install_environment.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('Unexpected download'))
    before = dll.stat().st_mtime_ns
    install_environment.patch_windows_cpu_backend(tmp_path)
    assert dll.stat().st_mtime_ns == before


def test_windows_update_branch_has_unique_labels_and_skips_model_installation():
    source = (ROOT / 'install_windows.bat').read_text()
    labels = re.findall(r'^:(\w+)$', source, re.MULTILINE)
    assert len(labels) == len(set(labels))
    assert set(re.findall(r'goto :(\w+)', source)) <= set(labels)
    checks = source.split('\n:arguments_ready\n')[1].split('\n:profile_ready\n')[0]
    assert 'pulid-runtime.json' in checks and 'update_options_error' in checks
    dispatch = source.index('if "%PULID_UPDATE_ONLY%"=="1" goto :update_dependencies')
    assert dispatch < source.index('-m pulid_app.installer')
    update = source.split('\n:update_dependencies\n')[1].split('\n:update_options_error\n')[0]
    assert 'bootstrap_windows.ps1" -Update' in update
    assert 'exit /b 0' in update and 'pulid_app.installer' not in update


def test_real_uv_incremental_sync_preserves_existing_packages(tmp_path):
    """Vraies synchronisations hors ligne, avec wheels minuscules créées sur place."""
    uv = os.environ.get('PULID_TEST_UV')
    if not uv:
        pytest.skip('Set PULID_TEST_UV to the pinned uv executable for the offline installation test')
    expected_uv = (ROOT / '.uv-version').read_text().strip()
    assert subprocess.check_output([uv, '--version'], text=True).split()[1] == expected_uv
    environment = install_environment.clean_environment(tmp_path, tmp_path / 'models')
    environment['UV_PYTHON_PREFERENCE'] = 'only-system'  # Temporary venv from the test interpreter.

    def run(*args):
        return subprocess.run([uv, *args, '--no-config', '--offline', '--no-python-downloads'],
            cwd=tmp_path, env=environment, capture_output=True, text=True, check=True)

    for name in ('heavy_fixture', 'kernel_fixture', 'extra_fixture'):
        wheel = tmp_path / f'{name}-1.0-py3-none-any.whl'
        dist = f'{name}-1.0.dist-info'
        files = {
            f'{name}/__init__.py': 'VERSION = "1.0"\n',
            f'{dist}/METADATA': f'Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n',
            f'{dist}/WHEEL': 'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
        }
        files[f'{dist}/RECORD'] = ''.join(f'{path},,\n' for path in files) + f'{dist}/RECORD,,\n'
        with zipfile.ZipFile(wheel, 'w') as archive:
            for name_in_archive, content in files.items():
                archive.writestr(name_in_archive, content)

    def project(with_kernel):
        dependencies = ['heavy-fixture', *(['kernel-fixture'] if with_kernel else [])]
        (tmp_path / 'pyproject.toml').write_text(
            '[project]\nname = "update-fixture"\nversion = "1.0"\nrequires-python = ">=3.11"\n'
            f'dependencies = {json.dumps(dependencies)}\n[dependency-groups]\nbuild = []\n'
            '[tool.uv]\npackage = false\n[tool.uv.sources]\n'
            'heavy-fixture = {path = "heavy_fixture-1.0-py3-none-any.whl"}\n'
            'kernel-fixture = {path = "kernel_fixture-1.0-py3-none-any.whl"}\n'
        )
        run('lock', '--python', sys._base_executable)

    project(False)
    common = ('sync', '--frozen', '--python', sys._base_executable, '--no-default-groups')
    run(*common)
    venv_python = tmp_path / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    run('pip', 'install', '--python', str(venv_python), str(tmp_path / 'extra_fixture-1.0-py3-none-any.whl'))
    packages = Path(subprocess.check_output([str(venv_python), '-I', '-c',
        'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
    existing = packages / 'heavy_fixture/__init__.py'
    existing.write_text('VERSION = "1.0"\n# Simulates an installed native DLL patch.\n')
    before = existing.read_bytes(), existing.stat().st_mtime_ns, existing.stat().st_ino
    marker = tmp_path / '.venv/keep.txt'
    marker.write_text('preserve the environment')
    project(True)
    run(*common, '--inexact', '--only-group', 'build', '--no-install-project')
    run(*common, '--inexact', '--group', 'build', '--no-build-isolation')
    assert (existing.read_bytes(), existing.stat().st_mtime_ns, existing.stat().st_ino) == before
    assert marker.read_text() == 'preserve the environment'
    assert (packages / 'extra_fixture/__init__.py').is_file()
    assert (packages / 'kernel_fixture/__init__.py').is_file()


@pytest.mark.parametrize('options', [['--update'], ['--update', '--production'], ['--network', '--update']])
def test_native_windows_update_skips_full_installer(tmp_path, options):
    if sys.platform != 'win32':
        pytest.skip('Requires native cmd.exe')
    project = tmp_path / 'PuLID with spaces'
    (project / '.venv').mkdir(parents=True)
    (project / '.venv/pulid-runtime.json').write_text('{}')
    models = project / 'PuLID_models'
    models.mkdir()
    source = (ROOT / 'install_windows.bat').read_text().replace(
        'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_DIR%scripts\\bootstrap_windows.ps1" -Update',
        'echo UPDATE_CALLED',
    )
    script = project / 'install_windows.bat'
    script.write_bytes(source.replace('\n', '\r\n').encode())
    result = subprocess.run([os.environ['COMSPEC'], '/d', '/c', str(script), *options],
        cwd=tmp_path, env=dict(os.environ, PULID_MODELS_ROOT=str(models)), stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=20)
    assert result.returncode == (0 if options == ['--update'] else 2), result.stdout + result.stderr
    assert ('UPDATE_CALLED' in result.stdout) == (options == ['--update'])
    assert 'recreation de .venv' not in result.stdout
    assert 'reparation des modeles' not in result.stdout
