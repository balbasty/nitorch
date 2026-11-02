"""CLI error reporting: `[ERROR] <message>` without traceback, and a
non-zero exit status from the `nitorch` executable on failure (#171)."""
import os
import subprocess
import sys

import pytest

import nitorch.cli  # noqa: F401  (imports and registers all commands)
from nitorch.cli.cli import cli, commands, _status

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(*args):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [ROOT] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    return subprocess.run([sys.executable, '-m', 'nitorch', *args],
                          capture_output=True, text=True, env=env, timeout=600)


# Commands whose generic `except Exception` handler was restored, with
# arguments pointing at an input file that does not exist.
MISSING = '/nonexistent/nitorch-missing-input.nii.gz'
RUNTIME_FAILURES = {
    'affreg': ['-m', MISSING, '-f', MISSING],
    'simplereg': ['-m', MISSING, '-f', MISSING],
    'register': ['@loss', 'mse', '@@fix', MISSING, '@@mov', MISSING],
    'meanspace': [MISSING],
    'vopt': ['--input', '0', '1', MISSING],
    'reslice': [MISSING],
    'smooth': [MISSING],
}


@pytest.mark.parametrize('cmd', sorted(RUNTIME_FAILURES))
def test_missing_input_prints_error_not_traceback(cmd, capsys):
    rc = commands[cmd](RUNTIME_FAILURES[cmd])
    out, err = capsys.readouterr()
    assert rc == 1
    assert '[ERROR]' in err
    assert 'Traceback' not in err + out


@pytest.mark.parametrize('cmd', ['register', 'reslice', 'affopt', 'vopt'])
def test_parse_error_prints_help_and_error(cmd, capsys):
    rc = commands[cmd](['--this-option-does-not-exist'])
    out, err = capsys.readouterr()
    assert rc == 1
    assert '[ERROR]' in err
    assert out.strip()  # help


def test_programmatic_call_does_not_exit():
    # returns a status, never raises SystemExit
    assert cli(['reslice', MISSING]) == 1
    assert cli(['reslice', '-h']) == 0


def test_status():
    assert _status(None) == 0
    assert _status(0) == 0
    assert _status(1) == 1
    assert _status(True) == 0
    assert _status(object()) == 0


@pytest.mark.slow
def test_exit_status_parse_error():
    res = _run('register', '--this-option-does-not-exist')
    assert res.returncode != 0
    assert '[ERROR]' in res.stderr


@pytest.mark.slow
@pytest.mark.parametrize('cmd', ['reslice', 'register'])
def test_exit_status_runtime_error(cmd):
    res = _run(cmd, *RUNTIME_FAILURES[cmd])
    assert res.returncode != 0
    assert '[ERROR]' in res.stderr
    assert 'Traceback' not in res.stderr


@pytest.mark.slow
def test_exit_status_help():
    res = _run('reslice', '-h')
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip()
