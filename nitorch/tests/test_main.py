import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(*args):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [ROOT] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    return subprocess.run([sys.executable, '-m', 'nitorch', *args],
                          capture_output=True, text=True, env=env, timeout=600)


@pytest.mark.parametrize('args', [['--help'], pytest.param(['help', 'register'], marks=pytest.mark.slow)])
def test_python_m_nitorch_help(args):
    res = _run(*args)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip()


@pytest.mark.slow
def test_python_m_nitorch_unknown_command():
    res = _run('this-command-does-not-exist')
    assert res.returncode != 0


@pytest.mark.slow
def test_python_m_nitorch_help_unknown_command():
    res = _run('help', 'this-command-does-not-exist')
    assert res.returncode != 0
    assert '<function' not in res.stdout
    assert 'Unknown command' in res.stderr
