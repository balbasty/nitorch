"""Error messages and exit status of `vopt`, `optinorm`, `resize` and
`autoreg` (#171)."""
import pytest

import nitorch.cli  # noqa: F401  (imports and registers all commands)
from nitorch.cli.cli import commands

MISSING = '/nonexistent/nitorch-missing-input.nii.gz'


def test_vopt_stray_value_names_problem(capsys):
    rc = commands['vopt']([MISSING])
    _, err = capsys.readouterr()
    assert rc == 1
    assert f"Unexpected value '{MISSING}'" in err
    assert '-i/--input' in err
    assert 'Unknown tag' not in err


def test_vopt_unknown_tag_still_reported(capsys):
    rc = commands['vopt'](['--nope'])
    _, err = capsys.readouterr()
    assert rc == 1
    assert 'Unknown tag --nope' in err


def test_optinorm_runtime_error_is_reported(tmp_path, capsys):
    # two missing inputs: parsing succeeds, the pipeline fails
    rc = commands['optinorm']([MISSING, MISSING, '-o', str(tmp_path),
                               '--steps', 'affopt'])
    out, err = capsys.readouterr()
    assert rc == 1
    assert '[ERROR]' in err
    assert 'Traceback' not in out + err


def test_optinorm_parse_error_returns_1(capsys):
    # option without its value
    rc = commands['optinorm']([MISSING, MISSING, '-a'])
    out, err = capsys.readouterr()
    assert rc == 1
    assert '[ERROR]' in err
    assert out.strip()  # help
