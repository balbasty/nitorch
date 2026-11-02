import pytest

from nitorch.core.cli import Conversions


@pytest.mark.parametrize('word', ['t', 'T', 'true', 'True', 'y', 'yes', 'YES', '1'])
def test_bool_true(word):
    assert Conversions.bool(word) is True


@pytest.mark.parametrize('word', ['f', 'F', 'false', 'False', 'n', 'no', 'NO', '0'])
def test_bool_false(word):
    assert Conversions.bool(word) is False


def test_bool_other_word_is_returned_unchanged():
    assert Conversions.bool('auto') == 'auto'


# --- istag / isvalue: negative floats are values, not tags (#171) -----

from nitorch.core.cli import istag, isvalue, next_isvalue, check_next_isvalue


@pytest.mark.parametrize('arg', ['-.5', '-inf', '-nan', '-1e-3', '-1',
                                 '-0.5', '-Infinity', '-.5e2', '-', '1'])
def test_istag_numbers_are_values(arg):
    assert istag(arg) is False
    assert isvalue(arg)
    assert next_isvalue([arg])
    check_next_isvalue([arg], '-x')  # does not raise


@pytest.mark.parametrize('arg', ['-o', '--foo', '-in', '-n', '-e', '--inf',
                                 '-.', '-.o', '-infx'])
def test_istag_tags_are_tags(arg):
    assert istag(arg) is True
    assert not next_isvalue([arg])


def test_crop_center_accepts_dot_and_inf_negative_floats():
    from nitorch.cli.misc.crop.parser import parse
    opt = parse(['in.nii', '-c', '-.5', '-1e-3', '-inf', '-o', 'out.nii'])
    assert opt.files == ['in.nii']
    assert opt.center == [-0.5, -0.001, float('-inf')]
    assert opt.output == ['out.nii']
