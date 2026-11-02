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


# --- istag / next_istag -----------------------------------------------
from nitorch.core.cli import istag, next_istag  # noqa: E402


@pytest.mark.parametrize('arg', ['-l', '--linear', '-iv', '-x'])
def test_next_istag_default_accepts_dash_tags(arg):
    assert next_istag([arg, 'other'])
    assert istag(arg)


@pytest.mark.parametrize('arg', ['extra.nii', 'vel.json', 'x1.nii', 'b',
                                 '-1', '-0.5', '-'])
def test_next_istag_default_rejects_values(arg):
    # Regression: the default symbol used to be '' so that any argument of
    # length >= 2 whose second character is not a digit (e.g. a file name)
    # was reported as a tag.
    assert not next_istag([arg])
    assert not istag(arg)


def test_next_istag_empty():
    assert not next_istag([])


def test_next_istag_custom_symbol():
    assert next_istag(['@group'], symbol='@')
    assert not next_istag(['-l'], symbol='@')
