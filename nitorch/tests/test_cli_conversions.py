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
