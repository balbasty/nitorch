"""Error reporting of the shared `nitorch.core.cli.CommandParser` (#171):
invalid `choices` values are named, unknown options are rejected."""
import pytest

from nitorch.core import cli
from nitorch.core.cli import ParseError


def _parser(**kwargs):
    p = cli.CommandParser('toy', **kwargs)
    p.add_positional('files', nargs='+')
    p.add_option('mode', ('-m', '--mode'), nargs=1, default='a',
                 validation=cli.Validations.choice(['a', 'b', 'c']))
    p.add_option('num', ('-n', '--num'), nargs='+', default=[0.],
                 convert=float)
    p.add_option('out', ('-o', '--output'), nargs='*')
    p.add_option('verbose', ('-v', '--verbose'), nargs='?', default=0,
                 convert=int, action=cli.Actions.store_value(1))
    return p


def _grouped_parser():
    p = cli.CommandParser('toy')
    p.add_option('verbose', '-v', nargs=0, default=False)
    grp = cli.Group('item', ('-i', '--item'), n='+')
    grp.add_positional('name', nargs='?')
    grp.add_option('files', ('-f', '--files'), nargs='+')
    grp.add_option('kind', '--kind', nargs=1,
                   validation=cli.Validations.choice(['x', 'y']))
    p.add_group(grp)
    return p


# --- choices ------------------------------------------------------------

@pytest.mark.parametrize('tag', ['-m', '--mode'])
def test_valid_choice(tag):
    assert _parser().parse(['f.nii', tag, 'b']).mode == 'b'


@pytest.mark.parametrize('tag', ['-m', '--mode'])
def test_invalid_choice_names_value_and_choices(tag):
    with pytest.raises(ParseError) as e:
        _parser().parse(['f.nii', tag, 'foo'])
    assert str(e.value) == "Invalid value 'foo' for --mode; choose from {a, b, c}"


@pytest.mark.parametrize('args', [
    ['f.nii', '--mode'],                 # end of command line
    ['f.nii', '--mode', '-v'],           # followed by a known option
])
def test_missing_choice_still_not_enough_arguments(args):
    with pytest.raises(ParseError, match='Not enough arguments for option mode'):
        _parser().parse(args)


def test_invalid_converted_value_is_named():
    with pytest.raises(ParseError, match="Invalid value 'abc' for --num"):
        _parser().parse(['f.nii', '--num', 'abc'])


def test_invalid_choice_in_group():
    with pytest.raises(ParseError) as e:
        _grouped_parser().parse(['-i', '--kind', 'z'])
    assert str(e.value) == "Invalid value 'z' for --kind; choose from {x, y}"


def test_choice_validation_is_still_a_predicate():
    validate = cli.Validations.choice(['a', 'b'])
    assert validate('a') and not validate('z')
    assert list(validate.choices) == ['a', 'b']


# --- unknown options ----------------------------------------------------

@pytest.mark.parametrize('args', [
    ['--nope', 'f.nii'],                 # before positionals
    ['f.nii', '--nope', 'g.nii'],        # among positionals
    ['f.nii', '-o', 'x', '--nope'],      # after an option with nargs='*'
    ['f.nii', '-n', '1', '--nope'],      # after an option with nargs='+'
])
def test_unknown_long_option(args):
    with pytest.raises(ParseError) as e:
        _parser().parse(args)
    assert str(e.value) == 'Unknown option --nope'


@pytest.mark.parametrize('args', [['-z', 'f.nii'], ['f.nii', '-z']])
def test_unknown_short_option(args):
    with pytest.raises(ParseError) as e:
        _parser().parse(args)
    assert str(e.value) == 'Unknown option -z'


def test_unknown_option_in_group():
    with pytest.raises(ParseError, match='Unknown option --nope'):
        _grouped_parser().parse(['-i', 'a', '-f', 'x.nii', '--nope'])


def test_group_options_are_known():
    out = _grouped_parser().parse(['-v', '-i', 'a', '-f', 'x.nii', 'y.nii',
                                   '--kind', 'y', '--item', '-f', 'z.nii'])
    assert out.verbose is True
    assert out.item[0].files == ['x.nii', 'y.nii']
    assert out.item[0].kind == 'y'
    assert out.item[1].files == ['z.nii']


@pytest.mark.parametrize('value', ['-1', '-0.5', '-.5', '-1e-3', '-inf'])
def test_negative_numbers_are_values(value):
    out = _parser().parse(['f.nii', '-n', value, '2'])
    assert out.num == [float(value), 2.]
    out = _parser().parse([value, 'f.nii'])
    assert out.files == [value, 'f.nii']


def test_single_dash_is_a_value():
    assert _parser().parse(['-', '--output', '-']).files == ['-']


@pytest.mark.parametrize('args', [['-h'], ['--help'], ['f.nii', '-h'],
                                  ['-h', '--nope'], ['--nope', '--help']])
def test_help_still_works(args):
    assert _parser().parse(args).help is True


def test_opt_out_keeps_unknown_options_as_values():
    out = _parser(allow_unknown_options=True).parse(['f.nii', '--nope'])
    assert out.files == ['f.nii', '--nope']


# --- real commands ------------------------------------------------------

def test_topup_invalid_loss_and_unknown_option(capsys):
    from nitorch.cli.qmri.topup.main import cli as topup
    assert topup(['fit', 'pos.nii', 'neg.nii', '-l', 'foo']) == 1
    err = capsys.readouterr().err
    assert ("Invalid value 'foo' for --loss; "
            "choose from {mse, ncc, lncc, gmm, lgmm}") in err
    assert topup(['apply', 'f.nii', '-p', 'a.nii', '--nope', '-n', 'b.nii']) == 1
    assert 'Unknown option --nope' in capsys.readouterr().err


def test_epic_invalid_direction_and_unknown_option():
    from nitorch.cli.qmri.epic.parser import parser
    with pytest.raises(ParseError, match=r"Invalid value 'foo' for --direction; "
                                         r"choose from \{r, l, a, p,"):
        parser.parse(['e1.nii', '-d', 'foo'])
    with pytest.raises(ParseError, match='Unknown option --nope'):
        parser.parse(['e1.nii', '--nope', 'e2.nii'])
    # `-` is a valid polarity value, not an option
    assert parser.parse(['e1.nii', '-p', '-']).polarity == '-'


def test_register_unknown_option_in_file_group():
    from nitorch.cli.registration.register.parser import parser
    args = ['@loss', 'mse', '@@fix', 'a.nii', '--nope', '@@mov', 'b.nii']
    with pytest.raises(ParseError, match='Unknown option --nope'):
        parser.parse(args)
    args = ['-v', '@loss', 'mse', '-w', '2', '@@fix', 'a.nii', '-o',
            '@@mov', 'b.nii', '@nonlin', '-h', '@pyramid', '-l', '0:3']
    out = parser.parse(args)
    assert out.loss[0].fix.files == ['a.nii']
    assert out.loss[0].mov.files == ['b.nii']
