"""`-h, --help [LEVEL]` is only advertised by commands whose parser
takes a help level (issue #171)."""
import importlib
import pytest


@pytest.mark.parametrize('module, parser_name', [
    ('nitorch.cli.registration.meanspace.parser', 'parser'),
    ('nitorch.cli.registration.affreg.parser', 'parser'),
    ('nitorch.cli.qmri.topup.parser', 'parser_fit'),
    ('nitorch.cli.qmri.topup.parser', 'parser_apply'),
    ('nitorch.cli.qmri.estatics.parser', 'parser'),
    ('nitorch.cli.denoising.denoise.parser', 'parser'),
])
def test_help_matches_parser(module, parser_name):
    """`-h` takes no value: the help must not advertise a LEVEL."""
    module = importlib.import_module(module)
    parser = getattr(module, parser_name)
    assert parser.parse(['-h']).help
    help_option = [o for o in parser.options if o.name == 'help'][0]
    assert help_option.nargs == 0
    lines = [l for l in module.help.splitlines() if '--help' in l]
    assert lines
    assert all('[LEVEL]' not in l for l in lines)


def test_register_help_levels():
    """`register -h` does take a level, and each level has a help text."""
    from nitorch.cli.registration.register.parser import parser, help
    assert parser.parse(['-h']).help == 1
    for level in (1, 2, 3):
        assert parser.parse(['-h', str(level)]).help == level
        assert level in help
    assert '[LEVEL]' in help[2]
