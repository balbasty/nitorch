import pytest

from nitorch.cli.registration.register.parser import parser, default_hessian

BASE = ['@loss', 'mse', '@@fix', 'fix.nii', '@@mov', 'mov.nii', '@nonlin', 'svf']


@pytest.mark.parametrize('args, expected', [
    ([], False),                                # not written by default
    (['-h'], default_hessian),                  # default path
    (['-h', 'out/hess.nii.gz'], 'out/hess.nii.gz'),
    (['-h', 'true'], True),                     # mapped to default path by cli
    (['-h', 'false'], False),
])
def test_register_nonlin_hessian_option(args, expected):
    options = parser.parse(BASE + args)
    assert options.nonlin.hessian == expected
