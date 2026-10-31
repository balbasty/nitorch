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


IMAGES = ['@@fix', 'fix.nii', '@@mov', 'mov.nii']


@pytest.mark.parametrize('name', ['lcc', 'lgmmi', 'lgmm'])
def test_register_loss_patch_option(name):
    options = parser.parse(['@loss', name, '-p', '5'] + IMAGES)
    assert options.loss[0].patch == [5.0]


def test_register_mi_rejects_patch_option():
    # MI has no patch: -p is not offered (it used to be silently dropped)
    with pytest.raises(Exception):
        parser.parse(['@loss', 'mi', '-p', '5'] + IMAGES)
