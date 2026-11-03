import pytest

from nitorch.cli.registration.register.parser import parser, default_hessian

BASE = ['@loss', 'mse', '@@fix', 'fix.nii', '@@mov', 'mov.nii', '@nonlin', 'svf']


@pytest.mark.parametrize('args, expected', [
    ([], False),                                # not written by default
    (['-h'], default_hessian),                  # default path
    (['-h', 'out/hess.nii.gz'], 'out/hess.nii.gz'),
    (['-h', 'true'], default_hessian),          # same as bare -h
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


@pytest.mark.parametrize('args, factor, groups', [
    # group factors default to 1 (precision = FACTOR), FACTOR to 0 (off)
    ([], [0.], (1., 1., 1., 1.)),
    (['3'], [3.], (1., 1., 1., 1.)),
    (['3', '-t', '2', '--shear-penalty', '0.5'], [3.], (2., 1., 1., .5)),
    (['3', '--rotation-penalty', '4', '-z', '5'], [3.], (1., 4., 5., 1.)),
])
def test_register_affine_penalty_factors(args, factor, groups):
    options = parser.parse(['@loss', 'mse'] + IMAGES +
                           ['@affine', 'affine'] + args)
    assert options.affine.factor == factor
    assert (options.affine.translation, options.affine.rotation,
            options.affine.zoom, options.affine.shear) == groups
