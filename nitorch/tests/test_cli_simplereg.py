"""`nitorch simplereg`: affine + nonlinear wrapper around `nitorch register`."""
import pytest
import nitorch.cli  # noqa: F401  (imports and registers all commands)
from nitorch.cli.cli import commands, cli
from nitorch.core.cli import ParseError
from nitorch.cli.registration.simplereg.parser import parser
from nitorch.cli.registration.simplereg.cli import build_register_args
from nitorch.cli.registration.register.parser import parser as register_parser


def _register_options(*args):
    """simplereg args -> parsed `nitorch register` options"""
    options = parser.parse(['-m', 'mov.nii', '-f', 'fix.nii', *args])
    return register_parser.parse(build_register_args(options))


def test_simplereg_is_registered():
    assert 'simplereg' in commands


def test_simplereg_help(capsys):
    cli(['simplereg', '-h'])
    captured = capsys.readouterr()
    assert '[nitorch] simplereg' in captured.out
    assert 'affine + nonlinear' in captured.out
    assert 'Unknown command' not in captured.err


def test_simplereg_help_advertises_no_unparsed_options():
    from nitorch.cli.registration.simplereg.parser import help
    for opt in ('[m]', 'Multiple pairs'):
        assert opt not in help, opt


def test_simplereg_defaults():
    opt = _register_options()
    assert len(opt.loss) == 1
    assert opt.loss[0].name == 'mi'  # register's alias for (normalized) mi
    assert opt.loss[0].norm == 'studholme'
    assert opt.loss[0].fix.files == ['fix.nii']
    assert opt.loss[0].mov.files == ['mov.nii']
    assert opt.affine.name == 'rigid'
    assert opt.affine.optim.name == 'pow'
    assert opt.affine.optim.max_iter == 50
    assert opt.affine.output == '{dir}/{name}.lta'
    # nonlinear defaults are those of `nitorch register` (as in the help)
    assert opt.nonlin.name == 'svf'
    assert opt.nonlin.factor == [1.]
    assert opt.nonlin.absolute == 1e-4
    assert opt.nonlin.membrane == 1e-3
    assert opt.nonlin.bending == 0.2
    assert tuple(opt.nonlin.lame) == (0.05, 0.2)
    assert opt.nonlin.optim.name == 'unset'
    assert opt.nonlin.output == '{dir}/{name}.nii.gz'
    assert opt.optim.name == 'interleaved'
    assert opt.optim.max_iter == 5
    assert list(opt.pyramid.levels) == [0, 1, 2]


def test_simplereg_options():
    opt = _register_options(
        '--affine', '-l', 'lcc', '-k', '5', 'vox', '-s', '-i', 'init.lta',
        '-a', 'aff.lta', '-w', 'warp.nii.gz', '-r', 'res.nii.gz',
        '--nonlin', 'shoot', '--penalty', '2', '--bending', '0.5',
        '--lame', '0.1', '0.3', '--nonlin-optimizer', 'lbfgs',
        '--nonlin-iter', '3', '--outer-iter', '2', '-n', '7', '-z', 'gn',
        '--fwhm', '2', '--bound', 'zero', '--order', '3',
        '--rescale', '5', '90', '--discretize', '-p', '1', '2')
    loss = opt.loss[0]
    assert loss.name == 'lcc' and loss.symmetric
    assert list(loss.patch) == [5., 'vox']
    for img in (loss.fix, loss.mov):
        assert list(img.fwhm) == [2.]
        assert img.bound == 'zero'
        assert img.order == 3
        assert tuple(img.rescale) == (5., 90.)
        assert img.discretize == 256
    assert loss.mov.affine == ['init.lta']
    assert loss.mov.resliced == 'res.nii.gz'
    assert opt.affine.name == 'affine'
    assert opt.affine.output == 'aff.lta'
    assert opt.affine.optim.name == 'gn'
    assert opt.affine.optim.max_iter == 7
    assert opt.nonlin.name == 'shoot'
    assert opt.nonlin.factor == [2.]
    assert opt.nonlin.bending == 0.5
    assert tuple(opt.nonlin.lame) == (0.1, 0.3)
    assert opt.nonlin.output == 'warp.nii.gz'
    assert opt.nonlin.optim.name == 'lbfgs'
    assert opt.nonlin.optim.max_iter == 3
    assert opt.optim.max_iter == 2
    assert list(opt.pyramid.levels) == [1, 2]


@pytest.mark.parametrize('flag', [('--no-nonlin',), ('--nonlin', 'none')])
def test_simplereg_no_nonlin(flag):
    opt = _register_options(*flag)
    assert opt.affine and not opt.nonlin
    assert opt.affine.name == 'rigid'


@pytest.mark.parametrize('flag', [('--no-affine',), ('--affine', 'none')])
def test_simplereg_no_affine(flag):
    opt = _register_options(*flag)
    assert opt.nonlin and not opt.affine
    assert opt.nonlin.name == 'svf'


@pytest.mark.parametrize('aff', [('--no-affine',), ('--affine', 'none')])
@pytest.mark.parametrize('nl', [('--no-nonlin',), ('--nonlin', 'none')])
def test_simplereg_no_nonlin_no_affine(aff, nl):
    with pytest.raises(ParseError):
        build_register_args(parser.parse(['-m', 'm', '-f', 'f', *aff, *nl]))


@pytest.mark.parametrize('args,name', [
    ((), 'rigid'),
    (('--affine',), 'affine'),
    (('--affine', 'affine'), 'affine'),
    (('--affine', 'rigid'), 'rigid'),
    (('--affine', 'similitude'), 'similitude'),
    (('--affine', 'translation'), 'translation'),
])
def test_simplereg_affine_type(args, name):
    opt = _register_options(*args)
    assert opt.affine.name == name
    assert opt.affine.output == '{dir}/{name}.lta'


def test_simplereg_bare_affine_before_other_option():
    opt = _register_options('--affine', '-l', 'mse')
    assert opt.affine.name == 'affine'
    assert opt.loss[0].name == 'mse'


def test_simplereg_bad_affine_type():
    with pytest.raises(ParseError):
        parser.parse(['-m', 'm', '-f', 'f', '--affine', 'foo'])


def test_simplereg_output_paths():
    opt = _register_options('-a', 'a.lta', '-w', 'w.nii.gz')
    assert opt.affine.output == 'a.lta'
    assert opt.nonlin.output == 'w.nii.gz'
    opt = _register_options('--output-affine', 'b.lta',
                            '--output-warp', 'v.nii.gz')
    assert opt.affine.output == 'b.lta'
    assert opt.nonlin.output == 'v.nii.gz'


def test_simplereg_old_output_flags_removed():
    for flag in ('-o', '--output', '--warp'):
        with pytest.raises(ParseError):
            parser.parse(['-m', 'm', '-f', 'f', flag, 'x'])


def test_simplereg_no_pyramid():
    opt = _register_options('--no-pyramid')
    assert list(opt.pyramid.levels) == [0]


def test_simplereg_smoke(tmp_path):
    import numpy as np
    nib = pytest.importorskip('nibabel')
    ndimage = pytest.importorskip('scipy.ndimage')
    rng = np.random.RandomState(0)
    for name in ('mov', 'fix'):
        img = ndimage.gaussian_filter(rng.rand(12, 12, 12), 1.5)
        nib.save(nib.Nifti1Image(img.astype('float32'), np.eye(4)),
                 str(tmp_path / f'{name}.nii.gz'))
    cli(['simplereg', '-m', str(tmp_path / 'mov.nii.gz'),
         '-f', str(tmp_path / 'fix.nii.gz'), '-n', '2', '-p', '0',
         '--nonlin-iter', '2', '--outer-iter', '1',
         '--affine', 'similitude',
         '-a', str(tmp_path / 'xfm.lta'),
         '-w', str(tmp_path / 'warp.nii.gz'),
         '-r', str(tmp_path / 'res.nii.gz')])
    assert (tmp_path / 'xfm.lta').exists()
    assert (tmp_path / 'warp.nii.gz').exists()
    assert (tmp_path / 'res.nii.gz').exists()
    assert (tmp_path / 'mov.moved.nii.gz').exists()
    warp = nib.load(str(tmp_path / 'warp.nii.gz'))
    assert warp.shape[:3] == (12, 12, 12) and warp.shape[-1] == 3


def test_simplereg_fixed_image_outputs_disabled():
    # #171: only the moving image's outputs are exposed (-x/-r)
    opt = _register_options()
    assert opt.loss[0].fix.output is False
    assert opt.loss[0].fix.resliced is False
    assert opt.loss[0].mov.output == '{dir}/{base}.moved{ext}'
    # preprocessing options given to the fixed image are kept
    opt = _register_options('--bound', 'zero', '--order', '3')
    assert opt.loss[0].fix.output is False
    assert opt.loss[0].fix.bound == 'zero'
    assert opt.loss[0].fix.order == 3


def test_simplereg_smoke_writes_no_fixed_moved(tmp_path):
    import os
    import numpy as np
    nib = pytest.importorskip('nibabel')
    rng = np.random.RandomState(0)
    for name in ('mov', 'fix'):
        nib.save(nib.Nifti1Image(rng.rand(12, 12, 12).astype('float32'),
                                 np.eye(4)),
                 str(tmp_path / f'{name}.nii.gz'))
    cli(['simplereg', '-m', str(tmp_path / 'mov.nii.gz'),
         '-f', str(tmp_path / 'fix.nii.gz'), '-n', '1', '-p', '0',
         '--no-nonlin', '-a', str(tmp_path / 'xfm.lta'), '-x', 'false',
         '-r', str(tmp_path / 'res.nii.gz')])
    assert sorted(os.listdir(tmp_path)) == ['fix.nii.gz', 'mov.nii.gz',
                                            'res.nii.gz', 'xfm.lta']
