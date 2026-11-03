"""`nitorch register` options from issue #171: -d/--dim, bare -o/-r on
@@fix/@@mov, prod/squeezed losses, `lcc --stride` short tag and
`mse -w auto`."""
import os
import numpy as np
import torch
import pytest

nib = pytest.importorskip('nibabel')

from nitorch.core.cli import ParseError
from nitorch.cli.registration.register.parser import (
    parser, default_moved, default_resliced, default_momentum
)
from nitorch.cli.registration.register.cli import (
    build_losses, check_options, cli, _cli
)
from nitorch.tools.registration import losses as L
from nitorch.tools.registration.pairwise_pyramid import pyramid_levels

IMAGES = ['@@fix', 'fix.nii', '@@mov', 'mov.nii']
BASE = ['@loss', 'mse'] + IMAGES + ['@affine']


def write(path, dat):
    nib.save(nib.Nifti1Image(np.asarray(dat, dtype='float32'), np.eye(4)),
             path)
    return path


@pytest.fixture
def images(tmp_path):
    rng = np.random.default_rng(0)
    fix = write(str(tmp_path / 'fix.nii.gz'), rng.random([8, 8, 8]))
    mov = write(str(tmp_path / 'mov.nii.gz'), rng.random([8, 8, 8]))
    return fix, mov


def built_loss(images, name, *args):
    fix, mov = images
    options = parser.parse(['@loss', name, *args, '@@fix', fix,
                            '@@mov', mov, '@affine', '@pyramid', '-l', '0'])
    shapes = [nib.load(f).shape for f in (fix, mov)]
    levels = list(pyramid_levels([[1.] * 3] * 2, shapes, **options.pyramid))
    losses, _ = build_losses(options, [{'fix': levels[0], 'mov': levels[1]}],
                             torch.device('cpu'))
    return options, losses[0][0].loss


# ----------------------------------------------------------------------
#   1. -d/--dim: validated (only 3 is supported)
# ----------------------------------------------------------------------

@pytest.mark.parametrize('args, ok', [
    ([], True), (['-d', '3'], True), (['--dim', '3'], True),
    (['-d', '2'], False), (['-d', '4'], False),
])
def test_dim_validated(args, ok):
    options = parser.parse(args + BASE)
    if ok:
        check_options(options)
        assert options.dim == 3
    else:
        with pytest.raises(ParseError, match='dim'):
            check_options(options)


def test_dim_rejected_before_reading_images(capsys):
    # files do not exist: the error must come from the -d check
    cli(['-d', '2'] + BASE)
    assert 'only 3 spatial dimensions' in capsys.readouterr().err


# ----------------------------------------------------------------------
#   2. bare -o / -r true on @@fix/@@mov
# ----------------------------------------------------------------------

@pytest.mark.parametrize('flag, args, expected', [
    ('output', [], default_moved),
    ('output', ['-o'], default_moved),
    ('output', ['-o', 'true'], default_moved),
    ('output', ['-o', 'false'], False),
    ('output', ['-o', 'out.nii'], 'out.nii'),
    ('resliced', [], False),
    ('resliced', ['-r'], default_resliced),
    ('resliced', ['-r', 'true'], default_resliced),
    ('resliced', ['-r', 'yes'], default_resliced),
    ('resliced', ['-r', 'false'], False),
    ('resliced', ['-r', 'out.nii'], 'out.nii'),
])
@pytest.mark.parametrize('side', ['fix', 'mov'])
def test_file_output_flags(side, flag, args, expected):
    fix = ['@@fix', 'fix.nii'] + (args if side == 'fix' else [])
    mov = ['@@mov', 'mov.nii'] + (args if side == 'mov' else [])
    options = parser.parse(['@loss', 'mse'] + fix + mov + ['@affine'])
    assert getattr(options.loss[0][side], flag) == expected
    check_options(options)


def test_file_output_bad_template_rejected():
    options = parser.parse(['@loss', 'mse', '@@fix', 'fix.nii',
                            '@@mov', 'mov.nii', '-r', '{nope}.nii',
                            '@affine'])
    with pytest.raises(ParseError, match='resliced'):
        check_options(options)


def test_bare_output_flags_write_default_paths(tmp_path):
    """Smoke run: bare -o / `-r true` used to crash after the fit."""
    rng = np.random.default_rng(0)
    fix = write(str(tmp_path / 'fix.nii.gz'), rng.random([12, 12, 12]))
    mov = write(str(tmp_path / 'mov.nii.gz'), rng.random([12, 12, 12]))
    _cli(['@loss', 'mse', '@@fix', fix, '-o', '-r', 'true',
          '@@mov', mov, '-o', '-r',
          '@affine', 'translation', '@@optim', 'gd', '-n', '2',
          '@optim', '-n', '1', '@pyramid', '-l', '0'])
    for base in ('fix', 'mov'):
        for kind in ('moved', 'resliced'):
            assert os.path.exists(tmp_path / f'{base}.{kind}.nii.gz')


# ----------------------------------------------------------------------
#   3. prod / squeezed losses
# ----------------------------------------------------------------------

@pytest.mark.parametrize('name, cls', [
    ('prod', L.ProdLoss), ('dot', L.ProdLoss), ('ndot', L.NormProdLoss),
    ('squeezed', L.SqueezedProdLoss), ('sqz', L.SqueezedProdLoss),
])
def test_prod_losses_build_and_evaluate(images, name, cls):
    _, loss = built_loss(images, name)
    assert isinstance(loss, cls)
    x = torch.rand([1, 8, 8, 8])
    ll = loss.loss(x, torch.rand([1, 8, 8, 8]), dim=3)
    assert torch.isfinite(torch.as_tensor(ll)).all()


@pytest.mark.parametrize('args, expected', [([], 1), (['-w', '2'], 2.)])
def test_squeezed_weight(images, args, expected):
    _, loss = built_loss(images, 'squeezed', *args)
    assert loss.lam == expected


# ----------------------------------------------------------------------
#   4. --stride has no short tag (-s is --symmetric)
# ----------------------------------------------------------------------

@pytest.mark.parametrize('name', ['lcc', 'lgmmi'])
def test_stride_long_form_and_symmetric_short_tag(name):
    options = parser.parse(['@loss', name, '--stride', '2', '-s'] + IMAGES)
    assert options.loss[0].stride == [2]
    assert options.loss[0].symmetric is True


@pytest.mark.parametrize('name', ['lcc', 'lgmmi'])
def test_stride_has_no_short_tag(name):
    with pytest.raises(ParseError):
        parser.parse(['@loss', name, '-s', '2'] + IMAGES)


# ----------------------------------------------------------------------
#   5. -w auto
# ----------------------------------------------------------------------

@pytest.mark.parametrize('name', ['mse', 'mad', 'tuk'])
@pytest.mark.parametrize('args, expected', [
    ([], None), (['-w', 'auto'], None), (['-w', '2'], 2.),
])
def test_weight_auto(images, name, args, expected):
    options, loss = built_loss(images, name, *args)
    assert options.loss[0].weight == expected
    assert loss.lam == expected


def test_weight_rejects_garbage():
    with pytest.raises(ParseError):
        parser.parse(['@loss', 'mse', '-w', 'banana'] + IMAGES)


# ----------------------------------------------------------------------
#   6. @@optim nes|ogm -o: bare / true -> default path, false -> none
# ----------------------------------------------------------------------

@pytest.mark.parametrize('group', ['@affine', '@nonlin'])
@pytest.mark.parametrize('name', ['nes', 'ogm'])
@pytest.mark.parametrize('args, expected', [
    ([], False),
    (['-o'], default_momentum),
    (['-o', 'true'], default_momentum),
    (['-o', 'yes'], default_momentum),
    (['-o', 'false'], False),
    (['-o', 'mom.nii.gz'], 'mom.nii.gz'),
])
def test_optim_momentum_output(group, name, args, expected):
    options = parser.parse(['@loss', 'mse'] + IMAGES +
                           [group, '@@optim', name] + args)
    opt = getattr(options, group[1:]).optim
    assert opt.output == expected
    check_options(options)


def test_optim_momentum_output_bad_template_rejected():
    options = parser.parse(['@loss', 'mse'] + IMAGES +
                           ['@nonlin', '@@optim', 'nes', '-o', '{nope}.nii'])
    with pytest.raises(ParseError, match='optim'):
        check_options(options)


# ----------------------------------------------------------------------
#   7. shooting `steps` saved in the JSON sidecar and used by reslice
# ----------------------------------------------------------------------

def test_shoot_steps_saved_and_used_by_reslice(tmp_path, monkeypatch):
    import json
    from nitorch import spatial
    from nitorch.cli.registration.reslice import main as reslice_main
    from nitorch.cli.registration.reslice import struct as reslice_struct
    rng = np.random.default_rng(0)
    fix = write(str(tmp_path / 'fix.nii.gz'), rng.random([12, 12, 12]))
    mov = write(str(tmp_path / 'mov.nii.gz'), rng.random([12, 12, 12]))
    vel = str(tmp_path / 'vel.nii.gz')
    _cli(['@loss', 'mse', '@@fix', fix, '-o', 'false', '@@mov', mov,
          '-o', 'false', '@nonlin', 'shoot', '-s', '3', '-o', vel,
          '@@optim', 'gd', '-n', '1', '@optim', '-n', '1',
          '@pyramid', '-l', '0'])
    with open(tmp_path / 'vel.json') as f:
        prm = json.load(f)
    assert prm['steps'] == 3
    assert prm['fast'] is True

    captured = {}
    shoot = spatial.shoot

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return shoot(*args, **kwargs)

    monkeypatch.setattr(spatial, 'shoot', spy)
    trf = reslice_struct.Velocity()
    trf.file = vel
    trf.json = str(tmp_path / 'vel.json')
    trf.inv = False
    trf.square = False
    reslice_main.exponentiate_transforms([trf])
    assert captured['steps'] == 3
    assert captured['fast'] is True



# ----------------------------------------------------------------------
#   8. @affine/@nonlin -o: bare / true -> default path, false -> none
# ----------------------------------------------------------------------

@pytest.mark.parametrize('group, default', [
    ('@affine', '{dir}{sep}{name}.lta'),
    ('@nonlin', '{dir}{sep}{name}.nii.gz'),
])
@pytest.mark.parametrize('args, expected', [
    ([], 'default'), (['-o'], 'default'), (['-o', 'true'], 'default'),
    (['-o', 'yes'], 'default'), (['-o', 'false'], False),
    (['-o', 'no'], False), (['-o', 'out.ext'], 'out.ext'),
])
def test_transform_output_bool(group, default, args, expected):
    options = parser.parse(['@loss', 'mse'] + IMAGES + [group] + args)
    expected = default if expected == 'default' else expected
    assert getattr(options, group[1:]).output == expected
    check_options(options)


@pytest.mark.parametrize('group', ['@affine', '@nonlin'])
def test_transform_output_bad_template(group):
    options = parser.parse(['@loss', 'mse'] + IMAGES + [group, '-o', '{x}'])
    with pytest.raises(ParseError, match=f'{group} -o'):
        check_options(options)


def test_affine_output_false_writes_nothing(tmp_path, monkeypatch):
    # before the fix, `-o false` wrote a file named `false` in the cwd
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    fix = write(str(tmp_path / 'fix.nii.gz'), rng.random([12, 12, 12]))
    mov = write(str(tmp_path / 'mov.nii.gz'), rng.random([12, 12, 12]))
    before = set(os.listdir(tmp_path))
    _cli(['@loss', 'mse', '@@fix', fix, '-o', 'false', '@@mov', mov,
          '-o', 'false', '@affine', 'translation', '-o', 'false',
          '@@optim', 'gd', '-n', '1', '@optim', '-n', '1',
          '@pyramid', '-l', '0'])
    assert set(os.listdir(tmp_path)) == before


# ----------------------------------------------------------------------
#   9. simplereg / affreg: false outputs, --bound nearest
# ----------------------------------------------------------------------

def _simplereg(*args):
    from nitorch.cli.registration.simplereg.parser import parser as sparser
    from nitorch.cli.registration.simplereg.cli import build_register_args
    options = sparser.parse(['-m', 'mov.nii', '-f', 'fix.nii', *args])
    return parser.parse(build_register_args(options))


@pytest.mark.parametrize('flag, group', [('-a', 'affine'), ('-w', 'nonlin')])
@pytest.mark.parametrize('value, expected', [
    ('false', False), ('no', False), ('true', 'default'), ('x.ext', 'x.ext'),
])
def test_simplereg_output_false(flag, group, value, expected):
    defaults = {'affine': '{dir}{sep}{name}.lta',
                'nonlin': '{dir}{sep}{name}.nii.gz'}
    options = _simplereg(flag, value)
    expected = defaults[group] if expected == 'default' else expected
    assert getattr(options, group).output == expected


@pytest.mark.parametrize('bound, expected', [
    ('nearest', 'replicate'), ('replicate', 'replicate'), ('dft', 'dft'),
])
def test_simplereg_bound(bound, expected):
    options = _simplereg('--bound', bound)
    assert options.loss[0].fix.bound == expected
    assert options.loss[0].mov.bound == expected


def test_affreg_output_false(monkeypatch):
    from nitorch.cli.registration.affreg import cli as affreg_cli
    captured = []
    monkeypatch.setattr(affreg_cli, 'cli_register',
                        lambda args: captured.append(args))
    affreg_cli._cli(['-m', 'mov.nii', '-f', 'fix.nii', '-o', 'false'])
    options = parser.parse(captured[0])
    assert options.affine.output is False


# ----------------------------------------------------------------------
#   10. stray values in @loss: name the loss and the option
# ----------------------------------------------------------------------

def test_loss_stray_value_after_flag():
    with pytest.raises(ParseError, match=r"@loss lcc: unexpected value '2' "
                                         r"after -s/--symmetric.*--stride"):
        parser.parse(['@loss', 'lcc', '-s', '2'] + IMAGES)


def test_loss_stray_value_after_factor():
    with pytest.raises(ParseError, match=r"@loss mse: unexpected value 'x'"):
        parser.parse(['@loss', 'mse', '3', 'x'] + IMAGES)


def test_loss_no_stray_value():
    options = parser.parse(['@loss', 'lcc', '--stride', '2', '-s'] + IMAGES
                           + ['@loss', 'mse'] + IMAGES + ['@affine'])
    assert options.loss[0].stride in (2, [2])
    assert options.loss[0].symmetric
    assert options.loss[1].name == 'mse'


# ----------------------------------------------------------------------
#   11. sqz -w auto: rejected (no automatic weighting)
# ----------------------------------------------------------------------

def test_sqz_weight_auto_rejected():
    options = parser.parse(['@loss', 'sqz', '-w', 'auto'] + IMAGES)
    with pytest.raises(ParseError, match='sqz -w/--weight: `auto`'):
        check_options(options)


@pytest.mark.parametrize('args, expected', [([], 1), (['-w', '2'], 2.)])
def test_sqz_weight(images, args, expected):
    options, loss = built_loss(images, 'sqz', *args)
    check_options(options)
    assert loss.lam == expected
