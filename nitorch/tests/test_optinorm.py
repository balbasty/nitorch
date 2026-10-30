import os

import numpy as np
import pytest
import torch

from nitorch.cli.registration.optinorm.parser import parser, steps
from nitorch.cli.registration.optinorm.main import OptinormPaths
from nitorch.spatial import optimal_svf


def test_optinorm_parser_defaults():
    options = parser.parse(['a.nii', 'b.nii'])
    assert options.input == ['a.nii', 'b.nii']
    assert options.output == 'norm'
    assert options.steps == list(steps)
    assert options.both is False
    assert options.cleanup is False
    assert options.loss == 'lcc'
    assert options.affine == 'rigid'
    assert options.affine_levels == ':4'
    assert options.nonlin_levels == ':6'
    assert options.membrane == 20
    assert options.order == 3
    assert options.clip is True
    assert options.device == ('cpu', None)


def test_optinorm_parser_options():
    options = parser.parse([
        'a.nii', 'b.nii', '-o', 'out', '-s', 'svf', 'svfopt', '--both',
        '--cleanup', '-m', '10', '--no-clip', '-i', '1', '--gpu', '1',
        '--affine', 'affine'])
    assert options.output == 'out'
    assert options.steps == ['svf', 'svfopt']
    assert options.both and options.cleanup
    assert options.membrane == 10
    assert options.clip is False
    assert options.order == 1
    assert options.device == ('gpu', 1)
    assert options.affine == 'affine'


def test_optinorm_paths():
    inputs = [f'/data/s{k}/img.nii.gz' for k in range(12)]
    paths = OptinormPaths(inputs, '/out', 'rigid')
    # pairs (1, 12) and (11, 2) must not collide
    assert paths.pair_dir('aff', 0, 11) != paths.pair_dir('aff', 10, 1)
    # identical basenames are disambiguated
    assert len(set(paths.normalised(i) for i in range(12))) == 12
    assert paths.normalised(0) == '/out/opt/1_img.norm.nii.gz'
    paths = OptinormPaths(['/data/a.nii', '/data/b.nii'], '/out', 'rigid')
    assert paths.normalised(1) == '/out/opt/b.norm.nii'
    assert paths.pair_affine(0, 1) == '/out/aff/1_2/rigid.lta'


@pytest.mark.parametrize('weighted', [False, True])
def test_optimal_svf_consistent_pairs(weighted):
    # pairwise SVFs that are exactly v_i - v_j are recovered
    torch.manual_seed(0)
    shape, n = [8, 8, 8], 3
    vel = torch.randn([n, *shape, 3])
    vel -= vel.mean(0)
    svf = {(i, j): vel[i] - vel[j]
           for i in range(n) for j in range(n) if i != j}
    weights = {}
    if weighted:
        hess = torch.zeros([*shape, 6])
        hess[..., :3] = 1
        weights = {k: hess for k in svf}
    prm = dict(absolute=1e-8, membrane=1e-8, bending=0, lame=0)
    opt = optimal_svf(svf, weights=weights, **prm)
    labels = list(set(label for pair in svf for label in pair))
    opt = torch.stack([opt[labels.index(i)] for i in range(n)])
    assert torch.allclose(opt, vel, atol=1e-3)


@pytest.mark.parametrize('n', [3, 6])
def test_optimal_affine_consistent_pairs(n):
    # pairwise rigid transforms that are exactly T_i @ inv(T_j) are recovered
    from nitorch.spatial import (optimal_affine, affine_matrix, affine_basis,
                                 affine_inv, affine_matmul)
    torch.manual_seed(0)
    z = 0.2 * torch.randn([n, 6])
    z[:, :3] *= 20      # translations (mm)
    z -= z.mean(0)
    mats = affine_matrix(z, affine_basis('SE', 3))
    pairs = {(i, j): affine_matmul(mats[i], affine_inv(mats[j]))
             for i in range(n) for j in range(n) if i != j}
    opt = optimal_affine(pairs, basis='SE')
    labels = list(set(label for pair in pairs for label in pair))
    opt = [opt[labels.index(i)] for i in range(n)]
    for (i, j), mat in pairs.items():
        assert torch.allclose(affine_matmul(opt[i], affine_inv(opt[j])), mat,
                              atol=1e-2)


def test_optinorm_end_to_end(tmp_path):
    nib = pytest.importorskip('nibabel')
    from scipy.ndimage import gaussian_filter, map_coordinates
    from nitorch.cli.registration.optinorm.main import optinorm

    # two warped and rigidly moved copies of a smooth phantom
    rng = np.random.default_rng(0)
    n = 24
    base = gaussian_filter(rng.random([n] * 3), 2)
    base = (base - base.min()) / (base.max() - base.min())
    grid = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing='ij'))
    base *= np.sqrt(((grid - (n - 1) / 2) ** 2).sum(0)) < 9
    inputs = []
    for k, (angle, shift) in enumerate([(4, (2, -1, 1)), (-3, (-1, 2, -2))]):
        disp = np.stack([gaussian_filter(rng.standard_normal([n] * 3), 4)
                         for _ in range(3)])
        disp *= 1 / np.abs(disp).max()
        dat = map_coordinates(base, grid + disp, order=3, mode='nearest')
        a = np.deg2rad(angle)
        aff = np.eye(4)
        aff[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        aff[:3, :3] *= 2
        aff[:3, 3] = -n + np.asarray(shift)
        path = os.path.join(tmp_path, f'img{k + 1}.nii.gz')
        nib.save(nib.Nifti1Image(dat.astype('float32'), aff), path)
        inputs.append(path)

    out = os.path.join(tmp_path, 'out')
    normalised = optinorm(inputs, out, affine_levels=':1', nonlin_levels=':1',
                          both=True, cleanup=True, verbose=0)

    assert normalised == [os.path.join(out, 'opt', f'img{k}.norm.nii.gz')
                          for k in (1, 2)]
    for mode in ('weighted', 'unweighted'):
        for k in (1, 2):
            assert os.path.exists(os.path.join(
                out, 'opt', 'svf', mode, f'svf_optim_{mode}_{k}.nii.gz'))
    # --cleanup removes the pairwise registrations
    assert not os.path.exists(os.path.join(out, 'aff'))
    assert not os.path.exists(os.path.join(out, 'svf'))

    # images are better aligned after rigid, and after nonlinear, alignment
    def corr(pattern):
        a, b = [nib.load(os.path.join(out, 'opt', pattern.format(k)))
                .get_fdata() for k in (1, 2)]
        mask = (a + b) > 0.1
        return np.corrcoef(a[mask], b[mask])[0, 1]

    rigid = corr('img{}.aligned.resliced.nii.gz')
    nonlin = corr('img{}.norm.nii.gz')
    assert nonlin > rigid > 0.5
    assert nonlin > 0.9
