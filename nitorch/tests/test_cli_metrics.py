"""`metrics` command: label/image metrics, groups, --append, SSIM/PSNR."""
import json
import math
import os

import numpy as np
import pytest
import torch

from nitorch.cli.misc.metrics.cli import _cli as metrics
from nitorch.cli.misc.metrics import cli as mcli
from nitorch.cli.misc.metrics import parser as mparser

nib = pytest.importorskip('nibabel')


def save(tmp_path, name, dat):
    fname = str(tmp_path / name)
    nib.save(nib.Nifti1Image(np.asarray(dat), np.eye(4)), fname)
    return fname


def run(args):
    metrics(args)
    oname = args[args.index('-o') + 1]
    with open(oname) as f:
        return json.load(f)


@pytest.fixture
def labels(tmp_path):
    ref = np.zeros([4, 5, 6], dtype='int16')
    ref[:2] = 1
    pred = np.zeros_like(ref)
    pred[:1] = 1
    pred[3, :2] = 1
    return (save(tmp_path, 'pred.nii.gz', pred),
            save(tmp_path, 'ref.nii.gz', ref), pred, ref)


@pytest.fixture
def images(tmp_path):
    rng = np.random.RandomState(0)
    ref = rng.rand(5, 6, 7).astype('float32') * 4.
    ref.flat[0] = 0.
    ref.flat[1] = 4.        # ref range = [0, 4]
    pred = ref + rng.randn(*ref.shape).astype('float32') * 0.5
    pred.flat[2] = 10.      # pred max != ref max
    return (save(tmp_path, 'pred.nii.gz', pred),
            save(tmp_path, 'ref.nii.gz', ref), pred, ref)


def stats(pred, ref, label=1):
    p, r = pred == label, ref == label
    return (p & r).sum(), (~p & ~r).sum(), (p & ~r).sum(), (~p & r).sum()


def manual_ssim(x, y, L, k1=0.01, k2=0.03):
    """Global SSIM (Wang et al. 2004), sample (N-1) statistics."""
    x = np.asarray(x, 'float64').ravel()
    y = np.asarray(y, 'float64').ravel()
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    cxy = ((x - mx) * (y - my)).sum() / (x.size - 1)
    return ((2 * mx * my + c1) * (2 * cxy + c2)
            / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))


def test_fnr(tmp_path, labels):
    fp, fr, pred, ref = labels
    out = str(tmp_path / 'o.json')
    res = run(['-p', fp, '-r', fr, '-m', 'fnr', 'tpr', '-o', out])
    tp, tn, fp_, fn = stats(pred, ref)
    assert res['fnr']['1'] == pytest.approx(fn / (tp + fn))
    assert res['fnr']['1'] + res['tpr']['1'] == pytest.approx(1)


def test_label_group(tmp_path, labels):
    fp, fr, pred, ref = labels
    out = str(tmp_path / 'o.json')
    res = run(['-p', fp, '-r', fr, '-m', 'labels', '-o', out])
    expected = [m for m in mparser.label_metrics if m != 'labels']
    assert set(res) == set(expected)
    assert 'fnr' in res
    tp, tn, fp_, fn = stats(pred, ref)
    assert res['dice']['1'] == pytest.approx(2 * tp / (2 * tp + fp_ + fn))


def test_image_group(tmp_path, images):
    fp, fr, pred, ref = images
    out = str(tmp_path / 'o.json')
    res = run(['-p', fp, '-r', fr, '-m', 'images', '-o', out])
    expected = [m for m in mparser.image_metrics if m != 'images']
    assert set(res) == set(expected)
    assert res['mse']['0'] == pytest.approx(((pred - ref) ** 2).mean(),
                                            rel=1e-5)


def test_label_and_image_mixed(tmp_path, images):
    fp, fr, pred, ref = images
    out = str(tmp_path / 'o.json')
    res = run(['-p', fp, '-r', fr, '-m', 'mse', 'dice', '-o', out])
    # image metrics are computed on floating point data
    assert res['mse']['0'] == pytest.approx(((pred - ref) ** 2).mean(),
                                            rel=1e-5)
    assert 'dice' in res


def test_append(tmp_path, images):
    fp, fr, pred, ref = images
    out = str(tmp_path / 'o.json')
    # append to a file that does not exist yet
    run(['-p', fp, '-r', fr, '-m', 'mse', '-a', '-o', out])
    res = run(['-p', fp, '-r', fr, '-m', 'rmse', '-a', '-o', out])
    assert res is not None
    assert set(res) == {'mse', 'rmse'}
    assert res['rmse']['0'] == pytest.approx(math.sqrt(res['mse']['0']),
                                             rel=1e-5)
    # re-appending an existing metric merges values without duplicates
    run(['-p', fp, '-r', fr, '-m', 'mse', '-s', 'macro', '-a', '-o', out])
    with open(out) as f:
        txt = f.read()
    res = json.loads(txt)
    assert set(res) == {'mse', 'rmse'}
    assert set(res['mse']) == {'0', 'macro'}
    assert txt.count('"0"') == 2
    # without --append, the file is overwritten
    res = run(['-p', fp, '-r', fr, '-m', 'mad', '-o', out])
    assert set(res) == {'mad'}


def test_rmse_micro(tmp_path, images):
    fp, fr, pred, ref = images
    fp2 = save(tmp_path, 'pred2.nii.gz', pred * 2)
    out = str(tmp_path / 'o.json')
    res = run(['-p', fp, fp2, '-r', fr, fr, '-m', 'rmse',
               '-s', 'micro', 'macro', '-o', out])
    pall = np.stack([pred, pred * 2])
    rall = np.stack([ref, ref])
    r0 = math.sqrt(((pred - ref) ** 2).mean())
    r1 = math.sqrt(((pred * 2 - ref) ** 2).mean())
    assert res['rmse']['pred'] == pytest.approx(r0, rel=1e-5)
    assert res['rmse']['pred2'] == pytest.approx(r1, rel=1e-5)
    assert res['rmse']['micro'] == pytest.approx(
        math.sqrt(((pall - rall) ** 2).mean()), rel=1e-5)
    assert res['rmse']['macro'] == pytest.approx((r0 + r1) / 2, rel=1e-5)


def test_psnr_reference_range(tmp_path, images):
    fp, fr, pred, ref = images
    out = str(tmp_path / 'o.json')
    res = run(['-p', fp, '-r', fr, '-m', 'psnr', '-s', 'micro', '-o', out])
    mse = ((pred.astype('float64') - ref) ** 2).mean()
    expected = 10 * math.log10(4. ** 2 / mse)   # peak = max(ref) = 4
    assert res['psnr']['0'] == pytest.approx(expected, rel=1e-5)
    assert res['psnr']['micro'] == pytest.approx(expected, rel=1e-5)
    # explicit range
    res = run(['-p', fp, '-r', fr, '-m', 'psnr', '--range', '2', '-o', out])
    assert res['psnr']['0'] == pytest.approx(10 * math.log10(4. / mse),
                                             rel=1e-5)


def test_ssim_full(tmp_path, images):
    fp, fr, pred, ref = images
    out = str(tmp_path / 'o.json')
    res = run(['-p', fp, '-r', fr, '-m', 'ssim', '--ssim-fwhm', 'full',
               '-o', out])
    assert res['ssim']['0'] == pytest.approx(manual_ssim(pred, ref, 4.),
                                             rel=1e-4)
    # identical images -> 1
    res = run(['-p', fr, '-r', fr, '-m', 'ssim', '--ssim-fwhm', 'full',
               '-o', out])
    assert res['ssim']['0'] == pytest.approx(1, rel=1e-6)


def test_ssim_window(tmp_path, images):
    fp, fr, pred, ref = images
    out = str(tmp_path / 'o.json')
    res = run(['-p', fp, '-r', fr, '-m', 'ssim', '--ssim-fwhm', '3',
               '-s', 'micro', '-o', out])
    vals = []
    nx, ny, nz = ref.shape
    for i in range(nx - 2):
        for j in range(ny - 2):
            for k in range(nz - 2):
                sl = (slice(i, i + 3), slice(j, j + 3), slice(k, k + 3))
                vals.append(manual_ssim(pred[sl], ref[sl], 4.))
    expected = np.mean(vals)
    assert res['ssim']['0'] == pytest.approx(expected, rel=1e-4)
    assert res['ssim']['micro'] == pytest.approx(expected, rel=1e-4)
    res = run(['-p', fr, '-r', fr, '-m', 'ssim', '-o', out])
    assert res['ssim']['0'] == pytest.approx(1, rel=1e-5)


def test_ssim_gauss_fwhm(tmp_path, images):
    fp, fr, pred, ref = images
    tpred = torch.as_tensor(pred)[None].double()
    tref = torch.as_tensor(ref)[None].double()
    s1 = mcli.ssim(tpred, tref, [], fwhm=[1.5] * 3, gauss=True)[0]
    s2 = mcli.ssim(tpred, tref, [], fwhm=[4] * 3, gauss=True)[0]
    assert s1 != pytest.approx(s2)
    # identical images -> 1
    assert mcli.ssim(tref, tref, [], fwhm=[2] * 3,
                     gauss=True)[0] == pytest.approx(1)
    # Gaussian-weighted local statistics (Wang et al. 2004)
    from nitorch.spatial import smooth
    f = [2.] * 3
    sm = lambda x: smooth(x, fwhm=f, dim=3)
    L = 4.
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mp, mr = sm(tpred), sm(tref)
    vp, vr = sm(tpred * tpred) - mp ** 2, sm(tref * tref) - mr ** 2
    cv = sm(tpred * tref) - mp * mr
    m = ((2 * mp * mr + c1) * (2 * cv + c2)
         / ((mp ** 2 + mr ** 2 + c1) * (vp + vr + c2))).mean().item()
    assert mcli.ssim(tpred, tref, [], fwhm=f,
                     gauss=True)[0] == pytest.approx(m)
    # through the command line: --ssim-fwhm is honoured
    out = str(tmp_path / 'o.json')
    r1 = run(['-p', fp, '-r', fr, '-m', 'ssim', '--ssim-gauss',
              '--ssim-fwhm', '1.5', '-o', out])['ssim']['0']
    r2 = run(['-p', fp, '-r', fr, '-m', 'ssim', '--ssim-gauss',
              '--ssim-fwhm', '4', '-o', out])['ssim']['0']
    assert r1 == pytest.approx(s1, rel=1e-4)
    assert r2 == pytest.approx(s2, rel=1e-4)


def test_ext_in_output(tmp_path, images):
    fp, fr, pred, ref = images
    metrics(['-p', fp, '-r', fr, '-m', 'mse',
             '-o', '{dir}{sep}{base}{ext}.json'])
    assert os.path.exists(str(tmp_path / 'pred.nii.gz.json'))


def test_metrics_errors_return_status_1(tmp_path, capsys):
    """Parser and runtime errors print one line and return 1."""
    from nitorch.cli.misc.metrics.cli import cli
    assert cli(['-m', 'nope']) == 1
    assert cli(['-p', str(tmp_path / 'a.nii'), '-r', str(tmp_path / 'b.nii'),
                '-m', 'dice', '-o', str(tmp_path / 'o.json')]) == 1
    assert '[ERROR]' in capsys.readouterr().err


def test_args_not_mutated(tmp_path, images):
    fp, fr, pred, ref = images
    out = str(tmp_path / 'o.json')
    args = ['-p', fp, '-r', fr, '-m', 'mse', '-o', out]
    copy = list(args)
    metrics(args)
    assert args == copy
    from nitorch.cli.misc.metrics.cli import cli
    cli(args)
    assert args == copy


def test_weighted_hausdorff(tmp_path):
    ref = np.zeros([6, 6, 6], dtype='int16')
    ref[:3] = 1
    ref[3:, :2] = 2
    pred = np.array(ref)
    pred[5, 5, 5] = 1        # far false positive for label 1
    fp = save(tmp_path, 'pred.nii.gz', pred)
    fr = save(tmp_path, 'ref.nii.gz', ref)
    out = str(tmp_path / 'o.json')
    res = run(['-p', fp, '-r', fr, '-m', 'hausdorff',
               '-s', 'macro', 'weighted', '-o', out])['hausdorff']
    v1, v2 = (ref == 1).sum(), (ref == 2).sum()
    assert 'weighted' in res
    assert res['weighted'] == pytest.approx(
        (res['1'] * v1 + res['2'] * v2) / (v1 + v2))
    assert res['weighted'] != pytest.approx(res['macro'])


def test_weighted_image_metrics_error(tmp_path, images):
    from nitorch.cli.misc.metrics.cli import cli
    fp, fr, pred, ref = images
    out = str(tmp_path / 'o.json')
    with pytest.raises(mcli.ParseError, match='weighted'):
        metrics(['-p', fp, '-r', fr, '-m', 'mse', '-s', 'weighted',
                 '-o', out])
    assert cli(['-p', fp, '-r', fr, '-m', 'images', '-s', 'weighted',
                '-o', out]) == 1
    assert not os.path.exists(out)
    # label metrics still accept it
    run(['-p', fp, '-r', fr, '-m', 'dice', '-s', 'weighted', '-o', out])


def test_mad_wording():
    assert 'Median Absolute Deviation' not in mparser.help
    assert 'median absolute error' in mparser.help.lower()
    doc = (mcli.mad.__doc__ or '').lower()
    assert 'deviation' not in doc
    assert 'median absolute error' in doc


def test_multi_pair_keys(tmp_path, images):
    fp, fr, pred, ref = images
    fp2 = save(tmp_path, 'pred2.nii.gz', pred * 2)
    fr2 = save(tmp_path, 'ref2.nii.gz', ref)
    out = str(tmp_path / 'o.json')
    res = run(['-p', fp, fp2, '-r', fr, fr2, '-m', 'mse',
               '-s', 'macro', '-o', out])
    assert set(res['mse']) == {'pred', 'pred2', 'macro'}
    assert res['mse']['pred'] == pytest.approx(((pred - ref) ** 2).mean(),
                                               rel=1e-5)
    # --append with other pairs keeps pairs apart
    fp3 = save(tmp_path, 'pred3.nii.gz', pred * 3)
    res = run(['-p', fp3, fp, '-r', fr, fr, '-m', 'mse', '-a', '-o', out])
    assert set(res['mse']) == {'pred', 'pred2', 'pred3', 'macro'}
    # single pair: unchanged (channel index)
    res = run(['-p', fp, '-r', fr, '-m', 'mse', '-o', out])
    assert set(res['mse']) == {'0'}
