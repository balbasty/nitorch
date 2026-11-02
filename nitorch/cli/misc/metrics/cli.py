import os.path
import torch
import warnings
from nitorch.core.py import make_list, fileparts
from nitorch.core import py
from nitorch.core.utils import quantile
from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from nitorch.spatial import smooth, voxel_size
from nitorch import io
from .parser import parser, help, label_metrics, image_metrics
import sys
import json
import math


def cli(args=None):
    f"""Command-line interface for `metrics`
    
    {help}
    
    """

    # Exceptions are dealt with here
    try:
        _cli(args)
    except ParseError as e:
        print(help)
        print(f'[ERROR] {str(e)}', file=sys.stderr)
        return 1
    except Exception as e:
        print(f'[ERROR] {str(e)}', file=sys.stderr)
        return 1


commands['metrics'] = cli


def _cli(args):
    """Command-line interface for `metrics` without exception handling"""
    args = args or sys.argv[1:]

    options = parser(args)
    if options.help:
        print(help)
        return

    # For now, I'll assume that inputs are nifti-like files, where we
    # can assume an [X, Y, Z, T, C] layout

    device = setup_device(*options.device)

    # expand metric groups (and drop the group names themselves)
    metrics = []
    for metric in options.metrics:
        if metric == 'labels':
            group = label_metrics[1:]
        elif metric == 'images':
            group = image_metrics[1:]
        else:
            group = [metric]
        metrics += [m for m in group if m not in metrics]
    lab_metrics = [m for m in metrics if m in label_metrics]
    img_metrics = [m for m in metrics if m in image_metrics]

    # dynamic range (SSIM/PSNR)
    rng = options.range
    if len(rng) == 1:
        rng = (0, rng[0])
    elif len(rng) == 0:
        rng = None
    else:
        rng = tuple(rng)

    results = {}
    if lab_metrics:
        pred, _ = load_data(options.pred, True, device)
        ref, affine = load_data(options.ref, True, device)
        if pred.shape != ref.shape:
            raise ValueError('Incompatible shapes')
        results.update(compute_label_metrics(
            lab_metrics, pred, ref, options.average, options.exclude))
    if img_metrics:
        pred, _ = load_data(options.pred, False, device)
        ref, affine = load_data(options.ref, False, device)
        if pred.shape != ref.shape:
            raise ValueError('Incompatible shapes')
        vx = voxel_size(affine).tolist()

        # SSIM window
        fwhm = list(options.fwhm)
        unit = 'vox'
        if any(isinstance(f, str) and f.lower() == 'full' for f in fwhm):
            fwhm = 'full'
        else:
            if isinstance(fwhm[-1], str):
                *fwhm, unit = fwhm
                unit = unit.lower()
            if unit not in ('vox', 'mm') or not fwhm or \
                    any(isinstance(f, str) for f in fwhm):
                raise ParseError(f'Invalid --ssim-fwhm: {options.fwhm}')
            fwhm = make_list(fwhm, len(vx))[:len(vx)]
            if unit == 'mm':
                fwhm = [f/v for f, v in zip(fwhm, vx)]

        for metric in img_metrics:
            kwargs = dict()
            if metric == 'ssim':
                kwargs['fwhm'] = fwhm
                kwargs['gauss'] = options.gauss
                kwargs['data_range'] = rng
            elif metric == 'psnr':
                kwargs['data_range'] = rng
            results[metric] = globals()[metric](pred, ref, options.average,
                                                **kwargs)

    # JSON keys are always strings: convert now so that merging with an
    # existing file does not create duplicate keys (1 vs "1")
    results = json.loads(json.dumps(results))

    dir, base, ext = fileparts(options.pred[0])
    ofname = options.output.format(dir=dir or '.', sep=os.path.sep,
                                   base=base, ext=ext)
    if options.append and os.path.exists(ofname):
        with open(ofname) as f:
            old_results = json.load(f)
        for metric, value in results.items():
            if isinstance(old_results.get(metric), dict) \
                    and isinstance(value, dict):
                old_results[metric].update(value)
            else:
                old_results[metric] = value
        results = old_results
    with open(ofname, 'w') as f:
        json.dump(results, f, indent=4)


def setup_device(device='cpu', ndevice=0):
    if device == 'gpu' and not torch.cuda.is_available():
        warnings.warn('CUDA not available. Switching to CPU.')
        device, ndevice = 'cpu', None
    if device == 'cpu':
        device = torch.device('cpu')
        if ndevice:
            torch.set_num_threads(ndevice)
    else:
        assert device == 'gpu'
        if ndevice is not None:
            device = torch.device(f'cuda:{ndevice}')
        else:
            device = torch.device('cuda')
    return device


def ensure_CXYZ(f):
    shape = f.shape
    if f.dim > 4:
        if f.shape[3] == 1:
            f = f.squeeze(3)
        elif f.shape[4] == 1:
            f = f.squeeze(4)
    elif f.dim == 3:
        f = f.unsqueeze(-1)
    if f.dim != 4:
        raise ValueError(f'Cannot reformat file with shape '
                         f'{tuple(shape)} as [C, X, Y, Z]')
    f = f.movedim(-1, 0)
    return f


def load_data(fnames, is_label, device):
    dat = []
    for fname in fnames:
        dat.append(ensure_CXYZ(io.map(fname)))
    affine = dat[0].affine
    dat = io.cat(dat)
    dat = (dat.data(device=device, dtype=torch.long) if is_label else
            dat.fdata(device=device))
    return dat, affine


def compute_label_metrics(metrics, pred, ref, average=None, exclude=(0,)):

    @torch.jit.script
    def get_stats(pred, ref, label: int):
        pred = pred == label
        ref = ref == label
        tp = (pred & ref).sum()
        tn = (~pred & ~ref).sum()
        fp = (pred & ~ref).sum()
        fn = (~pred & ref).sum()
        return tp.cpu(), tn.cpu(), fp.cpu(), fn.cpu()

    tp, tn, fp, fn = {}, {}, {}, {}
    labels = set(ref.unique().tolist())
    labels.difference_update(set(exclude))
    for label in labels:
        tp[label], tn[label], fp[label], fn[label] = get_stats(pred, ref, label)

    results = {}
    for metric in metrics:
        if metric.startswith('hausdorff'):
            results[metric] = globals()[metric](pred, ref, average, exclude)
        else:
            results[metric] = results1 = {}
            func = globals()[metric]
            for label in labels:
                results1[label] = func(tp[label], tn[label], fp[label], fn[label])
            results0 = dict(results1)
            if 'macro' in average:
                results1['macro'] = sum(results0.values()) / len(results0)
            if 'micro' in average:
                results1['micro'] = func(sum(tp.values()), sum(tn.values()),
                                         sum(fp.values()), sum(fn.values()))
            if 'weighted' in average:
                results1['weighted'] = (
                    sum(results1[label] * (tp[label] + fn[label]) for label in labels)
                    / sum(tp[label] + fn[label] for label in labels)
                )

    for metric in results.keys():
        for label in results[metric].keys():
            if hasattr(results[metric][label], 'item'):
                results[metric][label] = results[metric][label].item()

    return results


def dice(tp, tn, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


def jaccard(tp, tn, fp, fn):
    return tp / (tp + fp + fn)


def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def ppv(tp, tn, fp, fn):
    return tp / (tp + fp)


def npv(tp, tn, fp, fn):
    return tn / (tn + fn)


def fdr(tp, tn, fp, fn):
    return fp / (tp + fp)


def tpr(tp, tn, fp, fn):
    return tp / (tp + fn)


def fpr(tp, tn, fp, fn):
    return fp / (tn + fp)


def fnr(tp, tn, fp, fn):
    return fn / (tp + fn)


def tnr(tp, tn, fp, fn):
    return tn / (tn + fp)


def plr(tp, tn, fp, fn):
    return tp * (tn + fp) / (fp * (tp + fn))


def nlr(tp, tn, fp, fn):
    return fn * (tn + fp) / (tn * (tp + fn))


def mse(pred, ref, average=()):
    results = {}
    acc = 0
    for c in range(len(ref)):
        pred1 = pred[c]
        ref1 = ref[c]
        results[c] = (pred1 - ref1).square().mean()
        if 'micro' in average:
            acc += results[c] * ref1.numel()

    results0 = dict(results)
    if 'micro' in average:
        results['micro'] = acc / ref.numel()
    if 'macro' in average:
        results['macro'] = sum(results0.values()) / len(results0)

    for label in results.keys():
        results[label] = results[label].item()
    return results


def rmse(pred, ref, average=()):
    results = {}
    acc = 0
    for c in range(len(ref)):
        pred1 = pred[c]
        ref1 = ref[c]
        results[c] = (pred1 - ref1).square().mean()
        if 'micro' in average:
            acc += results[c] * ref1.numel()
        results[c].sqrt_()

    results0 = dict(results)
    if 'micro' in average:
        results['micro'] = (acc / ref.numel()).sqrt_()
    if 'macro' in average:
        results['macro'] = sum(results0.values()) / len(results0)

    for label in results.keys():
        results[label] = results[label].item()
    return results


def mrse(pred, ref, average=()):
    results = {}
    for c in range(len(ref)):
        pred1 = pred[c]
        ref1 = ref[c]
        results[c] = (pred1 - ref1).abs().mean()

    results0 = dict(results)
    if 'micro' in average:
        results['micro'] = (pred - ref).square().sum(0).sqrt().mean()
    if 'macro' in average:
        results['macro'] = sum(results0.values()) / len(results0)

    for label in results.keys():
        results[label] = results[label].item()
    return results


def mad(pred, ref, average=()):
    results = {}
    for c in range(len(ref)):
        pred1 = pred[c]
        ref1 = ref[c]
        results[c] = (pred1 - ref1).abs().median()

    results0 = dict(results)
    if 'micro' in average:
        results['micro'] = (pred - ref).abs().median()
    if 'macro' in average:
        results['macro'] = sum(results0.values()) / len(results0)

    for label in results.keys():
        results[label] = results[label].item()
    return results


def _dynamic_range(ref, data_range=None):
    """Width of the dynamic range: max - min.

    By default (`data_range=None`), the min/max of the *reference* are used.
    """
    if data_range is None:
        data_range = (ref.min(), ref.max())
    mn, mx = data_range
    return mx - mn


def psnr(pred, ref, average=(), data_range=None):
    """Peak Signal-to-Noise Ratio = 10 * log10(peak**2 / MSE)

    The peak is the width of the dynamic range of the reference
    (max - min), or of `data_range` if provided.
    """
    results = {}
    for c in range(len(ref)):
        pred1 = pred[c]
        ref1 = ref[c]
        peak = _dynamic_range(ref1, data_range)
        mse = (pred1 - ref1).square().mean()
        results[c] = 10 * torch.log10(peak**2 / mse)

    results0 = dict(results)
    if 'micro' in average:
        peak = _dynamic_range(ref, data_range)
        mse = (pred - ref).square().mean()
        results['micro'] = 10 * torch.log10(peak**2 / mse)
    if 'macro' in average:
        results['macro'] = sum(results0.values()) / len(results0)

    for label in results.keys():
        results[label] = results[label].item()
    return results


def _ssim_map(pred_mean, ref_mean, pred_var, ref_var, cov, data_range,
              k1=0.01, k2=0.03):
    """SSIM (Wang et al., 2004) from (local) first and second moments

    SSIM = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) /
           ((mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2))
    with C1 = (k1 * L)**2, C2 = (k2 * L)**2 and L the dynamic range.
    """
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    ssim = (2 * pred_mean * ref_mean + c1) * (2 * cov + c2)
    ssim = ssim / ((pred_mean ** 2 + ref_mean ** 2 + c1) *
                   (pred_var + ref_var + c2))
    return ssim


def _ssim_average(_ssim, pred, ref, average, data_range):
    results = {}
    for c in range(len(ref)):
        pred1 = pred[c]
        ref1 = ref[c]
        results[c] = _ssim(pred1, ref1, _dynamic_range(ref1, data_range))

    results0 = dict(results)
    if 'micro' in average:
        results['micro'] = _ssim(pred, ref, _dynamic_range(ref, data_range))
    if 'macro' in average:
        results['macro'] = sum(results0.values()) / len(results0)

    for label in results.keys():
        results[label] = results[label].item()
    return results


def ssim(pred, ref, average=(), data_range=None, fwhm=3, gauss=False):
    """Structural Similarity Index Measure

    fwhm : 'full' -> global statistics
           otherwise, if gauss -> Gaussian window with this FWHM (in voxels)
           otherwise -> sliding square window of this size (in voxels)
    """
    if fwhm == 'full':
        return ssim_full(pred, ref, average, data_range)
    if gauss:
        return ssim_gauss(pred, ref, average, data_range, fwhm)

    ndim = pred.dim() - 1
    fwhm = make_list(fwhm, ndim)
    fwhm = [max(int(math.ceil(f)), 1) for f in fwhm]
    pool = getattr(torch.nn.functional, f'avg_pool{ndim}d')

    def box(x):
        # local mean over a sliding window (stride 1, valid region only)
        size = [min(f, s) for f, s in zip(fwhm, x.shape[-ndim:])]
        batch, spatial = x.shape[:-ndim], x.shape[-ndim:]
        x = pool(x.reshape([-1, 1, *spatial]), size, stride=1)
        return x.reshape([*batch, *x.shape[-ndim:]])

    def _ssim(p, r, l):
        size = [min(f, s) for f, s in zip(fwhm, p.shape[-ndim:])]
        n = py.prod(size)
        # sample (unbiased) statistics, as in Wang et al. (2004)
        unbias = n / max(n - 1, 1)
        pred_mean = box(p)
        ref_mean = box(r)
        pred_var = (box(p * p) - pred_mean * pred_mean).mul_(unbias)
        ref_var = (box(r * r) - ref_mean * ref_mean).mul_(unbias)
        cov = (box(p * r) - pred_mean * ref_mean).mul_(unbias)
        ssim = _ssim_map(pred_mean, ref_mean, pred_var, ref_var, cov, l)
        return ssim.mean().cpu()

    return _ssim_average(_ssim, pred, ref, average, data_range)


def ssim_gauss(pred, ref, average=(), data_range=None, fwhm=3):
    """SSIM with Gaussian-weighted local statistics (weights sum to one)"""

    ndim = pred.dim() - 1
    fwhm = make_list(fwhm, ndim)

    def _ssim(p, r, l):
        pred_mean = smooth(p, fwhm=fwhm, dim=ndim)
        ref_mean = smooth(r, fwhm=fwhm, dim=ndim)
        pred_var = smooth(p * p, fwhm=fwhm, dim=ndim)
        ref_var = smooth(r * r, fwhm=fwhm, dim=ndim)
        cov = smooth(p * r, fwhm=fwhm, dim=ndim)
        pred_var -= pred_mean * pred_mean
        ref_var -= ref_mean * ref_mean
        cov -= pred_mean * ref_mean
        ssim = _ssim_map(pred_mean, ref_mean, pred_var, ref_var, cov, l)
        return ssim.mean().cpu()

    return _ssim_average(_ssim, pred, ref, average, data_range)


def ssim_full(pred, ref, average=(), data_range=None):
    """SSIM with global statistics (single window covering the image)"""

    def _ssim(p, r, l):
        pred_mean = p.mean()
        pred_var = p.var()
        ref_mean = r.mean()
        ref_var = r.var()
        # sample (unbiased) covariance, consistent with `var()`
        cov = ((p - pred_mean) * (r - ref_mean)).sum() / max(p.numel() - 1, 1)
        ssim = _ssim_map(pred_mean, ref_mean, pred_var, ref_var, cov, l)
        return ssim.cpu()

    return _ssim_average(_ssim, pred, ref, average, data_range)


def hausdorff95(pred, ref, average=None, exclude=(0,)):
    return hausdorff(pred, ref, average, exclude, 0.95)


def hausdorff(pred, ref, average=None, exclude=(0,), percentile: float = 1):

    def _hausdorff(p, r, acc=None):
        from nitorch.spatial import euclidean_distance_transform
        if ~p.any():
            return torch.zeros([])
        r = euclidean_distance_transform(~r)
        p = r[p]
        if acc is not None:
            acc.append(p)
        if percentile == 1:
            return p.max().cpu()
        else:
            return quantile(p, percentile).cpu()

    results = {}
    labels = set(ref.unique().tolist())
    labels.difference_update(set(exclude))
    acc = [] if 'micro' in average else None
    for label in labels:
        results[label] = _hausdorff(pred == label, ref == label, acc)

    results0 = dict(results)
    if 'micro' in average:
        acc = torch.cat(acc)
        if percentile == 1:
            acc = acc.max().cpu()
        else:
            acc = quantile(acc, percentile).cpu()
        results['micro'] = acc
    if 'macro' in average:
        results['macro'] = sum(results0.values()) / len(results0)

    for label in results.keys():
        if hasattr(results[label], 'item'):
            results[label] = results[label].item()
    return results