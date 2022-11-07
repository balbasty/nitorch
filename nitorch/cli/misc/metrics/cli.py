import os.path
import torch
import warnings
from nitorch.core.py import make_list, fileparts
from nitorch.core.utils import unfold, quantile
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
    # except Exception as e:
    #     print(f'[ERROR] {str(e)}', file=sys.stderr)


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
    is_label = any(metric in label_metrics for metric in options.metrics)

    pred, _ = load_data(options.pred, is_label, device)
    ref, affine = load_data(options.ref, is_label, device)
    vx = voxel_size(affine).tolist()
    if pred.shape != ref.shape:
        raise ValueError('Incompatible shapes')

    fwhm = options.fwhm
    unit = 'vox'
    if isinstance(fwhm[-1], str):
        *fwhm, unit = fwhm
    fwhm = make_list(fwhm, len(vx))
    fwhm = [f/v for f, v in zip(fwhm, vx)] if unit == 'mm' else fwhm[:len(vx)]

    metrics = options.metrics
    if 'labels' in metrics:
        metrics = label_metrics
    if 'image' in metrics:
        metrics = image_metrics

    if 'ssim' in options.metrics or 'psnr' in metrics:
        rng = options.range
        if len(rng) == 1:
            rng = (0, rng[0])
        elif len(rng) == 0:
            rng = None

    results = {}
    if is_label:
        results = compute_label_metrics(
            metrics, pred, ref, options.average, options.exclude)
    else:
        for metric in options.metrics:
            kwargs = dict()
            if metric == 'ssim':
                kwargs['fwhm'] = fwhm
                kwargs['gauss'] = options.gauss
                kwargs['range'] = rng
            elif metric == 'psnr':
                kwargs['range'] = rng
            results[metric] = globals()[metric](pred, ref, options.average, **kwargs)

    dir, base, _ = fileparts(options.pred[0])
    ofname = options.output.format(dir=dir or '.', sep=os.path.sep, base=base)
    if options.append:
        with open(ofname) as f:
            results = json.load(f).update({
                (options.pred[0], options.ref[0]): results
            })
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


def mse(pred, ref, average=None):
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


def rmse(pred, ref, average=None):
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
        results['micro'] = (acc / ref.numel()).ref.sqrt_()
    if 'macro' in average:
        results['macro'] = sum(results0.values()) / len(results0)

    for label in results.keys():
        results[label] = results[label].item()
    return results


def mrse(pred, ref, average=None):
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


def mad(pred, ref, average=None):
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


def psnr(pred, ref, average=None, range=None):
    results = {}
    for c in range(len(ref)):
        pred1 = pred[c]
        ref1 = ref[c]
        range1 = range or (0, pred.max())
        peak = range1[1]
        mse = (pred1 - ref1).square().mean()
        results[c] = 10 * torch.log10(peak**2 / mse)

    results0 = dict(results)
    if 'micro' in average:
        range1 = range or (0, pred.max())
        peak = range1[1]
        mse = (pred - ref).square().mean()
        results['micro'] = 10 * torch.log10(peak**2 / mse)
    if 'macro' in average:
        results['macro'] = sum(results0.values()) / len(results0)

    for label in results.keys():
        results[label] = results[label].item()
    return results


def ssim(pred, ref, average=None, range=None, fwhm=3, gauss=False):
    if fwhm == 'full':
        return ssim_full(pred, ref, average, range)
    if gauss:
        return ssim_gauss(pred, ref, average, range)

    k1, k2 = 0.01, 0.03
    ndim = pred.dim() - 1
    fwhm = make_list(fwhm, ndim)
    fwhm = [int(math.ceil(f)) for f in fwhm]

    def _ssim(p, r, l):
        c1, c2 = (k1 * l) ** 2, (k2 * l) ** 2
        p = unfold(p, fwhm)
        dims = list(range(-ndim, 0))
        pred_mean = p.mean(dims)
        pred_var = p.var(dims)
        ref_mean = r.mean(dims)
        ref_var = r.var(dims)
        cov = ((p - pred_mean) * (r - ref_mean)).mean(dims)
        ssim = (2 * pred_mean * ref_mean) * (2 * cov + c2)
        ssim /= (pred_mean ** 2 + ref_mean ** 2 + c1) * (pred_var + ref_var + c2)
        return ssim.mean().cpu()

    results = {}
    for c in range(len(ref)):
        pred1 = pred[c]
        ref1 = ref[c]
        range1 = range or (0, pred.max())
        range1 = range1[1] - range1[0]
        results[c] = _ssim(pred1, ref1, range1)

    results0 = dict(results)
    if 'micro' in average:
        range1 = range or (0, pred.max())
        range1 = range1[1] - range1[0]
        results['micro'] = _ssim(pred, ref, range1)
    if 'macro' in average:
        results['macro'] = sum(results0.values()) / len(results0)

    for label in results.keys():
        results[label] = results[label].item()
    return results


def ssim_gauss(pred, ref, average=None, range=None, fwhm=3):

    k1, k2 = 0.01, 0.03
    ndim = pred.dim() - 1
    fwhm = make_list(fwhm, ndim)

    def _ssim(p, r, l):
        c1, c2 = (k1 * l) ** 2, (k2 * l) ** 2
        norm = p.numel()
        pred_mean = smooth(p, fwhm=fwhm, dim=ndim).div_(norm)
        pred_var = smooth(p * p, fwhm=fwhm, dim=ndim).div_(norm)
        ref_mean = smooth(r, fwhm=fwhm, dim=ndim).div_(norm)
        ref_var = smooth(r * r, fwhm=fwhm, dim=ndim).div_(norm)
        cov = smooth(p * r, fwhm=fwhm, dim=ndim).div_(norm)
        pred_var -= pred_mean * pred_mean
        ref_var -= ref_mean * ref_mean
        cov -= pred_mean * ref_mean
        ssim = (2 * pred_mean * ref_mean) * (2 * cov + c2)
        ssim /= (pred_mean ** 2 + ref_mean ** 2 + c1) * (pred_var + ref_var + c2)
        return ssim.mean()

    results = {}
    for c in range(len(ref)):
        pred1 = pred[c]
        ref1 = ref[c]
        range1 = range or (0, pred.max())
        range1 = range1[1] - range1[0]
        results[c] = _ssim(pred1, ref1, range1)

    results0 = dict(results)
    if 'micro' in average:
        range1 = range or (0, pred.max())
        range1 = range1[1] - range1[0]
        results['micro'] = _ssim(pred, ref, range1)
    if 'macro' in average:
        results['macro'] = sum(results0.values()) / len(results0)

    for label in results.keys():
        results[label] = results[label].item()
    return results


def ssim_full(pred, ref, average=None, range=None):

    k1, k2 = 0.01, 0.03

    def _ssim(p, r, l):
        c1, c2 = (k1 * l) ** 2, (k2 * l) ** 2
        pred_mean = p.mean().cpu()
        pred_var = p.var().cpu()
        ref_mean = r.mean().cpu()
        ref_var = r.var().cpu()
        cov = ((p - pred_mean) * (r - ref_mean)).mean().cpu()
        ssim = (2 * pred_mean * ref_mean) * (2 * cov + c2)
        ssim /= (pred_mean ** 2 + ref_mean ** 2 + c1) * (pred_var + ref_var + c2)
        return ssim


    results = {}
    for c in range(len(ref)):
        pred1 = pred[c]
        ref1 = ref[c]
        range1 = range or (0, pred.max())
        range1 = range1[1] - range1[0]
        results[c] = _ssim(pred1, ref1, range1)

    results0 = dict(results)
    if 'micro' in average:
        range1 = range or (0, pred.max())
        range1 = range1[1] - range1[0]
        results['micro'] = _ssim(pred, ref, range1)
    if 'macro' in average:
        results['macro'] = sum(results0.values()) / len(results0)

    for label in results.keys():
        results[label] = results[label].item()
    return results


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