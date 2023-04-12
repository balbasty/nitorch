from nitorch import spatial, io
from nitorch.core import py
import torch
import os


def info(inp, meta=None, stat=False):
    """Print information on a volume.

    Parameters
    ----------
    inp : str or (tensor, tensor)
        Either a path to a volume file or a tuple `(dat, affine)`, where
        the first element contains the volume data and the second contains
        the orientation matrix.
    meta : sequence of str
        List of fields to print.
        By default, a list of common fields is used.
    stat : bool, default=False
        Compute intensity statistics

    """

    meta = meta or []
    metadata = {}
    is_file = isinstance(inp, str)
    if is_file:
        fname = inp
        f = io.volumes.map(inp)
        if stat:
            inp = (f.fdata(), f.affine)
        else:
            inp = (f.shape, f.affine)
        metadata = f.metadata(meta)
        metadata['dtype'] = f.dtype
    dat, aff = inp
    if not is_file:
        metadata['dtype'] = dat.dtype
    if torch.is_tensor(dat):
        shape = dat.shape
    else:
        shape = dat

    pad = max([0] + [len(m) for m in metadata.keys()])
    if not meta:
        more_fields = ['shape', 'layout', 'filename']
        pad = max(pad, max(len(f) for f in more_fields))
    title = lambda tag: ('{tag:' + str(pad) + 's}').format(tag=tag)

    if not meta:
        if is_file:
            print(f'{title("filename")} : {fname}')
        print(f'{title("shape")} : {tuple(shape)}')
        layout = spatial.affine_to_layout(aff)
        layout = spatial.volume_layout_to_name(layout)
        print(f'{title("layout")} : {layout}')
        center = torch.as_tensor(shape[:3], dtype=torch.float)/2
        center = spatial.affine_matvec(aff, center)
        print(f'{title("center")} : {tuple(center.tolist())} mm (RAS)')
        if stat and torch.is_tensor(dat):
            chandim = dat.shape[3:]
            if not chandim:
                vmin = dat.min().tolist()
                vmax = dat.max().tolist()
                vmean = dat.mean().tolist()
            else:
                dat1 = dat.reshape([-1, *chandim])
                vmin = dat1.min(dim=0).values.tolist()
                vmax = dat1.max(dim=0).values.tolist()
                vmean = dat1.mean(dim=0).tolist()
            print(f'{title("min")} : {vmin}')
            print(f'{title("max")} : {vmax}')
            print(f'{title("mean")} : {vmean}')

    for key, value in metadata.items():
        if value is None and not meta:
            continue
        if torch.is_tensor(value):
            value = str(value.numpy())
            value = value.split('\n')
            value = ('\n' + ' ' * (pad+3)).join(value)
        print(f'{title(key)} : {value}')
