"""Pad-or-tile wrapper presenting a fixed-input-size model as one that
accepts any spatial shape.

`AnatomixViT` (and architectures like it) require an exact 128x128x128
input. `SlidingWindowRunner` hides that constraint from callers: it pads
small inputs up to the fixed window and crops back, or tiles large inputs
into overlapping windows and reassembles them with blended overlap -- see
``specs/003-anatomix-vit-preprocessing/research.md`` §2.
"""
import torch
from torch.nn import functional as F

__all__ = ['SlidingWindowRunner']


def _triangular_weight_1d(n, device, dtype):
    """A 1D weight ramping up then down, peaking at the center, used to
    blend overlapping tiles smoothly instead of showing a seam at tile
    borders. Never zero, so every voxel contributes to its tile's output.
    """
    idx = torch.arange(n, device=device, dtype=dtype)
    return torch.minimum(idx + 1, n - idx)


def _tile_starts(size, window, stride):
    """Start offsets for tiles of `window` covering [0, size), including a
    final tile flush with the right edge so the whole extent is covered."""
    if size <= window:
        return [0]
    starts = list(range(0, size - window + 1, stride))
    if starts[-1] != size - window:
        starts.append(size - window)
    return starts


class SlidingWindowRunner:
    """Wrap a fixed-input-size model to accept any spatial shape.

    Parameters
    ----------
    model : callable
        Maps a `(1, C_in, window, window, window)` tensor to a
        `(1, C_out, window, window, window)` tensor.
    window : int
        The model's required fixed spatial size, per axis.
    overlap : float, default=0.25
        Fraction of `window` adjacent tiles overlap by, when tiling.

    """

    def __init__(self, model, window, overlap=0.25):
        self.model = model
        self.window = window
        self.overlap = overlap

    def __call__(self, x):
        """
        Parameters
        ----------
        x : (1, C_in, *spatial) tensor
            Any spatial shape.

        Returns
        -------
        (1, C_out, *spatial) tensor
            Same spatial shape as the input.

        """
        spatial = x.shape[2:]
        window = self.window
        if all(s <= window for s in spatial):
            return self._run_padded(x)
        return self._run_tiled(x)

    def _run_padded(self, x):
        spatial = x.shape[2:]
        pad = []
        for s in reversed(spatial):  # F.pad takes dims in reverse order
            pad += [0, self.window - s]
        x_padded = F.pad(x, pad, mode='replicate')
        y = self.model(x_padded)
        crop = tuple(slice(0, s) for s in spatial)
        return y[(..., *crop)]

    def _run_tiled(self, x):
        window = self.window
        stride = max(1, int(round(window * (1 - self.overlap))))
        spatial = x.shape[2:]
        # per axis: tile start positions, and how many real (unpadded)
        # voxels each tile has along that axis (< window for a short axis)
        starts_per_axis = [_tile_starts(s, window, stride) for s in spatial]
        extract_per_axis = [min(window, s) for s in spatial]

        out_sum = None
        weight_sum = None
        w1d_full = _triangular_weight_1d(window, x.device, x.dtype)
        for i0 in starts_per_axis[0]:
            for j0 in starts_per_axis[1]:
                for k0 in starts_per_axis[2]:
                    ei, ej, ek = extract_per_axis
                    real_sl = (slice(i0, i0 + ei), slice(j0, j0 + ej),
                               slice(k0, k0 + ek))
                    tile_in = x[(..., *real_sl)]
                    pad = []
                    for e in (ek, ej, ei):  # F.pad takes dims in reverse order
                        pad += [0, window - e]
                    if any(pad):
                        tile_in = F.pad(tile_in, pad, mode='replicate')
                    tile_out = self.model(tile_in)[(..., slice(0, ei), slice(0, ej), slice(0, ek))]

                    weight_tile = (
                        w1d_full[:ei, None, None]
                        * w1d_full[None, :ej, None]
                        * w1d_full[None, None, :ek]
                    )[None, None]

                    if out_sum is None:
                        out_channels = tile_out.shape[1]
                        out_sum = x.new_zeros((x.shape[0], out_channels, *spatial))
                        weight_sum = x.new_zeros((1, 1, *spatial))

                    out_sum[(..., *real_sl)] += tile_out * weight_tile
                    weight_sum[(..., *real_sl)] += weight_tile

        return out_sum / weight_sum
