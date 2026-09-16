"""Anatomix's experimental 3D Vision Transformer (`anatomix-dev-vit`).

Wraps the third-party `dynamic_network_architectures` package's `PrimusV2`
architecture (a hybrid residual-CNN tokenizer + EVA-style transformer +
transpose-conv decoder) with anatomix's own small additions: per-head
query/key `LayerNorm` in every attention block, and a stateless channel
-demean output normalization. This is a thin wrapper around a real,
published architecture -- not a from-scratch reimplementation. See
``specs/003-anatomix-vit-preprocessing/research.md`` §4 for how these exact
parameters and additions were verified against the real published
`anatomix-dev-vit` checkpoint (a strict, zero-mismatch `load_state_dict`).

Requires the optional `dynamic_network_architectures` dependency
(``pip install nitorch[anatomix-vit]``).
"""
from torch import nn

__all__ = ['AnatomixViT', 'ARCHITECTURE_DEFAULTS', 'INPUT_SIZE']

#: Fixed working resolution this architecture was trained at and requires.
INPUT_SIZE = 128

#: Architecture kwargs matching the published `anatomix-dev-vit` checkpoint.
ARCHITECTURE_DEFAULTS = dict(
    input_channels=1,
    num_classes=32,
    embed_dim=396,
    eva_depth=12,
    eva_numheads=6,
    patch_embed_size=(8, 8, 8),
    input_shape=(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE),
    num_register_tokens=8,
    init_values=0.1,
    scale_attn_inner=True,
)


def _import_primus_v2():
    try:
        from dynamic_network_architectures.architectures.primus import (
            PrimusV2,
        )
    except ImportError as e:
        raise ImportError(
            "AnatomixViT requires the optional `dynamic_network_architectures` "
            "package. Install it with `pip install nitorch[anatomix-vit]` or "
            "`pip install dynamic_network_architectures`."
        ) from e
    return PrimusV2


class _ChannelDemean(nn.Module):
    """Subtract each channel's own spatial mean.

    Stateless (no learned parameters); matches anatomix's
    ``out_norm='demean'`` output normalization exactly.
    """

    def forward(self, x):
        return x - x.mean(dim=tuple(range(2, x.dim())), keepdim=True)


class AnatomixViT(nn.Module):
    """anatomix's experimental 3D Vision Transformer (`anatomix-dev-vit`).

    A hybrid architecture: a 4-stage residual CNN tokenizer downsamples a
    ``(1, 1, 128, 128, 128)`` volume to a ``(1, 396, 16, 16, 16)`` token
    grid, 8 learned register tokens are prepended, 12 EVA-style transformer
    blocks (6 heads, SwiGLU MLP, LayerScale, per-head QK-LayerNorm) process
    the sequence, and a 3-stage transpose-conv decoder upsamples back to
    ``(1, 32, 128, 128, 128)``, followed by a per-channel demean.

    Parameters
    ----------
    **kwargs
        Overrides for `ARCHITECTURE_DEFAULTS`. Changing these makes the
        module architecturally incompatible with the published
        `anatomix-dev-vit` checkpoint -- only override for experimentation
        with a differently-trained checkpoint.

    """

    def __init__(self, **kwargs):
        super().__init__()
        primus_v2 = _import_primus_v2()
        config = dict(ARCHITECTURE_DEFAULTS)
        config.update(kwargs)
        self.output_nc = config['num_classes']
        #: Fixed spatial resolution this configured instance requires, per
        #: axis (derived from `input_shape`, not hardcoded -- so a
        #: differently-configured instance, e.g. a small one built for
        #: testing, reports its own true required size).
        self.input_size = config['input_shape'][0]
        self._primus = primus_v2(**config)

        head_dim = None
        for block in self._primus.eva.blocks:
            attn = block.attn
            if head_dim is None:
                head_dim = getattr(attn, 'head_dim', None) or (
                    attn.q_proj.out_features // attn.num_heads)
            attn.q_norm = nn.LayerNorm(head_dim)
            attn.k_norm = nn.LayerNorm(head_dim)

        self.out_norm = _ChannelDemean()

    def forward(self, x):
        """Extract dense modality-invariant features from a volume.

        Parameters
        ----------
        x : (1, 1, 128, 128, 128) tensor
            Single-channel input volume, exactly 128 voxels per axis.

        Returns
        -------
        (1, output_nc, 128, 128, 128) tensor
            Dense feature volume, same spatial shape as the input.

        """
        return self.out_norm(self._primus(x))
