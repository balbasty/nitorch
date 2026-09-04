"""Anatomix modality-agnostic 3D feature extraction.

Anatomix (Dey et al.) is a pretrained 3D U-Net that maps a single-channel
biomedical image volume to a modality-agnostic feature representation.
This package vendors a minimal, dependency-light reimplementation of its
architecture (`unet.py`) and weight loading (`weights.py`), and exposes the
public `extract_features` function used both standalone (see
``specs/001-anatomix-registration-features/spec.md`` User Story 2) and by
`nitorch.tools.registration.pairwise_makeobj.make_image`'s `anatomix=`
feature-transform option (User Story 1).

No new hard dependency is introduced: only `torch` (already required by
nitorch) is needed for inference; the optional weight download path uses
the standard library's `urllib`.
"""

from .unet import AnatomixUNet
from .weights import AnatomixWeightsError, resolve_weights_path, load_state_dict_into

__all__ = ['AnatomixFeatureExtractor', 'extract_features', 'AnatomixWeightsError']

#: Architecture defaults matching the published `anatomix` checkpoint.
ARCHITECTURE_DEFAULTS = dict(
    num_downs=4,
    ngf=16,
    output_nc=16,
    norm='batch',
    interp='nearest',
    pooling='max',
)


class AnatomixFeatureExtractor:
    """Frozen anatomix feature extractor.

    Lazily builds and loads the vendored U-Net on first use, then reuses
    it across calls. Stateless from the caller's perspective beyond the
    loaded, frozen weights.

    Parameters
    ----------
    weights_path : str, optional
        Local path to a pretrained anatomix `.pth` checkpoint. Required
        unless `auto_download=True`.
    auto_download : bool, default=False
        Opt in to downloading the checkpoint from the official anatomix
        HuggingFace Hub distribution and caching it locally.
    cache_dir : str, optional
        Directory to cache downloaded weights in, if `auto_download=True`.
        Defaults to a nitorch-specific user cache directory.
    num_downs, ngf, output_nc, norm, interp, pooling
        Architecture parameters forwarded to `AnatomixUNet`. Default to
        `ARCHITECTURE_DEFAULTS`, matching the published checkpoint; only
        override these when loading a non-default checkpoint variant.

    Raises
    ------
    AnatomixWeightsError
        If no usable weight source resolves (FR-006), raised eagerly at
        construction time rather than lazily on first `extract` call.

    """

    def __init__(self, weights_path=None, auto_download=False, cache_dir=None,
                 **architecture):
        self.weights_path = resolve_weights_path(
            weights_path=weights_path, auto_download=auto_download,
            cache_dir=cache_dir)
        config = dict(ARCHITECTURE_DEFAULTS)
        config.update(architecture)
        self.architecture = config
        self._model = None
        self._device = None

    def _build(self, device):
        model = AnatomixUNet(input_nc=1, **self.architecture)
        load_state_dict_into(model, self.weights_path)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        self._model = model
        self._device = device

    def __call__(self, volume):
        """Extract anatomix features from a volume.

        Parameters
        ----------
        volume : (1, 1, *spatial) or (*spatial,) tensor
            Single-channel 3D image volume.

        Returns
        -------
        (1, output_nc, *spatial) tensor
            Extracted feature map, on the same device as `volume`.

        """
        if volume.dim() < 4:
            volume = volume.reshape(1, 1, *volume.shape)
        device = volume.device
        if self._model is None or self._device != device:
            self._build(device)
        return self._model(volume)


def extract_features(volume, weights_path=None, auto_download=False, **architecture):
    """Extract anatomix modality-agnostic features from a 3D volume.

    Standalone entry point (spec.md User Story 2 / FR-007): usable
    independently of the registration workflow. Internally reused by
    `pairwise_makeobj.make_image`'s `anatomix=` wiring.

    Parameters
    ----------
    volume : (1, 1, *spatial) or (*spatial,) tensor
        Single-channel 3D image volume.
    weights_path : str, optional
        Local path to a pretrained anatomix `.pth` checkpoint. Required
        unless `auto_download=True`.
    auto_download : bool, default=False
        Opt in to downloading the checkpoint automatically.
    num_downs, ngf, output_nc, norm, interp, pooling
        Architecture overrides, see `AnatomixFeatureExtractor`.

    Returns
    -------
    (1, output_nc, *spatial) tensor

    Raises
    ------
    AnatomixWeightsError
        If no usable weight source resolves, or the checkpoint fails to
        load (FR-006, SC-005).

    """
    extractor = AnatomixFeatureExtractor(
        weights_path=weights_path, auto_download=auto_download, **architecture)
    return extractor(volume)
