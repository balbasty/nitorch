"""Weight resolution and loading for the vendored anatomix U-Net.

By default, callers must supply a local path to a pretrained ``.pth``
checkpoint (safe for offline/network-restricted environments such as HPC
clusters). Automatic download from the official anatomix HuggingFace Hub
distribution is available only as an explicit opt-in (``auto_download=True``).
See ``specs/001-anatomix-registration-features/research.md`` (decision 3).
"""

import os
import urllib.error
import urllib.request

import torch

#: Official anatomix HuggingFace Hub repository.
HF_REPO_ID = 'neeldey/anatomix'
#: Default model variant (matches the published, 6M-parameter checkpoint).
DEFAULT_VARIANT = 'anatomix'


class AnatomixWeightsError(RuntimeError):
    """Raised when anatomix pretrained weights cannot be resolved or loaded.

    Always carries a human-readable message naming the missing prerequisite
    and a remedy (spec.md FR-006, SC-005).
    """


def _default_cache_dir():
    """Return the local directory used to cache auto-downloaded weights."""
    try:
        import appdirs
        return appdirs.user_cache_dir('nitorch', 'nitorch')
    except ImportError:
        return os.path.join(os.path.expanduser('~'), '.cache', 'nitorch')


def _download(variant, repo_id, cache_dir):
    """Download a checkpoint from the anatomix HuggingFace Hub distribution.

    Uses a plain HTTPS GET (no `huggingface_hub` dependency), matching
    HuggingFace Hub's public file-resolution URL scheme.

    Parameters
    ----------
    variant : str
    repo_id : str
    cache_dir : str

    Returns
    -------
    str
        Local path to the downloaded (or already-cached) checkpoint.

    """
    os.makedirs(cache_dir, exist_ok=True)
    filename = f'{variant}.pth'
    local_path = os.path.join(cache_dir, filename)
    if os.path.isfile(local_path):
        return local_path

    url = f'https://huggingface.co/{repo_id}/resolve/main/{filename}'
    tmp_path = local_path + '.part'
    try:
        urllib.request.urlretrieve(url, tmp_path)
    except (urllib.error.URLError, OSError) as e:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise AnatomixWeightsError(
            f"Failed to download anatomix weights from {url!r}: {e}. "
            f"Check your network connection, or supply a local checkpoint "
            f"path via `weights_path` instead of `auto_download=True`."
        ) from e
    os.replace(tmp_path, local_path)
    return local_path


def resolve_weights_path(weights_path=None, auto_download=False,
                          variant=DEFAULT_VARIANT, repo_id=HF_REPO_ID,
                          cache_dir=None):
    """Resolve a local filesystem path to a usable anatomix checkpoint.

    Parameters
    ----------
    weights_path : str, optional
        Local path to a `.pth` checkpoint. Takes precedence over
        `auto_download` if both are given.
    auto_download : bool, default=False
        If `weights_path` is not given, opt into downloading the checkpoint
        from the official anatomix HuggingFace Hub distribution.
    variant : str, default='anatomix'
        Which anatomix variant to download, if `auto_download=True`.
    repo_id : str, default='neeldey/anatomix'
        HuggingFace Hub repository to download from.
    cache_dir : str, optional
        Directory to cache downloaded weights in. Defaults to a
        nitorch-specific user cache directory.

    Returns
    -------
    str
        A local filesystem path to a checkpoint file.

    Raises
    ------
    AnatomixWeightsError
        If neither a valid `weights_path` nor a successful download is
        available.

    """
    if weights_path is not None:
        if not os.path.isfile(weights_path):
            raise AnatomixWeightsError(
                f"anatomix weights path {weights_path!r} does not exist or "
                f"is not a file. Supply a valid local `.pth` checkpoint "
                f"path, or set `auto_download=True` to fetch one "
                f"automatically from the official anatomix distribution."
            )
        return weights_path

    if auto_download:
        return _download(variant, repo_id, cache_dir or _default_cache_dir())

    raise AnatomixWeightsError(
        "No anatomix weights available: no `weights_path` was supplied and "
        "`auto_download` is False. Either pass a local path to a "
        "pretrained anatomix `.pth` checkpoint via `weights_path`, or set "
        "`auto_download=True` to fetch one automatically from the official "
        "anatomix distribution (requires network access)."
    )


def load_state_dict_into(model, weights_path):
    """Load a checkpoint's weights into `model` in place.

    Transparently strips a `_orig_mod.` prefix from state-dict keys, which
    `torch.compile()`-saved checkpoints (including some anatomix releases)
    may carry.

    Parameters
    ----------
    model : torch.nn.Module
    weights_path : str

    Raises
    ------
    AnatomixWeightsError
        If the checkpoint cannot be read, or its keys/shapes are
        incompatible with `model` (e.g. architecture parameters that don't
        match the checkpoint).

    """
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
    except Exception as e:
        raise AnatomixWeightsError(
            f"Failed to read anatomix checkpoint at {weights_path!r}: {e}. "
            f"The file may be corrupt or not a valid PyTorch checkpoint."
        ) from e

    state_dict = checkpoint.get('state_dict', checkpoint) \
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint \
        else checkpoint
    if not isinstance(state_dict, dict):
        raise AnatomixWeightsError(
            f"Checkpoint at {weights_path!r} does not contain a recognizable "
            f"state_dict (got a {type(state_dict).__name__})."
        )
    state_dict = {
        (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
        for k, v in state_dict.items()
    }

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise AnatomixWeightsError(
            f"Checkpoint at {weights_path!r} is incompatible with the "
            f"configured anatomix architecture (num_downs/ngf/output_nc/"
            f"norm/interp/pooling). Verify these match the checkpoint you "
            f"are loading. Original error: {e}"
        ) from e
