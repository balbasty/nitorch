import pytest
import torch

from nitorch._models.anatomix import AnatomixViTFeatureExtractor, AnatomixWeightsError
from nitorch._models.anatomix.vit import AnatomixViT

# A tiny, structurally-valid PrimusV2 config (not the real 27M-param
# anatomix-dev-vit) for fast tests: same overall architecture (CNN
# tokenizer -> EVA transformer -> demean), but ~1000 params instead of 27M,
# and a small fixed working resolution (16 instead of 128).
TINY_ARCH = dict(
    input_channels=1, num_classes=4, embed_dim=24, eva_depth=1,
    eva_numheads=2, patch_embed_size=(8, 8, 8), input_shape=(16, 16, 16),
    num_register_tokens=2, init_values=0.1, scale_attn_inner=True,
)


@pytest.fixture
def fake_checkpoint(tmp_path):
    """A synthetic anatomix-dev-vit checkpoint: a tiny model's own raw
    (upstream-format, unprefixed) state_dict."""
    model = AnatomixViT(**TINY_ARCH)
    path = tmp_path / 'fake_anatomix_vit.pth'
    torch.save(model._primus.state_dict(), path)
    return str(path)


def test_extractor_shape_and_device(fake_checkpoint):
    volume = torch.rand(1, 1, 16, 16, 16)
    extractor = AnatomixViTFeatureExtractor(weights_path=fake_checkpoint, **TINY_ARCH)
    features = extractor(volume)
    assert features.shape == (1, TINY_ARCH['num_classes'], 16, 16, 16)
    assert features.device == volume.device


def test_extractor_standalone_bare_spatial_input(fake_checkpoint):
    # (*spatial,) input, no batch/channel dims
    volume = torch.rand(16, 16, 16)
    extractor = AnatomixViTFeatureExtractor(weights_path=fake_checkpoint, **TINY_ARCH)
    features = extractor(volume)
    assert features.shape == (1, TINY_ARCH['num_classes'], 16, 16, 16)


def test_extractor_rejects_multichannel_input(fake_checkpoint):
    volume = torch.rand(1, 2, 16, 16, 16)
    extractor = AnatomixViTFeatureExtractor(weights_path=fake_checkpoint, **TINY_ARCH)
    with pytest.raises(ValueError, match='single-channel'):
        extractor(volume)


def test_extractor_missing_weights_raises_descriptive_error():
    with pytest.raises(AnatomixWeightsError):
        AnatomixViTFeatureExtractor()


def test_extractor_bad_path_raises_descriptive_error(tmp_path):
    bad_path = str(tmp_path / 'does_not_exist.pth')
    with pytest.raises(AnatomixWeightsError, match='does not exist'):
        AnatomixViTFeatureExtractor(weights_path=bad_path)


def test_extractor_corrupt_checkpoint_raises_descriptive_error(tmp_path):
    # spec.md Edge Cases: "checkpoint file that does not match the
    # expected architecture (corrupt or wrong file)"
    bad_path = tmp_path / 'corrupt.pth'
    bad_path.write_bytes(b'not a real checkpoint')
    extractor = AnatomixViTFeatureExtractor(weights_path=str(bad_path), **TINY_ARCH)
    with pytest.raises(AnatomixWeightsError):
        extractor(torch.rand(1, 1, 16, 16, 16))


def test_extractor_mismatched_architecture_raises_descriptive_error(fake_checkpoint):
    # a checkpoint for one config loaded into a differently-shaped model
    mismatched = dict(TINY_ARCH, embed_dim=48)
    extractor = AnatomixViTFeatureExtractor(weights_path=fake_checkpoint, **mismatched)
    with pytest.raises(AnatomixWeightsError):
        extractor(torch.rand(1, 1, 16, 16, 16))


def test_extractor_pads_small_input(fake_checkpoint):
    # every axis < the fixed working resolution (16) -> pad path
    extractor = AnatomixViTFeatureExtractor(weights_path=fake_checkpoint, **TINY_ARCH)
    volume = torch.rand(1, 1, 6, 9, 4)
    features = extractor(volume)
    assert features.shape == (1, TINY_ARCH['num_classes'], 6, 9, 4)


def test_extractor_tiles_large_input(fake_checkpoint):
    # some axis > the fixed working resolution (16) -> sliding-window path
    extractor = AnatomixViTFeatureExtractor(weights_path=fake_checkpoint, **TINY_ARCH)
    volume = torch.rand(1, 1, 30, 20, 10)
    features = extractor(volume)
    assert features.shape == (1, TINY_ARCH['num_classes'], 30, 20, 10)


def test_extractor_tile_borders_have_no_sharp_discontinuity(fake_checkpoint):
    # blended overlap should vary smoothly across a tile border, not jump
    extractor = AnatomixViTFeatureExtractor(weights_path=fake_checkpoint, **TINY_ARCH)
    volume = torch.rand(1, 1, 30, 20, 10)
    features = extractor(volume)
    # gradient magnitude along the tiled axis should not have an outlier
    # spike at the tile boundary relative to its neighborhood
    diffs = (features[:, :, 1:] - features[:, :, :-1]).abs()
    assert diffs.max() < 10 * diffs.mean() + 1e-6


def test_extractor_auto_download_resolves_via_resolve_weights_path(monkeypatch, tmp_path):
    # confirm anatomix_vit's weight resolution reuses the existing
    # resolve_weights_path(variant='anatomix-dev-vit') machinery: mock the
    # network call the same way the existing U-Net test does, rather than
    # hitting the real network.
    import urllib.error
    import urllib.request

    def _fail(*a, **k):
        raise urllib.error.URLError('simulated network failure')

    monkeypatch.setattr(urllib.request, 'urlretrieve', _fail)
    with pytest.raises(AnatomixWeightsError, match='download'):
        AnatomixViTFeatureExtractor(auto_download=True, cache_dir=str(tmp_path))
