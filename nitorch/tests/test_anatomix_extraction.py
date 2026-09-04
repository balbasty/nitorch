import pytest
import torch

from nitorch._models.anatomix import extract_features, AnatomixWeightsError
from nitorch._models.anatomix.unet import AnatomixUNet

ARCH = dict(num_downs=2, ngf=4, output_nc=8)


@pytest.fixture
def fake_checkpoint(tmp_path):
    """A synthetic anatomix checkpoint: a small model's own state_dict."""
    model = AnatomixUNet(input_nc=1, **ARCH)
    path = tmp_path / 'fake_anatomix.pth'
    torch.save(model.state_dict(), path)
    return str(path)


def test_extract_features_shape_and_device(fake_checkpoint):
    volume = torch.rand(1, 1, 16, 16, 16)
    features = extract_features(volume, weights_path=fake_checkpoint, **ARCH)
    assert features.shape == (1, ARCH['output_nc'], 16, 16, 16)
    assert features.device == volume.device


def test_extract_features_standalone_no_registration_objects(fake_checkpoint):
    # (*spatial,) input, no batch/channel dims, no registration workflow involved
    volume = torch.rand(16, 16, 16)
    features = extract_features(volume, weights_path=fake_checkpoint, **ARCH)
    assert features.shape == (1, ARCH['output_nc'], 16, 16, 16)


def test_extract_features_missing_weights_raises_descriptive_error():
    volume = torch.rand(1, 1, 16, 16, 16)
    with pytest.raises(AnatomixWeightsError, match='weights_path'):
        extract_features(volume)


def test_extract_features_bad_path_raises_descriptive_error(tmp_path):
    volume = torch.rand(1, 1, 16, 16, 16)
    bad_path = str(tmp_path / 'does_not_exist.pth')
    with pytest.raises(AnatomixWeightsError, match='does not exist'):
        extract_features(volume, weights_path=bad_path)
