"""Shared `setup_device` helper (nitorch.cli.utils) and its callers."""
import warnings

import pytest
import torch

from nitorch.cli import utils as cli_utils
from nitorch.cli.utils import setup_device
from nitorch.cli.registration.register import cli as register_cli
from nitorch.cli.segmentation.uniseg import cli as uniseg_cli
from nitorch.cli.registration.meanspace import cli as meanspace_cli
from nitorch.cli.denoising.denoise_mri import cli as denoise_mri_cli


@pytest.fixture
def threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


@pytest.fixture
def cuda(monkeypatch):
    def set(available):
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
    return set


def test_cpu_default(threads):
    n = torch.get_num_threads()
    assert setup_device() == torch.device('cpu')
    assert setup_device('cpu', None) == torch.device('cpu')
    assert setup_device('cpu', 0) == torch.device('cpu')
    assert torch.get_num_threads() == n


def test_cpu_threads(threads):
    assert setup_device('cpu', 1) == torch.device('cpu')
    assert torch.get_num_threads() == 1


def test_gpu(cuda):
    cuda(True)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert setup_device('gpu', None) == torch.device('cuda')
        assert setup_device('gpu', 2) == torch.device('cuda:2')


def test_gpu_fallback(cuda, threads):
    cuda(False)
    n = torch.get_num_threads()
    with pytest.warns(UserWarning, match='CUDA not available'):
        # the index is dropped: it is not used as a thread count
        assert setup_device('gpu', 1) == torch.device('cpu')
    assert torch.get_num_threads() == n


def test_strict(cuda):
    cuda(True)
    with pytest.raises(AssertionError):
        setup_device('tpu', None)
    assert setup_device('tpu', None, strict=False) == torch.device('cuda')


def test_torch_string(cuda):
    cuda(True)
    assert setup_device('cpu', torch_string=True) == torch.device('cpu')
    assert setup_device('cuda:1', torch_string=True) == torch.device('cuda:1')
    cuda(False)
    with pytest.warns(UserWarning, match='CUDA not available'):
        assert setup_device('cuda:1', torch_string=True) == torch.device('cpu')


@pytest.mark.parametrize('module,args,kwargs', [
    (register_cli, ('gpu', 1), dict()),
    (uniseg_cli, ('gpu', 1), dict()),
    (meanspace_cli, ('gpu', 1), dict(strict=False)),
    (denoise_mri_cli, ('cuda:1',), dict(torch_string=True)),
])
def test_commands_delegate(monkeypatch, module, args, kwargs):
    seen = []

    def fake(*a, **k):
        seen.append((a, k))
        return torch.device('meta')

    monkeypatch.setattr(module, '_setup_device', fake)
    assert module.setup_device(*args) == torch.device('meta')
    assert seen == [(args, kwargs)]


@pytest.mark.parametrize('module', [
    register_cli, uniseg_cli, meanspace_cli, denoise_mri_cli])
def test_commands_use_shared(module):
    assert module._setup_device is cli_utils.setup_device


@pytest.mark.parametrize('module', [register_cli, uniseg_cli, meanspace_cli])
def test_commands_fallback(cuda, module):
    cuda(False)
    with pytest.warns(UserWarning, match='CUDA not available'):
        assert module.setup_device('gpu', 0) == torch.device('cpu')


def test_denoise_mri_parse_device(cuda):
    from nitorch.cli.denoising.denoise_mri.parser import parse
    cuda(False)
    opt = parse(['a.nii', '--gpu', '1'])
    assert opt.device == 'cuda:1'
    with pytest.warns(UserWarning, match='CUDA not available'):
        assert denoise_mri_cli.setup_device(opt.device) == torch.device('cpu')
