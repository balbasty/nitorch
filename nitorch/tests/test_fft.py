import pytest
import torch
from nitorch.core.optionals import try_import
from nitorch.core import fft as nifft
pyfft = try_import('torch.fft', _as=True)


_torch_has_old_fft = nifft._torch_has_old_fft
_torch_has_complex = nifft._torch_has_complex
_torch_has_fft_module = nifft._torch_has_fft_module
_torch_has_fftshift = nifft._torch_has_fftshift

norms = ('forward', 'backward', 'ortho')
ndims = (1, 2, 3, 4)


@pytest.mark.parametrize('norm', norms)
def test_fft(norm):
    if not _torch_has_fft_module or not _torch_has_old_fft:
        return True
    
    nifft._torch_has_complex = False
    nifft._torch_has_fft_module = False
    nifft._torch_has_fftshift = False

    x = torch.randn([16, 32, 2], dtype=torch.doubl)
    f1 = pyfft.fft(torch.complex(x[..., 0], x[..., 1]), norm=norm)
    f2 = nifft.fft(x, norm=norm)
    f2 = torch.complex(f2[..., 0], f2[..., 1])
    assert torch.allclose(f1, f2)

    x = torch.randn([16, 32, 2], dtype=torch.doubl)
    f1 = pyfft.fft(torch.complex(x[..., 0], x[..., 1]), dim=0, norm=norm)
    f2 = nifft.fft(x, dim=0, norm=norm)
    f2 = torch.complex(f2[..., 0], f2[..., 1])
    assert torch.allclose(f1, f2)

    x = torch.randn([16, 32], dtype=torch.doubl)
    f1 = pyfft.fft(x, norm=norm)
    f2 = nifft.fft(x, real=True, norm=norm)
    f2 = torch.complex(f2[..., 0], f2[..., 1])
    assert torch.allclose(f1, f2)


@pytest.mark.parametrize('norm', norms)
def test_ifft(norm):
    if not _torch_has_fft_module or not _torch_has_old_fft:
        return True
    
    nifft._torch_has_complex = False
    nifft._torch_has_fft_module = False
    nifft._torch_has_fftshift = False

    x = torch.randn([16, 32, 2], dtype=torch.doubl)
    f1 = pyfft.ifft(torch.complex(x[..., 0], x[..., 1]), norm=norm)
    f2 = nifft.ifft(x, norm=norm)
    f2 = torch.complex(f2[..., 0], f2[..., 1])
    assert torch.allclose(f1, f2)

    x = torch.randn([16, 32, 2], dtype=torch.doubl)
    f1 = pyfft.ifft(torch.complex(x[..., 0], x[..., 1]), dim=0, norm=norm)
    f2 = nifft.ifft(x, dim=0, norm=norm)
    f2 = torch.complex(f2[..., 0], f2[..., 1])
    assert torch.allclose(f1, f2)

    x = torch.randn([16, 32], dtype=torch.doubl)
    f1 = pyfft.ifft(x, norm=norm)
    f2 = nifft.ifft(x, real=True, norm=norm)
    f2 = torch.complex(f2[..., 0], f2[..., 1])
    assert torch.allclose(f1, f2)


@pytest.mark.parametrize('norm', norms)
@pytest.mark.parametrize('ndim', ndims)
@pytest.mark.parametrize('shuffle', (True, False))
def test_fftn(norm, ndim, shuffle):
    if not _torch_has_fft_module or not _torch_has_old_fft:
        return True
    
    nifft._torch_has_complex = False
    nifft._torch_has_fft_module = False
    nifft._torch_has_fftshift = False

    dims = [0, 1, -2, 3]
    if shuffle:
        import random
        random.shuffle(dims)
    dims = dims[:ndim]

    x = torch.randn([4, 9, 16, 33, 2], dtype=torch.double)
    f1 = pyfft.fftn(torch.complex(x[..., 0], x[..., 1]), norm=norm, dim=dims)
    f2 = nifft.fftn(x, norm=norm, dim=dims)
    f2 = torch.complex(f2[..., 0], f2[..., 1])
    assert torch.allclose(f1, f2)

    x = torch.randn([4, 9, 16, 33], dtype=torch.double)
    f1 = pyfft.fftn(x, norm=norm, dim=dims)
    f2 = nifft.fftn(x, real=True, norm=norm, dim=dims)
    f2 = torch.complex(f2[..., 0], f2[..., 1])
    assert torch.allclose(f1, f2, atol=1e-5)


@pytest.mark.parametrize('norm', norms)
@pytest.mark.parametrize('ndim', ndims)
@pytest.mark.parametrize('shuffle', (True, False))
def test_ifftn(norm, ndim, shuffle):
    if not _torch_has_fft_module or not _torch_has_old_fft:
        return True
    
    nifft._torch_has_complex = False
    nifft._torch_has_fft_module = False
    nifft._torch_has_fftshift = False

    dims = [0, 1, -2, 3]
    if shuffle:
        import random
        random.shuffle(dims)
    dims = dims[:ndim]

    x = torch.randn([4, 9, 16, 33, 2], dtype=torch.double)
    f1 = pyfft.ifftn(torch.complex(x[..., 0], x[..., 1]), norm=norm, dim=dims)
    f2 = nifft.ifftn(x, norm=norm, dim=dims)
    f2 = torch.complex(f2[..., 0], f2[..., 1])
    assert torch.allclose(f1, f2)

    x = torch.randn([4, 9, 16, 33], dtype=torch.double)
    f1 = pyfft.ifftn(x, norm=norm, dim=dims)
    f2 = nifft.ifftn(x, real=True, norm=norm, dim=dims)
    f2 = torch.complex(f2[..., 0], f2[..., 1])
    assert torch.allclose(f1, f2, atol=1e-5)

