"""Helpers shared by command-line tools."""
import warnings
import torch


def setup_device(device='cpu', ndevice=None, *, strict=True,
                 torch_string=False):
    """Return the torch device requested on the command line.

    Parameters
    ----------
    device : {'cpu', 'gpu'} or str, default='cpu'
        Device type, as parsed by the `--cpu`/`--gpu` options.
        If `torch_string`, any string understood by `torch.device`
        (e.g. 'cpu', 'cuda', 'cuda:1').
    ndevice : int, optional
        If 'cpu': number of threads (`torch.set_num_threads`; not set if
        falsy). If 'gpu': CUDA device index (default device if None).
        Ignored if `torch_string`.
    strict : bool, default=True
        If True, `device` must be 'cpu' or 'gpu' (AssertionError otherwise).
        If False, any value other than 'cpu' is treated as 'gpu'.
    torch_string : bool, default=False
        Interpret `device` as a torch device string.

    Returns
    -------
    device : torch.device
        A CUDA device falls back to the CPU, with a warning, if CUDA
        is not available.
    """
    if torch_string:
        if device.startswith('cuda') and not torch.cuda.is_available():
            warnings.warn('CUDA not available. Switching to CPU.')
            device = 'cpu'
        return torch.device(device)

    if device == 'gpu' and not torch.cuda.is_available():
        warnings.warn('CUDA not available. Switching to CPU.')
        device, ndevice = 'cpu', None
    if device == 'cpu':
        if ndevice:
            torch.set_num_threads(ndevice)
        return torch.device('cpu')
    if strict:
        assert device == 'gpu'
    if ndevice is not None:
        return torch.device(f'cuda:{ndevice}')
    return torch.device('cuda')
