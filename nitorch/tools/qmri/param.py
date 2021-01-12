from nitorch.tools.qmri import io as qio
import torch
import copy


class ParameterMap(qio.Volume3D):
    """
    Wrapper object for Parameter maps
    """

    min = None   # minimum value
    max = None   # maximum value

    def __new__(cls, input=None, fill=None, dtype=None, device=None, **kwargs):
        """

        Parameters
        ----------
        input : sequence[int] or tensor_like or flie_like
            If a sequence[int], allocate a tensor of that shape
            Else, wrap the underlying `Volume3D` object.
        fill : number, optional
            A value to fill the tensor with
        dtype : torch.dtype, optional
        device : torch.device, optional
        kwargs : dict
            Attributes for `Volume3D`.
        """
        if isinstance(input, (list, tuple)):
            if fill is not None:
                volume = torch.full(input, fill, dtype=dtype, device=device)
            else:
                volume = torch.zeros(input, dtype=dtype, device=device)
            return cls.__new__(cls, volume, **kwargs)
        return super().__new__(cls, input, **kwargs)

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)
