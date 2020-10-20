"""
A base class for all nitorch modules.

The NiTorch module class extends PyTorch's by adding a `loss` keyword
to the call function. If `forward` implements this keyword, it should
compute losses on the fly along with the other (classical) outputs.
"""

import torch
import torch.nn as tnn
import inspect


def nitorchmodule(klass):
    """Decorator for modules to make them understand 'forward with loss'"""

    # copy original __call__ method
    call = klass.__call__

    # define new call method
    def __call__(self, *args, **kwargs):
        if '_loss' in kwargs.keys():
            if not '_loss' in inspect.signature(self.forward).parameters.keys():
                kwargs.pop('_loss')
        if '_metric' in kwargs.keys():
            if not '_metric' in inspect.signature(self.forward).parameters.keys():
                kwargs.pop('_metric')
        return call(self, *args, **kwargs)

    # define helper to store metrics
    @staticmethod
    def update_dict(old_dict, new_dict):
        for key, val in new_dict.items():
            if key in old_dict.keys():
                i = 1
                while '{}/{}'.format(key, i) in old_dict.keys():
                    i += 1
                key = '{}/{}'.format(key, i)
            old_dict[key] = val

    # assign new methods
    klass.__call__ = __call__
    klass.update_dict = update_dict
    return klass


@nitorchmodule
class Module(tnn.Module):
    pass
