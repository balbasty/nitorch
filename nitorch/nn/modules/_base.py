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
    def update_metrics(metrics, new_metrics):
        for key, val in new_metrics.items():
            if key in metrics.keys():
                i = 1
                while '{}/{}'.format(key, i) in metrics.keys():
                    i += 1
                key = '{}/{}'.format(key, i)
            metrics[key] = val

    # assign new methoda
    klass.__call__ = __call__
    klass.update_metrics = update_metrics
    return klass


Module = nitorchmodule(tnn.Module)
