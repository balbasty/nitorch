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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = {}
        self.metrics = {}
        self.tags = []

    def add_loss(self, tag, *loss_fn, **named_loss_fn):
        """Add one or more loss functions.

        Parameters
        ----------
        tag : str
            Tag/category of the loss
        loss_fn, named_loss_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.

        """
        if tag not in self.tags:
            raise ValueError('Loss tag "{}" not registered. '
                             'Registered tags are {}.'
                             .format(tag, self.tags))
        if tag not in self.losses.keys():
            self.losses[tag] = {}
        if '' not in self.losses[tag].keys():
            self.losses[tag][''] = []
        self.losses[tag][''] += list(loss_fn)
        self.losses[tag].update(dict(named_loss_fn))

    def set_image_loss(self, tag, *loss_fn, **named_loss_fn):
        """Set one or more image loss functions.

        Parameters
        ----------
        tag : str
            Tag/category of the loss
        loss_fn, named_loss_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.

        """
        if tag not in self.tags:
            raise ValueError('Loss tag "{}" not registered. '
                             'Registered tags are {}.'
                             .format(tag, self.tags))            
        self.losses[tag] = dict(named_loss_fn)
        self.losses[tag][''] = list(loss_fn)

    def compute_loss(self, *, prepend=False, **tag_args):
        """Compute all losses.

        Signature
        ---------
        self.compute_loss(tag1=[*args1], tag2=[*args2], prepend=False)

        Parameters
        ----------
        **tag_args: dict[list]
            Dictionary of (tag, arguments) pairs.
        prepend : bool, default=False
            Prepend the name of the class to all loss tags

        Returns
        -------
        loss : dict
            Dictionary of loss values
        metric : dict
            Dictionary of unweighted loss values
        """
        loss = {}
        metric = {}

        def add_loss(type, key, fn, *args):
            if isinstance(fn, (list, tuple)):
                fn, weight = fn
            else:
                weight = 1
            key = '{}/{}'.format(type, key)
            if prepend:
                key = '{}/{}'.format(self.__class__.__name__, key)
            val = fn(*args)
            metric[key] = val
            loss[key] = val if weight == 1 else weight * val

        for tag, losses in self.losses.items():
            args = tag_args[tag]
            for key, loss_fn in losses.items():
                if not key:
                    for key, loss_fn in enumerate(loss_fn):
                        add_loss(tag, key, loss_fn, *args)
                else:
                    add_loss(tag, key, loss_fn, *args)

        return loss, metric

    def add_metric(self, tag, **metric_fn):
        """Add one or more metric functions.

        Parameters
        ----------
        tag : str
            Tag/category of the metric
        metric_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.

        """
        if tag not in self.tags:
            raise ValueError('Metric tag "{}" not registered. '
                             'Registered tags are {}.'
                             .format(tag, self.tags))
        if tag not in self.metrics.keys():
            self.metrics[tag] = {}
        if '' not in self.metrics[tag].keys():
            self.metrics[tag][''] = []
        self.metrics[tag].update(dict(metric_fn))

    def set_metric(self, tag, **metric_fn):
        """Set one or more metric functions.

        Parameters
        ----------
        tag : str
            Tag/category of the metric
        metric_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.

        """
        if tag not in self.tags:
            raise ValueError('Metric tag "{}" not registered. '
                             'Registered tags are {}.'
                             .format(tag, self.tags))            
        self.metrics[tag] = dict(metric_fn)

    def compute_metric(self, *, prepend=False, **tag_args):
        """Compute all metrics.

        Signature
        ---------
        self.compute_metric(tag1=[*args1], tag2=[*args2], prepend=False)

        Parameters
        ----------
        **tag_args: dict[list]
            Dictionary of (tag, arguments) pairs.
        prepend : bool, default=False
            Prepend the name of the class to all loss tags

        Returns
        -------
        metric : dict
            Dictionary of metric values

        """
        metric = {}

        def add_metric(type, key, fn, *args):
            if isinstance(fn, (list, tuple)):
                _fn, weight = fn
                fn = lambda *a, **k: weight*_fn(*a, **k)
            key = '{}/{}'.format(type, key)
            if prepend:
                key = '{}/{}'.format(self.__class__.__name__, key)
            metric[key] = fn(*args)

        for tag, metrics in self.metrics.items():
            args = tag_args[tag]
            for key, metric_fn in metrics.items():
                if not key:
                    for key, metric_fn in enumerate(metric_fn):
                        add_metric(tag, key, metric_fn, *args)                
                else:
                    add_metric(tag, key, metric_fn, *args)

        return metric

    def compute(self, _loss=None, _metric=None, **tag_args):
        """Compute losses and metrics if necessary

        Parameters
        ----------
        _loss : dict
            Mutable dictionary of losses.
        _metric : dict
            Mutable dictionary of metrics.
        tag_args : dict[list]
            (tag, args) pairs
            Each tag (stored in self.tags) is associated with a
            list of arguments to be passed to the corresponding
            loss/metric function.

        """
        if _loss is not None:
            assert isinstance(_loss, dict)
            losses, metrics = self.compute_loss(**tag_args)
            self.update_dict(_loss, losses)
            if _metric is not None:
                self.update_dict(_metric, metrics)
        if _metric is not None:
            assert isinstance(_metric, dict)
            metrics = self.compute_metric(**tag_args)
            self.update_dict(_metric, metrics)


    def board(self, tb, inputs, outputs):
        """Defines model-specific tensorboard callback.

        Parameters
        ----------
        tb : torch.utils.tensorboard.writer.SummaryWriter
            TensorBoard writer object.
        inputs : tuple
            Model inputs.
        outputs : tuple
            Model outputs.

        """
        pass