"""
A base class for all nitorch modules.

The NiTorch module class extends PyTorch's by adding a `loss` keyword
to the call function. If `forward` implements this keyword, it should
compute losses on the fly along with the other (classical) outputs.
"""

import torch
import torch.nn as tnn
import inspect


class Module(tnn.Module):
    """A class that defines methods to manipulate losses, metrics, etc.

    Classes can either inherit it, or use the @nitorchmodule decorator.
    """
    _nitorchmodule = True

    def __init__(self):
        super().__init__()
        self.losses = {'train': {}, 'eval': {}}
        self.metrics = {'train': {}, 'eval': {}}
        self.tags = []
        self.augmenters = {'train': [], 'eval': []}

    def __call__(self, *args, **kwargs):
        mode = 'train' if self.training else 'eval'
        for augmenter in self.augmenters[mode]:
            args = augmenter(*args)
        if '_loss' in kwargs.keys():
            if not '_loss' in inspect.signature(self.forward).parameters.keys():
                kwargs.pop('_loss')
        if '_metric' in kwargs.keys():
            if not '_metric' in inspect.signature(self.forward).parameters.keys():
                kwargs.pop('_metric')
        return super().__call__(*args, **kwargs)

    def add_augmenter(self, augmenter, only='train'):
        """Add one or more augmenters.

        Parameters
        ----------
        augmenter : callable
            Function that applies some augmentation.
        only : {'train', 'eval', None}, default='train'

        """
        if not only:
            self.augmenters['train'].append(augmenter)
            self.augmenters['eval'].append(augmenter)
        elif only.lower() == 'train':
            self.augmenters['train'].append(augmenter)
        elif only.lower() == 'eval':
            self.augmenters['eval'].append(augmenter)
        else:
            raise ValueError(f'Unknown `only`: {only}')

    def add_loss(self, tag, *loss_fn, only=None, **named_loss_fn):
        """Add one or more loss functions.

        Parameters
        ----------
        tag : str
            Tag/category of the loss
        loss_fn, named_loss_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.
        only : {'train', 'eval', None}, optional
            Whether this loss should be computed only in train or eval
            mode. If None (default), it is computed in both cases.

        """
        if tag not in self.tags:
            raise ValueError('Loss tag "{}" not registered. '
                             'Registered tags are {}.'
                             .format(tag, self.tags))
        if tag not in self.losses['train'].keys():
            self.losses['train'][tag] = {}
        if tag not in self.losses['eval'].keys():
            self.losses['eval'][tag] = {}
        if '' not in self.losses['train'][tag].keys():
            self.losses['train'][tag][''] = []
        if '' not in self.losses['eval'][tag].keys():
            self.losses['eval'][tag][''] = []
        if not only:
            self.losses['train'][tag][''] += list(loss_fn)
            self.losses['train'][tag].update(dict(named_loss_fn))
            self.losses['eval'][tag][''] += list(loss_fn)
            self.losses['eval'][tag].update(dict(named_loss_fn))
        elif only.lower() == 'train':
            self.losses['train'][tag][''] += list(loss_fn)
            self.losses['train'][tag].update(dict(named_loss_fn))
        elif only.lower() == 'eval':
            self.losses['eval'][tag][''] += list(loss_fn)
            self.losses['eval'][tag].update(dict(named_loss_fn))
        else:
            raise ValueError(f'Unknown `only`: {only}')

    def set_loss(self, tag, *loss_fn, only=None, **named_loss_fn):
        """Set one or more image loss functions.

        Parameters
        ----------
        tag : str
            Tag/category of the loss
        loss_fn, named_loss_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.
        only : {'train', 'eval', None}, optional
            Whether this loss should be computed only in train or eval
            mode. If None (default), it is computed in both cases.

        """
        if tag not in self.tags:
            raise ValueError('Loss tag "{}" not registered. '
                             'Registered tags are {}.'
                             .format(tag, self.tags))
        if not only:
            self.losses['train'][tag] = dict(named_loss_fn)
            self.losses['train'][tag][''] = list(loss_fn)
            self.losses['eval'][tag] = dict(named_loss_fn)
            self.losses['eval'][tag][''] = list(loss_fn)
        elif only.lower() == 'train':
            self.losses['train'][tag] = dict(named_loss_fn)
            self.losses['train'][tag][''] = list(loss_fn)
        elif only.lower() == 'eval':
            self.losses['eval'][tag] = dict(named_loss_fn)
            self.losses['eval'][tag][''] = list(loss_fn)
        else:
            raise ValueError(f'Unknown `only`: {only}')

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
        mode = 'train' if self.training else 'eval'
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

        for tag, losses in self.losses[mode].items():
            if tag not in tag_args:
                continue
            args = tag_args[tag]
            for key, loss_fn in losses.items():
                if not key:
                    for key, loss_fn in enumerate(loss_fn):
                        add_loss(tag, key, loss_fn, *args)
                else:
                    add_loss(tag, key, loss_fn, *args)

        return loss, metric

    def add_metric(self, tag, only=None, **metric_fn):
        """Add one or more metric functions.

        Parameters
        ----------
        tag : str
            Tag/category of the metric
        metric_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.
        only : {'train', 'eval', None}, optional
            Whether this loss should be computed only in train or eval
            mode. If None (default), it is computed in both cases.

        """
        if tag not in self.tags:
            raise ValueError('Metric tag "{}" not registered. '
                             'Registered tags are {}.'
                             .format(tag, self.tags))
        if tag not in self.metrics['train'].keys():
            self.metrics['train'][tag] = {}
        if tag not in self.metrics['eval'].keys():
            self.metrics['eval'][tag] = {}
        if '' not in self.metrics['train'][tag].keys():
            self.metrics['train'][tag][''] = []
        if '' not in self.metrics['eval'][tag].keys():
            self.metrics['eval'][tag][''] = []
        if not only:
            self.metrics['train'][tag].update(dict(metric_fn))
            self.metrics['eval'][tag].update(dict(metric_fn))
        elif only.lower() == 'train':
            self.metrics['train'][tag].update(dict(metric_fn))
        elif only.lower() == 'eval':
            self.metrics['eval'][tag].update(dict(metric_fn))
        else:
            raise ValueError(f'Unknown `only`: {only}')

    def set_metric(self, tag, only=None, **metric_fn):
        """Set one or more metric functions.

        Parameters
        ----------
        tag : str
            Tag/category of the metric
        metric_fn : callable or [callable, float]
            Function of two arguments that returns a scalar value.
        only : {'train', 'eval', None}, optional
            Whether this loss should be computed only in train or eval
            mode. If None (default), it is computed in both cases.

        """
        if tag not in self.tags:
            raise ValueError('Metric tag "{}" not registered. '
                             'Registered tags are {}.'
                             .format(tag, self.tags))
        if not only:
            self.metrics['train'][tag] = dict(metric_fn)
            self.metrics['eval'][tag] = dict(metric_fn)
        elif only.lower() == 'train':
            self.metrics['train'][tag] = dict(metric_fn)
        elif only.lower() == 'eval':
            self.metrics['eval'][tag] = dict(metric_fn)
        else:
            raise ValueError(f'Unknown `only`: {only}')

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
        mode = 'train' if self.training else 'eval'
        metric = {}

        def add_metric(type, key, fn, *args):
            if isinstance(fn, (list, tuple)):
                _fn, weight = fn
                fn = lambda *a, **k: weight*_fn(*a, **k)
            key = '{}/{}'.format(type, key)
            if prepend:
                key = '{}/{}'.format(self.__class__.__name__, key)
            with torch.no_grad():
                metric[key] = fn(*args)

        for tag, metrics in self.metrics[mode].items():
            if tag not in tag_args:
                continue
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

    @staticmethod
    def update_dict(old_dict, new_dict):
        for key, val in new_dict.items():
            if key in old_dict.keys():
                i = 1
                while '{}/{}'.format(key, i) in old_dict.keys():
                    i += 1
                key = '{}/{}'.format(key, i)
            old_dict[key] = val

    def board(self, tb, inputs=None, outputs=None, epoch=None, minibatch=None,
              mode=None, loss=None, losses=None, metrics=None, *args, **kwargs):
        """Defines model-specific tensorboard callback.

        Parameters
        ----------
        tb : torch.utils.tensorboard.writer.SummaryWriter
            TensorBoard writer object.
        inputs : tuple
            Model inputs.
        outputs : tuple
            Model outputs.
        epoch : int
            Epoch index
        minibatch : int
            Minibatch index
        mode : {'train', 'eval'}
            Type of dataset processed
        loss : tensor
            Loss of the current minibatch
        losses : dict
            Loss components of the current minibatch
        metrics : dict
            Metrics the current minibatch

        """
        pass


def nitorchmodule(klass):
    """Decorator for modules to make them understand 'forward with loss'"""

    if hasattr(klass, '_nitorchmodule'):
        return klass
    klass._nitorchmodule = True

    init = klass.__init__
    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.losses = {'train': {}, 'eval': {}}
        self.metrics = {'train': {}, 'eval': {}}
        self.tags = []
        self.augmenters = {'train': [], 'eval': []}
    klass.__init__ = __init__

    call = klass.__call__
    def __call__(self, *args, **kwargs):
        mode = 'train' if self.training else 'eval'
        for augmenter in self.augmenters[mode]:
            args = augmenter(*args)
        if '_loss' in kwargs.keys():
            if not '_loss' in inspect.signature(self.forward).parameters.keys():
                kwargs.pop('_loss')
        if '_metric' in kwargs.keys():
            if not '_metric' in inspect.signature(self.forward).parameters.keys():
                kwargs.pop('_metric')
        return call(self, *args, **kwargs)
    klass.__call__ = __call__

# #     _getattr = klass.__getattr__
# #     def __getattr__(self, name):
# #         # overload pytorch's getter to allow access to properties
# #         # (pytorch only allows access to attributes stored in
# #         # _parameters, _modules or _buffers)
# # #         if (name not in self._parameters and
# # #                 name not in self._buffers and
# # #                 name not in self._modules and
# # #                 name in self.__class__.__dict__):
# # #             return type.__getattr__(self.__class__, name).fget(self)
# #         return _getattr(self, name)
# #     klass.__getattr__ = __getattr__

    if issubclass(klass, tnn.Sequential):
        if not hasattr(klass, 'in_channels'):
            klass.in_channels = property(lambda self: self[0].in_channels)
        if not hasattr(klass, 'out_channels'):
            klass.out_channels = property(lambda self: self[-1].out_channels)
        if not hasattr(klass, 'shape'):
            def shape(self, x):
                for layer in self:
                    x = layer.shape(x)
                return x
            klass.shape = shape

    for key, value in Module.__dict__.items():
        if key[0] != '_':
            setattr(klass, key, value)
    return klass


class Sequential(tnn.Sequential, Module):

    in_channels = property(lambda self: self[0].in_channels)
    out_channels = property(lambda self: self[-1].out_channels)

    def shape(self, x):
        for layer in self:
            x = layer.shape(x)
        return x


class ModuleList(tnn.ModuleList, Module):
    pass


class ModuleDict(tnn.ModuleList, Module):
    pass
