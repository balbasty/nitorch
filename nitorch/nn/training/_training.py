"""Tools to ease model training (like torch.ignite)"""

import torch
from nitorch.core.utils import benchmark
from nitorch.core.pyutils import make_tuple, make_list
from nitorch.nn.modules._base import Module, nitorchmodule
import string
import math
import os


def update_loss_dict(old, new, weight=1, inplace=True):
    """Update a dictionary of losses/metrics with a new batch

    Parameters
    ----------
    old : dict
        Previous (accumulated) dictionary of losses/metrics
    new : dict
        Dictionary of losses/metrics for the current batch
    weight : float, default=1
        Weight for the batch
    inplace : bool, default=True
        Modify the dictionary in-place

    Returns
    -------
    new : dict
        Updated (accumulated) dictionary of losses/metrics

    """
    if not inplace:
        old = dict(old)
    for key, val in new.items():
        if key in old.keys():
            old[key] += val * weight
        else:
            old[key] = val * weight
    return old


def normalize_loss_dict(losses, weight=1, inplace=True):
    """Normalize all losses in a dict.

    Parameters
    ----------
    losses : dict
        Accumulated dictionary of losses/metrics
    weight : float, default=1
        Sum of weights across all batches
    inplace : bool, default=True
        Modify the dictionary in-place

    Returns
    -------
    losses : dict
        Normalized dictionary of losses/metrics

    """
    if not inplace:
        losses = dict(losses)
    for key, val in losses.items():
        losses[key] /= weight
    return losses


class ModelTrainer:
    """A class that simplifies training a network."""

    _nb_steps = None
    _train_set = None
    _eval_set = None

    def __init__(self, model, train_set, eval_set=None,
                 optimizer=torch.optim.Adam,
                 nb_epoch=100,
                 nb_steps=None,
                 device=None,
                 dtype=None,
                 epoch=0,
                 log_interval=10,
                 benchmark=False,
                 save_model=None,
                 save_optimizer=None,
                 load_model=None,
                 load_optimizer=None,
                 show_losses=True,
                 show_metrics=False):
        """

        Parameters
        ----------
        model : Module
            Model to train.
            Its forward pass should accept a `loss` argument, and take as
            inputs the elements that pop out of the training set.
        train_set : sequence[tensor or tuple[tensor]]
            Training set.
            It should be a finite sequence of tensors or tuple of tensors.
        eval_set : sequence[tensor or tuple[tensor]], optional
            Evaluation set.
            It should be a finite sequence of tensors or tuple of tensors.
        optimizer : callable, default=Adam
            A function that takes trainable parameters as inputs and
            returns an Optimizer object.
        nb_epoch : int, default=100
            Number of epochs.
        nb_steps : int, default=`len(train_set) or 100`
            Number of steps per epoch.
            If the training set is a finite sequence (i.e., `len` is
            implemented), its length is used. Else, the training set
            is assumed to be infinite and the default number of steps
            is 100.

        Other Parameters
        ----------------
        device : torch.device, optional
            Device to use. By default, use the default cuda device if
            any, else use cpu.
        dtype : torch.dtype, optional
            Data type to use. By default use `torch.get_default_dtype`.
        epoch : int, default=0
            First epoch
        log_interval : int, default=float
            Print/save model
        benchmark : bool, default=False
            Use the cudnn benchmarking utility that uses the first forward
            pass to compare different convolution algorithms and select the
            best performing one. You should only use this option if the
            spatial shape of your input dat ais constant across mini batches.
        save_model : str, optional
            A path to save the model at each epoch. Can have a
            formatted component ('mymodel_{}.pth') for the epoch number.
        save_optimizer : str, optional
            A path to save the optimizer at each epoch. Can have a
            formatted component ('myoptim_{}.pth') for the epoch number.
        load_model : str, optional
            Path to saved weights to use to initialize the model.
        load_optimizer : str, optional
            Path to saved state to use to initialize the optimizer.
        show_losses : bool, default=True
            Print values of individual losses
        show_metrics : bool, default=False
            Print values of individual metrics
        """
        self.model = model
        self.train_set = train_set
        self.eval_set = eval_set
        self.optimizer = optimizer(model.parameters())
        self.log_interval = log_interval
        self.benchmark = benchmark
        self.save_model = save_model
        self.save_optimizer = save_optimizer
        self.load_model = load_model
        self.load_optimizer = load_optimizer
        self.show_losses = show_losses
        self.show_metrics = show_metrics
        self.nb_epoch = nb_epoch
        self.nb_steps = nb_steps
        self.epoch = epoch
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        self.dtype = dtype or torch.get_default_dtype()

        if self.load_model:
            self.model.load_state_dict(torch.load(self.load_model))
        if self.load_optimizer:
            self.optimizer.load_state_dict(torch.load(self.load_optimizer))

    def _update_nb_steps(self):
        def len_or(x, default):
            return len(x) if hasattr(x, '__len__') else default
        self._nb_train = self._nb_steps or len_or(self._train_set, 100)
        self._nb_eval = self._nb_steps or len_or(self._eval_set, 100)

    class _batch_iterator:
        def __init__(self, set, length):
            self.set = set
            self.length = length
        def __len__(self):
            return self.length
        def __iter__(self):
            d = 0
            while d < self.length:
                for batch in self.set:
                    if d >= self.length:
                        return
                    yield batch
                    d += 1

    @property
    def nb_steps(self):
        return self._nb_steps

    @nb_steps.setter
    def nb_steps(self, val):
        self._nb_steps = val
        self._update_nb_steps()

    @property
    def train_set(self):
        if self._train_set:
            return self._batch_iterator(self._train_set, self._nb_train)
        else:
            return None

    @train_set.setter
    def train_set(self, val):
        self._train_set = val
        self._update_nb_steps()

    @property
    def eval_set(self):
        if self._eval_set:
            return self._batch_iterator(self._eval_set, self._nb_eval)
        else:
            return None

    @eval_set.setter
    def eval_set(self, val):
        self._eval_set = val
        self._update_nb_steps()

    def _train(self, epoch=0):
        """Train for one epoch"""

        self.model.train()
        epoch_loss = 0
        epoch_losses = {}
        epoch_metrics = {}
        nb_batches = 0
        nb_steps = len(self.train_set)
        for n_batch, batch in enumerate(self.train_set):
            losses = {}
            metrics = {}
            # forward pass
            batch = make_tuple(batch)
            batch = tuple(torch.as_tensor(b, dtype=self.dtype, device=self.device) for b in batch)
            nb_batches += batch[0].shape[0]
            self.optimizer.zero_grad()
            self.model(*batch, _loss=losses, _metric=metrics)
            loss = sum(losses.values())
            # backward pass
            loss.backward()
            self.optimizer.step()
            # update average across batches
            with torch.no_grad():
                weight = float(batch[0].shape[0])
                epoch_loss += loss * weight
                update_loss_dict(epoch_losses, losses, weight)
                update_loss_dict(epoch_metrics, metrics, weight)
                # print
                if n_batch % self.log_interval == 0:
                    self._print('train', epoch, n_batch+1, nb_steps,
                                loss, losses, metrics)
        # print summary
        with torch.no_grad():
            epoch_loss /= nb_batches
            normalize_loss_dict(epoch_losses, nb_batches)
            normalize_loss_dict(epoch_metrics, nb_batches)
            self._print('train', epoch, nb_steps, nb_steps,
                        epoch_loss, epoch_losses, epoch_metrics, last=True)

    def _eval(self, epoch=0):
        """Evaluate once"""
        if self.eval_set is None:
            return

        self.model.eval()
        with torch.no_grad():
            epoch_loss = 0
            epoch_losses = {}
            epoch_metrics = {}
            nb_batches = 0
            nb_steps = len(self.eval_set)
            for n_batch, batch in enumerate(self.eval_set):
                losses = {}
                metrics = {}
                # forward pass
                batch = make_tuple(batch)
                batch = tuple(torch.as_tensor(b, dtype=self.dtype, device=self.device) for b in batch)
                nb_batches += batch[0].shape[0]
                self.optimizer.zero_grad()
                self.model(*batch, _loss=losses, _metric=metrics)
                loss = sum(losses.values())
                # update average across batches
                weight = float(batch[0].shape[0])
                epoch_loss += loss * weight
                update_loss_dict(epoch_losses, losses, weight)
                update_loss_dict(epoch_metrics, metrics, weight)
                # print
                if n_batch % self.log_interval == 0:
                    self._print('eval', epoch, n_batch + 1, nb_steps,
                                loss, losses, metrics)
            # print summary
            epoch_loss /= nb_batches
            normalize_loss_dict(epoch_losses, nb_batches)
            normalize_loss_dict(epoch_metrics, nb_batches)
            self._print('eval', epoch, nb_steps, nb_steps,
                        epoch_loss, epoch_losses, epoch_metrics, last=True)

    def _print(self, mode, n_epoch, n_batch, nb_steps, loss,
               losses=None, metrics=None, last=False):
        """Pretty printing

        Parameters
        ----------
        mode : {'train', 'eval'}
        n_epoch : int
            Index of current epoch (starts at one)
        n_batch : int
            Index of current batch (starts at one)
        nb_steps : int
            Total number of batches
        loss : () tensor
            Loss for this batch
        losses : dict[str: () tensor]
            Loss components for this batch
        metrics : dict[str: () tensor]
            Metrics for this batch
        last : bool, default=False
            Is this the end of the batch?
            If True, loss/losses/metrics should contain the average loss
            across all batches.

        """
        name = 'Train' if mode == 'train' else 'Eval '
        if last:
            pct = 1
            bar = '[' + '=' * 10 + ']'
        else:
            pct = n_batch/nb_steps
            len_arrow = min(math.floor(pct*10 + 0.5), 9)
            bar = '[' + '=' * len_arrow + '>' + ' ' * (9-len_arrow) + ']'

        lepoch = str(len(str(self.nb_epoch)))
        evolution = '{:s} | {:' + lepoch + 'd} | {:3.0f}% ' + bar + ' '
        evolution = evolution.format(name, n_epoch, pct*100)

        values = ''
        if mode == 'train':
            values += '| loss = {:12.6g} '.format(loss.item())
            if losses and self.show_losses:
                values += '|'
                for key, val in losses.items():
                    values += ' {}: {:12.6g} '.format(key, val.item())
        if metrics and (mode == 'eval' or self.show_metrics):
            values += '|'
            for key, val in metrics.items():
                values += ' {}: {:12.6g} '.format(key, val.item())

        print(evolution + values, end='\r', flush=True)
        if last:
            print('')

    def _hello(self, mode):
        if self.device.type == 'cuda':
            device = torch.cuda.get_device_name(self.device)
        else:
            assert self.device.type == 'cpu'
            device = 'CPU'
        dtype = str(self.dtype).split('.')[-1]
        if mode == 'train':
            hello = 'Training model {} for {} epochs (steps per epoch: {}) ' \
                    'on {} (dtype = {})'
            hello = hello.format(type(self.model).__name__, self.nb_epoch,
                                 len(self.train_set), device, dtype)
        else:
            hello = 'Evaluating model {} (minibatches: {}) on {} (dtype = {})'
            hello = hello.format(type(self.model).__name__,
                                 len(self.eval_set), device, dtype)
        print(hello, flush=True)

    def _save(self, epoch):
        """Save once"""
        if self.save_model:
            save_model = self._formatfile(self.save_model, epoch)
            dir_model = os.path.jdirname(save_model)
            os.makedirs(dir_model, exist_ok=True)
            torch.save(self.model.state_dict(), save_model)
        if self.save_optimizer:
            save_optimizer = self._formatfile(self.save_optimizer, epoch)
            dir_optimizer = os.path.jdirname(save_optimizer)
            os.makedirs(dir_optimizer, exist_ok=True)
            torch.save(self.optimizer.state_dict(), save_optimizer)

    @ staticmethod
    def _formatfile(file, epoch):
        """Format filename for an epoch"""
        keys = [tup[1] for tup in string.Formatter().parse(file)
                if tup[1] is not None]
        if len(keys) == 1:
            file = file.format(epoch)
        elif len(keys) > 1:
            raise ValueError('Cannot have more than one format key')
        return file

    def train(self):
        """Launch training"""
        self._hello('train')
        with benchmark(self.benchmark):
            self.model.to(dtype=self.dtype, device=self.device)
            self._eval(self.epoch)
            self._save(self.epoch)
            for self.epoch in range(self.epoch+1, self.nb_epoch+1):
                self._train(self.epoch)
                self._eval(self.epoch)
                self._save(self.epoch)

    def eval(self):
        """Launch evaluation"""
        self._hello('eval')
        self.model.to(dtype=self.dtype, device=self.device)
        self._eval()


class SupervisedModel(Module):
    """A wrapper class for classic supervised models.

    This makes `ModelTrainer` work with keras-like inputs and losses.
    """

    def __init__(self, model, losses=None, metrics=None):
        """

        Parameters
        ----------
        model : Module
            A model to train
        losses : callable or list[callable]
            Loss functions (this is what we optimize by backpropagation)
        metrics : callable or list[callable]
            Metric functions (this is not optimized, only monitored)
        """
        super().__init__(self)
        self.model = nitorchmodule(model)
        self.losses = make_list(losses) or []
        self.metrics = make_list(metrics) or []

    def forward(self, inputs, labels, *, _loss=None, _metric=None):
        """
        The forward pass is applied to the unpacked inputs.
        Then, for each triplet (output, label, loss), the loss
        is computed as `loss(output, label)`. Note that the number
        of losses can be smaller than the number of outputs (in this
        case the remaining outputs do not enter a loss). Similarly, the
        number of labels can be smaller than the number of outputs
        (in this case, the loss is called with only one argument).

        Parameters
        ----------
        inputs : tensor or sequence[tensor]
            Input tensors that are fed to the network.
        labels : tensor or sequence[tensor]
            Ground truth labels.

        Returns
        -------
        outputs : tensor or tuple[tensor]
            Outputs of the network
        """

        inputs = make_tuple(inputs)
        labels = make_tuple(labels)

        # forward pass
        outputs = self.model(*inputs, _loss=_loss, _metric=_metric)
        outputs = make_tuple(outputs)

        # compute losses
        if _loss is not None:
            losses = {}
            for i, loss in enumerate(self.losses):
                if len(labels) > i and labels[i] is not None:
                    loss_inputs = (outputs[i], labels[i])
                else:
                    loss_inputs = (outputs[i],)
                losses[str(i)] = loss(*loss_inputs)
            self.update_dict(_loss, losses)

        # compute metrics
        if _metric is not None:
            metrics = {}
            for i, metric in enumerate(self.metrics):
                if len(labels) > i and labels[i] is not None:
                    metric_inputs = (outputs[i], labels[i])
                else:
                    metric_inputs = (outputs[i],)
                metrics[str(i)] = metric(*metric_inputs)
            self.update_dict(_metric, metrics)

        return outputs
