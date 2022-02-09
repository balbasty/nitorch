"""Tools to ease model training (like torch.ignite)"""

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nitorch.core.utils import benchmark, isin
from nitorch.core.py import make_tuple
from nitorch.nn.modules import Module
import string
import math
import os
import random


try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    class autocast:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): pass

    class GradScaler:
        def scale(self, loss): return loss
        def step(self, optimizer): return optimizer.step()
        def update(self): pass


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    def SummaryWriter():
        raise ImportError('Optional dependency TensorBoard not found')


def split_train_val_test(data, split=(0.6, 0.1, 0.3), shuffle=False, seed=0):
    """Split sequence of data into train, validation and test.

    Parameters
    ----------
    data : [N,] list
        Input data.
    split : [3,] list, default=[0.6, 0.2, 0.2]
        Train, validation, test fractions.
    suffle : bool, default=False
        Randomly shuffle input data (with seed for reproducibility)
    seed : int, default=0
        Seed for random shuffling.

    Returns
    ----------
    train : [split[0]*N,] list
        Train split.
    val : [split[1]*N,] list
        Validation split.
    test : [split[2]*N,] list
        Test split.

    """
    N = len(data)
    # Ensure split is normalised
    split = [s / sum(split) for s in split]
    # Randomly shuffle input data (with seed for reproducibility)
    if shuffle:
        random.seed(seed)
        data = random.sample(data, N)
    # Do train/val/test split
    train, val, test = [], [], []
    for i, d in enumerate(data):
        if i < math.floor(split[0] * N):
            train.append(d)
        elif i < math.floor(sum(split[:2]) * N):
            val.append(d)
        elif i < math.floor(sum(split) * N):
            test.append(d)

    return train, val, test


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
            old[key] += val.detach() * weight
        else:
            old[key] = val.detach() * weight
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
    _tensorboard = None
    _tensorboard_callbacks = None
    random_state = []

    def __init__(self, model, train_set, eval_set=None,
                 optimizer=None,
                 nb_epoch=100,
                 nb_steps=None,
                 *, # the remaining parameters *must be* keywords
                 device=None,
                 dtype=None,
                 initial_epoch=0,
                 log_interval=None,
                 benchmark=False,
                 autocast=False,
                 seed=None,
                 tensorboard=None,
                 save_model=None,
                 save_optimizer=None,
                 load_model=None,
                 load_optimizer=None,
                 show_losses=True,
                 show_metrics=False,
                 scheduler=ReduceLROnPlateau):
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
        optimizer : type or callable or instance, default=Adam
            - If a type, it should be an Optimizer-like type, which will
              be instantiated by `optimizer(model.parameters())`.
            - If a function, it will be called with the model and should
              return an Optimizer-like instance. It makes it possible to
              instantiate optimizers in a more flexible way (e.g.,
              using groups of parameters).
            - If a (non-callable) instance, it should be an
              Optimizer-like instance.
        nb_epoch : int, default=100
            Number of epochs.
        nb_steps : int, default=`len(train_set) or 100`
            Number of steps per epoch.
            If the training set is a finite sequence (i.e., `len` is
            implemented), its length is used. Else, the training set
            is assumed to be infinite and the default number of steps
            is 100.
        scheduler : Scheduler, default=ReduceLROnPlateau

        Other Parameters
        ----------------
        device : torch.device, optional
            Device to use. By default, use the default cuda device if
            any, else use cpu.
        dtype : torch.dtype, optional
            Data type to use. By default use `torch.get_default_dtype`.
        initial_epoch : int, default=0
            First epoch
        log_interval : int, optional
            Number of steps between screen updates.
        benchmark : bool, default=False
            Use the cudnn benchmarking utility that uses the first forward
            pass to compare different convolution algorithms and select the
            best performing one. You should only use this option if the
            spatial shape of your input data is constant across mini batches.
        autocast : bool, default=False
            Automatically cast to half-precision when beneficial.
        seed : int, optional
            Manual seed to use for training. The seed is set when
            training starts. A context manager is used so that the global
            state is kept untouched. If `None`, use the global state.
        tensorboard : str, optional
            A path to the tensorboard log directory.
            If provided, losses and metrics are registered to the board
            by default.
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
        self.log_interval = log_interval
        self.benchmark = benchmark
        self.autocast = autocast
        self.seed = seed
        self.initial_seed = seed
        self.tensorboard = tensorboard
        self._tensorboard_callbacks = dict(train=dict(epoch=[], step=[]),
                                           eval=dict(epoch=[], step=[]))
        self.save_model = save_model
        self.save_optimizer = save_optimizer
        self.load_model = load_model
        self.load_optimizer = load_optimizer
        self.show_losses = show_losses
        self.show_metrics = show_metrics
        self.nb_epoch = nb_epoch
        self.nb_steps = nb_steps
        self.initial_epoch = initial_epoch
        self.epoch = initial_epoch
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        self.dtype = dtype or torch.get_default_dtype()
        self.scheduler = scheduler

        if self.load_model:
            print('load model')
            self.model.load_state_dict(torch.load(self.load_model))
        if optimizer is None:
            optimizer = torch.optim.Adam
        if isinstance(optimizer, type):  # Optimizer class
            optimizer = optimizer(model.parameters())
        elif callable(optimizer):  # function (more flexible)
            optimizer = optimizer(model)
        self.optimizer = optimizer
        if self.load_optimizer:
            print('load optim')
            self.optimizer.load_state_dict(torch.load(self.load_optimizer))
        if self.scheduler is not None:
            self.scheduler = self.scheduler(self.optimizer, verbose=True)

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
    def tensorboard(self):
        if self._tensorboard:
            return self._tensorboard
        else:
            return self._tensorboard

    @tensorboard.setter
    def tensorboard(self, val):
        if not val:
            self._tensorboard = val
        else:
            self._tensorboard = SummaryWriter(val)

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

    def _batch_to(self, elem):
        """Convert a minibatch to correct dtype and device"""
        if torch.is_tensor(elem):
            if elem.dtype.is_floating_point:
                elem = elem.to(dtype=self.dtype, device=self.device)
            else:
                elem = elem.to(device=self.device)
        return elem

    def _get_batch_size(self, batch):
        """Find a tensor in the batch tuple and return its length"""
        for elem in batch:
            if torch.is_tensor(elem):
                return len(elem)
        return 1

    def _backward(self, loss):
        if self.autocast:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def _train(self, epoch=0):
        """Train for one epoch"""

        self.model.train()

        if self.autocast and not getattr(self, 'grad_scaler', None):
            self.grad_scaler = GradScaler()

        # initialize
        epoch_loss = 0.
        epoch_losses = {}
        epoch_metrics = {}
        nb_batches = 0
        nb_steps = len(self.train_set)

        log_interval = self.log_interval or max(nb_steps // 200, 1)
        if hasattr(self.train_set, 'iter'):
            iterator = self.train_set.iter(device=self.device)
        else:
            iterator = self.train_set.__iter__()

        for n_batch, batch in enumerate(iterator):
            losses = {}
            metrics = {}
            # load input
            batch = tuple(map(self._batch_to, make_tuple(batch)))
            batch_size = self._get_batch_size(batch)
            nb_batches += batch_size
            # forward pass
            self.optimizer.zero_grad()
            with autocast(self.autocast):
                output = self.model(*batch, _loss=losses, _metric=metrics)
                loss = sum(losses.values())
            # backward pass
            self._backward(loss)
            # update average across batches
            with torch.no_grad():
                batch_size = float(batch_size)
                epoch_loss += loss.detach() * batch_size
                update_loss_dict(epoch_losses, losses, batch_size)
                update_loss_dict(epoch_metrics, metrics, batch_size)
                # print
                if n_batch % log_interval == 0:
                    self._print('train', epoch, n_batch+1, nb_steps,
                                loss, losses, metrics)
                # tb callback
                if self.tensorboard:
                    tbopt = dict(inputs=batch, outputs=output,
                                 epoch=epoch, minibatch=n_batch, mode='train',
                                 loss=loss, losses=losses, metrics=metrics)
                    self.model.board(self.tensorboard, **tbopt)
                    for func in self._tensorboard_callbacks['train']['step']:
                        func(self.tensorboard, **tbopt)
                    del tbopt
        # print summary
        with torch.no_grad():
            epoch_loss /= nb_batches
            normalize_loss_dict(epoch_losses, nb_batches)
            normalize_loss_dict(epoch_metrics, nb_batches)
            self._print('train', epoch, nb_steps, nb_steps,
                        epoch_loss, epoch_losses, epoch_metrics, last=True)
            self._board('train', epoch, epoch_loss, epoch_metrics)
            # tb callback
            if self.tensorboard:
                tbopt = dict(epoch=epoch, loss=epoch_loss, mode='train',
                             losses=epoch_losses, metrics=epoch_metrics)
                self.model.board(self.tensorboard, **tbopt)
                for func in self._tensorboard_callbacks['train']['epoch']:
                    func(self.tensorboard, **tbopt)

        return epoch_metrics, epoch_loss

    def _eval(self, epoch=0):
        """Evaluate once"""
        if self.eval_set is None:
            return None, None

        self.model.eval()
        with torch.no_grad():
            epoch_loss = 0
            epoch_losses = {}
            epoch_metrics = {}
            nb_batches = 0
            nb_steps = len(self.eval_set)
            log_interval = self.log_interval or max(nb_steps // 200, 1)
            if hasattr(self.eval_set, 'iter'):
                iterator = self.eval_set.iter(device=self.device)
            else:
                iterator = self.eval_set.__iter__()
            for n_batch, batch in enumerate(iterator):
                losses = {}
                metrics = {}
                batch = tuple(map(self._batch_to, make_tuple(batch)))
                batch_size = self._get_batch_size(batch)
                nb_batches += batch_size
                # forward pass
                with autocast(self.autocast):
                    output = self.model(*batch, _loss=losses, _metric=metrics)
                loss = sum(losses.values())
                # update average across batches
                batch_size = float(batch_size)
                epoch_loss += loss * batch_size
                update_loss_dict(epoch_losses, losses, batch_size)
                update_loss_dict(epoch_metrics, metrics, batch_size)
                # print
                if n_batch % log_interval == 0:
                    self._print('eval', epoch, n_batch + 1, nb_steps,
                                loss, losses, metrics)
                # tb callback
                if self.tensorboard:
                    tbopt = dict(inputs=batch, outputs=output,
                                 epoch=epoch, minibatch=n_batch, mode='eval',
                                 loss=loss, losses=losses, metrics=metrics)
                    self.model.board(self.tensorboard, **tbopt)
                    for func in self._tensorboard_callbacks['eval']['step']:
                        func(self.tensorboard, **tbopt)

            # print summary
            epoch_loss /= nb_batches
            normalize_loss_dict(epoch_losses, nb_batches)
            normalize_loss_dict(epoch_metrics, nb_batches)
            self._print('eval', epoch, nb_steps, nb_steps,
                        epoch_loss, epoch_losses, epoch_metrics, last=True)
            self._board('eval', epoch, epoch_loss, epoch_metrics)
            # tb callback
            if self.tensorboard:
                tbopt = dict(epoch=epoch, loss=epoch_loss, mode='eval',
                             losses=epoch_losses, metrics=epoch_metrics)
                self.model.board(self.tensorboard, **tbopt)
                for func in self._tensorboard_callbacks['eval']['epoch']:
                    func(self.tensorboard, **tbopt)

        return epoch_metrics, epoch_loss

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
        getitem = lambda x: x.item() if hasattr(x, 'item') else x

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
            values += '| loss = {:12.6g} '.format(getitem(loss))
            if losses and self.show_losses:
                values += '|'
                for key, val in losses.items():
                    values += ' {}: {:12.6g} '.format(key, getitem(val))
        if metrics and (mode == 'eval' or self.show_metrics):
            values += '|'
            for key, val in metrics.items():
                values += ' {}: {:12.6g} '.format(key, getitem(val))

        print(evolution + values, end='\r', flush=True)
        if last:
            print('')

    def _board(self, mode, epoch, loss, epoch_metrics):
        """Add losses and metrics to tensorboard."""
        if not self.tensorboard:
            return
        getitem = lambda x: x.item() if hasattr(x, 'item') else x
        tb = self.tensorboard
        tb.add_scalars('loss', {mode: getitem(loss)}, epoch)
        for tag, value in epoch_metrics.items():
            tb.add_scalars(tag, {mode: getitem(value)}, epoch)
        tb.flush()

    def add_tensorboard_callback(self, func, mode='train', trigger='epoch'):
        """Register tensorboard callbacks

        Parameters
        ----------
        func : callable
            If trigger 'step', with signature
                `(tb, input, output, epoch, step, loss, losses, metrics)`
            If trigger 'epoch', with signature:
                `(tb, epoch, loss, losses, metrics)`
        mode : {'train', 'eval'}
            Trigger either during a training or evaluation call.
        trigger : {'epoch', 'step'}
            Trigger either at the end of a step or at the end of an epoch.

        """
        if mode not in self._tensorboard_callbacks.keys():
            self._tensorboard_callbacks[mode] = dict()
        if trigger not in self._tensorboard_callbacks[mode].keys():
            self._tensorboard_callbacks[mode][trigger] = list()
        self._tensorboard_callbacks[mode][trigger].append(func)

    def _hello(self, mode):
        """Tell the use what we are going to do (mode, device, dtype, ...)

        Parameters
        ----------
        mode : {'train', 'eval'}

        """
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
            dir_model = os.path.dirname(save_model)
            if dir_model:
                os.makedirs(dir_model, exist_ok=True)
            torch.save(self.model.state_dict(), save_model)
        if self.save_optimizer:
            save_optimizer = self._formatfile(self.save_optimizer, epoch)
            dir_optimizer = os.path.dirname(save_optimizer)
            if dir_optimizer:
                os.makedirs(dir_optimizer, exist_ok=True)
            torch.save(self.optimizer.state_dict(), save_optimizer)

    def _append_results(self, results, results_batch):
        """append losses+metrics from one batch"""
        if not results:
            for key in results_batch:
                results[key] = results_batch[key][None]
        else:
            for key in results_batch:
                results[key] = \
                    torch.cat((results[key], results_batch[key][None]))
        return results

    @staticmethod
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
        """Launch training
        
        Returns
        ----------
        results : dict
            dictionary with train ('train') and validation ('val') results (losses+metrics).
            
        """
        self._hello('train')
        results = {
            'train': {}, 
            'val': {},
        }
        with torch.random.fork_rng(enabled=self.seed is not None):
            if self.seed is not None:
                torch.random.manual_seed(self.seed)
            self.initial_seed = torch.random.initial_seed()
            with benchmark(self.benchmark):
                self.to(dtype=self.dtype, device=self.device)
                self.epoch = self.initial_epoch
                self._eval(self.epoch)
                self._save(self.epoch)
                for self.epoch in range(self.epoch+1, self.nb_epoch+1):
                    # do training
                    train_results, train_loss = self._train(self.epoch)
                    # append results
                    results['train'] = self._append_results(results['train'], train_results)
                    # do evalutation
                    val_results, val_loss = self._eval(self.epoch)
                    if val_results is not None:
                        # append results
                        results['val'] = self._append_results(results['val'], val_results)
                    self._save(self.epoch)
                    # scheduler
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss or train_loss)
                    elif self.scheduler:
                        self.scheduler.step()
                        
        return results

    def eval(self):
        """Launch evaluation"""
        self._hello('eval')
        self.model.to(dtype=self.dtype, device=self.device)
        self._eval()

    def init(self):
        """Initialize the random state + run one evaluation."""
        with torch.random.fork_rng(enabled=self.seed is not None):
            if self.seed is not None:
                torch.random.manual_seed(self.seed)
            self.initial_seed = torch.random.initial_seed()
            self.epoch = self.initial_epoch
            self.to(dtype=self.dtype, device=self.device)
            self._eval(self.epoch)
            self._save(self.epoch)
            self.save_random_state()

    def set_random_state(self):
        """Populate the random state using a saved state."""
        if self.random_state:
            cpu_state, *gpu_states = self.random_state
            devices = list(range(torch.cuda.device_count()))
            torch.set_rng_state(self.random_state[0])
            for device, state in zip(devices, gpu_states):
                torch.cuda.set_rng_state(state, device)

    def save_random_state(self):
        """Save the current random state."""
        devices = list(range(torch.cuda.device_count()))
        self.random_state = [torch.get_rng_state()]
        self.random_state.extend(torch.cuda.get_rng_state(device)
                                 for device in devices)

    def pick_model(self, results, metric, dataset='val', epoch=None, arg_max=True):
        """pick a trained model
        
        Parameters
        ----------
        results : dict
            Output of ModelTrainer.train()
        metric : str
            Key in results dict.
        dataset : str, default='val'
            Dataset to chose model from ('train' or 'val').
        epoch : int, optional
            Overrides model selection and just picks based on training epoch.
        arg_max : bool, default=True
            argmax or argmin when chosing model.

        Returns
        ----------
        model : torch.model
            Selected model.

        """        
        if epoch is None or not isinstance(epoch, int):
            if metric not in results[dataset]:
                raise ValueError("metric {:} not in {:} results dictionary!" \
                    .format(metric, dataset))
            if arg_max:
                epoch = int(results[dataset][metric].argmax())
            else:
                epoch = int(results[dataset][metric].argmin())
            print("Returning model with {:} {:} {:} | epoch={:}, {:}={:.3f}." \
                .format('max' if arg_max else 'min', 
                    dataset, metric, epoch + 1, metric, results[dataset][metric][epoch]))
        file = self._formatfile(self.save_model, epoch + 1)
        self.model.load_state_dict(torch.load(file))
        self.model.to(self.device)
        return self.model

    def train1(self):
        """Train for one epoch."""
        with torch.random.fork_rng():
            self.set_random_state()
            self.to(dtype=self.dtype, device=self.device)
            self.epoch += 1
            train_loss = self._train(self.epoch)
            eval_loss = self._eval(self.epoch)
            # scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(eval_loss or train_loss)
            elif self.scheduler:
                self.scheduler.step()
            self._save(self.epoch)
            self.save_random_state()

    def to(self, dtype=None, device=None):
        # we need to track which parameters were optimized to reassign them
        # to the optimizer after they have been moved
        group_index = []
        for group in self.optimizer.param_groups:
            param_index = []
            for param in group['params']:
                found = False
                for i, model_param in enumerate(self.model.parameters()):
                    if param is model_param:
                        found = True
                        param_index.append(i)
                        break
                if not found:
                    raise ValueError(f'Could not find parameter {param} in '
                                     'the model.')
            group_index.append(param_index)

        # move model
        self.model = self.model.to(dtype=dtype, device=device)

        # update link to optimized parameters
        model_param = list(self.model.parameters())
        for group, index in zip(self.optimizer.param_groups, group_index):
            param = [model_param[i] for i in index]
            group['params'] = param

        # move state to new dtype/device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(dtype=dtype, device=device)


class SynthTrainer(ModelTrainer):
    def __init__(self, model, train_set=None, test_set=None,
                pretrain_crit=torch.nn.MSELoss(), nb_epoch_pretrain=None, type='seg',
                tf_weights=False, download_weights=False, model_url=None,
                **kwargs):
        if not train_set:
            if test_set:
                train_set = test_set
            else:
                raise RuntimeError('Must provide training data for training and/or test data for testing.')
            
        super().__init__(model=model, train_set=train_set, **kwargs)
        self.test_set = test_set
        self.pretrain_crit = pretrain_crit
        if download_weights:
            self.model.download_tf_weights(model_url=model_url)
        elif tf_weights:
            self.model.load_tf_weights(path_to_h5=tf_weights)

        if nb_epoch_pretrain:
            self.pretrain(nb_epoch_pretrain)
            
    def pretrain(self, nb_epoch_pretrain, verbose=False):
        if verbose:
            print('Pretraining using L2 loss on forward pass of labels.')

        self.model.train()

        if self.autocast and not getattr(self, 'grad_scaler', None):
            self.grad_scaler = GradScaler()

        nb_steps = len(self.train_set)

        if hasattr(self.train_set, 'iter'):
            iterator = self.train_set.iter(device=self.device)
        else:
            iterator = self.train_set.__iter__()

        for epch in range(nb_epoch_pretrain):
            epoch_loss = []
            for _, labels in iterator:
                # forward pass
                self.optimizer.zero_grad()
                with autocast(self.autocast):
                    output = self.model(labels)
                    losses = self.pretrain_crit(output, labels)
                    loss = sum(losses.values())
                # backward pass
                if self.autocast:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                epoch_loss.append(loss)

            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            if verbose:
                print('Epoch #{} - L2 Loss = {:.3f}'.format(epch, epoch_loss))


# from nitorch.nn.losses._wip_adversarial import Wasserstein, R1
# class AdvTrainer(ModelTrainer):
#     def __init__(self, model, discrim, train_set, eval_set=None,
#                  optimizer=None,
#                  optimizer_discrim=None,
#                  nb_epoch=100,
#                  nb_steps=None,
#                  gradient_penalty=None,
#                  lambda_gp=1,
#                  softplus=False,
#                  *, # the remaining parameters *must be* keywords
#                  device=None,
#                  dtype=None,
#                  initial_epoch=0,
#                  log_interval=None,
#                  benchmark=False,
#                  autocast=False,
#                  seed=None,
#                  tensorboard=None,
#                  save_model=None,
#                  save_optimizer=None,
#                  load_model=None,
#                  load_optimizer=None,
#                  save_model_discrim=None,
#                  save_optimizer_discrim=None,
#                  load_model_discrim=None,
#                  load_optimizer_discrim=None,
#                  show_losses=True,
#                  show_metrics=False,
#                  scheduler=ReduceLROnPlateau):
#         """
#         Inherits from ModelTrainer, with additional parameters for discriminator
#         """
#         super().__init__()
#         self.discrim = discrim
#         if optimizer_discrim is None:
#             optimizer_discrim = torch.optim.Adam(model.parameters())
#         self.optimizer_discrim = optimizer_discrim
#         self.save_model_discrim = save_model_discrim
#         self.save_optimizer_discrim = save_optimizer_discrim
#         self.load_model_discrim = load_model_discrim
#         self.load_optimizer_discrim = load_optimizer_discrim

#         if self.load_model_discrim:
#             self.discrim.load_state_dict(torch.load(self.load_model_discrim))
#         if self.load_optimizer_discrim:
#             self.optimizer_discrim.load_state_dict(torch.load(self.load_optimizer_discrim))

#         self.gradient_penalty = gradient_penalty
#         self.softplus = softplus
#         self.lambda_gp = lambda_gp

#     def _train_discrim(self, epoch=0):
#         """Train discriminator for one epoch"""

#         self.model.eval()
#         self.discrim.train()
#         epoch_loss = 0.
#         epoch_losses = {}
#         epoch_metrics = {}
#         nb_batches = 0
#         nb_steps = len(self.train_set)
#         for n_batch, batch in enumerate(self.train_set):
#             losses = {}
#             metrics = {}
#             # forward pass
#             batch = make_tuple(batch)
#             batch = tuple(torch.as_tensor(b, device=self.device) for b in batch)
#             batch = tuple(b.to(dtype=self.dtype)
#                           if b.dtype in (torch.half, torch.float, torch.double)
#                           else b for b in batch)
#             nb_batches += batch[0].shape[0]
#             self.optimizer_discrim.zero_grad()
#             output = self.model(*batch)
#             output_real = self.discrim(*batch)
#             output_fake = self.discrim(output)
#             if self.gradient_penalty == 'wasserstein' or 'Wasserstein' or 'wgan' or 'WGAN':
#                 gp = Wasserstein(self.discrim, *batch, fake=output)
#             elif self.gradient_penalty == 'r1' or 'R1':
#                 gp = R1(self.discrim, *batch)
#             else:
#                 gp = 0
#             if self.softplus:
#                 losses['adv'] = torch.mean(torch.nn.functional.softplus(-output_real)) + torch.mean(torch.nn.functional.softplus(output_fake)) + self.lambda_gp * gp
#             else:
#                 losses['adv'] = - torch.mean(output_real) + torch.mean(output_fake) + self.lambda_gp * gp
#             loss = sum(losses.values())
#             # backward pass
#             loss.backward()
#             self.optimizer_discrim.step()
#             # update average across batches
#             with torch.no_grad():
#                 weight = float(batch[0].shape[0])
#                 epoch_loss += loss * weight
#                 update_loss_dict(epoch_losses, losses, weight)
#                 update_loss_dict(epoch_metrics, metrics, weight)
#                 # print
#                 if n_batch % self.log_interval == 0:
#                     self._print('train', epoch, n_batch+1, nb_steps,
#                                 loss, losses, metrics)
#                 # tb callback
#                 if self.tensorboard:
#                     tbopt = dict(inputs=batch, outputs=output,
#                                  epoch=epoch, minibatch=n_batch, mode='train',
#                                  loss=loss, losses=losses, metrics=metrics)
#                     self.model.board(self.tensorboard, **tbopt)
#                     for func in self._tensorboard_callbacks['train']['step']:
#                         func(self.tensorboard, **tbopt)
#                     del tbopt
#         # print summary
#         with torch.no_grad():
#             epoch_loss /= nb_batches
#             normalize_loss_dict(epoch_losses, nb_batches)
#             normalize_loss_dict(epoch_metrics, nb_batches)
#             self._print('train', epoch, nb_steps, nb_steps,
#                         epoch_loss, epoch_losses, epoch_metrics, last=True)
#             self._board('train', epoch, epoch_loss, epoch_metrics)
#             # tb callback
#             if self.tensorboard:
#                 tbopt = dict(epoch=epoch, loss=epoch_loss, mode='train',
#                              losses=epoch_losses, metrics=epoch_metrics)
#                 self.model.board(self.tensorboard, **tbopt)
#                 for func in self._tensorboard_callbacks['train']['epoch']:
#                     func(self.tensorboard, **tbopt)

#         return epoch_loss
