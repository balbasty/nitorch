"""Tools to ease model training (like torch.ignite)"""

import torch
from nitorch.core.pyutils import make_tuple, make_list
from nitorch.nn.modules._base import Module, nitorchmodule
import string


class ModelTrainer:
    """A class that simplifies training a network."""

    def __init__(self, model, train_set, eval_set=None,
                 optimizer=torch.optim.Adam, nb_epoch=100, epoch=0,
                 log_interval=10, save_model=None, save_optimizer=None):
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
            Number of epochs
        epoch : int, default=0
            First epoch
        log_interval : int, default=float
            Print/save model
        save_model : str, optional
            A path to save the model at each epoch. Can have a
            formatted component ('mymodel_{}.pth') for the epoch number.
        save_optimizer : str, optional
            A path to save the optimizer at each epoch. Can have a
            formatted component ('myoptim_{}.pth') for the epoch number.
        """
        self.model = model
        self.train_set = train_set
        self.eval_set = eval_set
        self.optimizer = optimizer(model.parameters())
        self.log_interval = log_interval
        self.save_model = save_model
        self.save_optimizer = save_optimizer
        self.nb_epoch = nb_epoch
        self.epoch = epoch

        self._nb_train = len(train_set)
        self._nb_eval = len(eval_set)

    def _train(self, epoch):
        """Train for one epoch"""

        self.model.train()
        for n_batch, batch in enumerate(self.train_set):
            losses = []
            metrics = {}
            # forward pass
            batch = make_tuple(batch)
            self.optimizer.zero_grad()
            self.model(*batch, _loss=losses, _metric=metrics)
            loss = sum(losses).sum()
            # backward pass
            loss.backward()
            self.optimizer.step()
            # print / save
            if n_batch % self.log_interval == 0:
                train_print = 'Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, n_batch, self._nb_train,
                    100. * n_batch / self._nb_train)
                train_print += ' \tloss: {:.6f}'.format(loss.item())
                for key, val in metrics.items():
                    train_print += ' \t{}: {:.6f}'.format(key, val.item())
                print(train_print)

    def _save(self, epoch):
        """Save once"""
        if self.save_model:
            save_model = self._formatfile(self.save_model, epoch)
            torch.save(self.model.state_dict(), save_model)
        if self.save_optimizer:
            save_optimizer = self._formatfile(self.save_optimizer, epoch)
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

    def _eval(self, epoch):
        """Evaluate once"""
        if self.eval_set is None:
            return

        self.model.eval()
        with torch.no_grad():
            loss = 0
            metric = {}
            for n_batch, batch in enumerate(self.eval_set):
                losses = []
                metrics = {}
                batch = make_tuple(batch)
                self.model(*batch, _loss=losses, _metric=metrics)
                loss = loss + sum(losses)
                for key, val in metrics.items():
                    if key in metric.keys():
                        metric[key] += val
                    else:
                        metric[key] = val
            loss = loss / self._nb_eval
            for key, val in metric.items():
                metric[key] = metric[key] / self._nb_eval
        # print
        eval_print = 'Eval Epoch: {} \tloss: {:.6f}'.format(epoch, loss.item())
        for key, val in metric.items():
            eval_print += ' \t{}: {:.6f}'.format(key, val)
        print(eval_print)

    def train(self):
        """Launch training"""
        self._eval(self.epoch)
        self._save(self.epoch)
        for self.epoch in range(self.epoch+1, self.nb_epoch+1):
            self._train(self.epoch)
            self._eval(self.epoch)
            self._save(self.epoch)


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
            losses = []
            for i, loss in enumerate(self.losses):
                if len(labels) > i and labels[i] is not None:
                    loss_inputs = (outputs[i], labels[i])
                else:
                    loss_inputs = (outputs[i],)
                losses.append(loss(*loss_inputs))
            _loss += losses

        # compute metrics
        if _metric is not None:
            metrics = []
            for i, metric in enumerate(self.metrics):
                if len(labels) > i and labels[i] is not None:
                    metric_inputs = (outputs[i], labels[i])
                else:
                    metric_inputs = (outputs[i],)
                metrics.append(metric(*metric_inputs))
            _metric += metrics

        return outputs
