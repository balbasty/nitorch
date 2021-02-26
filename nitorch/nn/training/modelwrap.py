from nitorch.core.py import make_tuple, make_list
from nitorch.nn.modules import Module, nitorchmodule


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
