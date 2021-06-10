import torch
from nitorch import nn
from nitorch.core import utils, py
from .models import sandi


class SANDINet(nn.Module):
    """Unsupervised SANDI network"""

    def __init__(self, shells, fixed_parameters=None, random_shells=False,
                 Delta=80, delta=3, feed_parameters='b', log=True, net=None):
        """

        Parameters
        ----------
        shells : sequence[float]
            b value of each shell
        fixed_parameters : dict, optional
            Dictionary of fixed SANDI parameters that should not be
            estimated by the model.
        random_shells : int, optional
            Sample that number of random shells and only give these
            to the network.
        Delta : float, default=80
            Time interval between gradient pulses Δ (ms)
        delta : float, default=3
            Gradient duration δ (ms)
        feed_parameters : {'b', 'all'} or False, default='b'
            Feed selected parameters to the network as additional
            channels.
        log : bool, default=True
            Predict log parameters
        net : dict or callable(int, int)
            Dictionary of UNet parameters or function that takes
            an input and output number of channels and returns a
            network.
        """
        super().__init__()

        self.fixed_parameters = fixed_parameters or dict()
        self.shells = shells
        self.Delta = Delta
        self.delta = delta
        self.random_shells = random_shells
        self.feed_parameters = feed_parameters
        self.log = log
        nb_parameters = 6 - len(self.fixed_parameters)
        nb_input_shells = random_shells or len(shells)
        nb_input_channels = nb_input_shells
        if self.feed_parameters:
            nb_input_channels = nb_input_channels + 1  # b value
            if self.feed_parameters is True:  # all
                nb_input_channels += len(self.fixed_parameters) + 2

        net = net or dict()
        if isinstance(net, dict):
            net = nn.UNet2(nb_input_shells, nb_parameters, **net)
        else:
            net = net(nb_input_shells, nb_parameters)
        self.net = net

        self.tags = ['pred']

    def forward(self, x, _loss=None, _metric=None, **overload):
        """

        Parameters
        ----------
        x : (batch, shells, *spatial)
            Tensor of Powder-averaged shells
        shells : sequence[float], optional
            b value of each shell (overloads the initial value)
        Delta : [sequence of] float
        delta : [sequence of] float

        Returns
        -------
        prm : (batch, channels, *spatial)
            Fitted parameters ordered as
            (fextra, fneurite, radius, diff_extra, diff_neurite, diff_soma)

        """
        backend = utils.backend(x)
        dim = x.dim() - 2

        shells = overload.get('shells', self.shells)
        Delta = overload.get('Delta', self.Delta)
        delta = overload.get('delta', self.delta)

        # ensure shapes broadcast correctly
        shells = torch.as_tensor(shells, **backend)
        Delta = torch.as_tensor(Delta, **backend)
        delta = torch.as_tensor(delta, **backend)
        if shells.dim() == 1:
            shells = shells[None]
        shells = utils.unsqueeze(shells, -1, dim)
        if Delta.dim() == 0:
            Delta = Delta[None]
        Delta = utils.unsqueeze(Delta, -1, dim + 1)
        if delta.dim() == 0:
            delta = delta[None]
        delta = utils.unsqueeze(delta, -1, dim + 1)

        xin = x
        bin = shells
        if self.random_shells:
            sub = torch.randperm(self.random_shells, device=x.device)
            xin = xin[:, sub]
            bin = bin[:, sub]
        if self.feed_parameters == 'b':
            xin = torch.cat([xin, bin.expand(xin.shape)], dim=1)
        elif self.feed_parameters:
            batch, _, shape = xin.shape
            xin = [xin, bin.expand(xin.shape),
                   Delta.expand([batch, Delta.shape[1], *shape]),
                   delta.expand([batch, delta.shape[1], *shape])]
            xin = torch.cat(xin, dim=1)

        all_parameters = ('fextra', 'fneurite', 'radius',
                          'diff_extra', 'diff_neurite', 'diff_soma')
        fit_parameters = tuple(p for p in all_parameters
                               if p not in self.fixed_parameters)

        # network pass
        params = self.net(xin)

        params0 = params
        params = dict()
        if self.log:  # exponentiate
            for param, name in zip(params0.unbind(1), fit_parameters):
                if name[0] == 'f':  # fraction
                    param = param.neg().add_(1).reciprocal()
                else:
                    param = param.exp()
                params[name] = param
        else:
            for param, name in zip(params0.unbind(1), fit_parameters):
                params[name] = param

        # sandi pass
        pred = sandi(**params, **self.fixed_parameters,
                     b=shells, Delta=Delta, delta=delta)

        # compute loss and metrics
        self.compute(_loss, _metric, pred=[pred, x])

        return params, pred


