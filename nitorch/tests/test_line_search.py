import torch

from nitorch.cli.registration.helpers import BacktrackingLineSearch


def make_problem(**kwargs):
    param = torch.tensor([3., -2.], requires_grad=True)
    optim = BacktrackingLineSearch(torch.optim.SGD([param], **kwargs))

    def closure():
        optim.zero_grad()
        loss = param.square().sum()
        loss.backward()
        return loss

    return param, optim, closure


def test_line_search_decreases_loss():
    param, optim, closure = make_problem(lr=0.1)
    loss0 = param.detach().square().sum()
    loss = optim.step(closure)
    assert optim.last_ok
    assert loss < loss0
    assert torch.allclose(loss, param.detach().square().sum())


def test_line_search_restores_params_on_failure():
    # steps go uphill, so no backtracking step can decrease the loss
    param, optim, closure = make_problem(lr=0.1, maximize=True)
    param0 = param.detach().clone()
    loss0 = param0.square().sum()
    loss = optim.step(closure)
    assert not optim.last_ok
    assert torch.equal(param.detach(), param0)
    assert torch.allclose(loss, loss0)
