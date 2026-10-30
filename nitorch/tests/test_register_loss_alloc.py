"""Allocation-related checks of the EMMI and robust (MAD/Tukey) losses.

- The in-place / buffer-reusing implementation returns the same value,
  gradient and Hessian as the previous (allocating) one. The reference
  numbers below were computed with that previous implementation, from the
  same seeded inputs. They are summaries: the value, and the dot products
  of the gradient and Hessian with fixed random tensors.
- None of the loss functions modifies its inputs in place.
"""
import importlib
import torch
import pytest

emmi_mod = importlib.import_module('nitorch.tools.registration.losses.emmi')
robust_mod = importlib.import_module('nitorch.tools.registration.losses.robust')

SHAPE = [7, 6, 5]


def emmi_inputs(soft, K, J, batch, dt, mask, seed):
    g = torch.Generator().manual_seed(seed)
    B = [2] if batch else []
    mu = torch.rand([*B, K, *SHAPE], dtype=dt, generator=g).add_(.05)
    mu /= mu.sum(-4, keepdim=True)
    if soft:
        q = torch.rand([*B, J, *SHAPE], dtype=dt, generator=g).add_(.05)
        q[..., :K, :, :, :] += 2 * mu
        fix = q / q.sum(-4, keepdim=True)
    else:
        fix = torch.randint(0, J, [*B, 1, *SHAPE], generator=g)
    w = None
    if mask == 'bin':
        w = (torch.rand([*B, 1, *SHAPE], generator=g) > .2).to(dt)
    elif mask == 'soft':
        w = torch.rand([*B, 1, *SHAPE], dtype=dt, generator=g)
    return mu, fix, w


def robust_inputs(K, batch, dt, mask, zeros, seed):
    g = torch.Generator().manual_seed(seed)
    B = [3] if batch else []
    shape = [9, 8, 7]
    fix = torch.randn([*B, K, *shape], dtype=dt, generator=g)
    mov = fix + 0.3 * torch.randn([*B, K, *shape], dtype=dt, generator=g)
    mov[torch.rand([*B, K, *shape], generator=g) < .05] += 5   # outliers
    if zeros:
        mov.view(-1)[:zeros] = fix.view(-1)[:zeros]            # r = 0
    m = None
    if mask == 'bin':
        m = (torch.rand([*B, 1, *shape], generator=g) > .2).to(dt)
    elif mask == 'soft':
        m = torch.rand([*B, 1, *shape], dtype=dt, generator=g)
    return mov, fix, m


def project(x, seed):
    """<x, u> for a fixed random u (sensitive to every element)"""
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(x.shape, dtype=torch.double, generator=g)
    return (x.double() * u).sum().item()


# (soft, K, J, batch, dtype, mask, fwhm, max_iter)
EMMI_CASES = {
    'hard': (False, 3, 6, False, torch.double, 'bin', None, 8),
    'hard-batch-fwhm': (False, 4, 8, True, torch.double, 'soft', 1.5, 32),
    'hard-fixedH': (False, 3, 5, False, torch.double, None, None, 0),
    'soft': (True, 3, 5, False, torch.double, 'bin', None, 8),
    'soft-batch-fwhm': (True, 4, 6, True, torch.double, 'soft', 1.5, 32),
    'soft-float': (True, 3, 6, False, torch.float, None, None, 8),
}

# (cls, lam, joint, K, batch, dtype, mask, zeros)
ROBUST_CASES = {
    'mad-est-mask-zeros': ('MAD', None, False, 1, False, torch.double, 'bin', 20),
    'mad-est-joint-batch': ('MAD', None, True, 2, True, torch.double, None, 0),
    'mad-fixed': ('MAD', 2., False, 2, False, torch.double, 'soft', 0),
    'tukey-est-softmask': ('Tukey', None, False, 2, False, torch.double, 'soft', 0),
    'tukey-est-joint-batch': ('Tukey', None, True, 2, True, torch.double, 'bin', 20),
    'tukey-fixed-batch': ('Tukey', 3., False, 1, True, torch.double, None, 0),
    'tukey-est-float': ('Tukey', None, False, 1, False, torch.float, 'bin', 0),
}


def emmi_summary(name, emmi=None):
    emmi = emmi or emmi_mod.emmi
    soft, K, J, batch, dt, mask, fwhm, max_iter = EMMI_CASES[name]
    mu, fix, w = emmi_inputs(soft, K, J, batch, dt, mask, seed=len(name))
    ll, g, h, prior = emmi(mu, fix, dim=3, weights=w, fwhm=fwhm,
                           max_iter=max_iter, return_prior=True)
    return [ll.item(), project(g, 1), project(h, 2), project(prior, 3)]


def robust_summary(name, module=None):
    module = module or robust_mod
    cls, lam, joint, K, batch, dt, mask, zeros = ROBUST_CASES[name]
    mov, fix, m = robust_inputs(K, batch, dt, mask, zeros, seed=len(name))
    loss = getattr(module, cls)(lam=lam, joint=joint, dim=3)
    kw = dict(mask=m) if m is not None else {}
    out = []
    for _ in range(2):      # the second call starts from the cached lam
        ll, g, h = loss.loss_grad_hess(mov, fix, **kw)
        out += [ll.item(), project(g, 1), project(h, 2)]
    if torch.is_tensor(loss.lam):
        out += [project(loss.lam, 3)]
    return out


# Computed with the previous (allocating) implementation of
# losses/emmi.py and losses/robust.py, at a480672.
EMMI_REF = {
    'hard': [
        -0.020602131648001044, -1.5202935154335444, 2.9471420796192156,
        7.1399897742197265,
    ],
    'hard-batch-fwhm': [
        -0.014838435925538355, -1.9826975038904893, 4.882773699113675,
        26.797387674821685,
    ],
    'hard-fixedH': [
        -1.000050835792425e-12, -1.53496201344059, 2.8873458046862828,
        6.987786671961986,
    ],
    'soft': [
        -0.01817701032600183, -1.5093880491804073, 3.007260704662468,
        6.952909501566639,
    ],
    'soft-batch-fwhm': [
        -0.006922167597786988, -1.9707121466310413, 4.898773332159026,
        21.098120226578068,
    ],
    'soft-float': [
        -0.017862090474766643, -1.5167362274521308, 2.9982551529113866,
        7.724932344978009,
    ],
}
ROBUST_REF = {
    'mad-est-mask-zeros': [
        -0.05305157488474621, 0.05692833453346885, 13271.88983809347,
        -0.053051597055963406, 0.056928346423792305, 13271.895379482678,
        0.2266346690350638,
    ],
    'mad-est-joint-batch': [
        0.4083397270097402, 0.3432534645792609, 28.733262765562635,
        0.40833972700920285, 0.3432535483426967, 28.733269801187582,
        18.020723179402424,
    ],
    'mad-fixed': [
        1.7363569057348955, -0.005116693182795853, 38.903405801162094,
        1.7363569057348955, -0.005116693182795853, 38.903405801162094,
    ],
    'tukey-est-softmask': [
        -0.36085181224162066, -0.05681160192510708, 9.555091942812354,
        -0.360851822820093, -0.056811604433793864, 9.5550927599893,
        3.371224024593063,
    ],
    'tukey-est-joint-batch': [
        -2.7353914775724006, 0.032553518722601635, 27.284085573960557,
        -2.7353915080536293, 0.03255348622874257, 27.284089981851977,
        43.58774327120846,
    ],
    'tukey-fixed-batch': [
        0.8772765390904091, -0.007075631266697005, 4.010889042697366,
        0.8772765390904091, -0.007075631266697005, 4.010889042697366,
    ],
    'tukey-est-float': [
        -0.44400542974472046, 0.006739408100812748, 3.702968470724725,
        -0.4440053403377533, 0.006739410977890379, 3.7029690690314814,
        0.37771413833632306,
    ],
}


def tol(dt):
    return dict(rel=1e-5) if dt == torch.float else dict(rel=1e-10, abs=1e-12)


@pytest.mark.parametrize('name', list(EMMI_CASES))
def test_emmi_same_as_before(name):
    dt = EMMI_CASES[name][4]
    assert emmi_summary(name) == pytest.approx(EMMI_REF[name], **tol(dt))


@pytest.mark.parametrize('name', list(ROBUST_CASES))
def test_robust_same_as_before(name):
    dt = ROBUST_CASES[name][5]
    assert robust_summary(name) == pytest.approx(ROBUST_REF[name], **tol(dt))


def _check_unchanged(fn, *inputs):
    copies = [x.clone() if torch.is_tensor(x) else x for x in inputs]
    fn()
    for x, x0 in zip(inputs, copies):
        if torch.is_tensor(x):
            assert torch.equal(x, x0)


@pytest.mark.parametrize('soft', [False, True])
@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('method', ['loss', 'loss_grad', 'loss_grad_hess'])
def test_emmi_inputs_unchanged(soft, batch, method):
    mu, fix, w = emmi_inputs(soft, 3, 5, batch, torch.double, 'soft', 0)
    loss = emmi_mod.EMMI(dim=3, fwhm=1.)
    for _ in range(2):      # second call: warm start from the cached prior
        prior = getattr(loss, '_prior', None)

        def fn():
            getattr(loss, method)(mu, fix, mask=w)
        _check_unchanged(fn, mu, fix, w, prior)


@pytest.mark.parametrize('cls', ['MAD', 'Tukey'])
@pytest.mark.parametrize('lam', [None, 2.])
@pytest.mark.parametrize('joint', [False, True])
@pytest.mark.parametrize('method', ['loss', 'loss_grad', 'loss_grad_hess'])
def test_robust_inputs_unchanged(cls, lam, joint, method):
    mov, fix, m = robust_inputs(2, True, torch.double, 'bin', 10, 0)
    loss = getattr(robust_mod, cls)(lam=lam, joint=joint, dim=3)
    for _ in range(2):      # second call: starts from the cached lam

        def fn():
            getattr(loss, method)(mov, fix, mask=m)
        _check_unchanged(fn, mov, fix, m, loss.lam)
