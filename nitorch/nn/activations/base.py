import torch.nn as tnn
import inspect

_tnn_activations = [
    # linear units
    'ReLU',               # clip (, 0]
    'ReLU6',              # clip (, 0] and [6, )
    'LeakyReLU',          # mult factor for negative slope
    'PReLU',              # LeakyReLU with learnable factor
    'RReLU',              # LeakyReLU with random factor
    # sigmoid / softmax / soft functions
    'Sigmoid',
    'LogSigmoid',         # log(sigmoid)
    'Hardsigmoid',        # linear approximation of sigmoid
    'Softmax',            # multivariate sigmoid
    'LogSoftmax',         # log(softmax)
    # smooth approximations
    'Hardswish',          # 'smooth' RELU (quadratic)
    'Softplus',           # 'smooth' ReLU (logsumexp)
    'GELU',               # 'smooth' RELU (Gaussian cdf)
    'ELU',                # 'smooth' ReLU (exp-1)
    'SELU',               #               (scaled ELU)
    'CELU',               #               (~= SELU)
    'Softsign',           # 'smooth' sign function
    # shrinkage
    'Softshrink',         # soft-thresholding (subtract constant)
    'Hardshrink',         # clip [-lam, +lam]
    # tanh
    'Tanh',               # hyperbolic tangent
    'Hardtanh',           # linear approximation of tanh
    'Tanhshrink',         # shrink by tanh
]
_map_activations = {key.lower(): getattr(tnn, key) for key in _tnn_activations
                    if hasattr(tnn, key)}


def activation_from_name(name):
    """Return an activation Class from its name.

    Parameters
    ----------
    name : str
        Activation name. Registered activation functions are:
        - Rectified linear units:
            'ReLU'        : Clip (, 0]
            'ReLU6'       : Clip (, 0] and [6, )
            'LeakyReLU'   : Multiplicative factor for negative slope
            'PReLU'       : LeakyReLU with learnable factor
            'RReLU'       : LeakyReLU with random factor
        - Sigmoid, Softmax and Soft functions:
            'Sigmoid'     : Maps (-inf, +inf) to (0, 1)
            'LogSigmoid'  : log(sigmoid)
            'Hardsigmoid' : Linear approximation of sigmoid
            'Softmax'     : Multivariate sigmoid
            'LogSoftmax'  : log(softmax)
        - Smooth approximations:
            'Hardswish'   : Smooth RELU (quadratic)
            'Softplus'    : Smooth ReLU (logsumexp)
            'GELU'        : Smooth RELU (Gaussian cdf)
            'ELU'         : Smooth ReLU (exp-1)
            'SELU'        : Scaled ELU
            'CELU'        : (~= SELU)
            'Softsign'    : Smooth sign function
        - Shrinkage
            'Softshrink'  : Soft-thresholding (subtract a constant and clip)
            'Hardshrink'  : Clip [-lam, +lam]
        - Tanh
            'Tanh'        : Hyperbolic tangent
            'Hardtanh'    : Linear approximation of tanh
            'Tanhshrink'  : Shrink by tanh

    Returns
    -------
    Activation : type(Module)
        An activation class

    """
    name = name.lower()
    if name not in _map_activations:
        raise KeyError(f'Activation {name} is not registered.')
    return _map_activations[name]


def make_activation_from_name(name):
    """Return an instantiated activation from its name.

    Default values are used for functions that require parameters,

    Parameters
    ----------
    name : str
        Activation name. Registered activation functions are:
        - Rectified linear units:
            'ReLU'        : Clip (, 0]
            'ReLU6'       : Clip (, 0] and [6, )
            'LeakyReLU'   : Multiplicative factor (0.01) for negative slope
            'PReLU'       : Learnable LeakyReLU (common across channels, init=0.25)
            'RReLU'       : LeakyReLU with random factor (uniform in [1/8, 1/3])
        - Sigmoid, Softmax and Soft functions:
            'Sigmoid'     : Maps (-inf, +inf) to (0, 1)
            'LogSigmoid'  : log(sigmoid)
            'Hardsigmoid' : Linear approximation of sigmoid
            'Softmax'     : Multivariate sigmoid
            'LogSoftmax'  : log(softmax)
        - Smooth approximations:
            'Hardswish'   : Smooth RELU (quadratic)
            'Softplus'    : Smooth ReLU (logsumexp)
            'GELU'        : Smooth RELU (Gaussian cdf)
            'ELU'         : Smooth ReLU (exp-1)
            'SELU'        : Scaled ELU
            'CELU'        : (~= SELU)
            'Softsign'    : Smooth sign function
        - Shrinkage
            'Softshrink'  : Soft-thresholding (subtract 0.5 and clip)
            'Hardshrink'  : Clip [-0.5, +0.5]
        - Tanh
            'Tanh'        : Hyperbolic tangent
            'Hardtanh'    : Linear approximation of tanh
            'Tanhshrink'  : Shrink by tanh

    Returns
    -------
    activation : Module
        An instantiated activation module

    """
    klass = activation_from_name(name)
    if klass in (tnn.Softmax, tnn.Softmin, tnn.LogSoftmax):
        instance = klass(dim=1)
    else:
        instance = klass()
    return instance


def make_activation(activation):
    if not activation:
        return None
    if isinstance(activation, str):
        return make_activation_from_name(activation)
    if activation in (tnn.Softmax, tnn.Softmin, tnn.LogSoftmax):
        activation = activation(dim=1)
    else:
        activation = (activation() if inspect.isclass(activation)
                      else activation if callable(activation)
                      else None)
    return activation
