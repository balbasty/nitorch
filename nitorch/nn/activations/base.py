import torch.nn as tnn

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
