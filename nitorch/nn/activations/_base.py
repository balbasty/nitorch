import torch.nn as tnn

_tnn_activations = [
    # linear units
    'ReLU',                   # clip (, 0]
    'ReLU6',                  # clip (, 0] and [6, )
    'tnn.LeakyReLU',          # mult factor for negative slope
    'tnn.PReLU',              # LeakyReLU with learnable factor
    'tnn.RReLU',              # LeakyReLU with random factor
    # sigmoid / softmax / soft functions
    'tnn.Sigmoid',
    'tnn.LogSigmoid',         # log(sigmod)
    'tnn.Hardsigmoid',        # linear approximation of sigmoid
    'tnn.Softmax',            # multivariate sigmoid
    'tnn.LogSoftmax',         # log(softmax)
    # smooth approximations
    'tnn.Hardswish',          # 'smooth' RELU (quadratic)
    'tnn.Softplus',           # 'smooth' ReLU (logsumexp)
    'tnn.GELU',               # 'smooth' RELU (Gaussian cdf)
    'tnn.ELU',                # 'smooth' ReLU (exp-1)
    'tnn.SELU',               #               (scaled ELU)
    'tnn.CELU',               #               (~= SELU)
    'tnn.Softsign',           # 'smooth' sign function
    # shrinkage
    'tnn.Softshrink',         # soft-thresholding (subtract constant)
    'tnn.Hardshrink',         # clip [-lam, +lam]
    # tanh
    'tnn.Tanh',               # hyperbolic tangent
    'tnn.Hardtanh',           # linear approximation of tanh
    'tnn.Tanhshrink',         # shrink by tanh
]
_map_activations = {key.lower(): getattr(tnn, key) for key in _tnn_activations
                    if hasattr(tnn, key)}
