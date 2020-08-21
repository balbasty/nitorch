"""Python utilities."""


def pad_list(input, n, default=None):
    """Pad/crop a list so that its length is `n`.

    Parameters
    ----------
    input : scalar or iterable
        Input argument(s).
    n : int
        Target length.
    default : optional
        Default value to pad with.  If None, replicate the last value.

    Returns
    -------
    output : list or tuple
        Output arguments.
        If the input argument is not a list or tuple, the output
        type is `tuple`.

    """
    if not isinstance(input, (list, tuple)):
        input = (input,)
    return_type = type(input)
    input = list(input)
    input = input[:min(n, len(input))]
    if default is None:
        default = input[-1]
    default += [default] * max(0, n - len(input))
    return return_type(input)


def rep_list(input, n, interleaved=False):
    """Replicate a list-like object.

    Parameters
    ----------
    input : scalar or iterable
        Input argument(s).
    n : int
        Number of replicates.
    interleaved : bool, default=False
        Interleaved replication.

    Returns
    -------
    output : list or tuple
        Replicated list.
        If the input argument is not a list or tuple, the output
        type is `tuple`.

    """
    if not isinstance(input, (list, tuple)):
        input = (input,)
    return_type = type(input)
    input = list(input)
    if interleaved:
        input = [elem for sub in zip(*([input]*n)) for elem in sub]
    else:
        input = input * n
    return return_type(input)
