"""Python utilities."""


def make_list(input, n=None, *args, **kwargs):
    """Ensure that the input is a list and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence
        Input argument(s).
    n : int, optional
        Target length.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.

    Returns
    -------
    output : list
        Output arguments.

    """
    default = None
    has_default = False
    if len(args) > 0:
        default = args[0]
        has_default = True
    elif 'default' in kwargs.keys():
        default = kwargs['default']
        has_default = True

    if not isinstance(input, (list, tuple, range)):
        input = [input]
    input = list(input)
    if n is not None:
        input = input[:min(n, len(input))]
        if not has_default:
            default = input[-1]
        input += [default] * max(0, n - len(input))
    return input


def make_tuple(*args, **kwargs):
    """Ensure that the input is a tuple and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence
        Input argument(s).
    n : int, optional
        Target length.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.

    Returns
    -------
    output : tuple
        Output arguments.

    """
    return tuple(make_list(*args, **kwargs))


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


# backward compatibility
make_list = make_list
padlist = make_list
replist = rep_list


def getargs(kpd, args=None, kwargs=None, consume=False):
    """Read and remove argument from args/kwargs input.

    Args:
        kpd (list of tuple): List of (key, position, default) tuples with:
            key (str): argument name
            position (int): argument position
            default (optional): default value
        args (optional): list of positional arguments
        kwargs (optional): list of keyword arguments
        consume (bool, optional): consume arguments from args/kwargs

    Returns:
        values (list): List of values

    """

    args = [] if args is None else args
    kwargs = {} if kwargs is None else kwargs

    def raise_error(key):
        import inspect
        caller = inspect.stack()[1].function
        raise TypeError("{}() got multiple values for \
                        argument '{}}'".format(caller, key))

    # Sort argument by reverse position
    kpd = [(i,) + e for i, e in enumerate(kpd)]
    kpd = sorted(kpd, key=lambda x: x[2], reverse=True)

    values = []
    for elem in kpd:
        i = elem[0]
        key = elem[1]
        position = elem[2]
        default = elem[3] if len(elem) > 3 else None

        value = default
        if len(args) >= position:
            value = args[-1]
            if consume:
                del args[-1]
            if key in kwargs.keys():
                raise_error(key)
        elif key in kwargs.keys():
            value = kwargs[key]
            if consume:
                del kwargs[key]
        values.append((i, value))

    values = [v for _, v in sorted(values)]
    return values


def prod(sequence, inplace=False):
    """Perform the product of a sequence of elements.

    Parameters
    ----------
    sequence : any object that implements `__iter__`
        Sequence of elements for which the `__mul__` operator is defined.
    inplace : bool, default=False
        Perform the product inplace (using `__imul__` instead of `__mul__`).

    Returns
    -------
    product :
        Product of the elements in the sequence.

    """
    accumulate = None
    for elem in sequence:
        if accumulate is None:
            accumulate = elem
        elif inplace:
            accumulate *= elem
        else:
            accumulate = accumulate * elem
    return accumulate
