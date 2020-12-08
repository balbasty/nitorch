"""Python utilities."""
import os
import functools
from types import GeneratorType as generator
import warnings
from collections import Counter


def file_mod(s, nam='', prefix='', suffix='', odir='', ext=''):
    """Modify a file path.

    Parameters
    ----------
    s : str
        File path.
    nam : str, default=''
        Filename, if empty string, unchanged.
    prefix : str, default=''
        Filename prefix.
    suffix : str, default=''
        Filename suffix.
    odir : str, default=''
        Output directory, if empty string, unchanged.
    ext : str, default=''
        Extension, if empty string, unchanged.

    Returns
    ----------
    s : str
        Modified file path.

    """
    odir0, nam0 = os.path.split(s)
    parts = nam0.split('.')
    nam0 = parts[0]
    ext0 = '.' + '.'.join(parts[1:])
    if not odir:
        odir = odir0
    odir = os.path.abspath(odir)  # Get absolute path
    if not nam:
        nam = nam0
    if not ext:
        ext = ext0

    return os.path.join(odir, prefix + nam + suffix + ext)


def make_sequence(input, n=None, crop=True, *args, **kwargs):
    """Ensure that the input is a sequence and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence or generator
        Input argument(s).
    n : int, optional
        Target length.
    crop : bool, default=True
        Crop input sequence if longer than `n`.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.

    Returns
    -------
    output : list or tuple or generator
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

    if isinstance(input, generator):
        # special case for generators
        def make_gen():
            last = None
            i = None
            for i, elem in input:
                if crop and (i == n):
                    return
                last = elem
                yield elem
            if i is None:
                if n is None:
                    return
                if not has_default:
                    raise ValueError('Empty sequence')
                last = default
            for j in range(i+1, n):
                yield last
        return make_gen()
    else:
        # generic case -> induces a copy
        if not isinstance(input, (list, tuple, range)):
            input = [input]
        return_type = type(input) if isinstance(input, (list, tuple)) else list
        input = list(input)
        if len(input) == 0 and n is not None and not has_default:
            raise ValueError('Empty sequence')
        if n is not None:
            if crop:
                input = input[:min(n, len(input))]
            if not has_default:
                default = input[-1]
            input += [default] * max(0, n - len(input))
        return return_type(input)


def make_list(*args, **kwargs) -> list:
    """Ensure that the input is a list and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence generator
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
    return list(make_sequence(*args, **kwargs))


def make_tuple(*args, **kwargs) -> tuple:
    """Ensure that the input is a tuple and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence generator
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
    return tuple(make_sequence(*args, **kwargs))


def make_set(input) -> set:
    """Ensure that the input is a set.

    Parameters
    ----------
    input : scalar or sequence
        Input argument(s).

    Returns
    -------
    output : set
        Output arguments.

    """
    if not isinstance(input, (list, tuple, set, range, generator)):
        input = [input]
    return set(input)


def rep_sequence(input, n, interleaved=False):
    """Replicate a sequence.

    Parameters
    ----------
    input : scalar or iterable generator
        Input argument(s).
    n : int
        Number of replicates.
    interleaved : bool, default=False
        Interleaved replication.

    Returns
    -------
    output : list or tuple or generator
        Replicated list.
        If the input argument is not a list or tuple, the output
        type is `tuple`.

    """
    if isinstance(input, generator):
        # special case for generators
        if interleaved:
            def make_gen():
                for elem in input:
                    for _ in range(n):
                        yield elem
            return make_gen()
        else:
            warnings.warn('It is not efficient to replicate a generator '
                          'this way. We are holding *all* the data in '
                          'memory.', RuntimeWarning)
            input = list(input)

    # generic case for sequence -> induces a copy
    if not isinstance(input, (list, tuple, range)):
        input = [input]
    return_type = type(input) if isinstance(input, (list, tuple)) else list
    input = list(input)
    if interleaved:
        input = [elem for sub in zip(*([input]*n)) for elem in sub]
    else:
        input = input * n
    return return_type(input)


def rep_list(input, n, interleaved=False) -> list:
    """Replicate a list.

    Parameters
    ----------
    input : scalar or sequence or generator
        Input argument(s).
    n : int
        Number of replicates.
    interleaved : bool, default=False
        Interleaved replication.

    Returns
    -------
    output : list
        Replicated list.
        If the input argument is not a list or tuple, the output
        type is `tuple`.

    """
    return list(rep_sequence(input, n, interleaved))


# backward compatibility
padlist = functools.wraps(make_sequence)
replist = functools.wraps(rep_sequence)


def getargs(kpd, args=None, kwargs=None, consume=False):
    """Read and remove argument from args/kwargs input.

    Parameters
    ----------
    kpd : list of tuple
        List of (key, position, default) tuples with:
            * key (str): argument name
            * position (int): argument position
            * default (optional): default value
    args : sequence, optional
        List of positional arguments
    kwargs : dict, optional
        List of keyword arguments
    consume : bool, default=False
        Consume arguments from args/kwargs

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


def cumprod(sequence):
    """Perform the cumulative product of a sequence of elements.

    Parameters
    ----------
    sequence : any object that implements `__iter__`
        Sequence of elements for which the `__mul__` operator is defined.

    Returns
    -------
    product :
        Product of the elements in the sequence.

    """
    accumulate = None
    seq = []
    for elem in sequence:
        if accumulate is None:
            accumulate = elem
        else:
            accumulate = accumulate * elem
        seq.append(accumulate)
    return seq


def pop(obj, key=0, *args, **kwargs):
    """Pop an element from a mutable collection.

    Parameters
    ----------
    obj : dict or list
        Collection
    key : str or int
        Key or index
    default : optional
        Default value. Raise error if not provided.

    Returns
    -------
    elem
        Popped element

    """
    if isinstance(obj, dict):
        return obj.pop(key, *args, **kwargs)
    else:
        try:
            val = obj[key]
            del obj[key]
            return val
        except:
            if len(args) > 0:
                return args[0]
            else:
                return kwargs.get('default')


def majority(x):
    """Return majority element in a sequence.

    Parameters
    ----------
    x : sequence
        Input sequence of hashable elements

    Returns
    -------
    elem
        Majority element

    """
    count = Counter(x)
    return count.most_common(1)[0][0]
