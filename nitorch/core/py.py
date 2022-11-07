"""Python utilities."""
import os
import functools
from types import GeneratorType as generator
import warnings
from collections import Counter
from typing import List, Tuple, Iterable


def file_mod(s, nam=None, prefix='', suffix='', odir=None, ext=None):
    """Modify a file path.

    Parameters
    ----------
    s : str
        File path.
    nam : str, optional
        New basename (without extension). Default: same as input.
    prefix : str, default=''
        Filename prefix.
    suffix : str, default=''
        Filename suffix.
    odir : str, optional
        Output directory. Default: same as input.
    ext : str, optional
        New extension (with leading dot). Default: same as input.

    Returns
    ----------
    s : str
        Modified file path.

    """
    # Deprecated -> should use `file_replace`
    return file_replace(s, base=nam, prefix=prefix, suffix=suffix,
                        dir=odir, ext=ext)


def file_replace(fname, base=None, prefix='', dir=None, suffix='', ext=None):
    """Modify a file path.

    Parameters
    ----------
    fname : str
        Input file path.
    base : str, optional
        New basename (without extension). Default: same as input.
    prefix : str, default=''
        New basename prefix.
    suffix : str, default=''
        New basename suffix.
    dir : str, optional
        Output directory. Default: same as input.
    ext : str, optional
        New extension (with leading dot). Default: same as input.

    Returns
    ----------
    s : str
        Modified file path.

    """
    odir0, nam0 = os.path.split(fname)
    parts = nam0.split('.')
    nam0 = parts[0]
    ext0 = '.' + '.'.join(parts[1:])
    if dir is None:
        dir = odir0
    dir = os.path.abspath(dir)  # Get absolute path
    if base is None:
        base = nam0
    if ext is None:
        ext = ext0

    return os.path.join(dir, prefix + base + suffix + ext)


def fileparts(fname):
    """Extracts parts from filename

    Parameters
    ----------
    fname : str

    Returns
    -------
    dir : str
        Directory path
    base : str
        Base name, without extension
    ext : str
        Extension, with leading dot

    """
    dir = os.path.dirname(fname)
    base = os.path.basename(fname)
    base, ext = os.path.splitext(base)
    if ext.lower() in ('.gz', '.bz2'):
        base, ext0 = os.path.splitext(base)
        ext = ext0 + ext
    return dir, base, ext


def make_sequence(input, n=None, crop=True, *args, **kwargs) -> Iterable:
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
        if len(input) == 0 and n and not has_default:
            raise ValueError('Empty sequence')
        if n is not None:
            if crop:
                input = input[:min(n, len(input))]
            if len(input) < n:
                if not has_default:
                    default = input[-1]
                input += [default] * (n - len(input))
        return return_type(input)


@functools.wraps(make_sequence, assigned=[])
def make_list(*args, **kwargs) -> List:
    """Ensure that the input is a list and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence generator
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
    output : list
        Output arguments.

    """
    return list(elem for elem in make_sequence(*args, **kwargs))


def ensure_list(x, dim=None):
    """Ensure that an object is a list (of size at last dim)

    If x is a list, nothing is done (no copy triggered).
    If it is a tuple, it is converted into a list.
    Otherwise, it is placed inside a list.

    This function is less versatile (but much faster) than `make_list`.
    """
    if not isinstance(x, (list, tuple)):
        x = [x]
    elif isinstance(x, tuple):
        x = list(x)
    if dim and len(x) < dim:
        x += x[-1:] * (dim - len(x))
    return x


@functools.wraps(make_sequence, assigned=[])
def make_tuple(*args, **kwargs) -> Tuple:
    """Ensure that the input is a tuple and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence generator
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
    output : tuple
        Output arguments.

    """
    return tuple(elem for elem in make_sequence(*args, **kwargs))


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


def cumprod(sequence, reverse=False, exclusive=False):
    """Perform the cumulative product of a sequence of elements.

    Parameters
    ----------
    sequence : any object that implements `__iter__`
        Sequence of elements for which the `__mul__` operator is defined.
    reverse : bool, default=False
        Compute cumulative product from right-to-left:
        `cumprod([a, b, c], reverse=True) -> [a*b*c, b*c, c]`
    exclusive : bool, default=False
        Exclude self from the cumulative product:
        `cumprod([a, b, c], exclusive=True) -> [1, a, a*b]`

    Returns
    -------
    product : list
        Product of the elements in the sequence.

    """
    if reverse:
        sequence = reversed(sequence)
    accumulate = None
    seq = [1] if exclusive else []
    for elem in sequence:
        if accumulate is None:
            accumulate = elem
        else:
            accumulate = accumulate * elem
        seq.append(accumulate)
    if exclusive:
        seq = seq[:-1]
    if reverse:
        seq = list(reversed(seq))
    return seq


def cumsum(sequence, reverse=False, exclusive=False):
    """Perform the cumulative sum of a sequence of elements.

    Parameters
    ----------
    sequence : any object that implements `__iter__`
        Sequence of elements for which the `__sum__` operator is defined.
    reverse : bool, default=False
        Compute cumulative product from right-to-left:
        `cumprod([a, b, c], reverse=True) -> [a+b+c, b+c, c]`
    exclusive : bool, default=False
        Exclude self from the cumulative product:
        `cumprod([a, b, c], exclusive=True) -> [0, a, a+b]`

    Returns
    -------
    sum : list
        Sum of the elements in the sequence.

    """
    if reverse:
        sequence = reversed(sequence)
    accumulate = None
    seq = [0] if exclusive else []
    for elem in sequence:
        if accumulate is None:
            accumulate = elem
        else:
            accumulate = accumulate + elem
        seq.append(accumulate)
    if exclusive:
        seq = seq[:-1]
    if reverse:
        seq = list(reversed(seq))
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


def flatten(x):
    """Flatten nested sequences

    Parameters
    ----------
    x : tuple or list
        Nested sequence

    Returns
    -------
    x : tuple or list
        Flattened sequence

    """
    def _flatten(y):
        if isinstance(y, (list, tuple)):
            out = []
            for e in y:
                out.extend(_flatten(e))
            return out
        else:
            return [y]

    input_type = type(x)
    return input_type(_flatten(x))


def expand_list(x, n, crop=False, default=None):
    """Expand ellipsis in a list by substituting it with the value
    on its left, repeated as many times as necessary. By default,
    a "virtual" ellipsis is present at the end of the list.

    expand_list([1, 2, 3],       5)            -> [1, 2, 3, 3, 3]
    expand_list([1, 2, ..., 3],  5)            -> [1, 2, 2, 2, 3]
    expand_list([1, 2, 3, 4, 5], 3, crop=True) -> [1, 2, 3]

    Parameters
    ----------
    x : sequence
    n : int
        Target length
    crop : bool, default=False
        Whether to crop the list if it is longer than the target length
    default : optional
        default value to use if the ellipsis is the first element of the list

    Returns
    -------
    x : list
        List of length `n` (or `>= n`)
    """
    x = list(x)
    if Ellipsis not in x:
        x.append(Ellipsis)
    idx_ellipsis = x.index(Ellipsis)
    if idx_ellipsis == 0:
        fill_value = default
    else:
        fill_value = x[idx_ellipsis-1]
    k = len(x) - 1
    x = (x[:idx_ellipsis] +
         [fill_value] * max(0, n-k) +
         x[idx_ellipsis+1:])
    if crop:
        x = x[:n]
    return x


def move_to_permutation(length, source, destination):

    source = make_list(source)
    destination = make_list(destination)
    if len(destination) == 1:
        # we assume that the user wishes to keep moved dimensions
        # in the order they were provided
        destination = destination[0]
        if destination >= 0:
            destination = list(range(destination, destination+len(source)))
        else:
            destination = list(range(destination+1-len(source), destination+1))
    if len(source) != len(destination):
        raise ValueError('Expected as many source as destination positions.')
    source = [length + src if src < 0 else src for src in source]
    destination = [length + dst if dst < 0 else dst for dst in destination]
    if len(set(source)) != len(source):
        raise ValueError(f'Expected source positions to be unique but got '
                         f'{source}')
    if len(set(destination)) != len(destination):
        raise ValueError(f'Expected destination positions to be unique but got '
                         f'{destination}')

    # compute permutation
    positions_in = list(range(length))
    positions_out = [None] * length
    for src, dst in zip(source, destination):
        positions_out[dst] = src
        positions_in[src] = None
    positions_in = filter(lambda x: x is not None, positions_in)
    for i, pos in enumerate(positions_out):
        if pos is None:
            positions_out[i], *positions_in = positions_in

    return positions_out


def move_elem(x, source, destination):
    """Move the location of one or several elements in a list

    Parameters
    ----------
    x : list
        Input list
    source : [sequence of] int
        Input indices
    destination : [sequence of] int
        Output indices

    Returns
    -------
    x : list
        Permuted list

    """
    return [x[i] for i in move_to_permutation(len(x), source, destination)]


def argmax(x):
    """Return the index of the maximum element in an iterable

    Parameters
    ----------
    x : sequence of numbers

    Returns
    -------
    index : int
    """
    i = None
    v = -float('inf')
    for j, e in enumerate(x):
        if e > v:
            i = j
            v = e
    return i


def argmin(x):
    """Return the index of the minimum element in an iterable

    Parameters
    ----------
    x : sequence of numbers

    Returns
    -------
    index : int
    """
    i = None
    v = float('inf')
    for j, e in enumerate(x):
        if e < v:
            i = j
            v = e
    return i
