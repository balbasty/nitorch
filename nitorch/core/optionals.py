"""Check which optional modules are available."""
import importlib

# Numpy
try:
    import numpy
except ImportError:
    numpy = None

# Scipy
try:
    import scipy
except ImportError:
    scipy = None

# Matplotlib
try:
    import matplotlib
except ImportError:
    matplotlib = None


def try_import(path, keys=None, _as=True):
    """Try to import from a module.

    Parameters
    ----------
    path : str
        Path to module or variable in a module
    keys : str or list[str], optional
        Keys to load from the module
    _as : bool, defualt=True
        If False, perform recursive assignement as in
        >> # equivalent to: `import pack.sub.mod`
        >> pack = try_import('pack.sub.mod', _as=False)
        Else, it will look like a renamed import:
        >> # equivalent to: `import pack.sub.mod as my_mod`
        >> my_mod = try_import('pack.sub.mod', _as=True)


    Returns
    -------
    loaded_stuff : module or object or tuple
        A tuple is returned if `keys` is a list.
        Return None if import fails.

    """
    # check if the base package exists
    pack = path.split('.')[0]
    try:
        __import__(pack)
    except ImportError:
        if keys is None or isinstance(keys, str):
            return None
        else:
            keys = list(keys)
            return [None]*len(keys)

    if _as:
        # import a module
        module = importlib.import_module(path)
        # optional: extract attributes
        if keys is not None:
            if isinstance(keys, str):
                return getattr(module, keys)
            else:
                return tuple(getattr(module, key) for key in keys)
        return module
    else:
        # recursive import
        path = path.split('.')
        mod0 = importlib.import_module(path[0])
        cursor = mod0
        for i in range(1, len(path)):
            mod1 = importlib.import_module('.'.join(path[:i+1]))
            setattr(cursor, path[i], mod1)
            cursor = getattr(cursor, path[i])
        return mod0

