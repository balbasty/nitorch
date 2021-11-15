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

# torch amp (not there in all versions)
try:
    from torch.cuda.amp import custom_fwd, custom_bwd
except ImportError:
    custom_fwd = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda x: x)
    custom_bwd = lambda *a, **k: a[0] if a and callable(a[0]) else (lambda x: x)


def try_import(path, keys=None, _as=False):
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
    def fail(keys):
        if keys is None or isinstance(keys, str):
            return None
        else:
            keys = list(keys)
            return [None]*len(keys)
        
    def try_import_module(path):
        try:
            return importlib.import_module(path)
        except (ImportError, ModuleNotFoundError):
            return None

    # check if the base package exists
    pack = path.split('.')[0]
    try:
        __import__(pack)
    except (ImportError, ModuleNotFoundError):
        return fail(keys)

    if _as:
        # import a module
        module = try_import_module(path)
        if not module:
            return fail(keys)
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
        mod0 = try_import_module(path[0])
        if not mod0:
            return fail(keys)
        cursor = mod0
        for i in range(1, len(path)):
            mod1 = try_import_module('.'.join(path[:i+1]))
            if not mod1:
                return fail(keys)
            setattr(cursor, path[i], mod1)
            cursor = getattr(cursor, path[i])
        return mod0


def try_import_as(path, keys=None):
    return try_import(path, keys, _as=True)
