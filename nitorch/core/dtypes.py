"""Information and conversion of low-level data types.

Warning: We do not deal with numpy's byte orders here.
"""

from ..core.optionals import numpy as np
import torch

npdtype = getattr(np, 'dtype', lambda *args: None)


_dtypes = {
    'bool': {
        'str': ['bool'],
        'torch': torch.bool,
        'numpy': npdtype('bool'),
        'python': bool,
        'bytes': 1,
        'is_complex': False,
        'is_floating_point': False,
        'is_integer': False,
        'is_signed': False,
        'min': 0,
        'max': 1,
    },
    'uint8': {
        'str': ['uint8', 'char'],
        'torch': torch.uint8,
        'numpy': npdtype('uint8'),
        'python': None,
        'python_upcast': int,
        'bytes': 1,
        'is_complex': False,
        'is_floating_point': False,
        'is_integer': True,
        'is_signed': False,
        'min': 0,
        'max': 255,
    },
    'int8': {
        'str': ['int8'],
        'torch': torch.int8,
        'numpy': getattr(np, 'int8', None),
        'python': None,
        'python_upcast': int,
        'bytes': 1,
        'is_complex': False,
        'is_floating_point': False,
        'is_integer': True,
        'is_signed': True,
        'min': -128,
        'max': 127,
    },
    'uint16': {
        'str': ['uint16', 'unsigned short'],
        'torch': None,
        'torch_upcast': torch.int32,
        'numpy': npdtype('uint16'),
        'python': None,
        'python_upcast': int,
        'bytes': 2,
        'is_complex': False,
        'is_floating_point': False,
        'is_integer': True,
        'is_signed': False,
        'min': 0,
        'max': 65535,
    },
    'int16': {
        'str': ['int16', 'short'],
        'torch': torch.int16,
        'numpy': npdtype('int16'),
        'python': None,
        'python_upcast': int,
        'bytes': 1,
        'is_complex': False,
        'is_floating_point': False,
        'is_integer': True,
        'is_signed': True,
        'min': -32768,
        'max': 32767,
    },
    'uint32': {
        'str': ['uint32', 'unsigned', 'unsigned int'],
        'torch': None,
        'torch_upcast': torch.int64,
        'numpy': npdtype('uint32'),
        'python': None,
        'python_upcast': int,
        'bytes': 4,
        'is_complex': False,
        'is_floating_point': False,
        'is_integer': True,
        'is_signed': False,
        'min': 0,
        'max': 4294967295,
    },
    'int32': {
        'str': ['int32', 'int'],
        'torch': torch.int32,
        'numpy': npdtype('int32'),
        'python': None,
        'python_upcast': int,
        'bytes': 4,
        'is_complex': False,
        'is_floating_point': False,
        'is_integer': True,
        'is_signed': True,
        'min': -2147483648,
        'max': 2147483647,
    },
    'uint64': {
        'str': ['uint64', 'unsigned long'],
        'torch': None,
        'torch_upcast': torch.float64,
        'numpy': npdtype('uint64'),
        'python': None,
        'python_upcast': float,
        'bytes': 8,
        'is_complex': False,
        'is_floating_point': False,
        'is_integer': True,
        'is_signed': False,
        'min': 0,
        'max': 18446744073709551615,
    },
    'int64': {
        'str': ['int64', 'long'],
        'torch': torch.int64,
        'numpy': npdtype('int64'),
        'python': int,
        'bytes': 8,
        'is_complex': False,
        'is_floating_point': False,
        'is_integer': True,
        'is_signed': True,
        'min': -9223372036854775808,
        'max': 9223372036854775807,
    },
    'float16': {
        'str': ['float16', 'half'],
        'torch': torch.half,
        'numpy': npdtype('float16'),
        'python': None,
        'python_upcast': float,
        'bytes': 2,
        'is_complex': False,
        'is_floating_point': True,
        'is_integer': False,
        'is_signed': True,
        'min': -6.55040e+04,
        'max': 6.55040e+04,
        'eps': 1e-03,
    },
    'float32': {
        'str': ['float32', 'single'],
        'torch': torch.float,
        'numpy': npdtype('float32'),
        'python': None,
        'python_upcast': float,
        'bytes': 4,
        'is_complex': False,
        'is_floating_point': True,
        'is_integer': False,
        'is_signed': True,
        'min': -3.4028235e+38,
        'max': 3.4028235e+38,
        'eps': 1e-06,
    },
    'float64': {
        'str': ['float64', 'double'],
        'torch': torch.double,
        'numpy': npdtype('float64'),
        'python': float,
        'bytes': 8,
        'is_complex': False,
        'is_floating_point': True,
        'is_integer': False,
        'is_signed': True,
        'min': -1.7976931348623157e+308,
        'max': 1.7976931348623157e+308,
        'eps': 1e-15,
    },
    'complex32': {
        'str': ['complex32'],
        'torch': torch.complex32,
        'numpy': None,
        'numpy_upcast': npdtype('complex64'),
        'python': None,
        'python_upcast': complex,
        'bytes': 4,
        'is_complex': True,
        'is_floating_point': True,
        'is_integer': False,
        'is_signed': True,
        'min': -6.55040e+04,
        'max': 6.55040e+04,
        'eps': 1e-03,
    },
    'complex64': {
        'str': ['complex64'],
        'torch': torch.complex64,
        'numpy': npdtype('complex64'),
        'python': None,
        'python_upcast': complex,
        'bytes': 8,
        'is_complex': True,
        'is_floating_point': True,
        'is_integer': False,
        'is_signed': True,
        'min': -3.4028235e+38,
        'max': 3.4028235e+38,
        'eps': 1e-06,
    },
    'complex128': {
        'str': ['complex128'],
        'torch': torch.complex128,
        'numpy': npdtype('complex128'),
        'python': complex,
        'bytes': 16,
        'is_complex': True,
        'is_floating_point': True,
        'is_integer': False,
        'is_signed': True,
        'min': -1.7976931348623157e+308,
        'max': 1.7976931348623157e+308,
        'eps': 1e-15,
    },
}


def info(dtype):
    """Return information on a datatype.

    Parameters
    ----------
    dtype : str or type
        Generic data type (numpy, torch, python, string)

    Returns
    -------
    info : dict
        Dictionary with [optional] fields:
            *  'str'               : list of str
            *  'torch'             : torch.dtype or None
           [*] 'torch_upcast'      : torch.dtype
            *  'numpy'             : np.dtype or None
           [*] 'numpy_upcast'      : np.dtype
            *  'python'            : type
           [*] 'python_upcast'     : type
            *  'bytes'             : int
            *  'is_floating_point' : bool
            *  'is_integer'        : bool
            *  'is_signed'         : bool
            *  'min'               : int or float
            *  'max'               : int or float

    Raises
    ------
    ValueError
        If the input data type cannot be found in the dictionary.

    """
    def isin(dtype, list_dtype):
        found = False
        for dtype2 in list_dtype:
            if dtype2 is None:
                # Some np dtypes do compare positively to `None`
                # Since we use `None` to specify non-implemented types,
                # we need to treat it separately.
                continue
            try:
                found = dtype == dtype2
                if found:
                    break
            except (ValueError, TypeError):
                found = False
        return found

    dinfo = None
    for key, val in _dtypes.items():
        all_dtypes = (val['numpy'], val['torch'], val['python'], *val['str'])
        found = isin(dtype, all_dtypes)
        if found:
            dinfo = val
            break
        # try byteswapped version for numpy types
        if hasattr(val['numpy'], 'newbyteorder'):
            try:
                if dtype == val['numpy'].newbyteorder():
                    dinfo = val
                    break
            except TypeError:
                pass
    if dinfo is None:
        raise ValueError('Data type {} not found in the dictionary.'
                         .format(dtype))
    return dinfo


def asdtype(dtype, family):
    """Convert a data type to another family of data types.

    Parameters
    ----------
    dtype : str or type
        Generic data type (numpy, torch, python, string)
    family : {'numpy', 'torch', 'python'}
        A family of data types

    Returns
    -------
    dtype
        Converted data type

    """
    dinfo = info(dtype)
    return dinfo[family] or dinfo[family + '_upcast']


def asnumpy(dtype):
    """Convert a data type to a numpy dtype.

    Parameters
    ----------
    dtype : str or type
        Generic data type (numpy, torch, python, string)

    Returns
    -------
    dtype : np.number
        Numpy data type

    """
    return asdtype(dtype, 'numpy')


def astorch(dtype):
    """Convert a data type to a torch dtype.

    Parameters
    ----------
    dtype : str or type
        Generic data type (numpy, torch, python, string)

    Returns
    -------
    dtype : torch.dtype
        Torch data type

    """
    return asdtype(dtype, 'torch')


def aspython(dtype):
    """Convert a data type to a python dtype.

    Parameters
    ----------
    dtype : str or type
        Generic data type (numpy, torch, python, string)

    Returns
    -------
    dtype : type
        Python data type

    """
    return asdtype(dtype, 'python')

