import warnings
from nitorch.core import cli
from nitorch.core.cli import ParseError
from nitorch.core.struct import Structure


class Pool(Structure):
    """Structure that holds parameters of the `pool` command"""
    files: list = []
    window: list = 3
    stride: list = []
    method: str = 'mean'
    padding: list = 0
    bound: list = 0
    dim: int = 3
    output: list = '{dir}{sep}{base}.pool{ext}'
    device: str = 'cpu'


help = r"""[nitorch] Window-pooling of a volume

usage:
    nitorch pool *FILES [-w *WIN] [-s *STRD] [-p [BND] *(PAD [BND])] [-m METHOD]
                        [-d DIM] [-o *FILES] [-cpu|gpu]

    -w, --window *WIN      Window size per dimension (default: 3)
    -s, --stride *STRD     Stride between output elements (default: *WIN)
    -p, --padding *PAD     Padding on both sides. Can be 'same' (default: 0)
                           An optional boundary type (default: dct2) can be
                           given first (applies to all axes) and/or after
                           each size (applies to that axis).
    -m, --method METHOD    Pooling function: {'mean', 'sum', 'min', 'max', 'median', 'rms'} (default: 'mean')
                           rms = root mean square
    -d, --dimension DIM    Number of spatial dimensions (default: 3)
    -o, --output *FILES    Output filenames (default: {dir}/{base}.pool{ext})
    -cpu, -gpu             Device to use (default: cpu)
"""


def parse(args):
    """Parse the command-line arguments of the `nipool` command.

    Parameters
    ----------
    args : list of str
        List of arguments, without the command name.

    Returns
    -------
    Pool
        Filled structure

    """

    struct = Pool()

    struct.files = []
    while cli.next_isvalue(args):
        val, *args = args
        struct.files.append(val)

    while args:
        if cli.next_isvalue(args):
            raise ParseError(f'Value {args[0]} does not seem to belong '
                             f'to a tag.')
        tag, *args = args
        if tag in ('-w', '--window'):
            struct.window = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.window.append(int(val))
        elif tag in ('-s', '--stride'):
            struct.stride = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.stride.append(int(val))
        elif tag in ('-p', '--padding'):
            struct.padding = []
            struct.bound = []
            last_size = False       # last value was a size without type
            bound_default = 'dct2'
            has_default = False
            while cli.next_isvalue(args):
                val, *args = args
                try:
                    val = int(val)
                except ValueError:
                    val = val.lower()
                if isinstance(val, int) or val == 'same':
                    if last_size:
                        # previous size had no explicit type
                        struct.bound.append(bound_default)
                    last_size = True
                    struct.padding.append(val)
                elif len(struct.padding) == 0 and not has_default:
                    # first argument -> default padding mode
                    bound_default = val
                    has_default = True
                elif not last_size:
                    raise ParseError(
                        'Padding type must follow padding size'
                    )
                else:
                    struct.bound.append(val)
                    last_size = False
            if not struct.padding:
                raise ParseError('Padding type given without padding size')
            while len(struct.bound) < len(struct.padding):
                struct.bound.append(bound_default)
        elif tag in ('-m', '--method'):
            cli.check_next_isvalue(args, tag)
            struct.method, *args = args
            if struct.method.lower() == 'ssq':
                warnings.warn("pool: method 'ssq' has been renamed to 'rms' "
                              "(root mean square). 'ssq' will be removed in "
                              "a future version.", FutureWarning)
                struct.method = 'rms'
        elif tag in ('-cpu', '--cpu'):
            struct.device = 'cpu'
        elif tag in ('-gpu', '--gpu'):
            struct.device = 'cuda'
            if cli.next_isvalue(args):
                gpu, *args = args
                struct.device = 'cuda:{:d}'.format(int(gpu))
        elif tag in ('-d', '--dim', '--dimension'):
            cli.check_next_isvalue(args, tag)
            struct.dim, *args = args
            struct.dim = int(struct.dim)
        elif tag in ('-o', '--output'):
            struct.output = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.output.append(val)
        elif tag in ('-h', '--help'):
            print(help)
            return None
        else:
            raise ParseError(f'Unknown tag {tag}')

    return struct

