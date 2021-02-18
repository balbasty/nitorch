from nitorch.core import cli
from nitorch.core.cli import ParseError


class Pool(cli.ParsedStructure):
    """Structure that holds parameters of the `nipool` command"""
    files: list = []
    window: list = 3
    stride: list = []
    method: str = 'mean'
    dim: int = 3
    output: list = '{dir}{sep}{base}.pool{ext}'


help = r"""[nitorch] Window-pooling of a volume

usage:
    nipool *FILES [-w *WIN] [-s *STRD] [-m METHOD] [-d DIM] [-o *FILES] [-cpu|gpu]

    -w, --window *WIN      Window size per dimension (default: 3)
    -s, --stride *STRD     Stride between output elements (default: *WIN)
    -m, --method METHOD    Pooling function: {'mean', 'sum', 'min', 'max', 'median'} (default: 'mean')
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
        elif tag in ('-m', '--method'):
            cli.check_next_isvalue(args, tag)
            struct.method, *args = args
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

