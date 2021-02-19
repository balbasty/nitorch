from nitorch.core import cli
from nitorch.core.cli import ParseError


class InPaint(cli.ParsedStructure):
    """Structure that holds parameters of the `nipaint` command"""
    files: list = []
    missing: str = 'nan'
    output: list = '{dir}{sep}{base}.inpaint{ext}'
    device: str = 'cpu'
    verbose: int = 1


help = r"""[nitorch] Inpainting of missing data

usage:
    nitorch inpaint *FILES [-m *VAL] [-o *FILES] [-cpu|gpu]

    -m, --missing *VAL     Values considered missing (default: 'nan')
    -o, --output *FILES    Output filenames (default: {dir}/{base}.inpaint{ext})
    -cpu, -gpu             Device to use (default: cpu)
    -v, --verbose          Verbosity level (default: 1)
"""


def parse(args):
    """Parse the command-line arguments of the `inpaint` command.

    Parameters
    ----------
    args : list of str
        List of arguments, without the command name.

    Returns
    -------
    InPaint
        Filled structure

    """

    struct = InPaint()

    struct.files = []
    while cli.next_isvalue(args):
        val, *args = args
        struct.files.append(val)

    while args:
        if cli.next_isvalue(args):
            raise ParseError(f'Value {args[0]} does not seem to belong '
                             f'to a tag.')
        tag, *args = args
        if tag in ('-m', '--missing'):
            struct.missing = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.missing.append(float(val))
        elif tag in ('-cpu', '--cpu'):
            struct.device = 'cpu'
        elif tag in ('-gpu', '--gpu'):
            struct.device = 'cuda'
            if cli.next_isvalue(args):
                gpu, *args = args
                struct.device = 'cuda:{:d}'.format(int(gpu))
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

