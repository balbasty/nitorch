from nitorch.core import cli
from nitorch.core.cli import ParseError
from nitorch.core.struct import Structure


class Chunk(Structure):
    """Structure that holds parameters of the `chunk` command"""
    files: list = []
    chunk: int = 1
    dim: int = -1
    output: list = '{dir}{sep}{base}.{i}{ext}'
    transform: list = None


help = r"""[nitorch] Split a volume into chunks

usage:
    nitorch chunk *FILES -n SIZE [-d DIM] [-o *FILES] [-t *FILES] 

    -n, --chunk            Size of each chunk (default: 1)
    -d, --dimension DIM    Dimension to unstack (default: -1 = last)
    -o, --output *FILES    Output filenames (default: {dir}/{base}.{i}{ext})
    -t, --transform        Output transformation filename (default: none)
"""


def parse(args):
    """

    Parameters
    ----------
    args

    Returns
    -------

    """

    struct = Chunk()

    struct.files = []
    while cli.next_isvalue(args):
        val, *args = args
        struct.files.append(val)

    while args:
        if cli.next_isvalue(args):
            raise ParseError(f'Value {args[0]} does not seem to belong '
                             f'to a tag.')
        tag, *args = args
        if tag in ('-d', '--dim', '--dimension'):
            cli.check_next_isvalue(args, tag)
            struct.dim, *args = args
            struct.dim = int(struct.dim)
        elif tag in ('-n', '--chunk'):
            cli.check_next_isvalue(args, tag)
            struct.chunk, *args = args
            struct.chunk = int(struct.chunk)
        elif tag in ('-o', '--output'):
            struct.output = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.output.append(val)
        elif tag in ('t', '--transform'):
            if not cli.next_isvalue(args):
                struct.transform = ['{dir}{sep}{base}_2_{i}.lta']
            else:
                struct.transform = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.transform.append(val)
        elif tag in ('-h', '--help'):
            print(help)
            return None
        else:
            raise ParseError(f'Unknown tag {tag}')

    return struct

