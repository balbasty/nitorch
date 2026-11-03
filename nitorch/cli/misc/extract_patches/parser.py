from nitorch.core import cli
from nitorch.core.cli import ParseError
from nitorch.core.struct import Structure


class Patch(Structure):
    """Structure that holds parameters of the `chunk` command"""
    file: str = ''
    size: list = 64
    stride: list = None
    output: list = '{dir}{sep}{base}.{i}_{j}_{k}{ext}'
    transform: list = None


help = r"""[nitorch] Split a volume into patches

usage:
    nitorch extract_patches FILE -n *SIZE [-s *STRIDE] [-o *FILES] [-t *FILES] 

    -n, --size               Size of each patch (default: 64)
    -s, --stride             Stride between patches (default: same as size)
    -o, --output *FILES      Output filenames (default: {dir}/{base}.{i}_{j}_{k}{ext})
    -t, --transform *FILES   Output transformation filename (default: none)
"""


def parse(args):
    """

    Parameters
    ----------
    args

    Returns
    -------

    """

    struct = Patch()

    # `--` ends option parsing: what follows is the input file
    args, rest = cli.split_double_dash(args)
    if rest:
        struct.file = rest.pop(0)
        cli.check_rest_used(rest, 'extract_patches takes a single file')
    else:
        struct.file, = cli.pop_values(args, 'extract_patches', 1,
                                      names='<file>')

    while args:
        if cli.next_isvalue(args):
            raise ParseError(f'Value {args[0]} does not seem to belong '
                             f'to a tag.')
        tag, *args = args
        if tag in ('-n', '--size'):
            struct.size = []
            while cli.next_isvalue(args):
                struct.size += cli.pop_values(args, '-n/--size', 1, int)
        elif tag in ('-s', '--stride'):
            struct.stride = []
            while cli.next_isvalue(args):
                struct.stride += cli.pop_values(args, '-s/--stride', 1, int)
        elif tag in ('-o', '--output'):
            struct.output = []
            while cli.next_isvalue(args):
                struct.output.append(args.pop(0))
        elif tag in ('-t', '--transform'):
            if not cli.next_isvalue(args):
                struct.transform = ['{dir}{sep}{base}_to_{i}_{j}_{k}.lta']
            else:
                struct.transform = []
            while cli.next_isvalue(args):
                struct.transform.append(args.pop(0))
        elif tag in ('-h', '--help'):
            print(help)
            return None
        else:
            raise ParseError(f'Unknown tag {tag}')

    return struct

