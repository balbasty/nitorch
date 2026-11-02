from nitorch.core import cli
from nitorch.core.cli import ParseError
from nitorch.core.struct import Structure


class Info(Structure):
    """Structure that holds parameters of the `info` command"""
    files: list = []
    meta: list = []
    stat: bool = False


help = r"""[nitorch] Print volume information

usage:
    nitorch info *FILES [-m *FIELDS] [-s [BOOL]]

    -m, --meta             Specific fields that must be printed.
    -s, --stat [BOOL]      Compute intensity statistics (default: False)
                           BOOL can be true/false, yes/no or 1/0.
"""


def parse(args):
    """

    Parameters
    ----------
    args : list of str
        Command line arguments (without the command name)

    Returns
    -------
    Info
        Filled structure

    """

    struct = Info()

    struct.files = []
    while cli.next_isvalue(args):
        val, *args = args
        struct.files.append(val)

    while args:
        if cli.next_isvalue(args):
            raise ParseError(f'Value {args[0]} does not seem to belong '
                             f'to a tag.')
        tag, *args = args
        if tag in ('-m', '--meta'):
            struct.meta = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.meta.append(val)
        elif tag in ('-s', '--stat'):
            struct.stat = True
            if cli.next_isvalue(args):
                val, *args = args
                stat = cli.Conversions.bool(val)
                if not isinstance(stat, bool):
                    raise ParseError(f'Expected a boolean after {tag} '
                                     f'but got {val}')
                struct.stat = stat
        elif tag in ('-h', '--help'):
            print(help)
            return None
        else:
            raise ParseError(f'Unknown tag {tag}')

    return struct

