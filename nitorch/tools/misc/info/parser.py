from nitorch.core import cli
from nitorch.core.cli import ParseError


class Info(cli.ParsedStructure):
    """Structure that holds parameters of the `nireorient` command"""
    files: list = []
    meta: list = []


help = r"""[nitorch] Print volume information

usage:
    niinfo *FILES -m *FIELDS

    -m, --meta             Specific fields that must be printed.
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
        elif tag in ('-h', '--help'):
            print(help)
            return None
        else:
            raise ParseError(f'Unknown tag {tag}')

    return struct

