from nitorch.core import cli
from nitorch.core.cli import ParseError


class Convert(cli.ParsedStructure):
    """Structure that holds parameters of the `info` command"""
    files: list = []
    meta: list = []
    stat: bool = False


help = r"""[nitorch] Print volume information

usage:
    nitorch convert *FILES [-dt DTYPE] [-o *FILES]

    -c, --casting CAST           Casting type (default: unsafe) 
                                    unsafe, rescale, rescale_zero
    -dt, --dtype DTYPE           Output data type (default: same)
    -o, --output *FILES          Output filenames (default: {dir}/{base}{ext})
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
                if val.lower().startswith('f'):
                    val = False
                elif val.lower().startswith('t'):
                    val = True
                struct.stat = bool(int(val))
        elif tag in ('-h', '--help'):
            print(help)
            return None
        else:
            raise ParseError(f'Unknown tag {tag}')

    return struct

