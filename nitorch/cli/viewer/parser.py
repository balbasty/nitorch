from nitorch.core import cli
from nitorch.core.cli import ParseError
from nitorch.core.struct import Structure


class View(Structure):
    """Structure that holds parameters of the `denoise_mri` command"""
    files: list = []

help = r"""[nitorch] Interactive viewer for volumetric images.

usage:
    nitorch view *FILES
                
"""


def parse(args):
    """Parse the command-line arguments of the `view` command.

    Parameters
    ----------
    args : list of str
        List of arguments, without the command name.

    Returns
    -------
    View
        Filled structure

    """

    struct = View()

    struct.files = []
    while cli.next_isvalue(args):
        val, *args = args
        struct.files.append(val)

    while args:
        if cli.next_isvalue(args):
            raise ParseError(f'Value {args[0]} does not seem to belong '
                             f'to a tag.')
        tag, *args = args
        if tag in ('-h', '--help'):
            print(help)
            return None
        else:
            raise ParseError(f'Unknown tag {tag}')

    return struct

