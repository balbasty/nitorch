from nitorch.core import cli
from nitorch.core.cli import ParseError
from nitorch.core.struct import Structure


class Reorient(Structure):
    """Structure that holds parameters of the `nireorient` command"""
    files: list = []
    layout: str = 'RAS'
    output: list = '{dir}{sep}{base}.{layout}{ext}'
    transform: list = None


help = r"""[nitorch] Reorient volumes

The on-disk layout (or orientation) of a volume is the order in which 
dimensions are stored. A layout can be encoded by a permutation of three 
letters:
    - R (left to Right) or L (right to Left)
    - A (posterior to Anterior) or P (anterior to Posterior)
    - S (inferior to Superior) or I (superior to Inferior)

!! This command assumes that the orientation matrix in the input file is !!
!! CORRECT. It then shuffles the data to match a layout and updates the  !!
!! orientation matrix so that the world-space mapping is PRESERVED.      !!
!! If you wish to overwrite the orientation matrix and preserve the      !!
!! input data layout, use `orient` instead.                              !!

usage:
    nitorch reorient *FILES [-l LAYOUT] [-o *FILES] [-t *FILES] 

    -l, --layout LAYOUT    Target orientation (default: RAS)
    -o, --output *FILES    Output filenames (default: {dir}/{base}.{layout}{ext})
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

    struct = Reorient()

    struct.files = []
    while cli.next_isvalue(args):
        val, *args = args
        struct.files.append(val)

    while args:
        if cli.next_isvalue(args):
            raise ParseError(f'Value {args[0]} does not seem to belong '
                             f'to a tag.')
        tag, *args = args
        if tag in ('-l', '--layout'):
            cli.check_next_isvalue(args, tag)
            struct.layout, *args = args
        elif tag in ('-o', '--output'):
            struct.output = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.output.append(val)
        elif tag in ('t', '--transform'):
            if not cli.next_isvalue(args):
                struct.transform = ['{dir}{sep}{base}_2_{orientation}.lta']
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

