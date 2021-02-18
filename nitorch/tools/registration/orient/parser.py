from nitorch.core import cli
from nitorch.core.cli import ParseError


class Orient(cli.ParsedStructure):
    """Structure that holds parameters of the `nireorient` command"""
    files: list = []
    layout: str = 'RAS'
    voxel_size: list = []
    center: list = []
    like: str = None
    output: list = '{dir}{sep}{base}.{layout}{ext}'


help = r"""[nitorch] Orient volumes

!! This command assumes that the orientation matrix in the input file is !!
!! INCORRECT and must be OVERWRITTEN. The original vox2ras mapping will  !!
!! be LOST. If you wish to simply change the on-disk layout of the data  !!
!! and preserve the original world mapping, use `nireorient` instead.    !!

The on-disk layout (or orientation) of a volume is the order in which 
dimensions are stored. A layout can be encoded by a permutation of three 
letters:
    - R (left to Right) or L (right to Left)
    - A (posterior to Anterior) or P (anterior to Posterior)
    - S (inferior to Superior) or I (superior to Inferior)
    
usage:
    niorient *FILES [-l LAYOUT] [-v *VX] [-c *CTR] [-k FILE] [-o *FILES]

    -l, --layout LAYOUT    Target orientation (default: preserved)
    -v, --voxel-size *VX   Target voxel size (default: preserved)
    -c, --center *CTR      Target coordinates of the FOV center (default: preserved)
    -k, --like FILE        File from which to copy the orientation matrix (default: self)
    -o, --output *FILES    Output filenames (default: {dir}/{base}.{layout}{ext})

    To overrites the affine with a default RAS orientation and a voxel 
    size of [2, 2, 2], use:
        niorient broken_file.nii.gz -l RAS -c 0 -o 2
"""


def parse(args):
    """

    Parameters
    ----------
    args

    Returns
    -------

    """

    struct = Orient()

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
        elif tag in ('-v', '--voxel-size'):
            struct.voxel_size = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.voxel_size.append(float(val))
        elif tag in ('-c', '--center'):
            struct.center = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.center.append(float(val))
        elif tag in ('-k', '--like'):
            struct.like, *args = args
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

