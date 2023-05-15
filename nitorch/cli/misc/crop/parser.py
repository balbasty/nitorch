from nitorch.core import cli
from nitorch.core.cli import ParseError
from nitorch.core.struct import Structure


class Crop(Structure):
    """Structure that holds parameters of the `crop` command"""
    files: list = []
    size: list = []
    center: list = []
    size_space: str = 'vox'
    center_space: str = 'vox'
    bbox: float or bool = False
    like: str = None
    output: list = '{dir}{sep}{base}.crop{ext}'
    transform: list = None


help = r"""[nitorch] Crop a volume

usage:
    nitorch crop *FILES [-s *SIZE [SPACE]] [-c *CENTER [UNIT] [SPACE]]
                        [-k FILE] [-o *FILES] [-t *FILES] 

    -s, --size *SIZE [SPACE]     Size of the cropped region.
                                 Space in {vox (default), ras}
    -c, --center *SHIFT [SPACE]  Coordinate of the center of the cropped region 
                                 (default: center of the FOV)
    -k, --like FILE              Path to a pre-cropped volume to use as reference.
    -b, --bounding-box [THRESH]  Crop at the bounding box of `input > THRESH`.
    -o, --output *FILES          Output filenames (default: {dir}/{base}.crop{ext})
    -t, --transform              Input or output transformation filename (default: none)
                                    Input if none of s/c/l/k options are used.
                                    Output otherwise.
    
notes:
    Only one of --size, --like or --bounding-box can be used.
    
examples:
    # extract a patch of 10x10x10 voxels in the center of the FOV
    nicrop file.mgz -s 10 10 10
    
    # extract a patch 5x5x2 mm about known RAS coordinates
    nicrop file.mgz -s 5 5 2 ras -c 12.4 -6.4 3.2 ras
    
    # extract the same patch as one already extracted
    nicrop file1 -s 5 5 5 -o patch.mgz
    nicrop file2.mgz -k patch.mgz
    
    # apply a crop transform already generated
    nicrop file1 -s 5 5 5 -o patch.mgz -t crop.lta
    nicrop file2 -t crop.lta
    
"""


def parse(args):
    struct = Crop()

    struct.files = []
    while cli.next_isvalue(args):
        val, *args = args
        struct.files.append(val)

    while args:
        if cli.next_isvalue(args):
            raise ParseError(f'Value {args[0]} does not seem to belong '
                             f'to a tag.')
        tag, *args = args
        if tag in ('-s', '--size'):
            struct.size = []
            while cli.next_isvalue(args):
                val, *args = args
                if val.lower() in ('vox', 'ras'):
                    struct.size_space = val
                else:
                    struct.size.append(float(val))
        elif tag in ('-c', '--center'):
            struct.center = []
            while cli.next_isvalue(args):
                val, *args = args
                if val.lower() in ('vox', 'ras'):
                    struct.center_space = val
                else:
                    struct.center.append(float(val))
        elif tag in ('-b', '--bounding-box'):
            struct.bbox = True
            if cli.next_isvalue(args):
                struct.bbox = float(args.pop(0))
        elif tag in ('-k', '--like'):
            cli.check_next_isvalue(args, tag)
            struct.like = args.pop(0)
        elif tag in ('-o', '--output'):
            struct.output = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.output.append(val)
        elif tag in ('-t', '--transform'):
            if not cli.next_isvalue(args):
                struct.transform = ['{dir}{sep}{base}.crop.lta']
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

