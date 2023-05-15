from nitorch.core import cli, bounds
from fractions import Fraction


help = r"""[nitorch] Resize volumes

usage:
    nitorch resize *FILES [-f FACTOR] [-v VX] [-s SHAPE] [-a ANCHOR]
                   [-i ORDER] [-b BND] [-p FILTER] [-g] [-cpu|gpu]
    
    The resizing factor can be defined by one of three (exclusive) arguments:
    -f, --factor            Resizing factor
                                > 1: more/smaller voxels
                                < 1: fewer/larger voxels
    -v, --voxel-size        Voxel size of the resized image
    -s, --shape             Shape of the resized image
    
    Note that --shape can also be used on conjunction with -f or -v, in which
    case it defines the amount of padding to add at the bottom/right of
    the output image.
    
    -a, --anchor            {centers, edges, first, last} (default: centers)
    -i, --interpolation     Interpolation order (1)
    -b, --bound             Boundary conditions (dct2)
    -p, --prefilter         Apply spline prefilter {no, yes} (yes)
    -g, --grid              The input is a displacement (disp) or 
                            transformation (trf) grid (default: no) 
    -o, --output            Name of the output file (default: '*.resized*')
    -cpu, -gpu              Device to use (cpu)
   
"""


def check_anchor(x):
    if x[0] not in ('c', 'e', 'f', 'l'):
        raise ValueError(f'Unknown anchor {x}')
    return True


def check_interpolation(x):
    if x not in list(range(7)) and x not in \
            ('nearest', 'linear', 'quadratic', 'cubic',
             'fourth', 'fifth', 'sixth', 'seventh'):
        raise ValueError(f'Unknown interpolation {x}')
    return True


def convert_interpolation(x):
    try:
        return int(x)
    except Exception:
        return x


def convert_prefilter(x):
    if x[0].lower() in 'ty':
        return True
    if x[0].lower() in 'fn':
        return False
    return bool(int(x))


def convert_grid(x):
    if x[0].lower() == 'd':
        return 'd'
    if x[0].lower() == 't':
        return 't'
    return False


def convert_fraction(x):
    return float(Fraction(x))


parser = cli.CommandParser('resize', help=help)
parser.add_positional('files', nargs='+', help='Files')
parser.add_option('factor', ('-f', '--factor'), nargs='+', default=None,
                  convert=convert_fraction, help='Factor')
parser.add_option('voxel_size', ('-v', '--voxel-size'), nargs='+',
                  default=None, convert=float, help='Voxel size')
parser.add_option('shape', ('-s', '--shape'), nargs='+',
                  default=None, convert=int, help='Shape')
parser.add_option('anchor', ('-a', '--anchor'), nargs='+', default='c',
                  validation=check_anchor, help='Anchor')
parser.add_option('interpolation', ('-i', '--interpolation'), nargs='+', default=1,
                  convert=convert_interpolation, validation=check_interpolation,
                  help='Interpolation order')
parser.add_option('bound', ('-b', '--bound'), nargs='+', default='dct2',
                  convert=bounds.to_nitorch, help='Boundary condition')
parser.add_option('prefilter', ('-p', '--prefilter'), nargs=1, default=True,
                  convert=convert_prefilter, help='Spline prefilter')
parser.add_option('grid', ('-g', '--grid'), nargs=1, default=False,
                  convert=convert_grid, help='Displacement/Transformation')
parser.add_option('output', ('-o', '--output'), nargs='*',
                  default=['{dir}{sep}{base}.resized{ext}'],
                  help='Output files')
parser.add_option('gpu', '--gpu',
                  nargs='?', default='cpu', convert=int,
                  action=cli.Actions.store_value(0),
                  help='Use GPU (if available)')
parser.add_option('gpu', '--cpu', nargs=0,
                  action=cli.Actions.store_value('cpu'),
                  help='Use CPU')



