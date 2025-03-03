from nitorch.core import cli


help = r"""[nitorch] meanspace

Compute an average "oriented voxel space" from a series of oriented images.

usage:
    nitorch meanspace *PATHS [options]

Input options:
    *PATHS                          Input images

Output options:
    -o, --output PATH               Output path to mock mean image [{dir}/meanspace{ext}]
    -r, --resliced PATH             Output path to resliced image [{dir}/{base}.meanspace{ext}]
    -v, --voxel-size VAL [UNIT]     Voxel size and unit [100 %]
    -p, --pad VAL [UNIT]            Pad field of view by some amount [0 %]
    -l, --layout LAYOUT             Voxel layout (RAS, LPS, ...) [majority]

Other options:
    -h, --help [LEVEL]              Display this help
    --cpu [THREADS], --gpu          Device to use [cpu]

"""


def number_or_str(type=float):
    def _number_or_str(x):
        try:
            return type(x)
        except ValueError:
            return x
    return _number_or_str


def bool_or_str(x):
    if x.lower() in ('true', 'yes'):
        return True
    if x.lower() in ('false', 'no'):
        return False
    try:
        return bool(int(x))
    except ValueError:
        return x


def convert_device(device):
    def _convert(x):
        return (device, int(x))
    return _convert


parser = cli.CommandParser('meanspace', help=help)

parser.add_positional('input', nargs='+', help='Input images')
parser.add_option(
    'output', ('-o', '--output'), nargs=1,
    default='{dir}/meanspace{ext}',
    convert=bool_or_str
)
parser.add_option(
    'resliced', ('-r', '--resliced'), nargs='?',
    default=False,
    action=cli.Actions.store_value('{dir}/{base}.meanspace{ext}')
)
parser.add_option(
    'voxel_size', ('-v', '--voxel-size'), nargs='1*', default=[100, '%'],
    convert=number_or_str(float),
    help='Voxel size and unit'
)
parser.add_option(
    'pad', ('-p', '--pad'), nargs='1*', default=[0, '%'],
    convert=number_or_str(float),
    help='Pad field of view by some amount'
)
parser.add_option(
    'layout', ('-l', '--layout'), nargs=1, default=None,
    help='Voxel layout'
)
parser.add_option(
    'device', '--gpu', default=('cpu', None), nargs='?',
    convert=convert_device('gpu'),
    action=cli.Actions.store_value(('gpu', None)),
    help='Use GPU (if available)')
parser.add_option(
    'device', '--cpu', nargs='?',
    convert=convert_device('cpu'),
    action=cli.Actions.store_value(('cpu', None)),
    help='Use CPU')
