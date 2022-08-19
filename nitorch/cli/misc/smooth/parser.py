from nitorch.core import cli
from nitorch.core.bounds import all_bounds


help = r"""[nitorch] Smooth a volume

usage:
    nitorch smooth *FILES [-f *FWHM] [-m METHOD] [-o *FILES] ...

    -f, --fwhm [*F [UNIT]]  Full width at half maximum (default: 1 mm)
    -m, --method            Pooling function: {'gauss', 'tri', 'rect'} (default: 'gauss')
    -b, --basis             Basis function: {'tri', 'rect'} (default: 'tri')
    -p, --padding           Padding method: {'zero', 'replicate', 'reflect', 'circular'} (default: 'replicate') 
    -o, --output            Output filenames (default: {dir}/{base}.smooth{ext})
    --cpu [N=0]             Use CPU with N threads. If N=0: as many threads as CPUs.
    --gpu [ID=0]            Use GPU
"""


# Fit command
parser = cli.CommandParser('smooth', help=help)
parser.add_positional('files', nargs='+')
parser.add_option('fwhm', ('-f', '--fwhm'), nargs='+', default=[1],
                  convert=cli.Conversions.number_or_str(float))
parser.add_option('method', ('-m', '--method'), nargs=1, default='gauss',
                  validation=cli.Validations.choice(['gauss', 'tri', 'rect']))
parser.add_option('basis', ('-b', '--basis'), nargs=1, default='tri',
                  validation=cli.Validations.choice(['tri', 'rect']))
parser.add_option('padding', ('-p', '--padding'), nargs=1, default='replicate',
                  validation=cli.Validations.choice(all_bounds))
parser.add_option('output', ('-o', '--output'), nargs=1,
                  default='{dir}{sep}{base}.smooth{ext}')
parser.add_option('gpu', '--gpu', nargs='?', default=('cpu', 0),
                  convert=cli.Conversions.device,
                  action=cli.Actions.store_value(('gpu', 0)))
parser.add_option('gpu', '--cpu', nargs='?',
                  action=cli.Actions.store_value(('cpu', 0)))
