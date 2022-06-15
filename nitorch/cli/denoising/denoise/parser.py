from nitorch.core import cli

help = r"""[nitorch] denoise

total-variation denoising

usage: 
    nitorch denoise *FILES [options]

    *FILES                          Images to denoise (can be multiple channels)
    -o, --output PATH               Output path [{dir}/{base}.denoised{ext}]
    -j, --joint [true]              Use joint TV across channels
    -l, --lam [10]                  Regularization (per channel)
    -s, --sigma *VALUE              Noise standard deviation (per channel) [inferred]
    -n, --max-iter [10, 32]         Max number of IRLS and CG/RELAX iterations
    -t, --tolerance [1e-4]          Tolerance of IRLS and CG/RELAX iterations
    -x, --optim [cg]                Optimizer {cg, relax}
    -h, --help [LEVEL]              Display this help
    -v, --verbose [LVL]             Level of verbosity [1=print], 2=print more, 3=plot
    --cpu [THREADS], --gpu          Device to use [cpu]

"""

parser = cli.CommandParser('denoise', help=help)

parser.add_positional('files', nargs='+', help='Input images')
parser.add_option('output', ('-o', '--output'), nargs='*',
                  default='{dir}{sep}{base}.denoised{ext}')
parser.add_option('lam', ('-l', '--lam'), nargs='+', default=[10.],
                  convert=float)
parser.add_option('sigma', ('-s', '--sigma'), nargs='+', default=[],
                  convert=float)
parser.add_option('max_iter', ('-n', '--max-iter'), nargs='+', default=[10, 32],
                  convert=int)
parser.add_option('tolerance', ('-t', '--tolerance'), nargs='+', default=[1e-4],
                  convert=float)
optim_choices = ['cg', 'relax', 'fmg', 'fmg+cg', 'fmg+relax']
parser.add_option('optim', ('-x', '--optim'), nargs=1, default='cg',
                  validation=cli.Validations.choice(optim_choices))
parser.add_option('joint', ('-j', '--joint'), nargs='?', default=True,
                  convert=cli.Conversions.bool)
parser.add_option('verbose', ('-v', '--verbose'),
                  nargs='?', default=0, convert=int,
                  action=cli.Actions.store_value(1),
                  help='Level of verbosity')
parser.add_option('device', '--gpu', default=('cpu', None),
                  nargs='?', convert=cli.Conversions.device('gpu'),
                  action=cli.Actions.store_value(('gpu', None)),
                  help='Use GPU (if available)')
parser.add_option('device', '--cpu',
                  nargs='?', convert=cli.Conversions.device('cpu'),
                  action=cli.Actions.store_value(('cpu', None)),
                  help='Use CPU')
