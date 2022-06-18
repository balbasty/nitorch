from nitorch.core import cli
ParseError = cli.ParseError

help = r"""[nitorch] EPIC - Edge-Preserving B0 inhomogeneity correction

This function jointly denoises and corrects geometric distortions in 
bipolar multi-echo data using a pre-computed fieldmap. Furthermore, 
it assumes an exponential decay and constant echo spacing to add an 
additional constraint that makes the process more robust to imperfect 
fieldmaps.

usage: 
    nitorch epic *ECHOES -f FIELDMAP [options]
    
    *ECHOES                             Acquired bipolar echoes
    -r, --reversed *ECHOES              Reversed-polarity echoes
    -f, --fieldmap                      Fieldmap (in Hz) or voxel shift map
    -o, --output                        Path to corrected echoes [{dir}/{base}_epic{ext}]
    -d, --direction {rl,ap,is}          Readout direction [largest]
    -p, --polarity {[+],-}              Polarity of the first echo
    -b, --bandwidth [1]                 Acquisition bandwidth (Hz/pixel)
    -x, --no-synth                      Do not synthesize reverse echoes
    -e, --extrapolate [false]           Extrapolate first/last echo in synth
    -s, --slicewise [CHUNKSIZE]         Apply slicewise/chunkwise [false]
    -p, --penalty VAL                   Amount or regularization [100]
    -n, --max-iter IRLS [CG]            Maximum number of iterations [20 32]
    -t, --tolerance IRLS [CG]           Tolerance for early stopping [1e-5]
    --cpu, --gpu [THREAD/ID]            Device to use [cpu]
    -h, --help                          Display this help
    -v, --verbose [LVL]                 Level of verbosity [1=print], 2=more
    
References:
    "A new distortion correction approach for multi-contrast MRI"
    Divya Varadarajan, et al. ISMRM (2020)
"""

# Fit command
parser = cli.CommandParser('fit', help=help)
parser.add_positional('echoes', nargs='+', help='Bipolar echoes')
parser.add_option('fieldmap', ('-f', '--fieldmap'), nargs=1, help='Fieldmap')
parser.add_option('reversed', ('-r', '--reversed'), nargs='+', help='Reversed echoes')
parser.add_option('output', ('-o', '--output'), nargs='+',
                  default=['{dir}{sep}{base}_epic{ext}'],
                  help='Output file')
dir_choices = ['r', 'l', 'a', 'p', 'i', 's', 'rl', 'lr', 'ap', 'pa', 'is', 'si']
parser.add_option('direction', ('-d', '--direction'), nargs=1,
                  validation=cli.Validations.choice(dir_choices),
                  help='Readout direction')
parser.add_option('polarity', ('-p', '--polarity'), nargs=1, default='+',
                  validation=cli.Validations.choice(['+', '-']),
                  help='Polarity of first echo')
parser.add_option('bandwidth', ('-b', '--bandwidth'), nargs=1, default=1,
                  convert=float, help='Polarity of first echo')
parser.add_option('synth', ('-x', '--no-synth'), nargs=0, default=True,
                  convert=cli.Conversions.bool, action=cli.Actions.store_false,
                  help='Synthesize reverse echoes')
parser.add_option('extrapolate', ('-e', '--extrapolate'), nargs='?', default=False,
                  convert=cli.Conversions.bool, action=cli.Actions.store_true,
                  help='Extrapolate first/last echo')
parser.add_option('slicewise', ('-s', '--slicewise'), nargs='?', default=0,
                  convert=int, action=cli.Actions.store_value(1),
                  help='Perform slicewise or chunkwise')
parser.add_option('penalty', ('-p', '--penalty'), nargs=1, default=100,
                  convert=float, help='Regularization value')
parser.add_option('maxiter', ('-m', '--maxiter'), nargs='1*2', default=[20, 32],
                  convert=int, help='Maximum number of iterations')
parser.add_option('tolerance', ('-t', '--tolerance'), nargs='1*2', default=[1e-5],
                  convert=float, help='Tolerance')
# generic options
parser.add_option('verbose', ('-v', '--verbose'),
                  nargs='?', default=1, convert=int,
                  action=cli.Actions.store_value(2),
                  help='Level of verbosity')
parser.add_option('verbose', ('-q', '--quiet'),
                  nargs=0, action=cli.Actions.store_value(0),
                  help='No verbosity')
parser.add_option('gpu', '--gpu',
                  nargs='?', default='cpu',
                  convert=cli.Conversions.device('gpu'),
                  action=cli.Actions.store_value(('gpu', None)),
                  help='Use GPU (if available)')
parser.add_option('gpu', '--cpu', nargs=0,
                  convert=cli.Conversions.device('cpu'),
                  action=cli.Actions.store_value(('cpu', None)),
                  help='Use CPU')
