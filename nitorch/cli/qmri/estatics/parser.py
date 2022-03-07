from nitorch.core import cli
ParseError = cli.ParseError

help = r"""[nitorch] ESTATICS + MEETUP (nonlinear version)

This function performs a nonlinear exponential fit on multi-echo data.
If multiple contrasts are provided, the decay is common across contrasts.
If MEETUP is activated, contrast-specific distortion fields are estimated.

usage: 
    nitorch estatics [options] -c [NAME] [options] -e *FILE

Acquisition options:
    -c,  --contrast [NAME]               (default: contrasts are named by order)
    -te, --echo-time TE {[s],ms}         (default: try to read from file)
    -sp, --echo-spacing DELTA {[s],ms}   (default: unused)      
    -bw, --bandwidth [BW] [UNIT]         (default: unused)
    -rd, --readout {lr,is,ap}            (default: largest dim)
    -e,  --echo *FILE                    Path to individual echoes
    -b0, --distortion FIELD {[vox],hz}   B0 fieldmap 

Reconstruction options:
    --likelihood {gauss,chi}         Noise model (default: gauss)
    --register {yes,no}              Start by registrering contrasts
    --recon-space [NAME]             Name of a contrast or 'mean' (default: 'mean')
    --regularization {no,tkh,tv,jtv} Regularization type (default: jtv)
    --lam-intercept *VAL             Regularization of the intercepts (default: 50)
    --lam-decay VAL                  Regularization of the R2* decay (default: 0.05)
    --meetup {yes,no}                MEETUP distortion correction (default: no)
    --lam-meetup VAL                 Regularization of the distortion field (default: 1e5)

Optimization options
    --nb-levels [1]                  Number of resolutions
    --max-iter [10]                  Maximum number of outer iterations
    --tolerance [1e-5]               Tolerance for early stopping

General options:
    --cpu, --gpu                    Device to use [cpu]
    -o, --output-dir                Output directory [same as input files]
    -h, --help [LEVEL]              Display this help
    -v, --verbose [LVL]             Level of verbosity [1=print], 2=plot
    --framerate                     Framerate of plotting function, in Hz [1]

References:
    If you use this command, please cite:
        "Joint Total Variation ESTATICS for Robust Multi-Parameter Mapping"
        Balbastre et al., MICCAI (2020)
        https://arxiv.org/abs/2005.14247
    If you use the --meetup option, please cite:
        "Distortion correction in multi-echo MRI without field mapping or reverse encoding"
        Balbastre et al., Proc. ISMRM (2022)
    The original multi-contrast R2* log-fit was presented in: 
        "Estimating the apparent transverse relaxation time (R2*) from images 
         with different contrasts (ESTATICS) reduces motion artifacts"
        Weiskopf et al., Front Neurosci (2014)
        https://doi.org/10.3389/fnins.2014.00278
    Note that this implementation uses additional optimization tricks from:
        "Model-based multi-parameter mapping"
        Balbastre et al., Med Image Anal (2021)
        https://arxiv.org/abs/2102.01604
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


# Main options
parser = cli.CommandParser('estatics', help=help)

# Contrast group
contrast = cli.Group('contrast', ('-c', '--contrast'), n='+',
                     help='A contrast with multiple echoes')
contrast.add_positional('name', nargs='?', help='Name of the contrast')
contrast.add_option('bandwidth', ('-bw', '--bandwidth'), nargs='+2',
                    convert=number_or_str(float), help='Bandwidth [and unit]')
contrast.add_option('echo_spacing', ('-sp', '--echo-spacing'), nargs='+2',
                    convert=number_or_str(float), help='Echo spacing [and unit]')
contrast.add_option('te', ('-te', '--echo-time'), nargs='+',
                    convert=number_or_str(float), help='Echo time(s) [and unit]')
contrast.add_option('readout', ('-rd', '--readout'), nargs=1,
                    help='Readout direction')
contrast.add_option('echoes', ('-e', '--echo'), nargs='+', help='Echoes')
contrast.add_option('b0', ('-b0', '--distortion'), nargs='+3', help='B0 field', default=[])
parser.add_group(contrast)

# recon options
parser.add_option('likelihood', '--likelihood', nargs=1, default='gauss',
                  validation=cli.Validations.choice(['chi', 'gauss']))
parser.add_option('register', '--register', nargs='?', default=False,
                  convert=bool_or_str, action=cli.Actions.store_true)
parser.add_option('space', '--recon-space', nargs=1, default='mean',
                  convert=number_or_str(int))
parser.add_option('regularization', '--regularization', nargs=1, default='jtv',
                  validation=cli.Validations.choice(['no', 'tkh', 'tv', 'jtv']))
parser.add_option('meetup', '--meetup', nargs='?', default=False,
                  convert=bool_or_str, action=cli.Actions.store_true)
parser.add_option('lam_intercept', '--lam-intercept', nargs='+', default=[50.],
                  convert=float)
parser.add_option('lam_decay', '--lam-decay', nargs=1, default=0.05,
                  convert=float)
parser.add_option('lam_meetup', '--lam-meetup', nargs=1, default=1e5,
                  convert=float)

# optim options
parser.add_option('levels', '--nb-levels', nargs=1, default=1,
                  convert=int)
parser.add_option('iter', '--max-iter', nargs=1, default=10,
                  convert=int)
parser.add_option('tol', '--tolerance', nargs=1, default=1e-5,
                  convert=float)

# generic options
parser.add_option('verbose', ('-v', '--verbose'),
                  nargs='?', default=1, convert=int,
                  action=cli.Actions.store_value(1),
                  help='Level of verbosity')
parser.add_option('gpu', '--gpu',
                  nargs='?', default='cpu', convert=int,
                  action=cli.Actions.store_value(0),
                  help='Use GPU (if available)')
parser.add_option('gpu', '--cpu', nargs=0,
                  action=cli.Actions.store_value('cpu'),
                  help='Use CPU')
# parser.add_option('help', ('-h', '--help'),
#                   nargs='?', default=False, convert=int,
#                   action=cli.Actions.store_value(1),
#                   help='Display this help. Call it with 1, 2, or 3 for '
#                        'different levels of detail.')
parser.add_option('odir', ('-o', '--output-dir'), nargs=1,
                  help='Output directory')
parser.add_option('framerate', '--framerate', nargs=1, convert=float,
                  default=1., help='Framerate of plotting function, in Hz')

