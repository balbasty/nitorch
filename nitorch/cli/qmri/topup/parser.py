from nitorch.core import cli
ParseError = cli.ParseError

help = r"""[nitorch] TOPUP

This function estimates a 1D distortion field from reverse-polarity
images. In its default mode (mean-squared error loss), it is a 
reimplementation of FSL's TOPUP (with slight differences). However, we
also provide a cross-correlation loss that allows distortion fields to 
be estimated from reverse-polarity images with differing contrasts 
(e.g., the first two echoes of a multi-echo acquisition).

usage: 
    nitorch topup fit POS NEG [options]
    
    POS                                 File with positive polarity readout
    NEG                                 File with negative polarity readout
    -m, --mask                          Path to a mask in which to compute the loss
    -o, --output                        Path to output displacement field [{dir}/{base}_topup_b0{ext}]
    -r, --readout  {lr,is,ap}           Readout direction (default: largest dim)
    -bw, --bandwidth                    Readout bandwidth in Hz/pixel (1)
    -l, --loss {mse,ncc,lncc,gmm,lgmm}  Matching term [mse]
    -k, --kernel VAL [{vox,mm,%}]       LNCC kernel size [10 %] 
    -b, --bins VAL                      Number of (L)GMM classes [3]
    -m, --modulation {yes,no}           Jacobian modulation [yes]
    -d, --diffeo                        Use diffeomorphic model [no]
    -p, --penalty VAL [{memb,bend}]     Penalty on bending energy [100 bend]
    -w, --downsample                    Estimate field at a lower dimension [1 mm]
    -n, --max-iter                      Maximum number of iterations [50]
    -t, --tolerance                     Tolerance for early stopping [1e-4]
    --cpu, --gpu                        Device to use [cpu]
    -h, --help [LEVEL]                  Display this help
    -v, --verbose [LVL]                 Level of verbosity [1=print], 2=plot

usage: 
    nitorch topup apply FIELD -p *POS -n *NEG [options]
    
    FIELD                               Path to displacement field
    -p, --pos *POS                      Files with positive polarity readout
    -n, --neg *NEG                      Files with negative polarity readout
    -o, --output                        Path to output files [{dir}/{base}_topup_unwarped{ext}]
    -r, --readout  {lr,is,ap}           Readout direction (default: largest dim)
    -m, --modulation {yes,no}           Jacobian modulation [yes]
    -d, --diffeo                        Field is diffeomorphic [no]
    --cpu, --gpu                        Device to use [cpu]
    -h, --help [LEVEL]                  Display this help
    -v, --verbose [LVL]                 Level of verbosity [1=print], 2=plot
    
    
References:
    "How to correct susceptibility distortions in spin-echo echo-planar 
     images: application to diffusion tensor imaging"
    Andersson, Skare, Ashburner. NeuroImage (2003)
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


# Fit command
parser_fit = cli.CommandParser('fit', help=help)
parser_fit.add_positional('pos_file', nargs=1, help='Positive readout')
parser_fit.add_positional('neg_file', nargs=1, help='Negative readout')
parser_fit.add_option('mask', ('-m', '--mask'), nargs=1, help='Mask file')
parser_fit.add_option('output', ('-o', '--output'), nargs=1,
                      default='{dir}{sep}{base}_topup_b0{ext}',
                      help='Output file')
parser_fit.add_option('readout', ('-r', '--readout'), nargs=1,
                      help='Readout direction')
parser_fit.add_option('bandwidth', ('-bw', '--bandwidth'), nargs=1,
                      convert=float, help='Readout bandwidth')
losses = ['mse', 'ncc', 'lncc', 'gmm', 'lgmm']
parser_fit.add_option('loss', ('-l', '--loss'), nargs=1, default='mse',
                      validation=cli.Validations.choice(losses),
                      help='Matching loss')
parser_fit.add_option('kernel', ('-k', '--kernel'), nargs='+', default=[10],
                      convert=number_or_str(float), help='Kernel size')
parser_fit.add_option('bins', ('-b', '--bins'), nargs=1, default=3,
                      convert=int, help='Number of classes')
parser_fit.add_option('modulation', ('-m', '--modulation'), nargs=1, default=True,
                      convert=bool_or_str, action=cli.Actions.store_true,
                      help='Jacobian modulation')
parser_fit.add_option('diffeo', ('-d', '--diffeo'), nargs=1, default=False,
                      convert=bool_or_str, action=cli.Actions.store_true,
                      help='Diffeomorphic transform')
parser_fit.add_option('penalty', ('-p', '--penalty'), nargs='+', default=[100],
                      convert=number_or_str(float), help='Penalty (value and type)')
parser_fit.add_option('downsample', ('-w', '--downsample'), nargs='+', default=[8., 4., 1.],
                      convert=float, help='Downsampling voxel size')
parser_fit.add_option('max_iter', ('-n', '--max-iter'), nargs='+', default=[50],
                      convert=int, help='Max number of iterations')
parser_fit.add_option('tolerance', ('-t', '--tolerance'), nargs='+', default=[1e-4],
                      convert=float, help='Tolerance for early stopping')
# generic options
parser_fit.add_option('verbose', ('-v', '--verbose'),
                      nargs='?', default=1, convert=int,
                      action=cli.Actions.store_value(1),
                      help='Level of verbosity')
parser_fit.add_option('gpu', '--gpu',
                      nargs='?', default='cpu', convert=int,
                      action=cli.Actions.store_value(0),
                      help='Use GPU (if available)')
parser_fit.add_option('gpu', '--cpu', nargs=0,
                      action=cli.Actions.store_value('cpu'),
                      help='Use CPU')

# Fit command
parser_apply = cli.CommandParser('apply', help=help)
parser_apply.add_positional('dist_file', nargs=1, help='Distortion field')
parser_apply.add_option('file_pos', ('-p', '--pos'), nargs='*',
                        help='Positive readout files')
parser_apply.add_option('file_neg', ('-n', '--neg'), nargs='*',
                        help='Negative readout files')
parser_apply.add_option('output', ('-o', '--output'), nargs=1,
                        default='{dir}{sep}{base}_topup_unwarped{ext}',
                        help='Output file')
parser_apply.add_option('readout', ('-r', '--readout'), nargs=1,
                        help='Readout direction')
parser_apply.add_option('bandwidth', ('-bw', '--bandwidth'), nargs=1,
                        convert=float, help='Readout bandwidth')
parser_apply.add_option('modulation', ('-m', '--modulation'), nargs=1, default=True,
                        convert=bool_or_str, action=cli.Actions.store_true,
                        help='Jacobian modulation')
parser_apply.add_option('diffeo', ('-d', '--diffeo'), nargs=1, default=False,
                        convert=bool_or_str, action=cli.Actions.store_true,
                        help='Diffeomorphic transform')
# generic options
parser_apply.add_option('verbose', ('-v', '--verbose'),
                        nargs='?', default=1, convert=int,
                        action=cli.Actions.store_value(1),
                        help='Level of verbosity')
parser_apply.add_option('gpu', '--gpu',
                        nargs='?', default='cpu', convert=int,
                        action=cli.Actions.store_value(0),
                        help='Use GPU (if available)')
parser_apply.add_option('gpu', '--cpu', nargs=0,
                        action=cli.Actions.store_value('cpu'),
                        help='Use CPU')