from nitorch.core import cli


help = r"""[nitorch] uniseg

This is a reimplementation of Unified Segmentation from SPM12.
If there are any bugs, please blame me, not them.

usage: 
    nitorch uniseg *FILES [options]
    
inputs:
               *FILES               Path to input contrasts (same shape)
    -t, --tpm  *FILES               Path to tissue probability maps     [SPM12]
        --no-tpm                    Do not use spatial priors (= simple GMM)
    -m, --mask  FILE                Path to a mask or weight map
    
device:
    --cpu [N]                       Run on CPU with N threads           [all]
    --gpu [DEVICE]                  Run on GPU with DEVICE              [0] 

output: 
{space} can be `nat` (native), `mni` (affine) or `wrp` (nonlin)
Only native prob/labels are written by default.
    -o, --output PATH               Output directory                    [same as input]
        --labels-{space}  [FILE]    Labels                              [{dir}/{base}.labels.{space}{ext}]
        --prob-{space}    [FILE]    Posterior probabilities             [{dir}/{base}.prob.{space}{ext}]
        --prior-{space}   [FILE]    Prior probabilities                 [{dir}/{base}.prior.{space}{ext}]
        --nobias-{space}  [FILE]    Bias-corrected MRI                  [{dir}/{base}.nobias.{space}{ext}]
        --bias-{space}    [FILE]    Bias field                          [{dir}/{base}.bias.{space}{ext}]
        --warp-{space}    [FILE]    Warp field                          [{dir}/{base}.warp.{space}{ext}]
        --all-{space}               All outputs in this space           [no] 
        --mni-vx FLOAT              Voxel size of MNI space             [same as TPM]
By default, --prob-nat and --labels-nat are on. To deactivate them, use:
        --no-labels-nat             Do not write native labels
        --no-prob-nat               Do not write native posterior probabilities

components:
    -k, --clusters *INT             Number of clusters per class        [4 2 2 2 2 3]
    -b, --bias,  --no-bias          Perform bias correction             [yes]/no
    -a, --align, --no-align         Perform TPM alignment               [always]/once/no
    -w, --warp,  --no-warp          Perform TPM warping                 [yes]/no
        --mix,   --no-mix           Perform mixing proportion updates   [yes]/no
        --mrf,   --no-mrf           Perform Markov random field         learn/always/[once]/no
        --wish,  --no-wish          Perform Wishart regularization      [yes]/no
    -c, --clean, --no-clean         Perform postprocessing cleanup      [yes]/no
    -f, --flexi                     Try to find the correct orientation yes/[no]

regularization:
    --lam-prior FLOAT               Strength of spatial prior           [1]
    --lam-bias  FLOAT               Bias field regularization           [0.1]
    --lam-warp  FLOAT               Warp regularization                 [0.1]
    --lam-mix   FLOAT               Mixing proportion regularization    [100]
    --lam-mrf   FLOAT               MRF regularization                  [10]
    --lam-wish  FLOAT               Wishart regularization              [1]

optimization:
    -s, --spacing  *FLOAT           Space between sampled points, in mm [3]
    -n, --max-iter  INT             Maximum number of EM iterations     [30]
        --tolerance FLOAT           Tolerance for early stopping        [1e-3]
    
verbosity:
    -v, --verbose [INT]             Verbosity level                     0/[1]/2/3
    -p, --plot    [INT]             Plotting level                      [0]/1/2/3/4
    
reference:
    "Unified segmentation"
    Ashburner & Friston, NeuroImage (2005)
    https://doi.org/10.1016/j.neuroimage.2005.02.018
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


parser = cli.CommandParser('uniseg', help=help)
# --- inputs -----------------------------------------------------------
parser.add_positional('input', nargs='+')
parser.add_option('tpm', ('-t', '--tpm'), nargs='+', default=None)
parser.add_option('tpm', '--no-tpm', nargs=0,
                  action=cli.Actions.store_value(False))
parser.add_option('mask', ('-m', '--mask'), nargs='?', default=None)
# --- outputs ----------------------------------------------------------
parser.add_option('output', ('-o', '--output'), nargs=1, default=None)
for space in ('nat', 'mni', 'wrp'):
    default_fname = '{dir}{sep}{base}.prob.' + space + '{ext}'
    parser.add_option(f'prob_{space}', (f'--post-{space}', f'--prob-{space}'), nargs='?',
                      action=cli.Actions.store_value(default_fname),
                      default=default_fname if space == 'nat' else None)
    parser.add_option(f'prob_{space}', (f'--no-post-{space}', f'--no-prob-{space}'), nargs=0,
                      action=cli.Actions.store_value(False))
    default_fname = '{dir}{sep}{base}.labels.' + space + '{ext}'
    parser.add_option(f'labels_{space}', f'--labels-{space}', nargs='?',
                      action=cli.Actions.store_value(default_fname),
                      default=default_fname if space == 'nat' else None)
    parser.add_option(f'labels_{space}', f'--no-labels-{space}', nargs=0,
                      action=cli.Actions.store_value(False))
    default_fname = '{dir}{sep}{base}.prior.' + space + '{ext}'
    parser.add_option(f'prior_{space}', f'--prior-{space}', nargs='?',
                      action=cli.Actions.store_value(default_fname),
                      default=None)
    default_fname = '{dir}{sep}{base}.nobias.' + space + '{ext}'
    parser.add_option(f'nobias_{space}', f'--nobias-{space}', nargs='?',
                      action=cli.Actions.store_value(default_fname),
                      default=None)
    default_fname = '{dir}{sep}{base}.bias.' + space + '{ext}'
    parser.add_option(f'bias_{space}', f'--bias-{space}', nargs='?',
                      action=cli.Actions.store_value(default_fname),
                      default=None)
    default_fname = '{dir}{sep}{base}.warp.' + space + '{ext}'
    parser.add_option(f'warp_{space}', f'--warp-{space}', nargs='?',
                      action=cli.Actions.store_value(default_fname),
                      default=None)
    parser.add_option(f'all_{space}', f'--all-{space}', nargs=0,
                      action=cli.Actions.store_value(True), default=False)
parser.add_option('mni_vx', '--mni-vx', nargs=1, default=None, convert=float)
# --- components -------------------------------------------------------
mrf_aliases = {'yes': 'learn'}
mrf_choices = ['learn', 'always', 'once', 'no']
aff_aliases = {'yes': 'learn'}
aff_choices = ['always', 'once', 'no']
parser.add_option(f'clusters', ('-k', '--clusters'), nargs='+',
                  convert=int, default=None)
parser.add_option(f'bias', ('-b', '--bias'), nargs='?', convert=bool_or_str,
                  action=cli.Actions.store_value(True), default=True)
parser.add_option(f'bias', '--no-bias', nargs=0,
                  action=cli.Actions.store_value(False))
parser.add_option(f'align', ('-a', '--align'), nargs='?',
                  convert=lambda x: aff_aliases.get(x, x),
                  validation=cli.Validations.choice(aff_choices),
                  action=cli.Actions.store_value(True), default=True)
parser.add_option(f'align', '--no-align', nargs=0,
                  action=cli.Actions.store_value(False))
parser.add_option(f'warp', ('-w', '--warp'), nargs='?', convert=bool_or_str,
                  action=cli.Actions.store_value(True), default=True)
parser.add_option(f'warp', '--no-warp', nargs=0,
                  action=cli.Actions.store_value(False))
parser.add_option(f'mix', '--mix', nargs='?', convert=bool_or_str,
                  action=cli.Actions.store_value(True), default=True)
parser.add_option(f'mix', '--no-mix', nargs=0,
                  action=cli.Actions.store_value(False))
parser.add_option(f'mrf', '--mrf', nargs='?',
                  convert=lambda x: mrf_aliases.get(x, x),
                  validation=cli.Validations.choice(mrf_choices),
                  action=cli.Actions.store_value('learn'), default='once')
parser.add_option(f'mrf', '--no-mrf', nargs=0,
                  action=cli.Actions.store_value(False))
parser.add_option(f'wish', '--wish', nargs='?', convert=bool_or_str,
                  action=cli.Actions.store_value(True), default=True)
parser.add_option(f'wish', '--no-wish', nargs=0,
                  action=cli.Actions.store_value(False))
parser.add_option(f'clean', ('-c', '--clean'), nargs='?', convert=bool_or_str,
                  action=cli.Actions.store_value(True), default=True)
parser.add_option(f'clean', '--no-clean', nargs=0,
                  action=cli.Actions.store_value(False))
parser.add_option(f'flexi', ('-f', '--flexi'), nargs='?', convert=bool_or_str,
                  action=cli.Actions.store_value(True), default=False)
# --- regularization ---------------------------------------------------
parser.add_option(f'lam_prior', '--lam-prior', nargs=1, convert=float,
                  default=1)
parser.add_option(f'lam_bias', '--lam-bias', nargs=1, convert=float,
                  default=0.1)
parser.add_option(f'lam_warp', '--lam-warp', nargs=1, convert=float,
                  default=0.1)
parser.add_option(f'lam_mix', '--lam-mix', nargs=1, convert=float,
                  default=100)
parser.add_option(f'lam_mrf', '--lam-mrf', nargs=1, convert=float,
                  default=10)
parser.add_option(f'lam_wish', '--lam-wish', nargs=1, convert=float,
                  default=1)
# --- optimization -----------------------------------------------------
parser.add_option(f'spacing', ('-s', '--spacing'), nargs='+',
                  convert=float, default=[3])
parser.add_option(f'iter', ('-n', '--max-iter'), nargs=1,
                  convert=int, default=30)
parser.add_option(f'tolerance', '--tolerance', nargs=1,
                  convert=float, default=1e-3)
# --- verbosity --------------------------------------------------------
parser.add_option(f'verbose', ('-v', '--verbose'), nargs='?',
                  convert=int, default=1,
                  action=cli.Actions.store_value(2))
parser.add_option(f'verbose', ('-q', '--quiet'), nargs=0,
                  action=cli.Actions.store_value(0))
parser.add_option(f'plot', ('-p', '--plot'), nargs='?',
                  convert=int, default=0,
                  action=cli.Actions.store_value(1))
# --- device -----------------------------------------------------------
parser.add_option('device', '--gpu', default=('cpu', None),
                  nargs='?', convert=convert_device('gpu'),
                  action=cli.Actions.store_value(('gpu', None)),
                  help='Use GPU (if available)')
parser.add_option('device', '--cpu',
                  nargs='?', convert=convert_device('cpu'),
                  action=cli.Actions.store_value(('cpu', None)),
                  help='Use CPU')
