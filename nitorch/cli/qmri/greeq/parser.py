from nitorch.core import cli
ParseError = cli.ParseError

help = r"""[nitorch] GREEQ

Fits T1, T2* and (optionally) MTsat from multi-echo, multi-flip-angle data.
The complete nonlinear forward model is directly inverted from the raw data.

usage: 
    nitorch greeq [options] -c [NAME] [options] -e *FILES

Acquisition options:
    -c,  --contrast [NAME]              (default: contrasts are named by order)
    -fa, --flip-angle FA [UNIT]         (default: try to read from file)
    -tr, --repetition-time TR [UNIT]    (default: try to read from file)
    -te, --echo-time TE [UNIT]          (default: try to read from file)
    -sp, --echo-spacing DELTA [UNIT]    (default: unused)      
    -mt, --mt-pulse {yes,no}            (default: try to read from file, else `no`)
    -bw, --bandwidth [BW] [UNIT]        (default: unused)
    -e,  --echo *FILES                  Path to individual echoes
    -tf, --transmit FIELD [MAG] [UNIT]  Transmit fieldmap (unit: {[a.u.], pct}) 
    -rf, --receive FIELD [MAG] [UNIT]   Receive fieldmap (unit: {[a.u.], pct}) 
   [-b0, --b0-field FIELD [MAG] [UNIT]  B0 fieldmap] NOT IMPLEMENTED YET

Reconstruction options:
    --likelihood {[gauss],chi}          Noise model
    --register {yes,[no],field}         Start by registering contrasts
    --recon-space [NAME]                Name of a contrast or 'mean' (default: mean)
    --regularization {no,tkh,tv,[jtv]}  Regularization type
    --lam [10]                          Regularization (shared by all maps)
    --lam-pd VAL                        PD-specific regularization
    --lam-t1 VAL                        T1-specific regularization
    --lam-t2 VAL                        T2*-specific regularization
    --lam-mt VAL                        MTsat-specific regularization
   [--meetup {yes,[no]}                 MEETUP distortion correction] NOT IMPLEMENTED YET
   [--lam-meetup [1e5]]                 Regularization of the distortion field)] NOT IMPLEMENTED YET

Optimization options
    --nb-levels [1]                     Number of resolutions
    --max-iter [10]                     Maximum number of outer iterations
    --tolerance [1e-4]                  Tolerance for early stopping
    --solver {[cg], fmg}                Linear solver 

Output options
    -o, --output-dir                    Output directory   [same as input files]
    -u, --uncertainty                   Write Laplace uncertainty [no]

General options:
    --cpu, --gpu                        Device to use [cpu]
    -h, --help                          Display this help
    -v, --verbose [1]                   Level of verbosity [1=print], 2=plot
    --framerate                         Framerate of plotting function, in Hz [1]

References:
    If you use this command, please cite:
        "Model-based multi-parameter mapping"
        Balbastre et al., Med Image Anal (2021)
        https://arxiv.org/abs/2102.01604
    If you use the --meetup option, please cite:
        "Distortion correction in multi-echo MRI without field mapping or reverse encoding"
        Balbastre et al., Proc. ISMRM (2022)
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
parser = cli.CommandParser('greeq', help=help)

# Contrast group
contrast = cli.Group('contrast', ('-c', '--contrast'), n='+',
                     help='A contrast with multiple echoes')
contrast.add_positional('name', nargs='?', help='Name of the contrast')
contrast.add_option('fa', ('-fa', '--flip-angle'), nargs='+',
                    convert=number_or_str(float), help='Flip angle [and unit]')
contrast.add_option('tr', ('-tr', '--repetition-time'), nargs='+2',
                    convert=number_or_str(float), help='Repetition time [and unit]')
contrast.add_option('bandwidth', ('-bw', '--bandwidth'), nargs='+2',
                    convert=number_or_str(float), help='Bandwidth [and unit]')
contrast.add_option('echo_spacing', ('-es', '--echo-spacing'), nargs='+2',
                    convert=number_or_str(float), help='Echo spacing [and unit]')
contrast.add_option('te', ('-te', '--echo-time'), nargs='+',
                    convert=number_or_str(float), help='Echo time(s) [and unit]')
contrast.add_option('mt', ('-mt', '--mt-pulse'), nargs='?', default=None,
                    convert=bool_or_str, help='MT-weighted')
contrast.add_option('echoes', ('-e', '--echo'), nargs='+', help='Echoes')
contrast.add_option('transmit', ('-tf', '--transmit'), nargs='+3', help='Transmit field', default=[])
contrast.add_option('receive', ('-rf', '--receive'), nargs='+3', help='Receive field', default=[])
contrast.add_option('b0', ('-b0', '--b0-field'), nargs='+3', help='B0 field', default=[])
parser.add_group(contrast)

# shared fieldmaps
parser.add_option('transmit', ('-tf', '--transmit'), nargs='+3', help='Transmit field')
parser.add_option('receive', ('-rf', '--receive'), nargs='+3', help='Receive field')
parser.add_option('b0', ('-b0', '--b0-field'), nargs='+3', help='B0 field')

# recon options
parser.add_option('likelihood', '--likelihood', nargs=1, default='gauss',
                  validation=cli.Validations.choice(['chi', 'gauss']))
parser.add_option('register', '--register', nargs='?', default=False,
                  convert=bool_or_str, action=cli.Actions.store_true)
parser.add_option('uncertainty', ('-u', '--uncertainty'), nargs='?', default=False,
                  convert=bool_or_str, action=cli.Actions.store_true)
parser.add_option('space', '--recon-space', nargs=1, default='mean',
                  convert=number_or_str(int))
parser.add_option('crop', '--crop-space', nargs=1, default=0, convert=float)
parser.add_option('regularization', '--regularization', nargs=1, default='jtv',
                  validation=cli.Validations.choice(['no', 'tkh', 'tv', 'jtv']))
parser.add_option('meetup', '--meetup', nargs='?', default=False,
                  convert=bool_or_str, action=cli.Actions.store_true)
parser.add_option('lam', '--lam', nargs=1, default=10., convert=float)
parser.add_option('lam_pd', '--lam-pd', nargs=1, default=None, convert=float)
parser.add_option('lam_t1', '--lam-t1', nargs=1, default=None, convert=float)
parser.add_option('lam_t2', '--lam-t2', nargs=1, default=None, convert=float)
parser.add_option('lam_mt', '--lam-mt', nargs=1, default=None, convert=float)
parser.add_option('lam_meetup', '--lam-meetup', nargs=1, default=1e5, convert=float)

# optim options
parser.add_option('levels', '--nb-levels', nargs=1, default=1, convert=int)
parser.add_option('iter', '--max-iter', nargs=1, default=10, convert=int)
parser.add_option('tol', '--tolerance', nargs=1, default=1e-4, convert=float)
parser.add_option('solver', '--solver', nargs='+', default=['cg'])

# generic options
parser.add_option('verbose', ('-v', '--verbose'),
                  nargs='?', default=0, convert=int,
                  action=cli.Actions.store_value(1),
                  help='Level of verbosity')
parser.add_option('gpu', '--gpu',
                  nargs='?', default='cpu', convert=int,
                  action=cli.Actions.store_value(0),
                  help='Use GPU (if available)')
parser.add_option('gpu', '--cpu', nargs=0,
                  action=cli.Actions.store_value('cpu'),
                  help='Use CPU')
parser.add_option('odir', ('-o', '--output-dir'), nargs=1,
                  help='Output directory')
parser.add_option('framerate', '--framerate', nargs=1, convert=float,
                  default=1., help='Framerate of plotting function, in Hz')

