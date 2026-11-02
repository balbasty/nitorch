from nitorch.core import cli


help = r"""[nitorch] simplereg

Pairwise affine + nonlinear registration with a classical syntax.
This command is merely a wrapper around nitorch register, that exposes
fewer parameters. It registers a single pair of images with an affine
(rigid by default) transform and a nonlinear (stationary velocity field
by default) transform, optimized jointly (interleaved).
Use --no-nonlin for an affine-only registration (as nitorch affreg)
or --no-affine for a nonlinear-only registration.

usage:
    nitorch simplereg -m MOVING -f FIXED [options]

Input options:
    -m, --moving PATH               Moving image
    -f, --fixed PATH                Fixed image
    -i, --init PATH                 Initial affine to apply to `moving`.

Preprocessing options (applied to both images):
        --fwhm VAL [UNIT]           Smooth images before using them. Unit can be {vox,mm} [0 vox]
        --pad VAL [UNIT]            Pad images before using them. Unit can be {vox,mm} [0 vox]
        --bound BND                 Boundary condition {dct2,dct1,dst2,dst1,dft,zero,nearest} [dct2]
        --order N                   Interpolation order {0..7} [1]
        --rescale [MN] MX           Rescale images so that their MN/MX percentiles match (0, 1) [0 95]
                                    (--rescale 0: no rescaling)
        --discretize [N]            Discretize images into N bins [no; N=256 if no value]

Output options:
    -o, --output PATH               Output path to the affine [{dir}/{name}.lta]
    -w, --warp PATH                 Output path to the nonlinear field [{dir}/{name}.nii.gz]
    -x, --moved PATH                Output path to moved image [{dir}/{base}.moved{ext}]
    -r, --resliced PATH             Output path to resliced image [false: not written]
    {dir} is the directory of the fixed image, {name} is the transform
    type (rigid/affine for the affine; svf/shoot/smalldef for the nonlinear
    field), and {base}/{ext} are the moving image's base name and extension.

Loss options:
    -l, --loss                      Objective function
       [nmi]                            Normalized mutual information
        mse                             Mean squared error
        mad                             Median absolute deviation
        tuk                             Tukey's biweight function
        cc                              Correlation coefficient
        lcc                             Local correlation coefficient
        gmm                             Gaussian Mixture likelihood
        lgmm                            Local Gaussian Mixture likelihood
        emmi                            EM-based mutual information
    -k, --kernel [SIZE] [UNIT]      Kernel size (for lcc/lgmm). Unit can be {vox,mm,pct} [10 pct]
    -s, --symmetric                 Use a symmetric loss [false]

Affine options:
    -a, --affine                    12-param affine [default is 6-param rigid]
        --no-affine                 Do not estimate an affine transform
    -z, --optimizer [OPT]           Optimizer {gn,powell,lbfgs} [auto: powell for nmi, else as nonlin]
    -n, --max-iter VAL              Max number of iterations [50]
    -t, --tolerance VAL             Tolerance for early stopping [1e-5]
        --line-search VAL           Number of line search iterations [wolfe]

Nonlinear options:
        --nonlin NAME               Nonlinear model {svf,shoot,smalldef} [svf]
        --no-nonlin                 Do not estimate a nonlinear transform
        --penalty VAL               Global regularization factor [1]
        --absolute VAL              Penalty on absolute displacements [1e-4]
        --membrane VAL              Penalty on membrane energy [1e-3]
        --bending VAL               Penalty on bending energy [0.2]
        --lame VAL [VAL]            Penalty on linear elastic energy [0.05 0.2]
        --nonlin-optimizer OPT      Optimizer {gn,lbfgs} [auto: gn if the loss has a Hessian, else lbfgs]
        --nonlin-iter VAL           Max number of iterations [10]

Joint optimization options:
        --outer-iter VAL            Number of interleaved affine/nonlinear iterations [5]
    -p, --pyramid [LEVELS]          Pyramid levels to process [0:3]
        --no-pyramid                Do not use a pyramid

Other options:
    -h, --help                      Display this help
    -v, --verbose [LVL]             Level of verbosity [0], 1=print, 2=print more, 3=plot
    --cpu [THREADS], --gpu [ID]     Device to use [cpu]

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


def parse_range(x):
    if ':' not in x:
        return int(x)
    x = x.split(':')
    if len(x) == 2:
        x = [*x, '']
    elif len(x) == 1:
        x = ['', *x, '']
    start, stop, step = x
    start = int(start or 0)
    step = int(step or 1)
    stop = int(stop)
    return range(start, stop, step)


def convert_device(device):
    def _convert(x):
        return (device, int(x))
    return _convert


parser = cli.CommandParser('simplereg', help=help)

# Input options
parser.add_option('moving', ('-m', '--moving'), nargs=1, help='Moving image')
parser.add_option('fixed', ('-f', '--fixed'), nargs=1, help='Fixed image')
parser.add_option('init', ('-i', '--init'), nargs=1, help='Initial affine')

# Preprocessing options, applied to both images.
# None = not forwarded, i.e., nitorch register's default is used.
bounds = ['dct2', 'dct1', 'dst2', 'dst1', 'dft', 'zero', 'nearest']
parser.add_option('fwhm', '--fwhm', nargs='+', default=None,
                  convert=number_or_str(float))
parser.add_option('pad', '--pad', nargs='+', default=None,
                  convert=number_or_str(float))
parser.add_option('bound', '--bound', nargs=1, default=None,
                  validation=cli.Validations.choice(bounds))
parser.add_option('order', '--order', nargs=1, default=None, convert=int)
parser.add_option('rescale', '--rescale', nargs='1*2', default=None,
                  convert=float)
parser.add_option('discretize', '--discretize', nargs='?', default=None,
                  convert=int, action=cli.Actions.store_value(256))

# Output options
parser.add_option('output', ('-o', '--output'), nargs=1, default='{dir}/{name}.lta', convert=bool_or_str)
parser.add_option('warp', ('-w', '--warp'), nargs=1, default='{dir}/{name}.nii.gz', convert=bool_or_str)
parser.add_option('moved', ('-x', '--moved'), nargs=1, default='{dir}/{base}.moved{ext}', convert=bool_or_str)
parser.add_option('resliced', ('-r', '--resliced'), nargs=1, default=False, convert=bool_or_str)

# Loss options
losses = ['nmi', 'mse', 'mad', 'tuk', 'cc', 'lcc', 'gmm', 'lgmm', 'emmi']
parser.add_option('loss', ('-l', '--loss'), nargs=1, default='nmi',
                  validation=cli.Validations.choice(losses))
parser.add_option('kernel', ('-k', '--kernel'), nargs='+', default=[10, 'pct'],
                  convert=number_or_str(float))
parser.add_option('symmetric', ('-s', '--symmetric'), nargs='?', default=False,
                  convert=bool_or_str, action=cli.Actions.store_true)

# Affine options
parser.add_option('affine', ('-a', '--affine'), nargs='?', default=False,
                  convert=bool_or_str, action=cli.Actions.store_true)
parser.add_option('do_affine', '--no-affine', nargs=0, default=True,
                  action=cli.Actions.store_false)
optims = ['gn', 'lbfgs', 'powell']
parser.add_option('optimizer', ('-z', '--optimizer'), nargs=1, validation=cli.Validations.choice(optims))
parser.add_option('max_iter', ('-n', '--max-iter'), nargs=1, default=50, convert=int)
parser.add_option('tolerance', ('-t', '--tolerance'), nargs=1, default=1e-5, convert=float)
parser.add_option('search', '--line-search', nargs=1, default='wolfe', convert=number_or_str(int))

# Nonlinear options.
# None = not forwarded, i.e., nitorch register's default is used.
nonlins = ['svf', 'shoot', 'smalldef']
parser.add_option('nonlin', '--nonlin', nargs=1, default='svf',
                  validation=cli.Validations.choice(nonlins))
parser.add_option('do_nonlin', '--no-nonlin', nargs=0, default=True,
                  action=cli.Actions.store_false)
parser.add_option('penalty', '--penalty', nargs=1, default=None, convert=float)
parser.add_option('absolute', '--absolute', nargs=1, default=None, convert=float)
parser.add_option('membrane', '--membrane', nargs=1, default=None, convert=float)
parser.add_option('bending', '--bending', nargs=1, default=None, convert=float)
parser.add_option('lame', '--lame', nargs='1*2', default=None, convert=float)
parser.add_option('nonlin_optimizer', '--nonlin-optimizer', nargs=1, default=None,
                  validation=cli.Validations.choice(['gn', 'lbfgs']))
parser.add_option('nonlin_iter', '--nonlin-iter', nargs=1, default=None, convert=int)

# Joint optimization options
parser.add_option('outer_iter', '--outer-iter', nargs=1, default=None, convert=int)
parser.add_option('pyramid', ('-p', '--pyramid'), nargs='+', default=[range(3)],
                  convert=parse_range)
parser.add_option('pyramid', '--no-pyramid', nargs=0, action=cli.Actions.store_false)

# Other options
parser.add_option('verbose', ('-v', '--verbose'),
                  nargs='?', default=0, convert=int,
                  action=cli.Actions.store_value(1),
                  help='Level of verbosity')
parser.add_option('device', '--gpu', default=('cpu', None),
                  nargs='?', convert=convert_device('gpu'),
                  action=cli.Actions.store_value(('gpu', None)),
                  help='Use GPU (if available)')
parser.add_option('device', '--cpu',
                  nargs='?', convert=convert_device('cpu'),
                  action=cli.Actions.store_value(('cpu', None)),
                  help='Use CPU')
