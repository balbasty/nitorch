from nitorch.core import cli


help1 = r"""[nitorch] register

usage:
    nitorch register [options]
                     @loss [NAME] [FACTOR] @@fix *FILE @@mov *FILE ...
                     @affine [NAME] [FACTOR] @@optim [NAME] ...
                     @nonlin [NAME] [FACTOR] @@optim [NAME] ...
                     @optim [NAME] ...

    nitorch register -h [LEVEL]
        LEVEL can be:
           [1]      This help
            2       This help + NAME choices
            3       This help + NAME choices + options
"""


help2 = r"""[nitorch] register

usage:
    nitorch register [options]
                     @loss    [NAME] [FACTOR] @@fix *FILE @@mov *FILE ...
                     @affine  [NAME] [FACTOR] @@optim [NAME] ...
                     @nonlin  [NAME] [FACTOR] @@optim [NAME] ...
                     @optim   [NAME] ...
                     @pyramid [NAME] ...

@loss options:
    NAME can take values (with specific sub-options):
       [mi, nmi]                        Mutual information (can be normalized)
        mse, l2                         Mean squared error (can be weighted)
        mad, l1                         Median absolute deviation (can be weighted)
        tuk, tukey                      Tukey's biweight function (can be weighted)
        cc, ncc                         Correlation coefficient (= Normalized cross correlation)
        lcc, lncc                       Local correlation coefficient
        gmm                             Gaussian Mixture likelihood
        lgmm                            Local Gaussian Mixture likelihood
        emmi                            EM-based Mutual Information
        cat, cce                        Categorical cross-entropy
        dice, f1                        Dice coefficient

@affine options:
    NAME can take values:
        t, translation                  Translations only
        o, rotation                     Rotations only
       [r, rigid]                       Translations + Rotations
        s, similitude                   Translations + Rotations + Iso zoom
        a, affine                       Full affine

@nonlin options:
    NAME can take values:
       [v, svf]                         Stationary velocity field
        g, shoot                        Geodesic shooting
        d, smalldef                     Dense deformation field

@optim options:
    NAME can take values:
       [i, interleaved]                 Interleaved optimization of affine and nonlin
        s, sequential                   Sequential optimization of affine, then nonlin

@@optim options:
    NAME can take values:
        gn, gauss-newton                Second-order method. Not all losses can be used.
        gd, gradient-descent            Simple gradient descent
        mom, momentum                   Gradient descent with momentum
        nes, nesterov                   Gradient descent with Nesterov momentum
        ogm, optimized-gradient         Optimized gradient method of Kim & Fessler
        lbfgs                           Limited-memory BFGS

General options:
    --cpu [THREADS], --gpu          Device to use [cpu]
    -d, --dim [DIM]                 Number of spatial dimensions [try to guess]
    -o, --output-dir                Output directory [same as input files]
    -h, --help [LEVEL]              Display this help: [1=minimal], 2=normal, 3=more details
    -v, --verbose [LVL]             Level of verbosity [1=print], 2=plot
    -r, --framerate                 Framerate of plotting function, in Hz [1]

"""

help3 = r"""[nitorch] register

usage:
    nitorch register [options]
                    @loss [NAME] [FACTOR] ...
                        @@fix *FILE [-o *FILE] [-r *FILE] [-p *LVL] ...
                        @@mov *FILE [-o *FILE] [-r *FILE] [-p *LVL] ...
                    @affine [NAME] [FACTOR] [-i *FILE] ...
                        @@optim [NAME] ...
                    @nonlin [NAME] [FACTOR] [-i *FILE] ...
                        @@optim [NAME] ...
                    @optim [NAME] ...
                    @pyramid [NAME] ...

@loss options:
    NAME can take values (with specific sub-options):
       [mi, nmi]                        Mutual information (can be normalized)
            -m, --norm NAME                 Normalization: [studholme], arithmetic, geometric, no
            -b, --bins VAL [VAL]            Number of bins in the joint histogram [32]
            -f, --fwhm VAL                  Full width half-max of histogram smoothing [1]
            -n, --order VAL                 Spline order of parzen window [1]
        mse, l2                         Mean squared error (can be weighted)
            -w, --weight [VAL]              Weight (= Gaussian precision): [auto]
        mad, l1                         Median absolute deviation (can be weighted)
            -w, --weight [VAL]              Weight (= Laplace precision): [auto]
            -j, --joint                     Joint weight map across channels [false]
        tuk, tukey                      Tukey's biweight function (can be weighted)
            -w, --weight [VAL]              Weight (= Laplace precision): [auto]
            -j, --joint                     Joint weight map across channels [false]
        cc, ncc                         Correlation coefficient (= Normalized cross correlation)
        lcc, lncc                       Local correlation coefficient
            -p, --patch *VAL [UNIT]         Patch size [10] and unit: vox, mm, [pct]
        gmm                             Gaussian Mixture likelihood
            -b, --bins                      Number of clusters in the mixture [6]
            -n, --iter VAL                  Number of EM iterations [128]
        lgmm                            Local Gaussian Mixture likelihood
            -b, --bins                      Number of clusters in the mixture [6]
            -n, --iter VAL                  Number of EM iterations [128]
            -p, --patch *VAL [UNIT]         Patch size [10] and unit: vox, mm, [pct]
            -k, --kernel VAL                Kernel type: [gauss]/square
            -s, --stride *VAL               Stride between patches [0]
        emmi                            EM-based Mutual Information
        cat, cce                        Categorical cross-entropy
        dice, f1                        Dice coefficient
            -w, --weight *VAL               Weight per class [1]
    FACTOR must be a scalar value [1]
    Common options:
        -s, --symmetric                 Make loss symmetric [False]
        -z, --slicewise [AXIS=-1]       Make loss slice-wise [False]

@@fix/mov options:
    *FILES must be one or several filenames, which will be concatenated
    across the channel dimension.
    -o, --output [PATH]             Path to the output with minimal reslicing: [True={base}.moved.{ext}]
    -r, --resliced [PATH]           Path to the output resliced to the other image's space: [False], True={base}.resliced.{ext}
    -b, --bound BND                 Boundary conditions [dct2]
    -n, --order N                   Interpolation order [1]
    -a, --affine *PATH              Path to one or more affine transforms to apply
    -w, --world PATH                Path to an orientation matrix. Overrides the one from the data file.
    -f, --fwhm *F                   Smooth image before using it.
    -z, --pad *P                    Pad image before using it.
    -s, --rescale [[MN], MX]        Rescale image so that its MN/MX percentiles match (0, 1). [0, 95]
    -d, --discretize [N]            Discretize the image into N [default: 256] bins. [False]
    -l, --label *VAL                Specifies that the file is a label map [False]
                                    If no argument given, labels are [all but zero]
    -m, --mask FILE                 Path to a mask of voxels to *include* [all]
        --name NAME                 A name to use with `@nonlin --fov`
    -x, --missing *VAL              Values that should be considered missing [0]
        --no-missing                No value should be considered missing
        --mind  [FWHM=1 [RADIUS=0]] Compute MIND features
    -c, --channels *C               Channels to load. Can be a range start:stop:step [:]

@affine options:
    NAME can take values:
        t, translation                  Translations only
        o, rotation                     Rotations only
       [r, rigid]                       Translations + Rotations
        s, similitude                   Translations + Rotations + Iso zoom
        a, affine                       Full affine
    FACTOR must be a scalar value [1] and is a global penalty factor
    Common options:
        -p, --position                  Position of the affine: [sym], mov, fix
        -g, --progressive               Progressive optimization (t -> r -> s -> a) [false]
        -i, --input                     Path to initial transform
        -o, --output                    Path to the output transform: [{dir}/{name}.lta]
        -2d [AXIS=2]                    Force transform to be 2d about AXIS

@nonlin options:
    NAME can take values:
       [v, svf]                         Stationary velocity field
        g, shoot                        Geodesic shooting
        d, smalldef                     Dense deformation field
    FACTOR must be a scalar value [1] and is a global penalty factor
    Common options:
        -i, --input                     Path to initial transform
        -o, --output                    Path to the output transform: [{dir}/{name}.nii.gz]
        -h, --hessian                   Path to the output hessian: [{dir}/{name}_hessian.nii.gz]
        -s, --steps                     Number of integration steps
        -a, --absolute                  Penalty on absolute displacements (0th) [1e-4]
        -m, --membrane                  Penalty on membrane energy (1st) [1e-3]
        -b, --bending                   Penalty on bending energy (2nd) [0.2]
        -l, --lame                      Penalty on linear elastic energy [0.05, 0.2]]
        -v, --voxel-size VAL [UNIT]     Voxel size and unit [100 %]
        -f, --fov *NAMES                Name of inputs used to compute mean space [all]
        -p, --pad VAL [UNIT]            Pad field of view by some amount [0 %]
        -2d [AXIS=2]                    Force transform to be 2d about AXIS

@optim options:
    NAME can take values:
       [i, interleaved]                 Interleaved optimization of affine and nonlin
            -n, --max-iter                  Maximum number of iterations
            -t, --tolerance                 Tolerance for early stopping
        s, sequential                   Sequential optimization of affine, then nonlin

@@optim options:
    The default solver is chosen based on the loss.
    Usually, losses that define a Hessian are solved by Gauss-Newton,
    and the others by LBFGS for the affine component and Nesterov with
    Wolfe line search for the nonlinear component.

    NAME can take values (with options):
        gn, gauss-newton                Second-order method. Not all losses can be used.
            -m, --marquardt                  Levenberg-Marquardt regularization [auto]
            -q, --solver                     Linear solver: [cg], relax (used for @nonlin only)
            -j, --sub-iter                   Number of linear solver iterations [32]
        gd, gradient-descent            Simple gradient descent
        cg, conjugate-gradient          Conjugate gradient descent
            -b, --beta                      Update rule: [pr], fr, hs, dy
        mom, momentum                   Gradient descent with momentum
            -m, --momentum                  Momentum factor [0.9]
        nes, nesterov                   Gradient descent with Nesterov momentum
            -m, --momentum                  Momentum factor [0=auto]
            -r, --restart                   Automatic restart [True]
            -o, --output FILE               Write momentum maps
        ogm, optimized-gradient         Optimized gradient method of Kim & Fessler
            -m, --momentum                  Momentum factor [0=auto]
            -x, --relax                     Relaxation factor [0=auto]
            -r, --restart                   Automatic restart [True]
            -o, --output FILE               Write momentum maps
        lbfgs                           Limited-memory BFGS
            -h, --history                   History size [100]
        none                            Do not optimize
    Common options:
        -l, --lr                        Learning rate [1]
        -s, --line-search               Number of backtracking line search [wolfe]
        -t, --tolerance                 Tolerance for early stopping [1e-4]
        -c, --crit                      Stopping criterion: [diff], gain
        -n, --max-iter                  Maximum number of iterations
                                        [interleaved: affine=50,  nonlin=10,
                                         sequential:  affine=100, nonlin=50]

@pyramid options:
    NAME can take values:
       [g, gaussian]                    Gaussian pyramid
        a, average                      Average pooling
        m, median                       Median pooling
        s, stride                       Strides
    Common options:
            --max-size                  Maximum size along a dimension
            --min-size                  Minimum size along a dimension
            --max-vx                    Maximum voxel size along a dimension
            --min-vx                    Minimum voxel size along a dimension
        -l, --levels *LVL               Pyramid levels. Can be a range start:stop:step [:3]
        -c, --concurrent                Register pyramid levels concurrently [false]

General options:
    --cpu [THREADS], --gpu [ID]     Device to use [cpu]
    -d, --dim [DIM]                 Number of spatial dimensions [try to guess]
    -o, --output-dir                Output directory [same as input files]
    -h, --help [LEVEL]              Display this help: 0=minimal, [1=normal], 2=more details
    -v, --verbose [LVL]             Level of verbosity [0]

"""

help = {1: help1, 2: help2, 3: help3}


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
    if stop == '':
        out = slice(start, None, step)
    else:
        out = range(start, int(stop), step)
    return out


# Main options
def convert_device(device):
    def _convert(x):
        return (device, int(x))
    return _convert


parser = cli.CommandParser('register', help=help1, add_help=False)
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
parser.add_option('help', ('-h', '--help'),
                  nargs='?', default=False, convert=int,
                  action=cli.Actions.store_value(1),
                  help='Display this help. Call it with 1, 2, or 3 for '
                       'different levels of detail.')
parser.add_option('dim', ('-d', '--dim'), nargs=1, convert=int,
                  help='Number of spatial dimensions')
parser.add_option('odir', ('-o', '--output-dir'), nargs=1,
                  help='Output directory')
parser.add_option('framerate', ('-r', '--framerate'), nargs=1, convert=float,
                  default=1., help='Framerate of plotting function, in Hz')

# Loss group
loss_aliases = {'nmi': 'mi', 'l1': 'mse', 'l2': 'mad', 'tukey': 'tuk',
                'ncc': 'cc', 'lncc': 'lcc', 'cce': 'cat', 'f1': 'dice',
                'entropy': 'ent', 'sqz': 'squeezed'}
loss_choices = list(loss_aliases.values()) + ['gmm', 'lgmm', 'emmi', 'prod', 'dot', 'ndot', 'extra']
loss_choices = cli.Positional('name', nargs='?', default='mi',
                              validation=cli.Validations.choice(loss_choices),
                              convert=lambda x: loss_aliases.get(x, x),
                              help='Name of the image loss')
loss = cli.NamedGroup('loss', loss_choices, '@loss', n='+',
                      help='A loss between two images')
loss.add_positional('factor', nargs='*', default=[1.], convert=float,
                    help='Weight it this component in the global loss')
loss.add_option('symmetric', ('-s', '--symmetric'), nargs=0, default=False,
                help='Make the loss symmetric')
loss.add_option('slicewise', ('-z', '--slicewise'), nargs='?', default=False,
                action=cli.Actions.store_value(-1), convert=int,
                help='Make the loss slice-wise')
# conditional options
weight_option = cli.Option('weight', ('-w', '--weight'), nargs='1',
                           default=None, convert=number_or_str(float),
                           help='Weight (= precision) or the loss.')
joint_option = cli.Option('joint', ('-j', '--joint'), nargs='0',
                          default=False, action=cli.Actions.store_true,
                          help='Joint weight map across channels.')
patch_option = cli.Option('patch', ('-p', '--patch'), nargs='1*', default=10,
                          convert=number_or_str(float), help='Patch size')
stride_option = cli.Option('stride', ('-s', '--stride'), nargs='1*', default=1,
                           convert=int, help='Strides between patches.')
cluster_option = cli.Option('bins', ('-b', '--bins'), nargs=1, default=6,
                            convert=int, help='Number of bins in the mixture')
kernel_option = cli.Option('kernel', ('-k', '--kernel'), nargs=1, default='g',
                           help='Kernel type: [gauss], square')
iter_option = cli.Option('max_iter', ('-n', '--max-iter'), nargs=1, default=128,
                         convert=int, help='Maximum number of EM iterations')
loss.add_suboption('mi', 'norm', ('-m', '--norm'),
                   validation=cli.Validations.choice(['studholme', 'arithmetic', 'geometric', 'no']),
                   nargs='1', default='studholme',
                   help='Normalization method')
loss.add_suboption('mi', 'bins', ('-b', '--bins'), nargs='1*2?', default=32,
                   convert=int, help='Number of bins in joint histogram')
loss.add_suboption('mi', 'fwhm', ('-f', '--fwhm'), nargs='1*', default=0,
                   convert=float, help='Smoothing of the joint histogram, in bins.')
loss.add_suboption('mi', 'order', ('-n', '--order'), nargs='1*2?', default=[3],
                   convert=int, help='Spline order of the joint histogram')
loss.add_suboption('mi', patch_option)
loss.add_suboption('ent', 'bins', ('-b', '--bins'), nargs='1*2?', default=32,
                   convert=int, help='Number of bins in joint histogram')
loss.add_suboption('ent', 'fwhm', ('-f', '--fwhm'), nargs='1*', default=0,
                   convert=float, help='Smoothing of the joint histogram, in bins.')
loss.add_suboption('ent', 'order', ('-n', '--order'), nargs='1*2?', default=[3],
                   convert=int, help='Spline order of the joint histogram')
loss.add_suboption('lcc', patch_option)
loss.add_suboption('lcc', stride_option)
loss.add_suboption('lcc', kernel_option)
loss.add_suboption('mse', weight_option)
loss.add_suboption('mad', weight_option)
loss.add_suboption('tuk', weight_option)
loss.add_suboption('sqz', weight_option)
loss.add_suboption('mad', joint_option)
loss.add_suboption('tuk', joint_option)
loss.add_suboption('dice', 'weight', ('-w', '--weight'), nargs='*', default=False,
                   convert=float, action=cli.Actions.store_true,
                   help='Weight of each class')
loss.add_suboption('gmm', cluster_option)
loss.add_suboption('gmm', iter_option)
loss.add_suboption('lgmm', cluster_option)
loss.add_suboption('lgmm', patch_option)
loss.add_suboption('lgmm', stride_option)
loss.add_suboption('lgmm', kernel_option)
loss.add_suboption('lgmm', iter_option)
# fix/mov subgroups
file = cli.Group('file', n=1, help='Volume to register')
file.add_positional('files', nargs='1*', help='File names')
file.add_option('output', ('-o', '--output'), nargs='?',
                default='{dir}{sep}{base}.moved{ext}',
                convert=bool_or_str,
                help='Path to the output with minimal reslicing')
file.add_option('resliced', ('-r', '--resliced'), nargs='?',
                default=False, convert=bool_or_str,
                action=cli.Actions.store_value('{dir}{sep}{base}.resliced{ext}'),
                help='Path to the output resliced to the other \'s space')
file.add_option('fwhm', ('-f', '--fwhm'), nargs='0*',
                default=[0], convert=number_or_str(float),
                action=cli.Actions.store_value([1, 'vox']),
                help='Smooth the image. A unit can be provided (default: vox)')
file.add_option('pad', ('-z', '--pad'), nargs='0*',
                default=[0], convert=number_or_str(float),
                help='Pad the image. A unit can be provided (default: vox)')
file.add_option('bound', ('-b', '--bound'), nargs=1, default='dct2',
                validation=cli.Validations.choice(['dct2', 'dct1', 'dft', 'dst2', 'dst1', 'zero', 'replicate']),
                help='Boundary condition')
file.add_option('order', ('-n', '--order'), nargs=1, default=1,
                convert=int, help='Interpolation order')
file.add_option('extrapolate', ('-x', '--extrapolate'), nargs=0, default=True,
                convert=bool, help='Extrapolate out-of-bounds')
file.add_option('affine', ('-a', '--affine'), nargs='+', default=[],
                help='Path to one or more world-to-world affine transforms to apply')
file.add_option('world', ('-w', '--world'), nargs=1,
                help='Path to a voxel-to-world orientation matrix. Overrides the one form the data file')
file.add_option('rescale', ('-s', '--rescale'), nargs='1*2', convert=float,
                default=(0, 95), help='Rescale image so that its MN/MX percentiles match (0, 1)')
file.add_option('name', '--name', nargs=1,
                help='Give a unique name to te file so that it can be referenced in other options')
file.add_option('label', ('-l', '--label'), nargs='*', default=False,
                action=cli.Actions.store_true, convert=int,
                help='The file is a label map (and specify labels)')
file.add_option('discretize', ('-d', '--discretize'), nargs='?', convert=int,
                action=cli.Actions.store_value(256),
                default=0, help='Discretize the image into N discrete bins')
file.add_option('mask', ('-m', '--mask'), nargs=1, default=None,
                help='Path to a mask of voxels to include')
file.add_option('missing', ('-x', '--missing'), nargs='+', convert=float,
                default=[0], help='Missing values')
file.add_option('missing', '--no-missing', nargs=0,
                action=cli.Actions.store_value([]), help='No missing values')
file.add_option('mind', '--mind', nargs='*2', default=[], convert=float,
                action=cli.Actions.store_value([1, 2]))
file.add_option('channels', ('-c', '--channels'), nargs='1*', convert=parse_range, default=[slice(None)])
fix = cli.Group('fix', '@@fix', n=1)
fix.copy_from(file)
mov = cli.Group('mov', '@@mov', n=1)
mov.copy_from(file)
loss.add_group(fix)
loss.add_group(mov)

# optim subgroup
optim_aliases = {'gauss-newton': 'gn',
                 'gradient-descent': 'gd',
                 'conjugate-gradient': 'cg',
                 'momentum': 'mom',
                 'nesterov': 'nes',
                 'optimized-gradient': 'ogm',
                 'powell': 'pow'}
optim_choices = list(optim_aliases.values()) + ['lbfgs', 'unset', 'none']
optim_choices = cli.Positional('name', nargs='?', default='unset',
                               validation=cli.Validations.choice(optim_choices),
                               convert=lambda x: optim_aliases.get(x, x),
                               help='Name of the optimizer')
optim = cli.NamedGroup('optim', optim_choices, '@@optim', n='?')
optim.add_option('lr', ('-l', '--lr'), nargs=1, default=1, convert=float,
                 help='Learning rate')
optim.add_option('max_iter', ('-n', '--max-iter'), nargs=1, default=None,
                 convert=int, help='Maximum number of iterations')
optim.add_option('fmg', ('-g', '--fmg'), nargs=1, default=2,
                 convert=int, help='Number of FMG cycles (do no use FMG if 0)')
optim.add_option('line_search', ('-s', '--line-search'), nargs=1,
                 default='wolfe', convert=number_or_str(int),
                 help='Number of backtracking line search')
optim.add_option('tolerance', ('-t', '--tolerance'), nargs=1,
                 convert=float, default=1e-4, help='Tolerance for early stopping')
optim.add_option('crit', ('-c', '--crit'), nargs=1,
                 default='diff', validation=cli.Validations.choice(['gain', 'diff']),
                 help='Criterion for early stopping')
optim.add_suboption('gn', 'marquardt', ('-m', '--marquardt'), nargs=1,
                    default=None, convert=number_or_str(float),
                    help='Levenberg-Marquardt regularization')
optim.add_suboption('gn', 'solver', ('-q', '--solver'), nargs=1,
                    default='cg', validation=cli.Validations.choice(['cg', 'relax']),
                    help='Linear solver')
optim.add_suboption('gn', 'sub_iter', ('-j', '--sub-iter'), nargs=1,
                    default=None, convert=int,
                    help='Number of linear solver iterations')
optim.add_suboption('mom', 'momentum', ('-m', '--momentum'), nargs=1,
                    default=0.9, convert=float, help='Momentum factor')
optim.add_suboption('nes', 'momentum', ('-m', '--momentum'), nargs=1,
                    default=0, convert=float, help='Momentum factor')
optim.add_suboption('nes', 'restart', ('-r', '--restart'), nargs=1,
                    default=True, convert=bool, help='Automatic restart')
optim.add_suboption('nes', 'output', ('-o', '--output'), nargs='?',
                    default='', action=cli.Actions.store_value('{dir}{sep}momentum.{iter}.nii.gz'),
                    help='Write momentum maps')
optim.add_suboption('ogm', 'momentum', ('-m', '--momentum'), nargs=1,
                    default=0, convert=float, help='Momentum factor')
optim.add_suboption('ogm', 'relax', ('-x', '--relax'), nargs=1,
                    default=0, convert=float, help='Relaxation factor')
optim.add_suboption('ogm', 'restart', ('-r', '--restart'), nargs=1,
                    default=True, convert=bool, help='Automatic restart')
optim.add_suboption('ogm', 'output', ('-o', '--output'), nargs='?',
                    default='', action=cli.Actions.store_value('{dir}{sep}momentum.{iter}.nii.gz'),
                    help='Write momentum maps')
optim.add_suboption('lbfgs', 'history', ('-h', '--history'), nargs=1,
                    default=100, convert=int, help='History size')
cg_aliases = {'fletcher-reeves': 'fr',
              'polak-ribiere': 'pr',
              'hestenes-stiefel': 'hs',
              'dai-yuan': 'dy'}
cg_choices = list(cg_aliases.values())
optim.add_suboption('cg', 'beta', ('-b', '--beta'), nargs=1, default='pr',
                    validation=cli.Validations.choice(cg_choices),
                    convert=lambda x: cg_aliases.get(x, x))

# affine group
affine_aliases = {'t': 'translation',
                  'o': 'rotation',
                  'r': 'rigid',
                  's': 'similitude',
                  'a': 'affine'}
affine_choices = list(affine_aliases.values()) + ['unset']
affine_choices = cli.Positional('name', nargs='?', default='rigid',
                                validation=cli.Validations.choice(affine_choices),
                                convert=lambda x: affine_aliases.get(x, x),
                                help='Name of the affine transform')
affine = cli.NamedGroup('affine', affine_choices, '@affine', n='?', make_default=False)
affine.add_positional('factor', nargs='*', default=[1.], convert=float,
                      help='Penalty factor')
affine.add_option('position', ('-p', '--position'), default='sym', nargs=1,
                  validation=cli.Validations.choice(['sym', 'mov', 'fix']),
                  help='Position of the affine transform in the model')
affine.add_option('output', ('-o', '--output'), nargs=1,
                  default='{dir}{sep}{name}.lta',
                  help='Path to the output transform')
affine.add_option('init', ('-i', '--input', '--init'), nargs=1,
                  help='Path to the input transform')
affine.add_option('progressive', ('-g', '--progressive'), default=False,
                  nargs='?',  convert=bool_or_str)
affine.add_option('is2d', '-2d', default=False, nargs='?',
                  action=cli.Actions.store_value(2),
                  convert=number_or_str(int))
affine.add_group(optim)

# nonlin group
nonlin_aliases = {'v': 'svf',
                  'g': 'shoot',
                  'd': 'smalldef'}
nonlin_choices = list(nonlin_aliases.values()) + ['unset']
nonlin_choices = cli.Positional('name', nargs='?', default='svf',
                                validation=cli.Validations.choice(nonlin_choices),
                                convert=lambda x: nonlin_aliases.get(x, x),
                                help='Name of the nonlinear transform')
nonlin = cli.NamedGroup('nonlin', nonlin_choices, '@nonlin', n='?', make_default=False)
nonlin.add_positional('factor', nargs='*', default=[1.], convert=float,
                      help='Penalty factor')
nonlin.add_option('steps', ('-s', '--steps'), nargs=1, default=8, convert=int,
                  help='Number of integration steps')
nonlin.add_option('absolute', ('-a', '--absolute'), nargs=1, default=1e-4, convert=float,
                  help='Penalty on absolute displacements (0th derivatives)')
nonlin.add_option('membrane', ('-m', '--membrane'), nargs=1, default=1e-3, convert=float,
                  help='Penalty on membrane energy (1st derivatives)')
nonlin.add_option('bending', ('-b', '--bending'), nargs=1, default=0.2, convert=float,
                  help='Penalty on bending energy (2nd derivatives)')
nonlin.add_option('lame', ('-l', '--lame'), nargs='1*2?', default=(0.05, 0.2), convert=float,
                  help='Penalty on linear elastic energy (zooms and shears)')
nonlin.add_option('voxel_size', ('-v', '--voxel-size'), nargs='1*', default=[100, '%'],
                  convert=number_or_str(float), help='Voxel size and unit')
nonlin.add_option('pad', ('-p', '--pad'), nargs='1*', default=[0, '%'],
                  convert=number_or_str(float), help='Pad field of view by some amount')
nonlin.add_option('fov', ('-f', '--fov'), nargs='1*',
                  help='Name of inputs used to compute mean space')
nonlin.add_option('init', ('-i', '--input', '--init'), nargs=1,
                  help='Path to the input transform')
nonlin.add_option('output', ('-o', '--output'), nargs=1,
                  default='{dir}{sep}{name}.nii.gz',
                  help='Path to the output transform')
nonlin.add_option('hessian', ('-h', '--hessian'), nargs=1,
                  default='{dir}{sep}{name}_hessian.nii.gz',
                  help='Path to the output hessian')
nonlin.add_option('is2d', '-2d', default=False, nargs='?',
                  action=cli.Actions.store_value(2),
                  convert=number_or_str(int))
nonlin.add_group(optim)


# interleaved optim group
joptim_aliases = {'i': 'interleaved',
                  's': 'sequential'}
joptim_choices = list(joptim_aliases.values()) + ['unset']
joptim_choices = cli.Positional('name', nargs='?', default='interleaved',
                                validation=cli.Validations.choice(joptim_choices),
                                convert=lambda x: joptim_aliases.get(x, x),
                                help='Joint optimization strategy ')
joptim = cli.NamedGroup('optim', joptim_choices, '@optim', n='?')
joptim.add_suboption('interleaved', 'max_iter', ('-n', '--max-iter'), nargs=1,
                     default=5, convert=int, help='Maximum number of iterations')
joptim.add_suboption('interleaved', 'tolerance', ('-t', '--tolerance'), nargs=1,
                     convert=float, default=1e-5, help='Tolerance for early stopping')

# pyramid group
pyr_aliases = {'g': 'gaussian',
               'a': 'average',
               'm': 'median',
               's': 'stride',
               'n': 'none'}
pyr_choices = list(pyr_aliases.values()) + ['unset']
pyr_choices = cli.Positional('name', nargs='?', default='gaussian',
                             validation=cli.Validations.choice(pyr_choices),
                             convert=lambda x: pyr_aliases.get(x, x),
                             help='Pyramid strategy ')
pyr = cli.NamedGroup('pyramid', pyr_choices, '@pyramid', n='?')
pyr.add_option('min_size', '--min-size', nargs=1, convert=int)
pyr.add_option('max_size', '--max-size', nargs=1, convert=int)
pyr.add_option('min_vx', '--min-vx', nargs=1, convert=float)
pyr.add_option('max_vx', '--max-vx', nargs=1, convert=float)
pyr.add_option('levels', ('-l', '--levels'), nargs='1*', convert=parse_range, default=[range(3)])
pyr.add_option('concurrent', ('-c-', '--concurrent'), nargs=1, convert=bool_or_str, default=False)

# register groups
parser.add_group(loss)
parser.add_group(affine)
parser.add_group(nonlin)
parser.add_group(joptim)
parser.add_group(pyr)
