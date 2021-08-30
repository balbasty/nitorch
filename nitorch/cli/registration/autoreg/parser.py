"""This file parses the commandline arguments"""

from . import struct

help = r"""[nitorch] AUTOgrad REGistration tool

usage: 
    nitorch autoreg <*LOSS> [FACTOR] -fix *FILE [-o *FILE] [-r *FILE] [-pyr *LVL] ...
                                     -mov *FILE [-o *FILE] [-r *FILE] [-pyr *LVL] ...
                    <*TRF> [FACTOR] [-init PATH] [-lr LRNRATE] [-o FILE] [-pyr LVL] 
                           <*REG> [FACTOR]
                    <*OPT> [-nit MAXITER] [-lr LRNRATE] [-stop LIMIT] [-ls [MAXLS]]
                    [-all ...] [-prg] [-gpu|-cpu] [-v] [-h]

    <LOSS> can take values (with additional options): 
        -mi,  --mutual-info         Normalized Mutual Information
                                    (for images with different contrasts)
            -p, --patch *SIZE           Patch size for local MI (0 = global mi)
            -b, --bin BINS              Number of bins in the histogram (32)
            -f, --fwhm WIDTH            Full-width half-max of the bins (1)
            -t, --threshold VAL     Values under this threshold (in fixed) are excluded.
            -n, --order ORDER       Histogram spline order (3)
        -mse, --mean-squared-error  Mean Squared Error
                                    (for images with the same contrast)
        -jtv,  --total-variation    Normalized Joint Total Variation 
                                    (edge-based, for strong intensity non-uniformity)
        -dice, --dice               Dice coefficients (for soft/hard segments)
            -l, --labels *LABELS        Labels to use (default: all except 0)
            -w, ---weights *WEIGHTS     Label-wise weights (default: False)
        -cat,  --categorical        Categorical cross-entropy (for soft segments)
        -oth, --other               Not a loss -> other images to warp at the end.
    Note that the -fix and -mov tags must be placed directly after their 
    corresponding <LOSS> (The <LOSS> tag "opens a group"). Similarly, the 
    interpolation tags (-inter, -bnd, -ext) must be placed directly after 
    their -mov tag.
    Multiple <LOSS> statements can be used, resulting in a composite loss.
    Semi-supervised registration (that use both intensity images and segments)
    can be performed that way.

    <TRF> can take values (with additional options):
        -tr,  --translation         Translation
        -rig, --rigid               Rigid (tr + rot)
        -sim, --similitude          Similitude (tr + rot + iso)
        -aff, --affine              Affine (tr + rot + zoom + shear)
            -z, --zero 0|1              Move zero coord to center of FOV (True)
        -ffd, --free-form           Free-form deformation (cubic)
            -g, --grid *SIZE            Number of nodes (10)
        -dif, --diffeo              Diffeomorphic field
            -sd --smalldef              Penalize the exponentiated transform (False)
    Only one of -tr, -rig, -sim, -aff can be used.
    Only one of -ffd and -diffeo can be used.
    The ffd and diffeomorphic fields always live in the (mean) space of the 
    fixed images.

    <REG> can take values:
        -abs, --absolute            Absolute values
        -mem, --membrane            Membrane energy (first derivatives)
        -ben, --bending             Bending energy (second derivatives)
        -le,  --linear-elastic      Linear-Elastic (zooms and shears)
    By default, the bending energy of the FFD or velocity field is penalized, 
    and no penalty is placed on affine transforms.

    <OPT> can take values:
        -adam,  --adam               ADAM (default)
        -gd,    --gradient-descent   Gradient descent
        -ogm,  --optimized-gradient-descent
        -lbfgs, --lbfgs              L-BFGS
            -h, --history               History size (100)

    Generic options (within group `-all`):
        -inter, --interpolation      Interpolation order (1)
        -bnd,   --bound              Boundary conditions (dct2)
        -ex,    --extrapolate        Extrapolate out-of-bounds data (no)
        -o,     --output             Output file name or directory
        -r,     --resliced           Output file name or directory
        -init,  --init               Initial value: file or values (0) 
        -nit,   --max-iterations     Maximum number of iterations (1000)
        -lr,    --learning-rate      Initial learning rate (0.1)
        -stop,  --stop               Minimum LR value -> early stopping (lr*1e-4)
        -ls,    --line-search        Maximum number of backtracking line searches (0)
        -pyr,   --pyramid *LEVELS    Pyramid levels. Can be a range [start]:stop[:step] (1)
        -exc,   --exclude            Exclude a file from mean space computation.
    If these tags are present within a <LOSS>/<TRF>/<OPT> group, they only 
    apply to this group. If they are present within a `-all` group, they 
    apply to all <LOSS>/<TRF>/<OPT> groups that did not specify them in 
    their group.
    By default, only transformations  are written to disk. (the full 
    affine matrix for affine groups and subgroups, the grid of 
    parameters for ffd, the initial velocity for diffeo).

    Other options
        -cpu, -gpu                   Device to use (cpu)
        -prg, --progressive          Progressively increase degrees of freedom (true)
        -pad, --padding PAD [UNIT]   Padding of mean (nonlin) space (0 %)
        -v, --verbose [LVL]          Level of verbosity (1)

examples:
    # rigid registration between two images
    autoreg -mi -fix target.nii -mov source.nii

    # diffeo + similitude registration between two images
    autoreg -mi -fix target.nii -mov source.nii -dif -sim

    # diffeo + rigid between a template and a segment from SPM
    # > note that the tpm is the *moving* image but we use the -o tag
    #   in the -fix group so that the fixed image is also wrapped to 
    #   moving space.
    autoreg -cat -fix c1.nii c2.nii -o -mov tpm.nii -dif -rig

    # semi-supervised registration
    autoreg -mi   -fix target.nii     -mov source.nii \
            -dice -fix target_seg.nii -mov source_seg.nii \
            -dif

    # registration using gradient descent with a backtracking line 
    autoreg -mi -fix target.nii -mov source.nii -gd -ls

    # register based on intensity images and use it to warp segments
    autoreg -mi  -fix target.nii  -mov source.nii \
            -oth -mov segment.nii -inter 0
"""


class ParseError(RuntimeError):
    pass

# --------------------
#   DEFINE MAIN TAGS
# --------------------

# -
# matching loss
# -
mi = ('-mi', '--mutual-info', '--mutual-information')
mse = ('-mse', '--mean-squared-error')
jtv = ('-jtv', '--total-variation')
dice = ('-dice', '--dice')
cat = ('-cat', '--categorical')
oth = ('-oth', '--other')
match_losses = (*mi, *mse, *jtv, *dice, *cat, *oth)

# -
# transformations
# -
translation = ('-tr', '--translation')
rigid = ('-rig', '--rigid')
similitude = ('-sim', '--similitude')
affine = ('-aff', '--affine')
all_affine = (*translation, *rigid, *similitude, *affine)
diffeo = ('-dif', '--diffeo')
ffd = ('-ffd', '--free-form')
transformations = (*all_affine, *diffeo, *ffd)

# -
# regularizations
# -
absolute = ('-abs', '--absolute')
membrane = ('-mem', '--membrane')
bending = ('-ben', '--bending')
linelastic = ('-le', '--linear-elastic')
regularizations = (*absolute, *membrane, *bending, *linelastic)

# -
# optimizers
# -
adam = ('-adam', '--adam')
gd = ('-gd', '--gradient-descent')
ogm = ('-ogm', '--optimized-gradient-descent')
lbfgs = ('-lbfgs', '--lbfgs')
optim = (*adam, *gd, *ogm, *lbfgs)


# ----------------------------------------------------------------------
#                     PARSE COMMANDLINE ARGUMENTS
# ----------------------------------------------------------------------
def istag(x):
    return x.startswith('-')


def isvalue(x):
    return not istag(x)


def next_istag(args):
    return args and istag(args[0])


def next_isvalue(args):
    return args and isvalue(args[0])


def check_next_isvalue(args, group):
    if not next_isvalue(args):
        raise ParseError(f'Expected a value for tag {group} '
                         f'but found nothing.')


def parse_pyramid(args):
    levels = []
    while next_isvalue(args):
        level, *args = args
        if ':' in level:
            level = level.split(':')
            start = int(level[0] or 1)
            stop = (int(level[1] or start) if len(level) > 1 else start) + 1
            step = (int(level[2] or 1) if len(level) > 2 else 1)
            level = list(range(start, stop, step))
        else:
            level = [int(level)]
        levels.extend(level)
    return args, levels


def parse_file(args, opt):
    """Parse a moving or fixed file"""

    opt.files = list()
    while next_isvalue(args):
        val, *args = args
        opt.files.append(val)

    while args:
        if not next_istag(args):
            break
        tag, *args = args
        if tag in ('-o', '--output'):
            # Path to "updated" output file
            # (updated header + nonlin applied in original space)
            opt.updated = True
            while next_isvalue(args):
                if isinstance(opt.updated, bool):
                    opt.updated = []
                val, *args = args
                opt.updated.append(val)
        elif tag in ('-r', '--resliced'):
            # Path to "resliced" output file
            # (lin + nonlin applied, final space = target space)
            opt.resliced = True
            while next_isvalue(args):
                if isinstance(opt.resliced, bool):
                    opt.resliced = []
                val, *args = args
                opt.resliced.append(val)
        elif tag in ('-inter', '--interpolation'):
            check_next_isvalue(args, tag)
            opt.interpolation, *args = args
            opt.interpolation = int(opt.interpolation)
        elif tag in ('-bnd', '--bound'):
            check_next_isvalue(args, tag)
            opt.bound, *args = args
        elif tag in ('-ex', '--extrapolate'):
            opt.extrapolate = True
        elif tag in ('-pyr', '--pyramid'):
            args, levels = parse_pyramid(args)
            if levels:
                opt.pyramid = levels
            else:
                opt.pyramid = [1]
        else:
            args = [tag, *args]
            break
    return args


def parse_moving_and_target(args, opt):
    """Parse a loss that involves a moving and a target file"""

    if args and not next_istag(args):
        opt.factor, *args = args
        opt.factor = float(opt.factor)

    while args:
        if not next_istag(args):
            break
        tag, *args = args
        if tag in ('-mov', '--moving', '-src', '--source'):
            opt.moving = struct.MovingImageFile()
            args = parse_file(args, opt.moving)
        elif tag in ('-fix', '--fixed', '-trg', '--target'):
            opt.fixed = struct.FixedImageFile()
            args = parse_file(args, opt.fixed)
        elif tag in ('-inter', '--interpolation'):
            check_next_isvalue(args, tag)
            opt.interpolation, *args = args
            opt.interpolation = int(opt.interpolation)
        elif tag in ('-bnd', '--bound'):
            check_next_isvalue(args, tag)
            opt.bound, *args = args
        elif tag in ('-ex', '--extrapolate'):
            opt.extrapolate = True
        elif tag in ('-pyr', '--pyramid'):
            args, levels = parse_pyramid(args)
            if levels:
                opt.pyramid = levels
            else:
                opt.pyramid = [1]
        elif tag in ('-exc', '--exclude'):
            opt.exclude = True
            if next_isvalue(args):
                opt.exclude, *args = args
                if opt.exclude[0].lower() == 't':
                    opt.exclude = True
                elif opt.exclude[0].lower() == 'f':
                    opt.exclude = False
                else:
                    opt.exclude = bool(int(opt.exclude))
        elif isinstance(opt, struct.MILoss) and tag in ('-p', '--patch'):
            opt.patch = list()
            while next_isvalue(args):
                val, *args = args
                opt.patch.append(int(val))
        elif isinstance(opt, struct.MILoss) and tag in ('-f', '--fwhm'):
            check_next_isvalue(args, tag)
            opt.fwhm, *args = args
            opt.fwhm = float(opt.fwhm)
        elif isinstance(opt, struct.MILoss) and tag in ('-b', '--bins'):
            check_next_isvalue(args, tag)
            opt.bins, *args = args
            opt.bins = int(opt.bins)
        elif isinstance(opt, struct.MILoss) and tag in ('-t', '--threshold'):
            check_next_isvalue(args, tag)
            opt.threshold, *args = args
            opt.threshold = float(opt.threshold)
        elif isinstance(opt, struct.MILoss) and tag in ('-n', '--order'):
            check_next_isvalue(args, tag)
            opt.order, *args = args
            if opt.order != 'inf':
                opt.order = int(opt.order)
        elif isinstance(opt, struct.DiceLoss) and tag in ('-w', '--weights'):
            opt.weights = True
            while next_isvalue(args):
                val, *args = args
                if isinstance(opt.weights, bool):
                    opt.weights = []
                    opt.weights.append(float(val))
        elif isinstance(opt, struct.DiceLoss) and tag in ('-l', '--labels'):
            opt.labels = []
            while next_isvalue(args):
                val, *args = args
                opt.labels.append(int(val))
        else:
            args = [tag, *args]
            break
    return args


# --- WORKER FOR MATCHING GROUPS ---------------------------------------
def parse_match(args, options):
    """Parse a loss group"""
    loss, *args = args
    opt = (struct.MILoss if loss in mi else
           struct.MSELoss if loss in mse else
           struct.JTVLoss if loss in jtv else
           struct.DiceLoss if loss in dice else
           struct.CatLoss if loss in cat else
           struct.NoLoss)
    opt = opt()
    args = parse_moving_and_target(args, opt)
    options.losses.append(opt)
    return args


# --- WORKER FOR TRANSFORMATION GROUPS ---------------------------------
def parse_transform(args, options):
    """Parse a transformation group"""
    trf, *args = args
    opt = (struct.Translation if trf in translation else
           struct.Rigid if trf in rigid else
           struct.Similitude if trf in similitude else
           struct.Affine if trf in affine else
           struct.FFD if trf in ffd else
           struct.Diffeo if trf in diffeo else
           None)
    opt = opt()

    if next_isvalue(args):
        opt.factor, *args = args
        opt.factor = float(opt.factor)

    while args:
        if not next_istag(args):
            break
        tag, *args = args
        if tag in regularizations:
            optreg = (struct.AbsoluteLoss if tag in absolute else
                      struct.MembraneLoss if tag in membrane else
                      struct.BendingLoss if tag in bending else
                      struct.LinearElasticLoss if tag in linelastic else
                      None)
            optreg = optreg()
            islinelastic = isinstance(optreg, struct.LinearElasticLoss)
            factor = []
            while next_isvalue(args):
                val, *args = args
                factor.append(float(val))
            if factor:
                optreg.factor = (factor[0] if not islinelastic else factor)
            opt.losses.append(optreg)
        elif tag in ('-init', '--init'):
            check_next_isvalue(args, tag)
            opt.init, *args = args
        elif tag in ('-lr', '--learning-rate'):
            check_next_isvalue(args, tag)
            opt.lr, *args = args
            opt.lr = float(opt.lr)
        elif tag in ('-wd', '--weight-decay'):
            check_next_isvalue(args, tag)
            opt.weight_decay, *args = args
            opt.weight_decay = float(opt.weight_decay)
        elif tag in ('-stop', '--stop'):
            check_next_isvalue(args, tag)
            opt.stop, *args = args
            opt.stop = float(opt.stop)
        elif tag in ('-o', '--output'):
            opt.output = True
            if next_isvalue(args):
                opt.output, *args = args
        elif isinstance(opt, struct.Diffeo) and tag in ('-sd', '--smalldef'):
            opt.smalldef = True
            if next_isvalue(args):
                val, *args = args
                if val.lower().startswith('f'):
                    opt.smalldef = False
                elif val.lower().startswith('t'):
                    opt.smalldef = True
                else:
                    opt.smalldef = bool(int(val))
        elif tag in ('-pyr', '--pyramid'):
            args, levels = parse_pyramid(args)
            if levels:
                opt.pyramid = levels
            else:
                opt.pyramid = [1]
        elif isinstance(opt, struct.Linear) and tag in ('-z', '--zero'):
            opt.shift = True
            if next_isvalue(args):
                val, *args = args
                if val.lower().startswith('f'):
                    opt.shift = False
                elif val.lower().startswith('t'):
                    opt.shift = True
                else:
                    opt.shift = bool(int(val))
        elif isinstance(opt, struct.FFD) and tag in ('-g', '--grid'):
            opt.grid = []
            while next_isvalue(args):
                val, *args = args
                opt.grid.append(int(val))
        else:
            args = [tag, *args]
            break

    options.transformations.append(opt)
    return args


# --- WORKER FOR OPTIMIZER GROUPS --------------------------------------
def parse_optim(args, options):
    optim, *args = args

    opt = (struct.Adam if optim in adam else
           struct.GradientDescent if optim in gd else
           struct.OGM if optim in ogm else
           struct.LBFGS if optim in lbfgs else
           None)
    opt = opt()

    while args:
        if not next_istag(args):
            break
        tag, *args = args
        if tag in ('-nit', '--max-iter'):
            check_next_isvalue(args, tag)
            opt.max_iter, *args = args
            opt.max_iter = int(opt.max_iter)
        elif tag in ('-lr', '--learning-rate'):
            check_next_isvalue(args, tag)
            opt.lr, *args = args
            opt.lr = float(opt.lr)
        elif tag in ('-wd', '--weight-decay'):
            check_next_isvalue(args, tag)
            opt.weight_decay, *args = args
            opt.weight_decay = float(opt.weight_decay)
        elif tag in ('-stop', '--stop'):
            check_next_isvalue(args, tag)
            opt.stop, *args = args
            opt.stop = float(opt.stop)
        elif tag in ('-ls', '--line-search'):
            opt.ls = True
            if next_isvalue(args):
                opt.ls, *args = args
                opt.ls = int(opt.ls)
        elif isinstance(opt, struct.LBFGS) and tag in ('-h', '--history'):
            check_next_isvalue(args, tag)
            opt.history = int(args.pop(0))
        else:
            args = [tag, *args]
            break

    options.optimizers.append(opt)
    return args


# --- WORKER FOR DEFAULT GROUP -----------------------------------------
def parse_defaults(args, options):
    """Parse default parameters for all groups"""
    opt = options.defaults
    while args:
        if not next_istag(args):
            break
        tag, *args = args
        if tag in ('-nit', '--max-iter'):
            check_next_isvalue(args, tag)
            opt.max_iter, *args = args
            opt.max_iter = int(opt.max_iter)
        elif tag in ('-lr', '--learning-rate'):
            check_next_isvalue(args, tag)
            opt.lr, *args = args
            opt.lr = float(opt.lr)
        elif tag in ('-stop', '--stop'):
            check_next_isvalue(args, tag)
            opt.stop, *args = args
            opt.stop = float(opt.stop)
            args = args[2:]
        elif tag in ('-ls', '--line-search'):
            opt.ls = True
            if next_isvalue(args):
                opt.ls, *args = args
                opt.ls = int(opt.ls)
        elif tag in ('-inter', '--interpolation'):
            check_next_isvalue(args, tag)
            opt.interpolation, *args = args
            opt.interpolation = int(opt.interpolation)
        elif tag in ('-bnd', '--bound'):
            check_next_isvalue(args, tag)
            opt.bound, *args = args
        elif tag in ('-ex', '--extrapolate'):
            opt.extrapolate = True
        elif tag in ('-o', '--output'):
            check_next_isvalue(args, tag)
            opt.output, *args = args
        elif tag in ('-pyr', '--pyramid'):
            args, levels = parse_pyramid(args)
            if levels:
                opt.pyramid = levels
            else:
                opt.pyramid = [1]
        else:
            args = [tag, *args]
            break

    return args


# --- MAIN PARSER -----------------------------------------------------
def parse(args):
    """Parse AutoGrad's command-line arguments"""

    # This is the object that we will populate
    options = struct.AutoReg()

    while args:
        if next_isvalue(args):
            raise ParseError(f'Argument {args[0]} does not seem to '
                             f'belong to a group')
        tag, *args = args
        # Parse groups
        if tag in match_losses:
            args = parse_match([tag, *args], options)
        elif tag in transformations:
            args = parse_transform([tag, *args], options)
        elif tag in optim:
            args = parse_optim([tag, *args], options)
        elif tag in ('-all', '--all'):
            args = parse_defaults(args, options)

        # Help -> return empty option
        elif tag in ('-h', '--help'):
            print(help)
            return {}

        # Parse remaining top-level tags
        elif tag in ('-prg', '--progressive'):
            options.progressive = True
            if next_isvalue(args):
                options.progressive, *args = args
                if options.progressive[0] in ('t', 'y'):
                    options.progressive = True
                elif options.progressive[0] in ('f', 'n'):
                    options.progressive = False
                else:
                    options.progressive = bool(int(options.progressive))
        elif tag in ('-pad', '--padding'):
            check_next_isvalue(args, tag)
            options.pad, *args = args
            options.pad = float(options.pad)
            if next_isvalue(args):
                options.pad_unit, *args,  = args
        elif tag in ('-v', '--verbose'):
            options.verbose = 2
            if next_isvalue(args):
                options.verbose, *args,  = args
                options.verbose = int(options.verbose)
        elif tag in ('-cpu', '--cpu'):
            options.device = 'cpu'
        elif tag in ('-gpu', '--gpu'):
            options.device = 'cuda'
            if next_isvalue(args):
                gpu, *args = args
                options.device = 'cuda:{:d}'.format(int(gpu))

        # Something went wrong
        else:
            raise ParseError(f'Argument {tag} does not seem to '
                             f'belong to a group')

    return options
