"""This file parses the commandline arguments"""

import string
from . import struct

help = r"""[nitorch] AUTOgrad REGistration tool

usage: 
    nitorch autoreg <*LOSS> [FACTOR] -fix *FILE [-o *OUT] [-r *OUT] [-pyr *LVL] ...
                                     -mov *FILE [-o *OUT] [-r *OUT] [-pyr *LVL] ...
                    <*TRF> [FACTOR] [-init PATH] [-lr LRNRATE] [-o OUT] [-pyr LVL] 
                           <*REG> [FACTOR]
                    <*OPT> [-nit MAXITER] [-lr LRNRATE] [-stop LIMIT] [-ls [MAXLS]]
                    [-all ...] [-prg] [-gpu|-cpu] [-v] [-h]

    <LOSS> can take values (with additional options): 
        -mi,  --mutual-info         Normalized Mutual Information
                                    (for images with different contrasts)
            -p, --patch *SIZE           Patch size for local MI (0 = global mi)
            -b, --bins BINS             Number of bins in the histogram (32)
            -f, --fwhm WIDTH            Full-width half-max of the bins (1)
            -t, --threshold VAL     Values under this threshold (in fixed) are excluded.
            -n, --order ORDER       Histogram spline order (3), or inf
                                    for a Gaussian window
        -mse, --mean-squared-error  Mean Squared Error
                                    (for images with the same contrast)
        -jtv,  --total-variation    Normalized Joint Total Variation 
                                    (edge-based, for strong intensity non-uniformity)
        -dice, --dice               Dice coefficients (for soft/hard segments)
            -l, --labels *LABELS        Labels to use (default: all except 0)
            -w, --weights *WEIGHTS      Label-wise weights (default: False)
        -cat,  --categorical        Categorical cross-entropy (for soft segments)
        -oth, --other               Not a loss -> other images to warp at the end.
    Note that the -fix and -mov tags must be placed directly after their 
    corresponding <LOSS> (The <LOSS> tag "opens a group"). Similarly, the 
    interpolation tags (-inter, -bnd, -ext) must be placed directly after 
    their -mov tag.
    Multiple <LOSS> statements can be used, resulting in a composite loss.
    In a -fix/-mov group, -o writes the updated image (default: -mov only)
    and -r writes the image resliced to the other image's grid (default:
    no). Each takes one OUT per input file, which is either true/yes/1 (or
    no value: default name), false/no/0 (do not write), or a path. A path can
    contain {dir}, {base} and {ext}, which are replaced by the directory, base
    name and extension of the input file.
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
    At least one transformation is required (there is no default).
    The estimated transformation is written by default. -o takes true/yes/1
    (or no value: default path {name}{ext}), false/no/0 (do not write), or a
    path. A path can contain {name} and {ext}, which are replaced by the name
    and default extension of the transformation.
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
        -o,     --output             Output transform name or template (true)
        -init,  --init               Initial transform file (identity)
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
                                     UNIT: % (pct), vox (voxel[s]) or mm
        -v, --verbose [LVL]          Level of verbosity (1)

examples:
    # rigid registration between two images
    autoreg -mi -fix target.nii -mov source.nii -rig

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
    autoreg -mi -fix target.nii -mov source.nii -rig -gd -ls

    # register based on intensity images and use it to warp segments
    autoreg -mi  -fix target.nii  -mov source.nii \
            -oth -mov segment.nii -inter 0 \
            -rig
"""


# Shared with other nitorch parsers (so that `pop_values` errors are
# caught as parse errors)
from nitorch.core.cli import (  # noqa: E402
    ParseError, pop_values, convert_value, Conversions,
    split_double_dash, check_rest_used)

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
def _isnumber(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def istag(x):
    # A dash-prefixed argument that parses as a number (e.g., `-1`,
    # `-.5`, `-1e-3`) is a value, not a tag (same rule as nitorch.core.cli)
    return x.startswith('-') and not _isnumber(x)


def isvalue(x):
    return not istag(x)


def next_istag(args):
    return args and istag(args[0])


def next_isvalue(args):
    return args and isvalue(args[0])


def pop_bool(args, opt):
    """Pop an optional bool value (bare flag -> True)"""
    if not next_isvalue(args):
        return True
    return convert_value(args.pop(0), opt, Conversions.strict_bool, '<bool>')


def pop_all(args, opt, convert=float):
    """Pop (and convert) all values until the next tag"""
    return pop_values(args, opt, float('inf'), convert, nmin=0, istag=istag)


def pop_factor(args, opt):
    """Pop the optional leading factor of a group"""
    if next_isvalue(args):
        return convert_value(args.pop(0), opt, float, '<factor: float>')
    return None


def _spline_order(x):
    """<int> or inf (any spelling `float` accepts: inf, Inf, infinity...)
    -> int or 'inf' (the value nn.MutualInfoLoss expects)"""
    try:
        return int(x)
    except ValueError:
        if float(x) == float('inf'):
            return 'inf'
        raise


# Padding units understood by spatial.compute_fov (case-insensitive),
# mapped to the canonical spelling it expects
_pad_units = {'%': '%', 'pct': '%',
              'vox': 'vox', 'voxel': 'vox', 'voxels': 'vox',
              'mm': 'mm'}


def _pad_unit(x):
    try:
        return _pad_units[x.lower()]
    except KeyError:
        raise ValueError(x)


def bool_or_path(x):
    """'true'/'yes'/'1' -> True (default path), 'false'/'no'/'0' -> False
    (nothing written), anything else is kept as a path (template)."""
    if x.lower() in ('true', 'yes', '1'):
        return True
    if x.lower() in ('false', 'no', '0'):
        return False
    return x


def check_template(x, keys, tag):
    """Check that a path template only uses known format keys."""
    if not isinstance(x, str):
        return x
    try:
        fields = [f for _, f, _, _ in string.Formatter().parse(x)
                  if f is not None]
    except ValueError as e:
        raise ParseError(f'Invalid path template for {tag}: {x!r} ({e})')
    for f in fields:
        if f not in keys:
            raise ParseError(f'Invalid path template for {tag}: {x!r}. '
                             f'Unknown key {{{f}}}; valid keys are: '
                             + ', '.join('{%s}' % k for k in keys))
    return x


image_keys = ('dir', 'base', 'ext')
transform_keys = ('name', 'ext')


def parse_image_output(args, tag, rest=None):
    """Parse the values of -o/-r in a file group.

    Returns True (bare/true -> default path), False (false -> nothing
    written) or a list with one entry per file (path template, True or
    False).

    If this is the last option before `--`, the arguments after `--`
    (`rest`, emptied) are values too, even if dash-prefixed."""
    vals = []
    while next_isvalue(args):
        val, *args = args
        vals.append(check_template(bool_or_path(val), image_keys, tag))
    if not args and rest:
        vals += [check_template(bool_or_path(val), image_keys, tag)
                 for val in rest]
        del rest[:]
    if not vals:
        return args, True
    if len(vals) == 1 and isinstance(vals[0], bool):
        return args, vals[0]
    return args, vals


def _pyramid_level(level):
    if ':' in level:
        level = level.split(':')
        if len(level) > 3:
            raise ValueError(level)
        start = int(level[0] or 1)
        stop = (int(level[1] or start) if len(level) > 1 else start) + 1
        step = (int(level[2] or 1) if len(level) > 2 else 1)
        return list(range(start, stop, step))
    return [int(level)]


def parse_pyramid(args):
    levels = []
    while next_isvalue(args):
        level, *args = args
        levels.extend(convert_value(level, '-pyr/--pyramid', _pyramid_level,
                                    '<int> or [start]:stop[:step]'))
    return args, levels


def parse_file(args, opt, rest=None):
    """Parse a moving or fixed file

    If the file list is the last thing before `--`, the arguments after
    `--` (`rest`, emptied) are files too, even if dash-prefixed."""

    opt.files = list()
    while next_isvalue(args):
        val, *args = args
        opt.files.append(val)
    if not args and rest:
        opt.files += rest
        del rest[:]

    while args:
        if not next_istag(args):
            break
        tag, *args = args
        if tag in ('-o', '--output'):
            # Path to "updated" output file
            # (updated header + nonlin applied in original space)
            args, opt.updated = parse_image_output(args, tag, rest)
        elif tag in ('-r', '--resliced'):
            # Path to "resliced" output file
            # (lin + nonlin applied, final space = target space)
            args, opt.resliced = parse_image_output(args, tag, rest)
        elif tag in ('-inter', '--interpolation'):
            opt.interpolation, = pop_values(
                args, '-inter/--interpolation', 1, int, istag=istag)
        elif tag in ('-bnd', '--bound'):
            opt.bound, = pop_values(
                args, '-bnd/--bound', 1, istag=istag)
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
    check_output_count(opt, 'updated', '-o')
    check_output_count(opt, 'resliced', '-r')
    return args


def check_output_count(opt, key, tag):
    """One -o/-r value per input file, or a single one for all files"""
    vals = getattr(opt, key)
    nfiles = len(opt.files)
    if not isinstance(vals, list) or not nfiles or len(vals) == nfiles:
        return
    if len(vals) == 1:
        val = vals[0]
        if isinstance(val, str) and not any(
                f in ('dir', 'base') for _, f, _, _
                in string.Formatter().parse(val)):
            raise ParseError(
                f'{tag}: a single path for {nfiles} input files would '
                f'overwrite itself; use a template such as '
                f'{{base}}.moved{{ext}}, one value per file, or true/false')
        setattr(opt, key, vals * nfiles)
        return
    raise ParseError(f'{tag} expects 1 value per input file ({nfiles}), '
                     f'or a single template; got {len(vals)}')


def parse_moving_and_target(args, opt, group, rest=None):
    """Parse a loss that involves a moving and a target file"""

    factor = pop_factor(args, group)
    if factor is not None:
        opt.factor = factor

    while args:
        if not next_istag(args):
            break
        tag, *args = args
        if tag in ('-mov', '--moving', '-src', '--source'):
            opt.moving = struct.MovingImageFile()
            args = parse_file(args, opt.moving, rest)
        elif tag in ('-fix', '--fixed', '-trg', '--target'):
            opt.fixed = struct.FixedImageFile()
            args = parse_file(args, opt.fixed, rest)
        elif tag in ('-inter', '--interpolation'):
            opt.interpolation, = pop_values(
                args, '-inter/--interpolation', 1, int, istag=istag)
        elif tag in ('-bnd', '--bound'):
            opt.bound, = pop_values(
                args, '-bnd/--bound', 1, istag=istag)
        elif tag in ('-ex', '--extrapolate'):
            opt.extrapolate = True
        elif tag in ('-pyr', '--pyramid'):
            args, levels = parse_pyramid(args)
            if levels:
                opt.pyramid = levels
            else:
                opt.pyramid = [1]
        elif tag in ('-exc', '--exclude'):
            opt.exclude = pop_bool(args, '-exc/--exclude')
        elif isinstance(opt, struct.MILoss) and tag in ('-p', '--patch'):
            opt.patch = pop_all(args, '-p/--patch', int)
        elif isinstance(opt, struct.MILoss) and tag in ('-f', '--fwhm'):
            opt.fwhm, = pop_values(
                args, '-f/--fwhm', 1, float, istag=istag)
        elif isinstance(opt, struct.MILoss) and tag in ('-b', '--bins'):
            opt.bins, = pop_values(
                args, '-b/--bins', 1, int, istag=istag)
        elif isinstance(opt, struct.MILoss) and tag in ('-t', '--threshold'):
            opt.threshold, = pop_values(
                args, '-t/--threshold', 1, float, istag=istag)
        elif isinstance(opt, struct.MILoss) and tag in ('-n', '--order'):
            opt.order, = pop_values(
                args, '-n/--order', 1, istag=istag,
                convert=_spline_order, names="<int> or 'inf'")
        elif isinstance(opt, struct.DiceLoss) and tag in ('-w', '--weights'):
            opt.weights = pop_all(args, '-w/--weights') or True
        elif isinstance(opt, struct.DiceLoss) and tag in ('-l', '--labels'):
            opt.labels = pop_all(args, '-l/--labels', int)
        else:
            args = [tag, *args]
            break
    return args


# --- WORKER FOR MATCHING GROUPS ---------------------------------------
def parse_match(args, options, rest=None):
    """Parse a loss group"""
    loss, *args = args
    opt = (struct.MILoss if loss in mi else
           struct.MSELoss if loss in mse else
           struct.JTVLoss if loss in jtv else
           struct.DiceLoss if loss in dice else
           struct.CatLoss if loss in cat else
           struct.NoLoss)
    opt = opt()
    args = parse_moving_and_target(args, opt, loss, rest)
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

    factor = pop_factor(args, trf)
    if factor is not None:
        opt.factor = factor

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
            factor = pop_all(args, tag)
            if factor:
                optreg.factor = (factor[0] if not islinelastic else factor)
            opt.losses.append(optreg)
        elif tag in ('-init', '--init'):
            opt.init, = pop_values(
                args, '-init/--init', 1, istag=istag)
        elif tag in ('-lr', '--learning-rate'):
            opt.lr, = pop_values(
                args, '-lr/--learning-rate', 1, float, istag=istag)
        elif tag in ('-wd', '--weight-decay'):
            opt.weight_decay, = pop_values(
                args, '-wd/--weight-decay', 1, float, istag=istag)
        elif tag in ('-stop', '--stop'):
            opt.stop, = pop_values(
                args, '-stop/--stop', 1, float, istag=istag)
        elif tag in ('-o', '--output'):
            opt.output = True
            if next_isvalue(args):
                opt.output, *args = args
                opt.output = check_template(bool_or_path(opt.output),
                                            transform_keys, tag)
        elif isinstance(opt, struct.Diffeo) and tag in ('-sd', '--smalldef'):
            opt.smalldef = pop_bool(args, '-sd/--smalldef')
        elif tag in ('-pyr', '--pyramid'):
            args, levels = parse_pyramid(args)
            if levels:
                opt.pyramid = levels
            else:
                opt.pyramid = [1]
        elif isinstance(opt, struct.Linear) and tag in ('-z', '--zero'):
            opt.shift = pop_bool(args, '-z/--zero')
        elif isinstance(opt, struct.FFD) and tag in ('-g', '--grid'):
            opt.grid = pop_all(args, '-g/--grid', int)
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
            opt.max_iter, = pop_values(
                args, '-nit/--max-iter', 1, int, istag=istag)
        elif tag in ('-lr', '--learning-rate'):
            opt.lr, = pop_values(
                args, '-lr/--learning-rate', 1, float, istag=istag)
        elif tag in ('-wd', '--weight-decay'):
            opt.weight_decay, = pop_values(
                args, '-wd/--weight-decay', 1, float, istag=istag)
        elif tag in ('-stop', '--stop'):
            opt.stop, = pop_values(
                args, '-stop/--stop', 1, float, istag=istag)
        elif tag in ('-ls', '--line-search'):
            opt.ls = True
            if next_isvalue(args):
                opt.ls, = pop_values(
                    args, '-ls/--line-search', 1, int, istag=istag)
        elif isinstance(opt, struct.LBFGS) and tag in ('-h', '--history'):
            opt.history, = pop_values(
                args, '-h/--history', 1, int, istag=istag)
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
            opt.max_iter, = pop_values(
                args, '-nit/--max-iter', 1, int, istag=istag)
        elif tag in ('-lr', '--learning-rate'):
            opt.lr, = pop_values(
                args, '-lr/--learning-rate', 1, float, istag=istag)
        elif tag in ('-stop', '--stop'):
            opt.stop, = pop_values(
                args, '-stop/--stop', 1, float, istag=istag)
        elif tag in ('-ls', '--line-search'):
            opt.ls = True
            if next_isvalue(args):
                opt.ls, = pop_values(
                    args, '-ls/--line-search', 1, int, istag=istag)
        elif tag in ('-inter', '--interpolation'):
            opt.interpolation, = pop_values(
                args, '-inter/--interpolation', 1, int, istag=istag)
        elif tag in ('-bnd', '--bound'):
            opt.bound, = pop_values(
                args, '-bnd/--bound', 1, istag=istag)
        elif tag in ('-ex', '--extrapolate'):
            opt.extrapolate = True
        elif tag in ('-o', '--output'):
            # Default transform output: bare/true -> keep the default
            # template, false -> no transform written
            output = True
            if next_isvalue(args):
                output, *args = args
                output = check_template(bool_or_path(output),
                                        transform_keys, tag)
            if output is not True:
                opt.output = output
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

    # Empty command line -> help
    if not args:
        print(help)
        return {}

    # `--` ends option parsing: what follows are values of the -fix/-mov
    # file list (or of its -o/-r) placed just before `--`
    args, rest = split_double_dash(args)

    while args:
        if next_isvalue(args):
            raise ParseError(f'Argument {args[0]} does not seem to '
                             f'belong to a group')
        tag, *args = args
        # Parse groups
        if tag in match_losses:
            args = parse_match([tag, *args], options, rest)
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
            options.progressive = pop_bool(args, '-prg/--progressive')
        elif tag in ('-pad', '--padding'):
            options.pad, = pop_values(
                args, '-pad/--padding', 1, float, istag=istag)
            if next_isvalue(args):
                options.pad_unit = convert_value(
                    args.pop(0), '-pad/--padding', _pad_unit,
                    '<unit: ' + '|'.join(_pad_units) + '>')
        elif tag in ('-v', '--verbose'):
            options.verbose = 2
            if next_isvalue(args):
                options.verbose, = pop_values(
                    args, '-v/--verbose', 1, int, istag=istag)
        elif tag in ('-cpu', '--cpu'):
            options.device = 'cpu'
        elif tag in ('-gpu', '--gpu'):
            options.device = 'cuda'
            if next_isvalue(args):
                gpu, = pop_values(args, '-gpu/--gpu', 1, int, istag=istag)
                options.device = 'cuda:{:d}'.format(int(gpu))

        # Something went wrong
        else:
            raise ParseError(f'Argument {tag} does not seem to '
                             f'belong to a group')

    check_rest_used(rest, "use '--' just after the files of -fix/-mov "
                          "(or the values of their -o/-r)")

    if not options.transformations:
        raise ParseError('No transformation specified. Use at least one of '
                         '-tr, -rig, -sim, -aff, -ffd or -dif.')

    return options
