import torch
import os
from nitorch import io
from . import struct

help = r"""[AutoReg] AUTOgrad REGistration tool

usage: 
    autoreg <*LOSS> [FACTOR] -fix *FILE [-o *FILE] [-r *FILE] 
                             -mov *FILE [-o *FILE] [-r *FILE] 
                             [-inter N] [-bnd BND] [-ex]
            <*TRF> [FACTOR] [-init PATH/*VALUES] [-lr LEARNING_RATE] [-stop LIMIT] [-o FILE]
                   <*REG> [FACTOR]
            <*OPT> [-nit MAXITER] [-lr LEARNING_RATE] [-stop LIMIT] [-ls [MAXLS]]
            [+mov *FILE [-inter N] [-bnd BND] [-ex] [-o *FILE] [-r *FILE]]
            [-all ...] [-prg] [-pyr N] [-gpu|-cpu] [-h]

    <LOSS> can take values (with additional options): 
        -mi,  --mutual-info         Normalized Mutual Information
                                    (for images with different contrasts)
            -p, --patch *SIZE           Patch size for local MI (0 = global mi)
            -b, --bin BINS              Number of bins in the histogram (32)
            -f, --fwhm WIDTH            Full-width half-max of the bins (1)
        -mse, --mean-squared-error  Mean Squared Error
                                    (for images with the same contrast)
        -jtv,  --total-variation    Normalized Joint Total Variation 
                                    (edge-based, for strong intensity non-uniformity)
        -dice, --dice               Dice coefficients (for soft/hard segments)
            -l, --labels                Labels to use (default: all except 0)
            -w, ---weights *WEIGHTS     Label-wise weights 
        -cat,  --categorical        Categorical cross-entropy (for soft segments)
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
        -ffd, --free-form           Free-form deformation (cubic)
            -g, --grid *SIZE            Number of nodes (10)
        -diffeo, --diffeo           Diffeomorphic field
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
        -adam, --adam               ADAM (default)
        -gd,   --gradient-descent   Gradient descent

    Generic options:
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
    If these tags are present within a <LOSS>/<TRF>/<OPT> group, they only 
    apply to this group. If they are present within a `-all` group, they 
    apply to all <LOSS>/<TRF>/<OPT> groups that did not specify them in 
    their group.
    By default, only transformations  are written to disk. (the full 
    affine matrix for affine groups and subgroups, the grid of 
    parameters for ffd, the initial velocity for diffeo).

    Other options
        -cpu, -gpu                   Device to use (gpu if available)
        -prg, --progressive          Progressively increase degrees of freedom (no)
        -pyr, --pyramid *LEVELS      Pyramid levels. Can be a range [start]:stop[:step] (1)
        +mov, --other-moving         Additional moving images, without loss

examples:
    # rigid registration between two images
    autoreg -mi -fix target.nii -mov source.nii

    # diffeo + similitude registration between two images
    autoreg -mi -fix target.nii -mov source.nii -diffeo -sim

    # diffeo + rigid between a template and a segment from SPM
    # > note that the tpm is the *moving* image but we use the -o tag
    #   in the -fix group so that the fixed image is also wrapped to 
    #   moving space.
    autoreg -cat -fix c1.nii c2.nii -o -mov tpm.nii -diffeo -rig

    # semi-supervised registration
    autoreg -mi   -fix target.nii     -mov source.nii \
            -dice -fix target_seg.nii -mov source_seg.nii \
            -diffeo

    # registration using gradient descent with a backtracking line 
    autoreg -mi -fix target.nii -mov source.nii -gd -ls

    # register based on intensity images use it to warp segments
    autoreg -mi -fix target.nii -mov source.nii +mov segment.nii -inter 0
"""

# --------------------
#   DEFINE MAIN TAGS
# --------------------

# -
# matching loss
# -
mi = ('-mi', '--mutual-info', '--mutual-information')
jtv = ('-jtv', '--total-variation')
dice = ('-dice', '--dice')
cat = ('-cat', '--categorical')
mov2 = ('+mov', '--other-moving')
match_losses = (*mi, *jtv, *dice, *cat, *mov2)

# -
# transformations
# -
translation = ('-tr', '--translation')
rigid = ('-rig', '--rigid')
similitude = ('-sim', '--similitude')
affine = ('-aff', '--affine')
all_affine = (*translation, *rigid, *similitude, *affine)
diffeo = ('-diffeo', '--diffeo')
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
gd = ('-gd', '--gd')
optim = (*adam, *gd)


# ----------------------------------------------------------------------
#                     PARSE COMMANDLINE ARGUMENTS
# ----------------------------------------------------------------------
def istag(x):
    return x.startswith(('-', '+'))


def parse_file(args, opt):
    """Parse a moving or fixed file"""
    opt.files = list()
    while args and not istag(args[0]):
        opt.files.append(args[0])
        args = args[1:]
    while args:
        if args[0] in ('-o', '--output'):
            args = args[1:]
            opt.updated = True
            while args and not istag(args[0]):
                if isinstance(opt.updated, bool):
                    opt.updated = []
                opt.updated.append(args[0])
                args = args[1:]
        elif args[0] in ('-r', '--resliced'):
            args = args[1:]
            opt.resliced = True
            while args and not istag(args[0]):
                if isinstance(opt.resliced, bool):
                    opt.resliced = []
                opt.resliced.append(args[0])
                args = args[1:]
        else:
            break
    return args


def parse_moving_and_target(args, opt):
    """Parse a loss that involves a moving and a target file"""
    loss = opt.name

    if args and not args[0].startswith(('-', '+')):
        opt.factor = float(args[0])
        args = args[1:]

    while args:
        if args[0] in ('-mov', '--moving', '-src', '--source'):
            args = parse_file(args[1:], opt.moving)
        elif args[0] in ('-fix', '--fixed', '-trg', '--target'):
            args = parse_file(args[1:], opt.fixed)
        elif args[0] in ('-inter', '--interpolation'):
            opt['interpolation'] = int(args[1])
            args = args[2:]
        elif args[0] in ('-bnd', '--bound'):
            opt['bound'] = args[1]
            args = args[2:]
        elif args[0] in ('-ex', '--extrapolate'):
            opt['extrapolate'] = True
            args = args[1:]
        elif loss == 'mi' and args[0] in ('-p', '--patch'):
            opt.patch = list()
            args = args[1:]
            while not args[0].startswith(('-', '+')):
                opt.patch.append(int(args[0]))
                args = args[1:]
        elif loss == 'mi' and args[0] in ('-f', '--fwhm'):
            opt.fwhn = float(args[1])
            args = args[2:]
        elif loss == 'mi' and args[0] in ('-b', '--bins'):
            opt.bins = int(args[1])
            args = args[2:]
        else:
            break
    return args


def parse_moving(args, opt):
    """Parse a moving file that is not part of a loss"""
    opt.moving.files = []
    while args and not istag(args[0]):
        opt.moving.files.append(args[0])
        args = args[1:]
    while args:
        if args[0] in ('-inter', '--interpolation'):
            opt.interpolation = int(args[1])
            args = args[2:]
        elif args[0] in ('-bnd', '--bound'):
            opt.bound = args[1]
            args = args[2:]
        elif args[0] in ('-ex', '--extrapolate'):
            opt.extrapolate = True
            args = args[1:]
        elif args[0] in ('-o', '--output'):
            args = args[1:]
            opt.updated = True
            while args and not istag(args[0]):
                if isinstance(opt.updated, bool):
                    opt.updated = []
                opt.updated.append(args[0])
                args = args[1:]
        elif args[0] in ('-r', '--resliced'):
            args = args[1:]
            opt.resliced = True
            while args and not istag(args[0]):
                if isinstance(opt.resliced, bool):
                    opt.resliced = []
                opt.resliced.append(args[0])
                args = args[1:]
        else:
            break
    return args


# --- WORKER FOR MATCHING GROUPS ---------------------------------------
def parse_match(args, options):
    loss, *args = args

    if loss in ('+mov', '--other-moving'):
        opt = struct.NoLoss()
        args = parse_moving(args, opt)
    else:
        opt = (struct.MILoss if loss in mi else
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
    trf, *args = args
    print(trf)
    opt = (struct.Translation if trf in translation else
           struct.Rigid if trf in rigid else
           struct.Similitude if trf in similitude else
           struct.Affine if trf in affine else
           struct.FFD if trf in ffd else
           struct.Diffeo if trf in diffeo else
           None)
    opt = opt()
    while args:
        if args[0] in regularizations:
            reg, *args = args
            factor = []
            while args and not args[0].startswith(('-', '+')):
                factor.append(args[0])
                args = args[1:]
            optreg = (struct.AbsoluteLoss if reg in absolute else
                      struct.MembraneLoss if reg in membrane else
                      struct.BendingLoss if reg in bending else
                      struct.LinearElasticLoss if reg in linelastic else
                      None)
            optreg = optreg()
            if factor:
                optreg.factor = (factor[0] if reg not in linelastic
                                 else factor)
            opt.losses.append(optreg)
        elif args[0] in ('-init', '--init'):
            opt.init = args[1]
            args = args[2:]
        elif args[0] in ('-lr', '--learning-rate'):
            opt.lr = float(args[1])
            args = args[2:]
        elif args[0] in ('-stop', '--stop'):
            opt.stop = float(args[1])
            args = args[2:]
        elif args[0] in ('-o', '--output'):
            opt.output = args[1]
            args = args[2:]
        elif trf in ffd and args[0] in ('-g', '--grid'):
            grid = []
            args = args[1:]
            while not args[0].startswith(('-', '+')):
                grid.append(int(args[0]))
                args = args[1:]
            opt.grid = grid
        else:
            break

    options.transformations.append(opt)
    return args


# --- WORKER FOR OPTIMIZER GROUPS --------------------------------------
def parse_optim(args, options):
    optim, *args = args

    opt = (struct.Adam if optim in adam else
           struct.GradientDescent if optim in gd else None)
    opt = opt()

    while args:
        if args[0] in ('-nit', '--max-iter'):
            opt.max_iter = int(args[1])
            args = args[2:]
        elif args[0] in ('-lr', '--learning-rate'):
            opt.lr = float(args[1])
            args = args[2:]
        elif args[0] in ('-stop', '--stop'):
            opt.stop = float(args[1])
            args = args[2:]
        elif args[0] in ('-ls', '--line-search'):
            args = args[1:]
            if args and not args[0].startswith(('-', '+')):
                opt.ls = int(args[1])
                args = args[1:]
            else:
                opt.ls = True
        else:
            break

    options.optimizers.append(opt)
    return args


# --- WORKER FOR DEFAULT GROUP -----------------------------------------
def parse_defaults(args, options):
    """Parse default parameters for all groups"""
    opt = options.defaults
    while args:
        if args[0] in ('-nit', '--max-iter'):
            opt.max_iter = int(args[1])
            args = args[2:]
        elif args[0] in ('-lr', '--learning-rate'):
            opt.lr = float(args[1])
            args = args[2:]
        elif args[0] in ('-stop', '--stop'):
            opt.stop = float(args[1])
            args = args[2:]
        elif args[0] in ('-ls', '--line-search'):
            args = args[1:]
            if args and not args[0].startswith(('-', '+')):
                opt.ls = int(args[1])
                args = args[1:]
            else:
                opt.ls = True
        elif args[0] in ('-inter', '--interpolation'):
            opt.interpolation = int(args[1])
            args = args[2:]
        elif args[0] in ('-bnd', '--bound'):
            opt.bound = args[1]
            args = args[2:]
        elif args[0] in ('-ex', '--extrapolate'):
            opt.extrapolate = True
            args = args[1:]
        elif args[0] in ('-o', '--output'):
            opt.output = args[1]
            args = args[2:]
        else:
            break


# --- MAIN PARSER -----------------------------------------------------
def parse(args):
    """Parse AutoGrad's command-line arguments"""

    args = list(args)

    # This is the object that we will populate
    options = struct.AutoReg()

    args = args[1:]  # remove script name from list
    while args:
        # Parse groups
        if args[0] in match_losses:
            args = parse_match(args, options)
        elif args[0] in transformations:
            args = parse_transform(args, options)
        elif args[0] in optim:
            args = parse_optim(args, options)
        elif args[0] in ('-all', '--all'):
            args = parse_defaults(args[1:], options)

        # Help -> return empty option
        elif args[0] in ('-h', '--help'):
            print(help)
            return {}

        # Parse remaining top-level tags
        elif args[0] in ('-prg', '--progressive'):
            options.progressive = True
            args = args[1:]
        elif args[0] in ('-cpu', '--cpu'):
            if 'device' in options:
                raise ValueError('Cannot use both -cpu and -gpu')
            options.device = 'cpu'
            args = args[1:]
        elif args[0] in ('-gpu', '--gpu'):
            args = args[1:]
            if args and not args[0].startswith(('-', '+')):
                options.device = 'cuda:{:d}'.format(int(args[0]))
                args = args[1:]
            else:
                options.device = 'cuda'
        elif args[0] in ('-pyr', '--pyramid'):
            args = args[1:]
            levels = []
            while args and not args[0].startswith(('-', '+')):
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
            if levels:
                options.pyramid = levels
            else:
                options.pyramid = True

        # Something went wrong
        else:
            raise RuntimeError(f'Argument {args[0]} does not seem to '
                               f'belong to a group')

    return options
