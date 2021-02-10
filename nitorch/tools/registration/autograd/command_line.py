from sys import argv
import torch
import os
from nitorch import io, core, spatial, nn
from nitorch.core import dtypes
from nitorch.spatial import (identity_grid, affine_grid, grid_pull,
                             affine_matmul, affine_lmdiv, affine_matvec,
                             affine_conv)
from nitorch.core.utils import movedim
from ._utils import BacktrackingLineSearch
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
        -pyr, --pyramid              Number of pyramid levels (1)
        +mov, --other-moving         Additional moving images, without loss

examples:
    # rigid registration between two images
    autoreg -mi -fix target.nii -mov source.nii
    
    # diffeo + similitude registration between two images
    autoreg -mi -fix target.nii -mov source.nii -diffeo -sim
    
    # diffeo + rigid between a template and a segment from SPM
    # > note that the tpm is the *moving* image but we use the -inv tag
    #   so that the fixed image is also wrapped to moving space 
    autoreg -cat -fix c1.nii c2.nii -mov tpm.nii -inv -diffeo -rig
    
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
    opt['files'] = list()
    while args and not istag(args[0]):
        opt['files'].append(args[0])
        args = args[1:]
    while args:
        if args[0] in ('-o', '--output'):
            args = args[1:]
            opt['output'] = list()
            while args and not istag(args[0]):
                opt['output'].append(args[0])
                args = args[1:]
        elif args[0] in ('-r', '--resliced'):
            args = args[1:]
            opt['resliced'] = list()
            while args and not istag(args[0]):
                opt['resliced'].append(args[0])
                args = args[1:]
        else:
            break
    return args


def parse_moving_and_target(args, opt):
    """Parse a loss that involves a moving and a target file"""
    loss = opt['loss']

    if args and not args[0].startswith(('-', '+')):
        opt['factor'].append(args[0])
        args = args[1:]

    while args:
        if args[0] in ('-mov', '--moving', '-src', '--source'):
            opt['source'] = dict()
            args = parse_file(args[1:], opt['source'])
            args = args[2:]
        elif args[0] in ('-fix', '--fixed', '-trg', '--target'):
            opt['target'] = dict()
            args = parse_file(args[1:], opt['target'])
        elif args[0] in ('-inter', '--interpolation'):
            opt['interpolation'] = args[1]
            args = args[2:]
        elif args[0] in ('-bnd', '--bound'):
            opt['bound'] = args[1]
            args = args[2:]
        elif args[0] in ('-ex', '--extrapolate'):
            opt['extrapolate'] = True
            args = args[1:]
        elif args[0] in ('-o', '--output'):
            opt['output'] = args[1]
            args = args[2:]
        elif loss == 'mi' and args[0] in ('-p', '--patch'):
            opt['patch'] = list()
            args = args[1:]
            while not args[0].startswith(('-', '+')):
                opt['patch'].append(args[0])
                args = args[1:]
        elif loss == 'mi' and args[0] in ('-f', '--fwhm'):
            opt['fwhm'] = args[1]
            args = args[2:]
        elif loss == 'mi' and args[0] in ('-b', '--bins'):
            opt['bins'] = args[1]
            args = args[2:]
        else:
            break
    return args


def parse_moving(args, opt):
    """Parse a moving file that is not part of a loss"""
    opt['source'] = dict(files=list())
    while args and not istag(args[0]):
        opt['source']['files'].append(args[0])
        args = args[1:]
    while args:
        if args[0] in ('-inter', '--interpolation'):
            opt['interpolation'] = args[1]
            args = args[2:]
        elif args[0] in ('-bnd', '--bound'):
            opt['bound'] = args[1]
            args = args[2:]
        elif args[0] in ('-ex', '--extrapolate'):
            opt['extrapolate'] = True
            args = args[1:]
        elif args[0] in ('-o', '--output'):
            args = args[1:]
            opt['output'] = list()
            while args and not istag(args[0]):
                opt['output'].append(args[0])
                args = args[1:]
        elif args[0] in ('-r', '--resliced'):
            args = args[1:]
            opt['resliced'] = list()
            while args and not istag(args[0]):
                opt['resliced'].append(args[0])
                args = args[1:]
        else:
            break
    return args


# --- WORKER FOR MATCHING GROUPS ---------------------------------------
def parse_match(args, options):

    loss, *args = args
    opt = dict(loss=None)

    if loss in ('+mov', '--other-moving'):
        args = parse_moving(args, opt)
    else:
        opt['loss'] = ('mi' if loss in mi else
                       'njtv' if loss in jtv else
                       'dice' if loss in dice else
                       'cat' if loss in cat else
                       None)
        args = parse_moving_and_target(args, opt)

    options['matches'].append(opt)
    return args


# --- WORKER FOR TRANSFORMATION GROUPS ---------------------------------
def parse_transform(args, options):

    trf, *args = args
    opt = dict(regularizations=[])

    opt['transform'] = ('translation' if trf in translation else
                        'rigid' if trf in rigid else
                        'similitude' if trf in similitude else
                        'affine' if trf in affine else
                        'ffd' if trf in ffd else
                        'diffeo' if trf in diffeo else
                        None)
    while args:
        if args[0] in regularizations:
            reg, *args = args
            factor = []
            while args and not args[0].startswith(('-', '+')):
                factor.append(args[0])
                args = args[1:]
            optreg = dict()
            optreg['regularization'] = ('absolute' if reg in absolute else
                                        'membrane' if reg in membrane else
                                        'bending' if reg in bending else
                                        'linelastic' if reg in linelastic else
                                        None)
            if factor:
                optreg['factor'] = (factor[0] if reg not in linelastic
                                    else factor)
            opt['regularizations'].append(optreg)
        elif args[0] in ('-init', '--init'):
            opt['init'] = args[1]
            args = args[2:]
        elif args[0] in ('-lr', '--learning-rate'):
            opt['lr'] = args[1]
            args = args[2:]
        elif args[0] in ('-stop', '--stop'):
            opt['stop'] = args[1]
            args = args[2:]
        elif args[0] in ('-o', '--output'):
            opt['output'] = args[1]
            args = args[2:]
        elif trf in ffd and args[0] in ('-g', '--grid'):
            grid = []
            args = args[1:]
            while not args[0].startswith(('-', '+')):
                grid.append(int(args[0]))
                args = args[1:]
            opt['grid'] = grid
        else:
            break

    options['transforms'].append(opt)
    return args


# --- WORKER FOR OPTIMIZER GROUPS --------------------------------------
def parse_optim(args, options):

    optim, *args = args
    opt = dict()

    opt['optimizer'] = ('adam' if optim in adam else
                        'gd' if optim in gd else None)

    while args:
        if args[0] in ('-nit', '--max-iter'):
            opt['max_iter'] = args[1]
            args = args[2:]
        elif args[0] in ('-lr', '--learning-rate'):
            opt['lr'] = args[1]
            args = args[2:]
        elif args[0] in ('-stop', '--stop'):
            opt['min_lr'] = args[1]
            args = args[2:]
        elif args[0] in ('-ls', '--line-search'):
            args = args[1:]
            if args and not args[0].startswith(('-', '+')):
                opt['ls'] = int(args[1])
                args = args[1:]
            else:
                opt['ls'] = True
        else:
            break

    options['optimizers'].append(opt)
    return args


# --- WORKER FOR DEFAULT GROUP -----------------------------------------
def parse_defaults(args, options):
    """Parse default parameters for all groups"""
    opt = dict()
    while args:
        if args[0] in ('-nit', '--max-iter'):
            opt['max_iter'] = args[1]
            args = args[2:]
        elif args[0] in ('-lr', '--learning-rate'):
            opt['lr'] = args[1]
            args = args[2:]
        elif args[0] in ('-stop', '--stop'):
            opt['min_lr'] = args[1]
            args = args[2:]
        elif args[0] in ('-ls', '--line-search'):
            args = args[1:]
            if args and not args[0].startswith(('-', '+')):
                opt['ls'] = int(args[1])
                args = args[1:]
            else:
                opt['ls'] = True
        elif args[0] in ('-inter', '--interpolation'):
            opt['interpolation'] = args[1]
            args = args[2:]
        elif args[0] in ('-bnd', '--bound'):
            opt['bound'] = args[1]
            args = args[2:]
        elif args[0] in ('-ex', '--extrapolate'):
            opt['extrapolate'] = True
            args = args[1:]
        elif args[0] in ('-o', '--output'):
            opt['output'] = args[1]
            args = args[2:]
        else:
            break
    options['defaults'] = opt


# --- MAIN PARSER -----------------------------------------------------
def parse(args):
    """Parse AutoGrad's command-line arguments"""

    args = list(args)

    # This is the object that we will populate
    options = dict(matches=[], transforms=[], optims=[], defaults={})

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
            options['progressive'] = True
            args = args[1:]
        elif args[0] in ('-cpu', '--cpu'):
            if 'device' in options:
                raise ValueError('Cannot use both -cpu and -gpu')
            options['device'] = 'cpu'
            args = args[1:]
        elif args[0] in ('-gpu', '--gpu'):
            if 'device' in options:
                raise ValueError('Cannot use both -cpu and -gpu')
            args = args[1:]
            if args and not args[0].startswith(('-', '+')):
                options['device'] = 'cuda:{:d}'.format(int(args[0]))
                args = args[1:]
            else:
                options['device'] = 'cuda'
        elif args[0] in ('-pyr', '--pyramid'):
            args = args[1:]
            levels = []
            while args and not args[0].startswith(('-', '+')):
                level, *args = args
                if '-' in level:
                    level = level.split('-')
                    level = list(range(int(level[0]), int(level[1])+1))
                    levels.extend(level)
                else:
                    levels.append(int(level))
            if levels:
                options['pyramid'] = levels
            else:
                options['pyramid'] = True

        # Something went wrong
        else:
            raise RuntimeError(f'Argument {args[0]} does not seem to '
                               f'belong to a group')

    return options


# ----------------------------------------------------------------------
#                       FILL DEFAULTS VALUES
# ----------------------------------------------------------------------
def defaults(options):
    """Fill some (not all) default values"""
    if not options['optims']:
        options['optims'].append(dict(optim='adam'))
    if not options['transforms']:
        options['transforms'].append(dict(transform='rigid'))
    for transform in options['transforms']:
        if (transform['transform'] in ('ffd', 'diffeo') and
                not transform['regularizations']):
            transform['regularizations'].append(dict(regularization='bending'))
    if not torch.cuda.is_available():
        options['device'] = 'cpu'
    elif not options.get('device', None):
        options['device'] = 'cuda'
    return options


def input_info(options):
    """Extract some info about the input files."""

    def process_file(fname):
        file = dict()
        file['dirname'] = os.path.dirname(fname)
        file['basename'] = os.path.basename(fname)
        file['basename'], file['ext'] = os.path.splitext(file['basename'])
        if file['ext'] == '.gz':
            file['basename'], ext = os.path.splitext(file['basename'])
            file['ext'] = ext + file['ext']
        f = io.map(fname)
        file['shape'] = f.shape[:3]
        while len(f.shape) > 3:
            if f.shape[-1] == 1:
                f = f[..., 0]
                continue
            elif f.shape[3] == 1:
                f = f[:, :, :, 0, ...]
                continue
            else:
                break
        if len(f.shape) > 4:
            raise RuntimeError('Input has more than 1 channel dimensions.')
        if len(f.shape) > 3:
            file['channels'] = f.shape[-1]
        else:
            file['channels'] = 1
        return file

    def allsame(x):
        x = list(x)
        if x:
            x0 = x[0]
            for x1 in x[1:]:
                if x1 != x0:
                    return False
        return True

    def process_files(input, loss):
        input['files'] = [process_file(file) for file in input['files']]
        input['channels'] = sum(file['channels'] for file in input['files'])
        if not allsame(file['shape'] for file in input['files']):
            raise RuntimeError('All files should have the same (spatial) '
                               'shape for multi-file inputs.')
        input['shape'] = input['files'][0]['shape']
        if loss is None:
            input['type'] = None
        elif loss == 'cat':
            input['type'] = 'proba'
        elif loss == 'dice':
            if input['channels'] > 1:
                input['type'] = 'proba'
            elif input['files'][0]['float']:
                input['type'] = 'proba'
            else:
                input['type'] = 'labels'
        else:
            input['type'] = 'intensity'

    for match in options['matches']:
        process_files(match['source'], match['loss'])
        if 'target' in match:
            process_files(match['target'], match['loss'])
            if match['source']['type'] != match['target']['type']:
                raise RuntimeError('Source and target do not have the same '
                                   'modality type:\n'
                                   f'- source: {match["source"]["fname"]} '
                                   f'({match["source"]["type"]})\n'
                                   f'- target: {match["target"]["fname"]} '
                                   f'({match["target"]["type"]})\n')
    return options


def output_info(options):

    def process_file(file, odir=None, prefix='', suffix=''):
        basename = file['basename']
        basename = prefix + basename + suffix
        dirname = odir if odir is not None else file['dirname']
        if file['ext'] in ('.nii', '.nii.gz', '.mgh', '.mgz'):
            ext = file['ext']
        else:
            ext = '.nii.gz'
        oname = os.path.join(dirname, basename + ext)
        return oname

    def process_files(input, odir=None, prefix=''):
        if 'output' in input:
            if not input['output']:
                input['output'] = list()
                for file in input['files']:
                    oname = process_file(file, odir, prefix, '.reg')
                    input['output'].append(oname)
        if 'resliced' in input:
            if not input['resliced']:
                input['resliced'] = list()
                for file in input['files']:
                    oname = process_file(file, odir, prefix, '.resliced')
                    input['resliced'].append(oname)

    def process_transform(trf, odir, prefix=''):
        if 'output' in trf:
            if not trf['output']:
                basename = prefix + trf['transform']
                ext = '.nii.gz' if trf['transform'] in ('ffd', 'diffeo') \
                      else '.lta'
                trf['output'] = os.path.join(odir or '', basename + ext)

    odir = options.get('defaults', {}).get('output', None)
    for match in options['matches']:
        process_files(match['source'], odir)
        if 'target' in match:
            process_files(match['target'], odir)
    for trf in options['transforms']:
        process_transform(trf, odir)
    return options


def check_transforms(options):
    has_lin = False
    has_nonlin = False
    for trf in options['transformations']:
        if trf['transformation'] in ('ffd', 'diffeo'):
            if has_nonlin:
                raise RuntimeError('Cannot have both -ffd and -diffeo')
            has_nonlin = True
        else:
            if has_lin:
                raise RuntimeError('Cannot have multiple affines')
            has_lin = True


# ----------------------------------------------------------------------
#                               MAIN CODE
# ----------------------------------------------------------------------
def autoreg():

    # parse arguments
    options = parse(argv)
    if not options:
        return

    options = defaults(options)

    # -- DEBUG
    import pprint
    pprint.PrettyPrinter(indent=4).pprint(options)
    # --

    check_transforms(options)
    options = input_info(options)
    options = output_info(options)

    # -- DEBUG
    import pprint
    pprint.PrettyPrinter(indent=4).pprint(options)
    # --

    # MAIN STUFF
    data = load_data(options['matches'], options['defaults'],
                     options.get('pyramid', [1]))
    parameters = init_parameters(options)
    losses = init_losses(options)
    device = options['device']

    while not all_optimized(parameters):

        add_freedom(parameters, options)
        optimizers = init_optimizers(parameters, options['defaults'])

        for optimizer in optimizers:
            optimize(data, parameters, losses, optimizer, device=device)

    write_parameters(parameters, options)
    write_data(data, parameters, options)


def init_losses(options):

    def mi(factor, patch=None):
        def _mi(x, y):
            xs = x.unbind(0)
            ys = y.unbind(0)
            loss = 0
            nb_channels = len(xs)
            for x, y in zip(xs, ys):
                x = x[None, None]
                y = y[None, None]
                loss += nn.MutualInfoLoss(patch=patch)(x, y) / nb_channels
            # I take the average of MI across channels to be consistent
            # with how MSE works.
            return loss * factor
        return _mi

    def mse(factor):
        def _mse(x, y):
            return torch.nn.MSELoss(x[None], y[None]) * factor
        return _mse

    def dice(factor, weighted=False):
        def _dice(x, y):
            fn = nn.DiceLoss(discard_background=True, weighted=weighted)
            return fn(x[None], y[None]) * factor
        return _dice

    def cat(factor):
        def _cat(x, y):
            fn = nn.CategoricalLoss(log=False, implicit=True)
            return fn(x[None], y[None]) * factor
        return _cat

    defaults = options['defaults']

    losses = dict(match=[])
    for match in options['matches']:
        factor = match.get('factor', 1)
        if match['loss'] == 'mi':
            patch = match.get('patch', defaults.get('patch', 0))
            losses['matches'].append(mi(factor, patch))
        elif match['loss'] == 'mse':
            losses['matches'].append(mse(factor))
        elif match['loss'] == 'dice':
            weighted = match.get('weighted', defaults.get('weighted', False))
            losses['matches'].append(dice(factor, weighted))
        elif match['loss'] == 'cat':
            losses['matches'].append(cat(factor))

    for trf in options['transformations']:
        if trf['regularizations']:
            if trf['transformation'] in ('ffd', 'diffeo'):
                trfkey = trf['transformation']
            else:
                trfkey = 'affine'
            losses[trfkey] = []
            for reg in trf['regularizations']:
                factor = reg.get('factor', 1)
                if reg['regularization'] == 'absolute':
                    losses[trfkey].append(lambda v: factor * spatial.absolute_grid(v))
                elif reg['regularization'] == 'membrane':
                    losses[trfkey].append(lambda v: factor * spatial.membrane_grid(v))
                elif reg['regularization'] == 'bending':
                    losses[trfkey].append(lambda v: factor * spatial.bending_grid(v))
                elif reg['regularization'] == 'linearelastic':
                    factor = core.pyutils.make_list(factor, 2)
                    losses[trfkey].append(lambda v: factor[0] * spatial.lame_div(v))
                    losses[trfkey].append(lambda v: factor[1] * spatial.lame_shear(v))


def init_parameters(options):
    nonlin = ''
    affine = ''
    grid = None
    for trf in options['transformations']:
        if trf['tranformation'] in ('ffd', 'diffeo'):
            nonlin = trf['tranformation']
            if trf['tranformation'] == 'ffd':
                grid = trf['tranformation'].get('grid', [10])
                grid = core.pyutils.make_list(grid, 3)
        else:
            affine = trf['tranformation']

    parameters = dict()

    if affine:
        if affine == 'translation':
            nb_affine_prm = 3
        elif affine == 'rigid':
            nb_affine_prm = 6
        elif affine == 'similitude':
            nb_affine_prm = 7
        elif affine == 'affine':
            nb_affine_prm = 12
        parameters['affine'] = torch.zeros(nb_affine_prm)

    if nonlin:
        # compute mean space
        all_affines = []
        all_shapes = []
        for match in options['matches']:
            if not match['loss']:
                continue
            all_shapes.append(match['target']['shape'])
            f = io.map(match['target']['fname'])
            all_affines.append(f.affine)
        affine, shape = spatial.mean_space(all_affines, all_shapes)
        parameters[nonlin] = dict()
        parameters[nonlin]['affine'] = affine
        parameters[nonlin]['shape'] = shape
        if nonlin == 'ffd':
            parameters[nonlin]['parameters'] = torch.zeros([*grid, 3])
        else:
            parameters[nonlin]['parameters'] = torch.zeros([*shape, 3])

    return dict(hidden=parameters)


def write_parameters(parameters, options):
    nonlin = None
    affine = None
    for trf in options['transformations']:
        if trf['tranformation'] in ('ffd', 'diffeo'):
            nonlin = trf
        else:
            affine = trf

    if affine:
        fname = affine['output']
        param = parameters['affine']
        io.transforms.savef(param, fname, type=2)

    if nonlin:
        fname = nonlin['output']
        tname = nonlin['transformation']
        param = parameters[tname]['parameters']
        affine = parameters[tname]['affine']
        shape = parameters[tname]['shape']
        if tname == 'grid':
            factor = [s/g for s, g in zip(shape, param.shape[:-1])]
            affine, _ = spatial.affine_resize(affine, shape, factor)
        io.volumes.savef(param, fname, affine=affine)


def write_data(matches, parameters, options):

    def load_data1(match, tag):
        source = []
        affine = None
        for file in match[tag]['files']:
            f = io.map(file['fname'])
            if affine is None:
                affine = f.affine
            if match[tag]['type'] == 'labels':
                f = f.data(dtype=torch.int32)
            else:
                f = f.fdata(rand=True)
            f = f.reshape((*file['shape'], file['channels']))
            f = movedim(f, -1, 0)
            source.append(f)
        source = torch.cat(source, dim=0)

        if match[tag]['type'] == 'labels':
            # I don't have a nice pooling implemented for hard labels
            # There are plenty of ways to do this, I'll have to think.
            sources = [(source, affine)]
        else:
            sources = [(source, affine)] if 1 in levels else []
            for l in range(2, max_level+1):
                shape = source.shape[1:]
                ker = [min(2, d) for d in shape]

                source = torch.nn.functional.avg_pool3d(source[None], ker)[0]
                affine, _ = affine_conv(affine, shape, ker, ker)
                if l in levels:
                    sources.append((source, affine))

        return sources

    data = []
    for match in matches:
        if not match['loss']:
            continue

        default_order = 0 if match['source']['type'] == 'labels' else 1
        prm = {
            'interpolation': match.get('interpolation',
                                       defaults.get('interpolation',
                                                    default_order)),
            'bound': match.get('bound', defaults.get('bound', 'dct2')),
            'extrapolate': match.get('extrapolate',
                                     defaults.get('extrapolate', False)),
        }

        sources = load_data1(match, 'source')
        targets = load_data1(match, 'target')
        data_match = [(src, tgt) for src, tgt in zip(sources, targets)]
        data.append((data_match, prm))



def load_data(matches, defaults, pyramid=[1]):
    """Load the data that is used in the optimization + compute pyramid

    Parameters
    ----------
    matches : dict
    defaults : dict
    pyramid : list[int]

    Returns
    -------
    list[list[((source, source_aff), (target, target_aff))], prm)]

    """

    def load_data1(match, tag):
        source = []
        affine = None
        for file in match[tag]['files']:
            f = io.map(file['fname'])
            if affine is None:
                affine = f.affine
            if match[tag]['type'] == 'labels':
                f = f.data(dtype=torch.int32)
            else:
                f = f.fdata(rand=True)
            f = f.reshape((*file['shape'], file['channels']))
            f = movedim(f, -1, 0)
            source.append(f)
        source = torch.cat(source, dim=0)

        if match[tag]['type'] == 'labels':
            # I don't have a nice pooling implemented for hard labels
            # There are plenty of ways to do this, I'll have to think.
            sources = [(source, affine)]
        else:
            sources = [(source, affine)] if 1 in levels else []
            for l in range(2, max_level+1):
                shape = source.shape[1:]
                ker = [min(2, d) for d in shape]

                source = torch.nn.functional.avg_pool3d(source[None], ker)[0]
                affine, _ = affine_conv(affine, shape, ker, ker)
                if l in levels:
                    sources.append((source, affine))

        return sources

    levels = list(sorted(set(pyramid)))
    max_level = max(pyramid)
    data = []
    for match in matches:
        if not match['loss']:
            continue

        default_order = 0 if match['source']['type'] == 'labels' else 1
        prm = {
            'interpolation': match.get('interpolation',
                                       defaults.get('interpolation',
                                                    default_order)),
            'bound': match.get('bound', defaults.get('bound', 'dct2')),
            'extrapolate': match.get('extrapolate',
                                     defaults.get('extrapolate', False)),
        }

        sources = load_data1(match, 'source')
        targets = load_data1(match, 'target')
        data_match = [(src, tgt) for src, tgt in zip(sources, targets)]
        data.append((data_match, prm))

    return data


def init_optimizers(parameters, defaults):
    """Initialize optimizers and their step function.

    Parameters
    ----------
    parameters : dict
        {'affine': {'parameters': tensor, ...},
         'ffd':    {'parameters': tensor, ...},
         'diffeo': {'parameters': tensor, ...}}
    defaults : dict
        {'lr': float, 'min_lr': float, 'max_iter': int, 'ls': int}

    Returns
    -------
    list[{'max_iter': int, 'min_ls': float, 'step': callable}]

    """
    params = []
    for param in parameters.values():
        param = torch.nn.Parameter(param['parameters'], requires_grad=True)
        params.append(param)

    optimizers = []
    for optim in parameters['optim']:
        name = optim['optim']
        if name == 'adam':
            optim_klass = torch.optim.Adam
        else:
            optim_klass = torch.optim.SGD
        lr = optim.get('lr', defaults.get('lr', 0.1))
        min_lr = optim.get('min_lr', defaults.get('min_lr', lr*1e-4))
        max_iter = optim.get('max_iter', defaults.get('max_iter', 1000))
        ls = optim.get('ls', defaults.get('ls', 0))
        optim_obj = optim_klass(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_obj)
        if ls != 0:
            if ls is True:
                ls = 6
            optim_obj = BacktrackingLineSearch(optim_obj, max_iter=ls)

        def optim_step(fwd):
            optim_obj.zero_grad()
            loss = fwd()
            loss.backward()
            optim_obj.step()
            scheduler.step(loss)
            return loss

        def optim_lr():
            return optim_obj.param_groups[0]['lr']

        optim = dict(max_iter=max_iter, min_lr=min_lr,
                     lr=optim_lr, step=optim_step)
        optimizers.append(optim)

    return optimizers


def all_optimized(parameters):
    return not parameters['hidden']


def add_freedom(parameters, options):
    if options.get('progressive', False):
        if 'affine' in parameters['hidden']:
            parameters['affine'] = dict(parameters=parameters['hidden']['affine'])
            del parameters['hidden']['affine']
            nb_prm = len(parameters['affine']['parameters'])
            group = ('T' if nb_prm == 3 else
                     'SE' if nb_prm == 6 else
                     'CSO' if nb_prm == 7 else
                     'Aff+')
            backend = dict(dtype=parameters['affine']['parameters'].dtype,
                           device=parameters['affine']['parameters'].device)
            parameters['affine']['basis'] = spatial.affine_basis(group, 3, **backend)
        if 'ffd' in parameters['hidden']:
            parameters['ffd'] = parameters['hidden']['ffd']
            del parameters['hidden']['ffd']
        if 'diffeo' in parameters['hidden']:
            parameters['diffeo'] = parameters['hidden']['diffeo']
            del parameters['hidden']['diffeo']
    else:
        if 'affine' in parameters['hidden']:
            if 'affine' in parameters:
                nb_prm0 = len(parameters['affine']['parameters'])
            else:
                nb_prm0 = 0
            nb_prm = (3 if nb_prm0 == 0 else
                      6 if nb_prm0 == 3 else
                      7 if nb_prm0 == 6 else
                      12)
            subaffine = parameters['hidden']['affine'][:nb_prm]
            parameters['affine'] = dict(parameters=subaffine)
            if len(parameters['hidden']['affine']) == nb_prm:
                del parameters['hidden']['affine']
        elif 'ffd' in parameters['hidden']:
            parameters['ffd'] = parameters['hidden']['ffd']
            del parameters['hidden']['ffd']
        elif 'diffeo' in parameters['hidden']:
            parameters['diffeo'] = parameters['hidden']['diffeo']
            del parameters['hidden']['diffeo']


def optimize(data, parameters, losses, optimizer, device='cpu'):
    """Optimization loop.

    Parameters are updated in-place.

    Parameters
    ----------
    data : list[((source_dat, source_aff), (target_dat, target_aff), prm)]
        prm : {'interpolation': int, 'bound': str, 'extrapolate': bool}
    parameters : dict
        {'affine': {'parameters': tensor, 'basis': tensor},
         'ffd':    {'parameters': tensor, 'affine': tensor, 'shape': tensor},
         'diffeo': {'parameters': tensor, 'affine': tensor,}}
    losses : dict
        {'affine': callable, 'ffd': callable, 'diffeo': callable,
         'match': list[callable]}
    optimizer : dict
        {'max_iter': int, 'step': callable}
    device : str or torch.device, default='cpu'

    """

    device = torch.device(device)
    backend = dict(dtype=torch.float, device=device)

    def forward():
        """Forward pass up to the loss"""

        loss = 0

        # affine matrix
        if 'affine' in parameters:
            q = parameters['affine']['parameters'].to(**backend)
            B = parameters['affine']['basis'].to(**backend)
            A = core.linalg.expm(q, B)
            loss += losses['affine'](q)
        else:
            A = None

        # non-linear displacement field
        if 'ffd' in parameters:
            d = parameters['ffd']['parameters'].to(**backend)
            d = ffd_exp(d, parameters['ffd']['shape'], returns='disp')
            loss += losses['ffd'](d)
            d_aff = parameters['ffd']['affine'].to(**backend)
        elif 'diffeo' in parameters:
            d = parameters['diffeo']['parameters'].to(**backend)
            loss += losses['ffd'](d)
            d = spatial.exp(d, displacement=True)
            d_aff = parameters['diffeo']['affine'].to(**backend)
        else:
            d = None
            d_aff = None

        # loop over image pairs
        for (data1, prm), loss_fn in zip(data, losses['match']):
            nb_levels = len(data1)
            # loop over pyramid levels
            for source, target in data1:
                source_dat, source_aff = source
                target_dat, target_aff = target

                source_dat = source_dat.to(**backend)
                source_aff = source_aff.to(**backend)
                target_dat = target_dat.to(**backend)
                target_aff = target_aff.to(**backend)

                # affine-corrected source space
                if A is not None:
                    Ms = affine_matmul(A, source_aff)
                else:
                    Ms = source_aff

                if d is not None:
                    # target to param
                    Mt = affine_lmdiv(d_aff, target_aff)
                    if samespace(Mt, d.shape[:-1], target_dat.shape[1:]):
                        g = smalldef(d)
                    else:
                        g = affine_grid(Mt, target_dat.shape[1:], **backend)
                        g += pull_grid(d, g)
                    # param to source
                    Ms = affine_lmdiv(Ms, d_aff)
                    g = affine_matvec(Ms, g)
                else:
                    # target to source
                    Mt = target_aff
                    Ms = affine_lmdiv(Ms, Mt)
                    g = affine_grid(Ms, target_dat.shape[1:], **backend)

                # pull source image
                warped_dat = pull(source_dat, g, **prm)
                loss += loss_fn(warped_dat, target_dat) / float(nb_levels)

        return loss

    # optimization loop
    for n_iter in range(1, optimizer['max_iter']):
        loss = optimizer['step'](forward)
        print(f'{n_iter:4d} | {loss.item():12.6f}', end='\r')
        if optimizer['lr']() < optimizer['min_lr']:
            break
    print('')


# ----------------------------------------------------------------------
#                           HELPERS
# ----------------------------------------------------------------------


def samespace(aff, inshape, outshape):
    """Check if two spaces are the same

    Parameters
    ----------
    aff : (dim+1, dim+1)
    inshape : tuple of `dim` int
    outshape : tuple of `dim` int

    Returns
    -------
    bool

    """
    eye = torch.eye(4, dtype=aff.dtype, device=aff.device)
    return inshape == outshape and (aff - eye).allclose()


def ffd_exp(prm, shape, order=3, bound='dft', returns='disp'):
    """Transform FFD parameters into a displacement or transformation grid.

    Parameters
    ----------
    prm : (..., *spatial, dim)
        FFD parameters
    shape : sequence[int]
        Exponentiated shape
    order : int, default=3
        Spline order
    bound : str, default='dft'
        Boundary condition
    returns : {'disp', 'grid', 'disp+grid'}, default='grid'
        What to return:
        - 'disp' -> displacement grid
        - 'grid' -> transformation grid

    Returns
    -------
    disp : (..., *shape, dim), optional
        Displacement grid
    grid : (..., *shape, dim), optional
        Transformation grid

    """
    backend = dict(dtype=prm.dtype, device=prm.device)
    dim = prm.shape[-1]
    batch = prm.shape[:-(dim + 1)]
    prm = prm.reshape([-1, *prm.shape[-(dim + 1):]])
    disp = spatial.resize_grid(prm, type='displacement',
                               shape=shape,
                               interpolation=order, bound=bound)
    disp = disp.reshape(batch + disp.shape[1:])
    grid = disp + spatial.identity_grid(shape, **backend)
    if 'disp' in returns and 'grid' in returns:
        return disp, grid
    elif 'disp' in returns:
        return disp
    elif 'grid' in returns:
        return grid


def pull_grid(gridin, grid):
    """Sample a displacement field.

    Parameters
    ----------
    gridin : (*inshape, dim) tensor
    grid : (*outshape, dim) tensor

    Returns
    -------
    gridout : (*outshape, dim) tensor

    """
    gridin = movedim(gridin, -1, 0)[None]
    grid = grid[None]
    gridout = grid_pull(gridin, grid, bound='dft', extrapolate=True)
    gridout = movedim(gridout[0], 0, -1)
    return gridout


def pull(image, grid, interpolation=1, bound='dct2', extrapolate=False):
    """Sample a multi-channel image

    Parameters
    ----------
    image : (channel, *inshape) tensor
    grid : (*outshape, dim) tensor

    Returns
    -------
    imageout : (channel, *outshape)

    """
    image = image[None]
    grid = grid[None]
    image = grid_pull(image, grid, interpolation=interpolation,
                      bound=bound, extrapolate=extrapolate)[0]
    return image


def smalldef(disp):
    """Transform a displacement grid into a transformation grid

    Parameters
    ----------
    disp : (*shape, dim)

    Returns
    -------
    grid : (*shape, dim)

    """
    id = identity_grid(disp.shape[:-1], dtype=disp.dtype, device=disp.device)
    return disp + id
