from . import struct

help = r"""[nitorch] Reslice volumes

usage:
    nireslice *FILES <*TRF> FILE [-t FILE] [-o *FILE] 
              [-inter ORDER] [-bnd BND] [-ex] [-cpu|gpu]
    
    <TRF> can take values (with additional options):
    -l, --linear            Linear transform (i.e., affine matrix)
    -d, --displacement      Dense or free-form displacement field
        -n, --order             Order of the encoding splines (1)
    -v, --velocity          Diffeomorphic velocity field
   
    Each of these transforms can be inverted by prepending 'i' in the
    short form or appending '-inverse' in the long form:
    -il, --linear-inverse            Inverse of a L=linear transform
    -id, --displacement-inverse      Inverse of a dense or ffd displacement field
    -iv, --velocity-inverse          Inverse of a diffeomorphic velocity field
   
    Other tags are:
    -t,     --target            Defines the target space.
                                If not provided, minimal reslicing is performed.
    -o,     --output            Name of the output file (default: '*.resliced*')
    -inter, --interpolation     Interpolation order (1)
    -bnd,   --bound             Boundary conditions (dct2)
    -ex,    --extrapolate       Extrapolate out-of-bounds data (no)
    -cpu, -gpu                  Device to use (cpu)
   
   The output image is
    input_dat(inv(input_aff) o trf[0] o trf[1] o ... trf[-1] o target_aff)
    
    For example, if `autoreg` has been run to register 'mov.nii' to 'fix.nii'
    and has generated transforms 'affine.lta' and 'velocity.nii':
    - reslicing mov to fix:
        nireslice mov.nii -il affine.lta -v velocity.nii -t fix.nii
    - reslicing fix to mov:
        nireslice fix.nii -iv velocity.nii -l affine.lta -t mov.nii
    
"""


class ParseError(RuntimeError):
    pass


lin = ('-l', '--linear')
disp = ('-d', '--displacement')
vel = ('-v', '--velocity')
trf = [*lin, *disp, *vel]
itrf = [t + '-inverse' if t.startswith('--') else '-i' + t[1:]
        for t in trf]

# --- HELPERS ----------------------------------------------------------
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
        raise RuntimeError(f'Expected a value for tag {group} '
                           f'but found nothing.')

# --- TRF PARSER -------------------------------------------------------
def parse_transform(args, options):

    tag, *args = args

    # is it an inversed transform?
    inv = False
    if tag in itrf:
        inv = True
        if tag.startswith('--'):
            tag = tag.split('--inverse')[0]
        else:
            tag = '-' + tag[2:]

    opt = (struct.Linear if tag in lin else
           struct.Displacement if tag in disp else
           struct.Velocity if tag in vel else None)
    opt = opt()
    opt.inv = inv

    check_next_isvalue(args, tag)
    opt.file, *args = args

    while next_istag(args):
        tag, *args = args
        if isinstance(opt, struct.Displacement) and tag in ('-n', '--order'):
            check_next_isvalue(args, tag)
            opt.order, *args = args
            opt.order = int(opt.order)
        else:
            args = [tag, *args]
            break

    options.transformations.append(opt)

    return args


# --- MAIN PARSER ------------------------------------------------------
def parse(args):
    """Parse AutoGrad's command-line arguments"""

    # This is the object that we will populate
    options = struct.Reslicer()

    _, *args = args  # remove script name from list

    # Get input files
    while next_isvalue(args):
        val, *args = args
        options.files.append(val)

    while args:
        if not next_istag(args):
            break
        tag, *args = args

        # Parse transforms
        if tag in (*trf, *itrf):
            args = parse_transform([tag, *args], options)

        # Help -> return empty option
        elif tag in ('-h', '--help'):
            print(help)
            return {}

        # Parse remaining top-level tags
        elif tag in ('-t', '--target'):
            check_next_isvalue(args, tag)
            options.target, *args = args
        elif tag in ('-o', '--output'):
            options.output = []
            while next_isvalue(args):
                val, *args = args
                options.output.append(val)
        elif tag in ('-inter', '--interpolation'):
            check_next_isvalue(args, tag)
            options.interpolation, *args = args
            options.interpolation = int(options.interpolation)
        elif tag in ('-bnd', '--bound'):
            check_next_isvalue(args, tag)
            options.bound, *args = args
        elif tag in ('-ex', '--extrapolate'):
            options.extrapolate = False
        elif tag in ('-cpu', '--cpu'):
            options.device = 'cpu'
        elif tag in ('-gpu', '--gpu'):
            options.device = 'cuda'
            if next_isvalue(args):
                gpu, *args = args
                options.device = 'cuda:{:d}'.format(int(gpu))

        # Something went wrong
        else:
            print(help)
            raise RuntimeError(f'Argument {tag} does not seem to '
                               f'belong to a group')

    return options

