from nitorch.core import cli
from nitorch.core.cli import ParseError
from nitorch.core.struct import Structure


class FileWithInfo(Structure):
    fname: str = None               # Full path
    shape: tuple = None             # Spatial shape
    affine = None                   # Orientation matrix
    dir: str = None                 # Directory
    base: str = None                # Base name (without extension)
    ext: str = None                 # Extension
    channels: int = None            # Number of channels
    float: bool = True              # Is raw dtype floating point


class Transform(Structure):
    file: str = None
    inv: bool = False


class Linear(Transform):
    pass


class Displacement(Transform):
    order: int = 1
    unit: str = 'vox'
    spline: int = 1


class Velocity(Transform):
    spline: int = 1
    pass


class Composer(Structure):
    transformations: list = []
    target: str = None
    output: str = 'composed{ext}'
    output_unit: str = 'vox'
    device: str = 'cpu'


help = r"""[nitorch] Compose spatial transformations

usage:
    nitorch compose <*TRF> FILE [-t FILE] [-o *FILE] [-ou UNIT] [-cpu|gpu]
    
    <TRF> can take values (with additional options):
    -l, --linear            Linear transform (i.e., affine matrix)
    -d, --displacement      Dense or free-form displacement field
        -n, --order             Order of the encoding splines (1)
        -u, --unit              Unit/Space of the displacement (mm or [vox])
    -v, --velocity          Diffeomorphic velocity field
   
    Each of these transforms can be inverted by prepending 'i' in the
    short form or appending '-inverse' in the long form:
    -il, --linear-inverse            Inverse of a linear transform
    -id, --displacement-inverse      Inverse of a dense or ffd displacement field
    -iv, --velocity-inverse          Inverse of a diffeomorphic velocity field
   
    Other tags are:
    -t,  --target            Defines the target space.
                             If not provided, use the space of the last non-linear transform.
    -o,  --output            Name of the output file (default: 'composed{ext}')
    -ou, --output-unit       Unit of the output nonlinear displacement field (mm or [vox])
    -cpu, -gpu               Device (cpu)
    
    If all transforms are linear, a linear transform is returned.
    Else, a dense non-linear displacement field is returned.
    
"""

lin = ('-l', '--linear')
disp = ('-d', '--displacement')
vel = ('-v', '--velocity')
trf = [*lin, *disp, *vel]
itrf = [t + '-inverse' if t.startswith('--') else '-i' + t[1:]
        for t in trf]


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

    opt = (Linear if tag in lin else
           Displacement if tag in disp else
           Velocity if tag in vel else None)
    opt = opt()
    opt.inv = inv

    cli.check_next_isvalue(args, tag)
    opt.file, *args = args

    while cli.next_istag(args):
        tag, *args = args
        if isinstance(opt, Displacement) and tag in ('-n', '--order'):
            cli.check_next_isvalue(args, tag)
            opt.order, *args = args
            opt.order = int(opt.order)
        if isinstance(opt, Displacement) and tag in ('-u', '--unit'):
            cli.check_next_isvalue(args, tag)
            opt.unit, *args = args
        else:
            args = [tag, *args]
            break

    options.transformations.append(opt)

    return args


# --- MAIN PARSER ------------------------------------------------------
def parse(args):
    """Parse AutoGrad's command-line arguments"""

    # This is the object that we will populate
    options = Composer()

    while args:
        if cli.next_isvalue(args):
            raise ParseError(f'Value {args[0]} does not seem to belong '
                             f'to a tag.')
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
            cli.check_next_isvalue(args, tag)
            options.target, *args = args
        elif tag in ('-o', '--output'):
            cli.check_next_isvalue(args, tag)
            options.output, *args = args
        elif tag in ('-ou', '--output-unit'):
            cli.check_next_isvalue(args, tag)
            options.output_unit, *args = args
        elif tag in ('-cpu', '--cpu'):
            options.device = 'cpu'
        elif tag in ('-gpu', '--gpu'):
            options.device = 'cuda'
            if cli.next_isvalue(args):
                gpu, *args = args
                options.device = 'cuda:{:d}'.format(int(gpu))

        # Something went wrong
        else:
            raise ParseError(f'Unknown tag {tag}')

    return options
