from . import struct
from nitorch.core import cli

help = r"""[nitorch] Reslice volumes

usage:
    nitorch reslice *FILES <*TRF> FILE [-t FILE] [-o *FILE] 
                    [-i ORDER] [-b BND] [-p] [-x] [-cpu|gpu]
    
    <TRF> can take values (with additional options):
    -l, --linear            Linear transform (i.e., affine matrix)
    -d, --displacement      Dense or free-form displacement field
        -n, --order             Order of the encoding splines (1)
        -u, --unit              Unit/Space of the displacement (mm or [vox])
    -v, --velocity          Diffeomorphic velocity field 
                            [and JSON file of shooting parameters]
                            If no JSON file, assume stationary velocity.
   
    Each of these transforms can be inverted by prepending 'i' in the
    short form or appending '-inverse' in the long form:
    -il, --linear-inverse            Inverse of a linear transform
    -id, --displacement-inverse      Inverse of a dense or ffd displacement field
    -iv, --velocity-inverse          Inverse of a diffeomorphic velocity field
   
    Other tags are:
    -t, --target            Defines the target space.
                            If not provided, minimal reslicing is performed.
    -o, --output            Name of the output file (default: '*.resliced*')
    -i, --interpolation     Interpolation order. Use `l` for labels. (1)
    -p, --prefilter         Apply spline prefilter (yes)
    -b, --bound             Boundary conditions (dct2)
    -x, --extrapolate       Extrapolate out-of-bounds data (no)
    -v, --voxel-size        Voxel size of the resliced space (default: from target)
    -cpu, -gpu              Device to use (cpu)
   
   The output image is
    input_dat(inv(inpt_aff) o trf[0] o trf[1] o ... trf[-1] o target_aff)
    
    For example, if `autoreg` has been run to register 'mov.nii' to 'fix.nii'
    and has generated transforms 'affine.lta' and 'velocity.nii':
    - reslicing mov to fix:
        nireslice mov.nii -il affine.lta -v velocity.nii -t fix.nii
    - reslicing fix to mov:
        nireslice fix.nii -iv velocity.nii -l affine.lta -t mov.nii
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

    opt = (struct.Linear if tag in lin else
           struct.Displacement if tag in disp else
           struct.Velocity if tag in vel else None)
    opt = opt()
    opt.inv = inv

    cli.check_next_isvalue(args, tag)
    opt.file, *args = args
    if isinstance(opt, struct.Velocity) and cli.next_isvalue(args):
        opt.json, *args = args

    while cli.next_istag(args):
        tag, *args = args
        if isinstance(opt, struct.Displacement) and tag in ('-n', '--order'):
            cli.check_next_isvalue(args, tag)
            opt.order, *args = args
            opt.order = int(opt.order)
        if isinstance(opt, struct.Displacement) and tag in ('-u', '--unit'):
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
    options = struct.Reslicer()

    # Get input files
    while cli.next_isvalue(args):
        val, *args = args
        options.files.append(val)

    while args:
        if not cli.next_istag(args):
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
            cli.check_next_isvalue(args, tag)
            options.target, *args = args
        elif tag in ('-o', '--output'):
            options.output = []
            while cli.next_isvalue(args):
                val, *args = args
                options.output.append(val)
        elif tag in ('-i', '-inter', '--interpolation'):
            cli.check_next_isvalue(args, tag)
            options.interpolation, *args = args
            try:
                options.interpolation = int(options.interpolation)
            except ValueError:
                if options.interpolation[0] != 'l':
                    raise cli.ParseError('interpolation should be an integer of "l"')
                options.interpolation = 'l'
        elif tag in ('-b', '-bnd', '--bound'):
            cli.check_next_isvalue(args, tag)
            options.bound, *args = args
        elif tag in ('-p', '--prefilter'):
            cli.check_next_isvalue(args, tag)
            options.prefilter, *args = args
            options.prefilter = (
                True if options.prefilter.lower()[0] in 'ty' else
                False if options.prefilter.lower()[0] in 'fn' else
                bool(int(options.prefilter)))
        elif tag in ('-x', '-ex', '--extrapolate'):
            options.extrapolate = False
        elif tag in ('-v', '-vx', '--voxel-size'):
            options.voxel_size = []
            while cli.next_isvalue(args):
                val, *args = args
                options.voxel_size.append(float(val))
        elif tag in ('-cpu', '--cpu'):
            options.device = 'cpu'
        elif tag in ('-gpu', '--gpu'):
            options.device = 'cuda'
            if cli.next_isvalue(args):
                gpu, *args = args
                options.device = 'cuda:{:d}'.format(int(gpu))

        # Something went wrong
        else:
            print(help)
            raise cli.ParseError(f'Argument {tag} does not seem to '
                                 f'belong to a group')

    return options

