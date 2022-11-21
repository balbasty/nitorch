from nitorch.core import cli
from nitorch.core.cli import ParseError
from nitorch.core.struct import Structure


class DenoiseMRI(Structure):
    """Structure that holds parameters of the `denoise_mri` command"""
    files: list = []
    lam_scl: float = 10.0
    learning_rate: float = 1e1
    max_iter: int = 10000
    tolerance: float = 1e-8
    verbose: bool = True
    device: str = 'cuda'
    do_write: bool = True
    dir_out: str = None

help = r"""[nitorch] Total variation denoising of magnetic resonance images (MRIs). For multi-
sequence denoising, requires all images to have the same dimensions.

usage:
    nitorch denoise_mri *FILES <*OPT>

    <OPT> can take values:
        -s,   --lam_scl        Scaling of regularisation values ({:})
        -lr,  --learning_rate  Optimiser learning rate ({:})
        -n,   --max_iter N     Maximum number of fitting iterations ({:})
        -t,   --tolerance TOL  Convergence threshold (when to stop iterating) ({:})
        -v,   --verbose        Print to terminal? ({:})
        -o,   --dir_out        Directory where to write output (default is same as input)   
        -cpu, -gpu             Device to use (default: gpu)
        
examples:
    # single-channel MR image denoising
    nitorch denoise_mri T1w.nii
            
    # multi-channel MR image denoising
    nitorch denoise_mri T1w.nii T2w.nii -o './output'
                
""".format(DenoiseMRI.lam_scl, DenoiseMRI.learning_rate, DenoiseMRI.max_iter, DenoiseMRI.tolerance,
           DenoiseMRI.verbose)


def parse(args):
    """Parse the command-line arguments of the `denoise_mri` command.

    Parameters
    ----------
    args : list of str
        List of arguments, without the command name.

    Returns
    -------
    DenoiseMRI
        Filled structure

    """

    struct = DenoiseMRI()

    struct.files = []
    while cli.next_isvalue(args):
        val, *args = args
        struct.files.append(val)

    while args:
        if cli.next_isvalue(args):
            raise ParseError(f'Value {args[0]} does not seem to belong '
                             f'to a tag.')
        tag, *args = args
        if tag in ('-s', '--lam_scl'):
            cli.check_next_isvalue(args, tag)
            struct.lam_scl, *args = args
            struct.lam_scl = float(struct.lam_scl)
        elif tag in ('-lr', '--learning_rate'):
            cli.check_next_isvalue(args, tag)
            struct.learning_rate, *args = args
            struct.learning_rate = float(struct.learning_rate)
        elif tag in ('-n', '--max_iter'):
            cli.check_next_isvalue(args, tag)
            struct.max_iter, *args = args
            struct.max_iter = int(struct.max_iter)
        elif tag in ('-t', '--tolerance'):
            cli.check_next_isvalue(args, tag)
            struct.tolerance, *args = args
            struct.tolerance = float(struct.tolerance)
        elif tag in ('-cpu', '--cpu'):
            struct.device = 'cpu'
        elif tag in ('-gpu', '--gpu'):
            struct.device = 'cuda'
            if cli.next_isvalue(args):
                gpu, *args = args
                struct.device = 'cuda:{:d}'.format(int(gpu))
        elif tag in ('-o', '--dir_out'):
            cli.check_next_isvalue(args, tag)
            struct.dir_out, *args = args
            struct.dir_out = str(struct.dir_out)
        elif tag in ('-v', '--verbose'):
            cli.check_next_isvalue(args, tag)
            struct.verbose, *args = args
            struct.verbose = bool(struct.verbose)
        elif tag in ('-h', '--help'):
            print(help)
            return None
        else:
            raise ParseError(f'Unknown tag {tag}')

    return struct

