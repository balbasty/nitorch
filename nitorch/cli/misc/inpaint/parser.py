from nitorch.core import cli
from nitorch.core.cli import ParseError
from nitorch.core.struct import Structure


class InPaint(Structure):
    """Structure that holds parameters of the `nipaint` command"""
    files: list = []
    missing: str = 'nan'
    max_rls = 10
    tol_rls = 1e-5
    max_cg = 32
    tol_cg = 1e-5
    output: list = '{dir}{sep}{base}.inpaint{ext}'
    device: str = 'cpu'
    verbose: int = 1


help = r"""[nitorch] Inpainting of missing data

usage:
    nitorch inpaint *FILES [-m *VAL] [-o *FILES] [-cpu|gpu] ...

    -m,    --missing *VAL   Values considered missing (default: 'nan')
    -o,    --output *FILES  Output filenames (default: {dir}/{base}.inpaint{ext})
    -nrls, --max-rls N      Maximum number of RLS iterations (10)
    -trls, --tol-rls TOL    RLS tolerance (1e-5)
    -ncg,  --max-cg N       Maximum number of CG iterations (32)
    -tcg,  --tol-cg TOL     CG tolerance (1e-5)
    -v,    --verbose        Verbosity level (default: 1)
    -cpu, -gpu              Device to use (default: cpu)
"""


def parse(args):
    """Parse the command-line arguments of the `inpaint` command.

    Parameters
    ----------
    args : list of str
        List of arguments, without the command name.

    Returns
    -------
    InPaint
        Filled structure

    """

    struct = InPaint()

    struct.files = []
    while cli.next_isvalue(args):
        val, *args = args
        struct.files.append(val)

    while args:
        if cli.next_isvalue(args):
            raise ParseError(f'Value {args[0]} does not seem to belong '
                             f'to a tag.')
        tag, *args = args
        if tag in ('-m', '--missing'):
            struct.missing = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.missing.append(float(val))
        elif tag in ('-nrls', '--max-rls'):
            cli.check_next_isvalue(args, tag)
            struct.max_rls, *args = args
            struct.max_rls = int(struct.max_rls)
        elif tag in ('-trls', '--tol-rls'):
            cli.check_next_isvalue(args, tag)
            struct.tol_rls, *args = args
            struct.tol_rls = float(struct.tol_rls)
        elif tag in ('-ncg', '--max-cg'):
            cli.check_next_isvalue(args, tag)
            struct.max_cg, *args = args
            struct.max_cg = int(struct.max_cg)
        elif tag in ('-tcg', '--tol-cg'):
            cli.check_next_isvalue(args, tag)
            struct.tol_cg, *args = args
            struct.tol_cg = float(struct.tol_cg)
        elif tag in ('-cpu', '--cpu'):
            struct.device = 'cpu'
        elif tag in ('-gpu', '--gpu'):
            struct.device = 'cuda'
            if cli.next_isvalue(args):
                gpu, *args = args
                struct.device = 'cuda:{:d}'.format(int(gpu))
        elif tag in ('-o', '--output'):
            struct.output = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.output.append(val)
        elif tag in ('-v', '--verbose'):
            struct.verbose = 1
            if cli.next_isvalue(args):
                struct.verbose, *args = args
                struct.verbose = int(struct.verbose)
        elif tag in ('-h', '--help'):
            print(help)
            return None
        else:
            raise ParseError(f'Unknown tag {tag}')

    return struct

