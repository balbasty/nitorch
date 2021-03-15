from nitorch.core import cli
from nitorch.core.cli import ParseError


class Vexp(cli.ParsedStructure):
    """Structure that holds parameters of the `crop` command"""
    files: list = []
    type: str = 'd'
    unit: str = 'v'
    bound: str = 'dft'
    inv: bool = False
    nb_steps: int = 8
    output: list = '{dir}{sep}{base}.vexp{ext}'


help = r"""[nitorch] Exponentiate a stationary velocity field

usage:
    nitorch vexp *FILES [-t TYPE] [-u UNIT] [-inv] [-n STEP] [-o *FILES]

    -t,   --type TYPE       {d[isplacement], t[ransformation]} (default: d)
    -u,   --unit UNIT       {v[oxel], mm} (default: v)
    -inv, --inverse         Generate the inverse transform (default: false)
    -n,   --nb-steps STEPS  Number of scaling and squaring steps (default: 8)
    -b,   --bound BND       {dft, dct1, dct2, dst1, dst2, zero}, default=dft
    -o,   --output *FILES   Output filenames (default: {dir}/{base}.vexp{ext})
    
"""


def parse(args):
    struct = Vexp()

    struct.files = []
    while cli.next_isvalue(args):
        val, *args = args
        struct.files.append(val)

    while args:
        if cli.next_isvalue(args):
            raise ParseError(f'Value {args[0]} does not seem to belong '
                             f'to a tag.')
        tag, *args = args
        if tag in ('-t', '--type'):
            cli.check_next_isvalue(args, tag)
            struct.type, *args = args
        if tag in ('-u', '--unit'):
            cli.check_next_isvalue(args, tag)
            struct.unit, *args = args
        if tag in ('-b', '--bound'):
            cli.check_next_isvalue(args, tag)
            struct.bound, *args = args
        if tag in ('-n', '--nb-steps'):
            cli.check_next_isvalue(args, tag)
            struct.nb_steps, *args = args
            struct.nb_steps = int(struct.nb_steps)
        if tag in ('-inv', '--inverse'):
            val = False
            if cli.next_isvalue(args):
                val, *args = args
            if val[0] == 'f':
                val = False
            elif val[0] == 't':
                val = True
            else:
                val = bool(int(val))
            struct.inv = val
        elif tag in ('-o', '--output'):
            struct.output = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.output.append(val)
        elif tag in ('-h', '--help'):
            print(help)
            return None
        else:
            raise ParseError(f'Unknown tag {tag}')

    return struct

