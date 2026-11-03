import ast

from nitorch.core import cli
from nitorch.core.cli import ParseError
from nitorch.core.struct import Structure
from nitorch.core import bounds


def parse_bound(val):
    """Parse a boundary condition: a bound name/alias or a constant."""
    if val.lower() == 'dct':
        return 'dct2'
    try:
        val = bounds.to_nitorch(val)
        return 0. if val == 'zero' else val
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        raise ParseError(f'Unknown boundary condition {val}. Expected a '
                         f'number or one of {("dct",) + bounds.all_bounds}.')


class Pad(Structure):
    """Structure that holds parameters of the `pad` command"""
    files: list = []
    size: list = []
    size_space: str = 'vox'
    like: str = None
    bound: str or float = 0
    output: list = '{dir}{sep}{base}.pad{ext}'
    transform: list = None


help = r"""[nitorch] Pad a volume

usage:
    nitorch pad *FILES [-s *SIZE [SPACE]] [-b BOUND]
                       [-k FILE] [-o *FILES] [-t *FILES] 

    -s, --size *SIZE [SPACE]     Padding size. Space in {vox (default), ras}
                                 Each size element can be a pair of
                                 lower/upper (or negative/positive) padding
                                 size, e.g. "(0,16)" or "(0, 16)".
    -b, --bound BOUND            Boundary conditions: a number (constant fill)
                                 or {zero (default), replicate, dct2, dft, ...}
                                 Any bound name/alias known to nitorch is
                                 accepted (`dct` is an alias for `dct2`).
    -k, --like FILE              Path to a pre-padded volume to use as reference.
    -o, --output *FILES          Output filenames (default: {dir}/{base}.pad{ext})
    -t, --transform [*FILES]     Input or output transformation filename
                                 (default: none; {dir}/{base}.pad.lta if
                                 the tag is given without a value)
                                    Input if none of s/k options are used.
                                    Output otherwise.
    
notes:
    Only one of --size or --like can be used.
    
"""


def parse(args):
    struct = Pad()

    struct.files = []
    while cli.next_isvalue(args):
        val, *args = args
        struct.files.append(val)

    while args:
        if cli.next_isvalue(args):
            raise ParseError(f'Value {args[0]} does not seem to belong '
                             f'to a tag.')
        tag, *args = args
        if tag in ('-s', '--size'):
            struct.size = []
            while cli.next_isvalue(args):
                val, *args = args
                if val.lower() in ('vox', 'ras'):
                    struct.size_space = val
                elif val.startswith(('(', '[')):
                    # A pair `(lower, upper)`, either in one word or
                    # split across several words.
                    bracket = ')' if val.startswith('(') else ']'
                    while bracket not in val and cli.next_isvalue(args):
                        tmp, *args = args
                        val += tmp
                    if bracket not in val:
                        raise ParseError(f'Unclosed pair in --size: {val}')
                    try:
                        pair = ast.literal_eval(val)
                        pair = tuple(float(v) for v in pair)
                    except Exception:
                        raise ParseError(f'Could not parse pair {val}')
                    if len(pair) != 2:
                        raise ParseError(f'Expected a pair (lower, upper) '
                                         f'but got {val}')
                    struct.size.append(pair)
                else:
                    try:
                        struct.size.append(float(val))
                    except ValueError:
                        raise ParseError(f'Could not parse size {val}')
        elif tag in ('-k', '--like'):
            struct.like, = cli.pop_values(args, '-k/--like', 1,
                                          names='<file>')
        elif tag in ('-b', '--bound'):
            # Negative constants such as `-.5` or `-inf` are values
            bound, = cli.pop_values(args, '-b/--bound', 1, names='<bound>')
            struct.bound = parse_bound(bound)
        elif tag in ('-o', '--output'):
            struct.output = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.output.append(val)
        elif tag in ('-t', '--transform'):
            if not cli.next_isvalue(args):
                struct.transform = ['{dir}{sep}{base}.pad.lta']
            else:
                struct.transform = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.transform.append(val)
        elif tag in ('-h', '--help'):
            print(help)
            return None
        else:
            raise ParseError(f'Unknown tag {tag}')

    return struct

