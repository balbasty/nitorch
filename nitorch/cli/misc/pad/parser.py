import ast

from nitorch.core import cli
from nitorch.core.cli import ParseError
from nitorch.core.struct import Structure


class Pad(Structure):
    """Structure that holds parameters of the `crop` command"""
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
                                 lower/upper (or negative/positive) padding size.
    -b, --bound BOUND            Boundary conditions 
                                 {zero (default), replicate, dct, dct2, dft}
    -k, --like FILE              Path to a pre-padded volume to use as reference.
    -o, --output *FILES          Output filenames (default: {dir}/{base}.crop{ext})
    -t, --transform              Input or output transformation filename (default: none)
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
                elif '(' in val or '[' in val:
                    bracket = ')' if '(' in val else ']'
                    while cli.next_isvalue(args):
                        tmp, *args = args
                        val += tmp
                        if bracket in tmp:
                            val = ast.literal_eval(val)
                            struct.size.append(val)
                            break
                else:
                    struct.size.append(float(val))
        elif tag in ('-k', '--like'):
            cli.check_next_isvalue(args, tag)
            struct.like = args.pop(0)
        elif tag in ('-b', '--bound'):
            cli.check_next_isvalue(args, tag)
            struct.bound = args.pop(0)
            if struct.bound not in ('zero', 'dft', 'dct', 'dct2', 'replicate'):
                struct.bound = float(struct.bound)
            if struct.bound == 'zero':
                struct.bound = 0.
        elif tag in ('-o', '--output'):
            struct.output = []
            while cli.next_isvalue(args):
                val, *args = args
                struct.output.append(val)
        elif tag in ('-t', '--transform'):
            if not cli.next_isvalue(args):
                struct.transform = ['{dir}{sep}{base}.crop.lta']
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

