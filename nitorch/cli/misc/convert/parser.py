from nitorch.core import cli
from nitorch.core.cli import ParseError
from nitorch.core.struct import Structure


class Convert(Structure):
    """Structure that holds parameters of the `info` command"""
    files: list = []
    meta: dict = {}
    dtype: str = ''
    cast: str = 'unsafe'
    format: str = None
    output: list or str = '{dir}{sep}{base}{ext}'


help = r"""[nitorch] Convert between file formats

usage:
    nitorch convert *FILES [-dt DTYPE] [-c CAST] [-m *META] [-o *FILES]

    -dt, --dtype DTYPE           Output data type (default: same)
    -c, --casting CAST           Casting type (default: unsafe) 
                                    unsafe, rescale, rescale_zero
    -m, --metadata KEY=VAL       List of (key, value) pairs.
    -f,--format FMT              Format name (default: guessed from extension)
    -o, --output *FILES          Output filenames (default: {dir}/{base}{ext})
    
formats:
    nifti : {.nii, .nii.gz}  read/write
    mgh   : {.mgh, .mgz}     read/write
    tiff  : {.tif, .tiff}    read
"""


def parse(args):
    """

    Parameters
    ----------
    args : list of str
        Command line arguments (without the command name)

    Returns
    -------
    Convert
        Filled structure

    """

    struct = Convert()

    struct.files = []
    while cli.next_isvalue(args):
        val, *args = args
        struct.files.append(val)

    while args:
        if cli.next_isvalue(args):
            raise ParseError(f'Value {args[0]} does not seem to belong '
                             f'to a tag.')
        tag, *args = args
        if tag in ('-m', '--meta'):
            struct.meta = {}
            while cli.next_isvalue(args):
                val, *args = args
                if '=' not in val:
                    raise ValueError(f'Metadata should have format KEY=VAL '
                                     f'without whitespaces. Found {val} '
                                     f'alone.')
                key, val = val.split('=')
                struct.meta[key] = val
        elif tag in ('-c', '--casting'):
            cli.check_next_isvalue(args, tag)
            struct.cast, *args = args
        elif tag in ('-dt', '--dtype'):
            cli.check_next_isvalue(args, tag)
            struct.dtype, *args = args
        elif tag in ('-f', '--format'):
            cli.check_next_isvalue(args, tag)
            struct.format, *args = args
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

