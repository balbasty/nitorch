import sys

from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from nitorch.io.transforms import loadf, savef
from nitorch.spatial import optimal_affine


def cli(args=None):
    f"""Command-line interface for `affopt`

    {help}

    """

    # Exceptions are dealt with here
    try:
        _cli(args)
    except AskForHelp:
        print(help)
        return
    except ParseError as e:
        print(help)
        print('[ERROR]', e)
        return
    # except Exception as e:
    #     print(f'[ERROR] {str(e)}', file=sys.stderr)


commands['affopt'] = cli

help = r"""[nitorch] affopt

Compute optimal "template-to-subject" matrices from "subject-to-subject" pairs.

notes:
    In Leung et al, all possible pairs of images are registered (in both
    forward and backward directions), such that the "subject-to-template"
    transforms can easily be computed as T[i,tpl] = expm(mean_j(logm(T[i,j]))).

    Our implementation differs in several aspects:
    - We allow some transformation pairs to be missing, at the cost of
      introducing bias in the mean space estimation. This bias can be
      overcome in the statistical sense if the number of subjects is large
      and evaluated pairs are randomly sampled.
    - Instead of first symmetrizing pairwise transforms, we fit the mean
      space from all possible forward and backward transformations.
    - Instead of minimizing the L2 norm in the matrix Lie algebra
      (which is done implicitly by Leung et al's method), we add
      the possibility to minimize the L2 norm in the embedding space (i.e.,
      the Frobenius norm of affine matrices). This method is more accurate
      when pairwise transformations are large, in which case affine
      composition is badly approximated by log-matrix addition.

usage:
    nitorch affopt --input <fix> <mov> <path> [--input ...]

arguments:
    -i, --input         Affine transform for one pair of images
                          <fix>   Index (or label) of fixed image
                          <mov>   Index (or label) of moving image
                          <path>  Path to an LTA file that warps <mov> to <fix>
    -o, --output        Path to output transforms (default: {label}_optimal.lta)
    -l, --log           Minimize L2 in Lie algebra (default: L2 in matrix space)
    -a, --affine        Assume transforms are all affine (default)
    -s, --similitude    Assume transforms are all similitude
    -r, --rigid         Assume transforms are all rigid

example:
    nitorch affopt \
      -i mtw pdw mtw_to_pdw.lta \
      -i mtw t1w mtw_to_t1w.lta \
      -i pdw mtw pdw_to_mtw.lta \
      -i pdw t1w pdw_to_t1w.lta \
      -i t1w mtw t1w_to_mtw.lta \
      -i t1w pdw t1w_to_pdw.lta \
      -o out/{label}_to_mean.lta

references:
    "Consistent multi-time-point brain atrophy estimation from the
    boundary shift integral"
    Leung, Ridgway, Ourselin, Fox
    NeuroImage (2011)

    "Symmetric Diffeomorphic Modeling of Longitudinal Structural MRI"
    Ashburner, Ridgway
    Front Neurosci. (2012)
"""


class AskForHelp(Exception):
    pass


def parse(args):

    args = list(args)
    if not args:
        raise ParseError('No arguments')

    inputs = {}
    output = log = affine = similitude = rigid = None

    tags = (
        '-i', '--input',
        '-o', '--output',
        '-l', '--lie', '--log',
        '-a', '--aff', '--affine',
        '-s', '--sim', '--similitude',
        '-r', '--rig', '--rigid'
    )

    while args:
        tag = args.pop(0)
        if tag in ('-h', '--help'):
            raise AskForHelp
        elif tag in ('-i', '--input'):
            fix = args.pop(0)
            if fix in tags:
                raise ParseError(f'Expected <fix> <mov> <path> after {tag}')
            mov = args.pop(0)
            if mov in tags:
                raise ParseError(f'Expected <fix> <mov> <path> after {tag}')
            path = args.pop(0)
            if path in tags:
                raise ParseError(f'Expected <fix> <mov> <path> after {tag}')
            inputs[(fix, mov)] = path
        elif tag in ('-o', '--output'):
            out = args.pop(0)
            if out in tags:
                raise ParseError(f'Expected <path> after {tag}')
            if output is not None:
                raise ParseError(f'Max one {tag} accepted')
            output = out
        elif tag in ('-l', '--log', '--lie'):
            if log is not None:
                raise ParseError(f'Max one {tag} accepted')
            log = True
        elif tag in ('-a', '--aff', '--affine'):
            if affine is not None:
                raise ParseError(f'Max one {tag} accepted')
            affine = True
        elif tag in ('-s', '--sim', '--similitude'):
            if similitude is not None:
                raise ParseError(f'Max one {tag} accepted')
            similitude = True
        elif tag in ('-r', '--rig', '--rigid'):
            if rigid is not None:
                raise ParseError(f'Max one {tag} accepted')
            rigid = True
        else:
            raise ParseError(f'Unknown tag {tag}')

    output = output or '{label}_optimal.lta'
    affine = affine or False
    similitude = similitude or False
    rigid = rigid or False
    if int(rigid) + int(similitude) + int(affine) > 1:
        raise ParseError('Max one of --rigid, --similitude, --affine accepeted')
    if int(rigid) + int(similitude) + int(affine) == 0:
        affine = True
    if affine:
        basis = 'Aff+'
    elif similitude:
        basis = 'CSO'
    else:
        basis = 'SE'

    return inputs, output, log, basis


def _cli(args):

    inputs, output, log, basis = parse(args)

    # LinearTransformArray(v).matrix() returns array with shape
    # [N, 4, 4], although N is always 1 in our case.
    inputs = {k: loadf(v).squeeze()
              for k, v in inputs.items()}
    optimal = optimal_affine(inputs, basis=basis,
                             loss='log' if log else 'exp')

    labels = list(set([label for pair in inputs for label in pair]))
    for i, label in enumerate(labels):
        savef(optimal[i], output.format(label=label), type='ras')
