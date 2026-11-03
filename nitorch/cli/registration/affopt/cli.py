import sys

from nitorch.cli.cli import commands
from nitorch.core.cli import (
    ParseError, pop_values, split_double_dash, check_rest_used)
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
        print(f'[ERROR] {str(e)}', file=sys.stderr)
        return 1
    except Exception as e:
        print(f'[ERROR] {str(e)}', file=sys.stderr)
        return 1


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
    nitorch affopt ... -i <fix> <mov> -- <path>  (<path> may start with -)

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

    # `--` ends option parsing: what follows completes the values of
    # the option just before it (e.g., `-i fix mov -- -odd.lta`)
    args, rest = split_double_dash(args)
    if not args and not rest:
        raise ParseError('No arguments')

    inputs = {}
    output = log = affine = similitude = rigid = None

    while args:
        tag = args.pop(0)
        if tag in ('-h', '--help'):
            raise AskForHelp
        elif tag in ('-i', '--input'):
            fix, mov, path = pop_values(
                args, '-i/--input', 3, names='<fix> <mov> <path>',
                rest=rest)
            inputs[(fix, mov)] = path
        elif tag in ('-o', '--output'):
            out, = pop_values(args, '-o/--output', 1, names='<path>',
                              rest=rest)
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
    check_rest_used(rest, "use '--' just before a value of the last "
                          "option (e.g., -i <fix> <mov> -- <path>)")

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
