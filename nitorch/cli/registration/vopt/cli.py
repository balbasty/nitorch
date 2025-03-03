import os.path as op

from nitorch.cli.cli import commands
from nitorch.core.cli import ParseError
from nitorch import io
from nitorch.spatial import optimal_svf
from nitorch.core.py import fileparts


def cli(args=None):
    f"""Command-line interface for `vopt`

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


commands['vopt'] = cli

help = r"""[nitorch] vopt

Compute optimal "template-to-subject" SVF from "subject-to-subject" pairs.

usage:
    nitorch vopt --input <fix> <mov> <path> [--input ...]

note:
    In USLR and JUMP, Casamitjana et al. use a L1 objective when fitting
    "mean-to-subject" SVFs from "subject-to-subject" pairs. Although
    they use integral programming, their approach could have been
    implemented in a reweighted fashion, where L2 weights are iteratively
    updated. Here, we instead use fixed L2 weights obtained, at the same
    time as the SVF pairs, from the Hessian of the registration loss.

arguments:
    -i, --input         Affine transform for one pair of images
                          <fix>   Index (or label) of fixed image
                          <mov>   Index (or label) of moving image
                          <path>  Path to an SVF file that warps <mov> to <fix>
    -w, --weight        Precision (= squared weights) for one pair of images
    -o, --output        Path to output transforms (default: {dir}/{label}_optimal{ext})
    -x, --bound [dft]   Boundary condition: {dct2,dct1,dst2,dst1,dft,zero,nearest}
    -a, --absolute      Penalty on absolute displacements (0th) [1e-4]
    -m, --membrane      Penalty on membrane energy (1st) [1e-3]
    -b, --bending       Penalty on bending energy (2nd) [0.2]
    -l, --lame          Penalty on linear elastic energy [0.05, 0.2]]

example:
    nitorch vopt \
      -i mtw pdw mtw_to_pdw.nii.gz \
      -i mtw t1w mtw_to_t1w.nii.gz \
      -i pdw mtw pdw_to_mtw.nii.gz \
      -i pdw t1w pdw_to_t1w.nii.gz \
      -i t1w mtw t1w_to_mtw.nii.gz \
      -i t1w pdw t1w_to_pdw.nii.gz \
      -o out/{label}_to_mean.nii.gz

references:
    "JUMP: A joint multimodal registration pipeline for neuroimaging
    with minimal preprocessing"
    Casamitjana, Iglesias, Tudela, Ninerola-Baizan, Sala-Llonch
    ISBI (2024)

    "USLR: an open-source tool for unbiased and smooth longitudinal
    registration of brain MR"
    Casamitjana, Sala-Llonch, Lekadir, Iglesias
    Arxiv (2023)

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

    inputs, weights = {}, {}
    output = bound = absolute = membrane = bending = lame = None

    tags = (
        '-i', '--input',
        '-w', '--weight',
        '-o', '--output',
        '-x', '--bound',
        '-a', '--absolute',
        '-m', '--membrane',
        '-b', '--bending',
        '-l', '--lame',
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
        elif tag in ('-w', '--weight'):
            fix = args.pop(0)
            if fix in tags:
                raise ParseError(f'Expected <fix> <mov> <path> after {tag}')
            mov = args.pop(0)
            if mov in tags:
                raise ParseError(f'Expected <fix> <mov> <path> after {tag}')
            path = args.pop(0)
            if path in tags:
                raise ParseError(f'Expected <fix> <mov> <path> after {tag}')
            weights[(fix, mov)] = path
        elif tag in ('-o', '--output'):
            out = args.pop(0)
            if out in tags:
                raise ParseError(f'Expected <path> after {tag}')
            if output is not None:
                raise ParseError(f'Max one {tag} accepted')
            output = out
        elif tag in ('-x', '--bound'):
            if bound is not None:
                raise ParseError(f'Max one {tag} accepted')
            bound = args.pop(0)
        elif tag in ('-a', '--absolute'):
            if absolute is not None:
                raise ParseError(f'Max one {tag} accepted')
            absolute = float(args.pop(0))
        elif tag in ('-m', '--membrane'):
            if membrane is not None:
                raise ParseError(f'Max one {tag} accepted')
            membrane = float(args.pop(0))
        elif tag in ('-b', '--bending'):
            if bending is not None:
                raise ParseError(f'Max one {tag} accepted')
            bending = float(args.pop(0))
        elif tag in ('-l', '--lame'):
            if lame is not None:
                raise ParseError(f'Max one {tag} accepted')
            lame = [float(args.pop(0))]
            if args and args[0][:1] != '-':
                lame.append(float(args.pop(0)))
        else:
            raise ParseError(f'Unknown tag {tag}')

    output = output or '{dir}{sep}{label}_optimal.lta'
    bound = bound or 'dft'
    absolute = 1e-4 if absolute is None else absolute
    membrane = 1e-4 if membrane is None else membrane
    bending = 1e-4 if bending is None else bending
    lame = 1e-4 if lame is None else lame

    prm = dict(bound=bound, absolute=absolute, membrane=membrane, bending=bending, lame=lame)
    return inputs, weights, output, prm


def _cli(args):

    inputs, weights, output, prm = parse(args)

    first_input = list(iter(inputs.values()))[0]
    dir, _, ext = fileparts(op.abspath(first_input))

    inputs = {k: io.map(v) for k, v in inputs.items()}
    weights = {k: io.map(v) for k, v in weights.items()}

    aff = list(iter(inputs.values()))[0].affine
    prm["voxel_size"] = aff.square().sum(0).sqrt().tolist()

    optimal = optimal_svf(inputs, weights=weights, **prm)

    labels = list(set([label for pair in inputs for label in pair]))
    for i, label in enumerate(labels):
        ofname = output.format(label=label, sep=op.sep, dir=dir, ext=ext)
        io.savef(optimal[i], ofname, like=first_input)
