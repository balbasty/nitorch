from nitorch.core import cli


help = r"""[nitorch] optinorm

Unbiased (groupwise) normalisation of several scans of the same subject
(e.g., sessions or contrasts) to their mean space.

All pairs of images are registered (rigidly, then diffeomorphically),
and the pairwise transformations are combined into optimal
"mean-to-image" transformations, so that no image is privileged.

usage:
    nitorch optinorm *INPUTS [-o DIR] [-s *STEPS] [options]

Input/output options:
    *INPUTS                         Input images (at least two)
    -o, --output DIR                Output directory [norm]
    -s, --steps *STEPS              Steps to run [all]. Each step reads the
                                    outputs of the previous ones (from DIR),
                                    so that a run can be resumed.
        aff                             Pairwise affine registrations
        affopt                          Optimal mean-to-image affines
        meanspace                       Mean space of the aligned images
        svf                             Pairwise nonlinear registrations (SVF)
        svfopt                          Optimal mean-to-image SVFs
        norm                            Reslice the images to the mean space
    --both                          Also compute unweighted optimal SVFs
                                    [only Hessian-weighted]
    --cleanup                       Delete pairwise registrations once all
                                    steps have succeeded [keep everything]

Registration options:
    --loss NAME                     Matching loss of the pairwise
                                    registrations (see `nitorch register`) [lcc]
    --affine NAME                   Affine model: {rigid, similitude, affine}
                                    [rigid]
    --affine-levels LEVELS          Pyramid levels of the affine
                                    registrations [:4]
    --nonlin-levels LEVELS          Pyramid levels of the nonlinear
                                    registrations [:6]
    -a, --absolute VAL              Penalty on absolute displacements [0]
    -m, --membrane VAL              Penalty on membrane energy [20]
    -b, --bending VAL               Penalty on bending energy [0]
    -l, --lame VAL [VAL]            Penalty on linear-elastic energy [0 0]

Reslicing options:
    -i, --interpolation ORDER       Interpolation order of the outputs [3]
    --no-clip                       Do not clip resliced values to the
                                    input range [clip]

Other options:
    -h, --help [LEVEL]              Display this help
    -v, --verbose [LVL]             Level of verbosity [1]
    --cpu [THREADS], --gpu [ID]     Device to use [cpu]

Output layout (in DIR):
    aff/{i}_{j}/{affine}.lta        Affine from image j (fix) to image i (mov)
    svf/{i}_{j}/svf.nii.gz          SVF from image j (fix) to image i (mov)
    svf/{i}_{j}/hessian.nii.gz      Hessian of the SVF matching term
    opt/aff/{affine}_optim_{i}.lta  Optimal mean-to-image affine
    opt/aff/{affine}_optim_{i}_inv.lta
    opt/{base}.aligned{ext}         Image aligned to the mean (header only)
    opt/meanspace.nii.gz            Mean space
    opt/{base}.aligned.resliced{ext}
    opt/svf/weighted/svf_optim_weighted_{i}.nii.gz
    opt/svf/unweighted/svf_optim_unweighted_{i}.nii.gz  (--both)
    opt/{base}.norm{ext}            Image normalised to the mean space
    (Indices i and j start at 1, in the order of INPUTS.)

"""

steps = ('aff', 'affopt', 'meanspace', 'svf', 'svfopt', 'norm')
affine_models = ('rigid', 'similitude', 'affine')


def convert_device(device):
    def _convert(x):
        return (device, int(x))
    return _convert


parser = cli.CommandParser('optinorm', help=help)
parser.add_positional('input', nargs='+', help='Input images')
parser.add_option('output', ('-o', '--output'), nargs=1, default='norm',
                  help='Output directory')
parser.add_option('steps', ('-s', '--steps'), nargs='+', default=list(steps),
                  validation=cli.Validations.choice(steps),
                  help='Steps to run')
parser.add_option('both', '--both', nargs=0, default=False,
                  action=cli.Actions.store_true,
                  help='Also compute unweighted optimal SVFs')
parser.add_option('cleanup', '--cleanup', nargs=0, default=False,
                  action=cli.Actions.store_true,
                  help='Delete pairwise registrations at the end')
parser.add_option('loss', '--loss', nargs=1, default='lcc',
                  help='Matching loss')
parser.add_option('affine', '--affine', nargs=1, default='rigid',
                  validation=cli.Validations.choice(affine_models),
                  help='Affine model')
parser.add_option('affine_levels', '--affine-levels', nargs=1, default=':4',
                  help='Pyramid levels of the affine registrations')
parser.add_option('nonlin_levels', '--nonlin-levels', nargs=1, default=':6',
                  help='Pyramid levels of the nonlinear registrations')
parser.add_option('absolute', ('-a', '--absolute'), nargs=1, default=0.,
                  convert=float, help='Penalty on absolute displacements')
parser.add_option('membrane', ('-m', '--membrane'), nargs=1, default=20.,
                  convert=float, help='Penalty on membrane energy')
parser.add_option('bending', ('-b', '--bending'), nargs=1, default=0.,
                  convert=float, help='Penalty on bending energy')
parser.add_option('lame', ('-l', '--lame'), nargs='1*', default=[0., 0.],
                  convert=float, help='Penalty on linear-elastic energy')
parser.add_option('order', ('-i', '--interpolation'), nargs=1, default=3,
                  convert=int, help='Interpolation order')
parser.add_option('clip', '--no-clip', nargs=0, default=True,
                  action=cli.Actions.store_false,
                  help='Do not clip resliced values')
parser.add_option('verbose', ('-v', '--verbose'), nargs='?', default=1,
                  convert=int, action=cli.Actions.store_value(1),
                  help='Level of verbosity')
parser.add_option('device', '--gpu', default=('cpu', None), nargs='?',
                  convert=convert_device('gpu'),
                  action=cli.Actions.store_value(('gpu', None)),
                  help='Use GPU (if available)')
parser.add_option('device', '--cpu', nargs='?',
                  convert=convert_device('cpu'),
                  action=cli.Actions.store_value(('cpu', None)),
                  help='Use CPU')
