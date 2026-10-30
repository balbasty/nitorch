"""Unbiased (groupwise) normalisation of a few images of the same subject.

This is a port of `scripts/optinorm.bash`, which chains:
    1. pairwise affine registrations      (`nitorch register`)
    2. optimal mean-to-image affines      (`nitorch affopt`)
    3. mean space of the aligned images   (`nitorch meanspace`)
    4. pairwise nonlinear registrations   (`nitorch register`)
    5. optimal mean-to-image SVFs         (`nitorch vopt`)
    6. reslicing to the mean space        (`nitorch reslice`)
"""
import os
import shutil

from nitorch import io, spatial
from nitorch.core.py import fileparts
from nitorch.cli.registration.register.cli import _cli as register
from nitorch.cli.registration.meanspace.cli import _cli as meanspace
from nitorch.cli.registration.reslice.main import reslice
from .parser import steps as all_steps

# basis used by `optimal_affine` for each affine model of `register`
affine_bases = {'rigid': 'SE', 'similitude': 'CSO', 'affine': 'Aff+'}


def optinorm(inputs, output='norm', steps=all_steps, both=False,
             cleanup=False, loss='lcc', affine='rigid',
             affine_levels=':4', nonlin_levels=':6',
             absolute=0, membrane=20, bending=0, lame=(0, 0),
             order=3, clip=True, verbose=1, device=('cpu', None)):
    """Normalise a few images of the same subject to their mean space.

    Parameters
    ----------
    inputs : sequence[str]
        Input images (at least two)
    output : str, default='norm'
        Output directory
    steps : sequence of {'aff', 'affopt', 'meanspace', 'svf', 'svfopt', 'norm'}
        Steps to run. Each step reads the outputs of the previous ones
        from `output`, so that a run can be resumed.
    both : bool, default=False
        Also compute unweighted optimal SVFs.
    cleanup : bool, default=False
        Delete the pairwise registrations once all steps have succeeded.
    loss : str, default='lcc'
        Matching loss of the pairwise registrations
    affine : {'rigid', 'similitude', 'affine'}, default='rigid'
        Affine model
    affine_levels, nonlin_levels : str, default=':4', ':6'
        Pyramid levels of the affine and nonlinear registrations
    absolute, membrane, bending, lame : float
        Regularisation of the nonlinear registrations
        (and of the optimal SVFs)
    order : int, default=3
        Interpolation order of the resliced outputs
    clip : bool, default=True
        Clip resliced values to the input range
    verbose : int, default=1
        Level of verbosity
    device : (str, int or None), default=('cpu', None)
        Device ('cpu' or 'gpu') and number of threads (cpu) or id (gpu)

    Returns
    -------
    normalised : list[str]
        Paths to the normalised images (if the 'norm' step was run)

    """
    inputs = [os.path.abspath(path) for path in inputs]
    if len(inputs) < 2:
        raise ValueError('At least two input images are required')
    unknown = set(steps) - set(all_steps)
    if unknown:
        raise ValueError(f'Unknown steps: {sorted(unknown)}')
    if affine not in affine_bases:
        raise ValueError(f'Unknown affine model: {affine}')
    lame = list(lame) if isinstance(lame, (list, tuple)) else [lame]

    paths = OptinormPaths(inputs, output, affine)
    paths.make_dirs()
    pairs = paths.pairs

    # options forwarded to `register` and `reslice`
    device_name, device_id = device
    register_opt = [f'--{device_name}']
    if device_id is not None:
        register_opt += [str(device_id)]
    register_opt += ['-v', str(verbose)]
    reslice_opt = ['-' + device_name]
    if device_name == 'gpu' and device_id is not None:
        reslice_opt += [str(device_id)]
    if not verbose:
        reslice_opt += ['-q']
    reslice_opt += ['-i', str(order)] + (['--clip'] if clip else [])

    def log(*args):
        if verbose:
            print('[optinorm]', *args)

    # ------------------------------------------------------------------
    # Pairwise affine registrations
    # ------------------------------------------------------------------
    if 'aff' in steps:
        for i, j in pairs:
            log(f'affine registration: {i+1} (mov) <- {j+1} (fix)')
            register([
                *register_opt, '-o', paths.pair_dir('aff', i, j),
                '@loss', loss,
                '@@mov', inputs[i], '-o', 'false',
                '@@fix', inputs[j], '-o', 'false',
                '@affine', affine, '-g', '-o', paths.pair_affine(i, j),
                '@pyramid', '--levels', affine_levels,
            ])
            check_exists('aff', paths.pair_affine(i, j))

    # ------------------------------------------------------------------
    # Optimal mean-to-image affines
    # ------------------------------------------------------------------
    if 'affopt' in steps:
        log('optimal affines')
        check_exists('affopt', *[paths.pair_affine(i, j) for i, j in pairs])
        # `pairs[(i, j)]` maps fixed (j) world coordinates to moving (i)
        # world coordinates, which is the convention of `optimal_affine`.
        # Labels are integers so that their (set) ordering is stable.
        matrices = {(i, j): io.transforms.loadf(paths.pair_affine(i, j))
                    .squeeze() for i, j in pairs}
        optimal = spatial.optimal_affine(matrices, basis=affine_bases[affine])
        labels = list(set(label for pair in matrices for label in pair))
        for i in range(len(inputs)):
            mat = optimal[labels.index(i)]
            io.transforms.savef(mat, paths.opt_affine(i), type='ras')
            io.transforms.savef(spatial.affine_inv(mat),
                                paths.opt_affine(i, inverse=True), type='ras')

        # apply optimal affines (without reslicing)
        for i, path in enumerate(inputs):
            reslice([path, '-l', paths.opt_affine(i),
                     '-o', paths.aligned(i), *reslice_opt])
        check_exists('affopt', *[paths.aligned(i) for i in range(len(inputs))])

    # ------------------------------------------------------------------
    # Mean space
    # ------------------------------------------------------------------
    if 'meanspace' in steps:
        log('mean space')
        aligned = [paths.aligned(i) for i in range(len(inputs))]
        check_exists('meanspace', *aligned)
        meanspace([*aligned, '-o', paths.meanspace])
        check_exists('meanspace', paths.meanspace)

        # reslice to mean space (for visualisation)
        for i, path in enumerate(inputs):
            reslice([path, '-l', paths.opt_affine(i), '-t', paths.meanspace,
                     '-o', paths.aligned_resliced(i), *reslice_opt])

    # ------------------------------------------------------------------
    # Pairwise nonlinear registrations
    # ------------------------------------------------------------------
    if 'svf' in steps:
        check_exists('svf', paths.meanspace,
                     *[paths.opt_affine(i) for i in range(len(inputs))])
        for i, j in pairs:
            log(f'nonlinear registration: {i+1} (mov) <- {j+1} (fix)')
            register([
                *register_opt, '-o', paths.pair_dir('svf', i, j),
                '@loss', loss,
                '@@mov', inputs[i], '-a', paths.opt_affine(i), '-o', 'false',
                '@@fix', inputs[j], '-a', paths.opt_affine(j), '-o', 'false',
                '@nonlin', 'svf',
                '-a', str(absolute), '-m', str(membrane),
                '-b', str(bending), '-l', *map(str, lame),
                '-i', paths.meanspace,
                '-o', paths.pair_svf(i, j),
                '-h', paths.pair_hessian(i, j),
                '@pyramid', '--levels', nonlin_levels,
            ])
            check_exists('svf', paths.pair_svf(i, j), paths.pair_hessian(i, j))

    # ------------------------------------------------------------------
    # Optimal mean-to-image SVFs
    # ------------------------------------------------------------------
    if 'svfopt' in steps:
        check_exists('svfopt', *[paths.pair_svf(i, j) for i, j in pairs],
                     *[paths.pair_hessian(i, j) for i, j in pairs])
        svfs = {(i, j): io.map(paths.pair_svf(i, j)) for i, j in pairs}
        affine0 = svfs[pairs[0]].affine
        prm = dict(
            bound='dft',
            absolute=absolute,
            membrane=membrane,
            bending=bending,
            lame=lame,
            voxel_size=spatial.voxel_size(affine0).tolist(),
        )
        modes = ['weighted', 'unweighted'] if both else ['weighted']
        for mode in modes:
            log(f'optimal SVFs ({mode})')
            weights = {}
            if mode == 'weighted':
                weights = {(i, j): io.map(paths.pair_hessian(i, j))
                           for i, j in pairs}
            optimal = spatial.optimal_svf(svfs, weights=weights, **prm)
            labels = list(set(label for pair in svfs for label in pair))
            for i in range(len(inputs)):
                io.savef(optimal[labels.index(i)], paths.opt_svf(i, mode),
                         like=paths.pair_svf(*pairs[0]))
        check_exists('svfopt', *[paths.opt_svf(i, mode)
                                 for i in range(len(inputs)) for mode in modes])

    # ------------------------------------------------------------------
    # Reslice to mean space
    # ------------------------------------------------------------------
    normalised = []
    if 'norm' in steps:
        log('normalisation')
        for i, path in enumerate(inputs):
            check_exists('norm', paths.opt_affine(i), paths.opt_svf(i),
                         paths.meanspace)
            reslice([path, '-l', paths.opt_affine(i),
                     '-v', paths.opt_svf(i), '-t', paths.meanspace,
                     '-o', paths.normalised(i), *reslice_opt])
            normalised.append(paths.normalised(i))
        check_exists('norm', *normalised)

    if cleanup:
        log('cleanup')
        for name in ('aff', 'svf'):
            shutil.rmtree(os.path.join(paths.output, name), ignore_errors=True)

    return normalised


def check_exists(step, *paths):
    """Raise an error if an expected file does not exist"""
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise RuntimeError(f'[optinorm:{step}] Missing file(s): {missing}')


class OptinormPaths:
    """Paths of all inputs and outputs"""

    def __init__(self, inputs, output, affine):
        self.inputs = inputs
        self.output = os.path.abspath(output)
        self.affine = affine
        n = len(inputs)
        self.pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        # use the input basenames in output names, unless they collide
        bases = [fileparts(path)[1] for path in inputs]
        unique = len(set(bases)) == len(bases)
        self.bases = bases if unique else [f'{i+1}_{base}'
                                           for i, base in enumerate(bases)]
        self.exts = [fileparts(path)[2] for path in inputs]

    def make_dirs(self):
        for i, j in self.pairs:
            os.makedirs(self.pair_dir('aff', i, j), exist_ok=True)
            os.makedirs(self.pair_dir('svf', i, j), exist_ok=True)
        os.makedirs(os.path.join(self.output, 'opt', 'aff'), exist_ok=True)
        for mode in ('weighted', 'unweighted'):
            os.makedirs(os.path.join(self.output, 'opt', 'svf', mode),
                        exist_ok=True)

    def pair_dir(self, kind, i, j):
        # separator, so that (1, 12) and (11, 2) do not collide
        return os.path.join(self.output, kind, f'{i+1}_{j+1}')

    def pair_affine(self, i, j):
        return os.path.join(self.pair_dir('aff', i, j), f'{self.affine}.lta')

    def pair_svf(self, i, j):
        return os.path.join(self.pair_dir('svf', i, j), 'svf.nii.gz')

    def pair_hessian(self, i, j):
        return os.path.join(self.pair_dir('svf', i, j), 'hessian.nii.gz')

    def opt_affine(self, i, inverse=False):
        suffix = '_inv' if inverse else ''
        return os.path.join(self.output, 'opt', 'aff',
                            f'{self.affine}_optim_{i+1}{suffix}.lta')

    def opt_svf(self, i, mode='weighted'):
        return os.path.join(self.output, 'opt', 'svf', mode,
                            f'svf_optim_{mode}_{i+1}.nii.gz')

    @property
    def meanspace(self):
        return os.path.join(self.output, 'opt', 'meanspace.nii.gz')

    def _opt_image(self, i, suffix):
        return os.path.join(self.output, 'opt',
                            f'{self.bases[i]}.{suffix}{self.exts[i]}')

    def aligned(self, i):
        return self._opt_image(i, 'aligned')

    def aligned_resliced(self, i):
        return self._opt_image(i, 'aligned.resliced')

    def normalised(self, i):
        return self._opt_image(i, 'norm')
