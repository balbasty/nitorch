---
icon: octicons/terminal-24
---

# simplereg

A wrapper around [`register`](register.md) with a simplified, classical
interface (`-m MOVING -f FIXED`). It is meant as the entry point for people
who want a registration without learning the nested syntax of `register`.

!!! warning "Status in this version of NITorch"
    `simplereg` is documented as an *affine plus nonlinear* tool, but the
    code is, at present, a copy of [`affreg`](affreg.md): it performs an
    **affine-only** registration and accepts exactly the options of
    `affreg`. In addition, the module is not imported by
    `nitorch.cli.registration`, so the command `nitorch simplereg` is
    **not available** in a regular installation (it answers `Unknown
    command`). For an affine registration use [`affreg`](affreg.md); for
    anything involving a nonlinear transformation use
    [`register`](register.md).

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch simplereg -m FILE -f FILE [options]
    ```

    The options are those of [`affreg`](affreg.md#options).

## Description

The built-in help of `simplereg` describes a command that registers several
pairs of images with an affine and a nonlinear transformation, with
options that can be repeated and that apply to the matching pair, and a set
of preprocessing options (`--fwhm`, `--bound`, `--order`, `--pad`,
`--rescale`, `--discretize`) and an `emmi` loss. None of this is
implemented: the parser and the command-line handler of `simplereg` are
identical to those of `affreg`, line for line, up to the name of the
command and the help text. As a consequence:

- Only one pair of images (one `-m`, one `-f`) can be registered.
- The transformation is rigid, or a full affine with `-a`. There is no
  nonlinear transformation.
- The options listed above are rejected as unknown by the parser, and `emmi`
  is not an accepted value of `-l`.

Everything else, including the default loss (normalised mutual information),
the optimiser, the pyramid and the files that are written, is therefore the
same as for [`affreg`](affreg.md), and described there. Under the hood,
the command runs `register` with an `@affine` group only.

### Running it anyway

If you want to run the command as it exists, you have to import its module
first, which registers it. From Python:

```shell
python -c "
import nitorch.cli.registration.simplereg
from nitorch.cli import cli
cli(['simplereg', '-m', 'FLAIR.nii.gz', '-f', 'T1w.nii.gz'])
"
```

This is equivalent to `nitorch affreg -m FLAIR.nii.gz -f T1w.nii.gz`, and
there is no reason to prefer it.

## Options

See the [options of `affreg`](affreg.md#options). They are
`-m`, `-f`, `-i`, `-o`, `-x`, `-r`, `-l` (with `nmi`, `mse`, `mad`, `tuk`,
`cc`, `lcc`, `gmm`, `lgmm`), `-k`, `-s`, `-a`, `-p`, `-z`, `-n`, `-t`,
`--line-search`, `-v`, `--cpu`, `--gpu` and `-h`.

## Examples

Rigid registration of a FLAIR scan to a T1-weighted scan (once the command is
available, as described above):

```shell
nitorch simplereg -m FLAIR.nii.gz -f T1w.nii.gz -r FLAIR_in_T1w.nii.gz
```

The equivalent, with the command that is always installed:

```shell
nitorch affreg -m FLAIR.nii.gz -f T1w.nii.gz -r FLAIR_in_T1w.nii.gz
```

A nonlinear registration needs `register`:

```shell
nitorch register -v \
    @loss mi @@fix T1w.nii.gz -o false @@mov FLAIR.nii.gz -o false -r \
    @affine rigid \
    @nonlin svf
```

## See also

- [`affreg`](affreg.md) — the working equivalent.
- [`register`](register.md) — full control, including nonlinear
  registration.
