---
icon: octicons/terminal-24
---

# simplereg

A wrapper around [`register`](register.md) with a simplified, classical
interface (`-m MOVING -f FIXED`). It is meant as the entry point for people
who want a registration without learning the nested syntax of `register`.

!!! note "Affine only"
    Despite its original description as an *affine plus nonlinear* tool,
    `simplereg` performs an **affine-only** (rigid, or full affine with `-a`)
    registration of a single pair of images, and accepts exactly the options
    of [`affreg`](affreg.md). For anything involving a nonlinear
    transformation use [`register`](register.md).

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch simplereg -m FILE -f FILE [options]
    ```

    The options are those of [`affreg`](affreg.md#options).

## Description

The parser and the command-line handler of `simplereg` are identical to
those of [`affreg`](affreg.md), up to the name of the command. As a
consequence:

- Only one pair of images (one `-m`, one `-f`) can be registered.
- The transformation is rigid, or a full affine with `-a`. There is no
  nonlinear transformation.

Everything else, including the default loss (normalised mutual information),
the optimiser, the pyramid and the files that are written, is therefore the
same as for [`affreg`](affreg.md), and described there. Under the hood,
the command runs `register` with an `@affine` group only.

## Options

See the [options of `affreg`](affreg.md#options). They are
`-m`, `-f`, `-i`, `-o`, `-x`, `-r`, `-l` (with `nmi`, `mse`, `mad`, `tuk`,
`cc`, `lcc`, `gmm`, `lgmm`), `-k`, `-s`, `-a`, `-p`, `-z`, `-n`, `-t`,
`--no-pyramid`, `--line-search`, `-v`, `--cpu`, `--gpu` and `-h`.

## Examples

Rigid registration of a FLAIR scan to a T1-weighted scan:

```shell
nitorch simplereg -m FLAIR.nii.gz -f T1w.nii.gz -r FLAIR_in_T1w.nii.gz
```

The equivalent, with `affreg`:

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

- [`affreg`](affreg.md) — the equivalent command.
- [`register`](register.md) — full control, including nonlinear
  registration.
