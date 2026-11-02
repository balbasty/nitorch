---
icon: octicons/terminal-24
---

# extract_patches

Cut a volume into a regular grid of patches and save each patch as its own
file. Use it to prepare training data for a neural network, or to process a
large image piece by piece.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch extract_patches FILE [-n SIZE...] [-s STRIDE...] [-o FILE...]
                                 [-t [FILE...]]
    ```

## Description

`extract_patches` reads a single volume and slides a cubic (or box-shaped)
window across its three spatial dimensions. Every position of the window
produces one output volume. The window moves by `STRIDE` voxels at a time;
by default the stride equals the patch size, so the patches tile the image
without overlapping. A smaller stride gives overlapping patches, and a larger
one leaves gaps between them.

Patches are only taken where the window fits entirely inside the image. No
padding is applied, so voxels that remain after the last complete patch along
an axis are discarded. Along an axis of length $L$, there are
$\lfloor (L - \text{size}) / \text{stride} \rfloor + 1$ patches. If the patch
size exceeds the image size along any axis, the command fails.

Only the first three dimensions are cut. Any further dimension (for example
the channels of a 4D file) is kept whole in every patch.

Each patch is written with the header of the input file as a template and
with its own orientation matrix: the patch keeps the voxel size and
orientation of the original image, and its origin is moved to the patch
position, so that it still lines up with the original image in world space.
Data are written with the data type of the input file.

Patches are numbered by their position on the grid, counting from 1, with `i`
along the first voxel axis, `j` along the second and `k` along the third. The
file name template can use these numbers, as described in
[Output files](#output-files).

## Arguments

| Name   | Description                                                      |
| ------ | ---------------------------------------------------------------- |
| `FILE` | The input volume. Exactly one file; it must come before options. |

## Options

| Option             | Value          | Description                                                                                                                                                       | Default                                    |
| ------------------ | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `-n`, `--size`     | `SIZE...`      | Patch size, in voxels. One value is used for all three dimensions; two values are completed by repeating the last one.                                            | `64`                                       |
| `-s`, `--stride`   | `STRIDE...`    | Step between neighbouring patches, in voxels. Same completion rule as `--size`.                                                                                   | same as the patch size                     |
| `-o`, `--output`   | `FILE...`      | Output file names or [templates](#output-files). Give one template, or exactly one name per patch.                                                                | `{dir}{sep}{base}.{i}_{j}_{k}{ext}`        |
| `-t`, `--transform` | `[FILE...]`  | Also write, for every patch, a transformation file relating the patch to the original image. Without values, the default template is used. [See below](#transformations). | not written                                |

## Output files

The output names are Python format strings. Each placeholder is replaced for
every patch:

| Placeholder | Expands to                                                        |
| ----------- | ----------------------------------------------------------------- |
| `{dir}`     | Directory of the input file (`.` if it has none).                 |
| `{base}`    | File name of the input without its extension (`.nii.gz` counts as one extension). |
| `{ext}`     | Extension of the input, including the leading dot.                |
| `{sep}`     | The path separator of your operating system.                      |
| `{i}`, `{j}`, `{k}` | Position of the patch on the grid, starting at 1.         |

The patches are visited with `i` as the slowest index and `k` as the
fastest. If you pass a list of names to `-o`, the first name is given to patch
`(1, 1, 1)`, the second to `(1, 1, 2)`, and so on. A list shorter than the
number of patches is completed by repeating its last name, and a single
template is reused for every patch. So that patches do not overwrite one
another, a template must contain at least `{i}`, `{j}` and `{k}` (or you must
list every name).

For the general rules on templates see
[Output filenames](../../conventions.md#output-filenames).

## Transformations

With `--transform`, `extract_patches` writes one transformation file per
patch. Each file stores the identity transformation in world space, between
the original image (source) and the patch (target). Its purpose is to record
the two geometries in a form that other tools can read, for instance to map
voxel coordinates of the patch back into the original image. The format is
chosen from the file extension; the default template is
`{dir}{sep}{base}_to_{i}_{j}_{k}.lta`, which writes FreeSurfer `.lta` files.
The templates accept the same placeholders as the output names. As for the
images, a single template is reused for all patches, and a list is read in the
same order. See
[Spatial transformations](../../conventions.md#spatial-transformations).

## Examples

Split a volume into non-overlapping 64-voxel cubes (the defaults). A
`256 x 256 x 160` image yields `4 x 4 x 2` patches, named
`T1w.1_1_1.nii.gz` to `T1w.4_4_2.nii.gz`:

```shell
nitorch extract_patches T1w.nii.gz
```

Extract 32-voxel cubes that overlap by half:

```shell
nitorch extract_patches T1w.nii.gz -n 32 -s 16
```

Use a box-shaped patch, thinner along the third axis:

```shell
nitorch extract_patches T1w.nii.gz -n 96 96 32
```

Store the patches in a separate folder, using a custom template:

```shell
nitorch extract_patches T1w.nii.gz -n 64 -o patches/sub-01_{i}-{j}-{k}.nii.gz
```

Also save the transformations relating each patch to the original image:

```shell
nitorch extract_patches T1w.nii.gz -n 64 --transform
```

## See also

- [`chunk`](chunk.md) — split a volume into chunks along a single dimension.
- [`unstack`](unstack.md) — split a volume along one dimension into separate files.
- [`crop`](crop.md) — extract a single region.
