---
icon: octicons/terminal-24
---

# reorient

Permute and flip the voxel array of a volume so that it is stored in a chosen layout, such as `RAS`, and update the orientation matrix so that every voxel stays at the same place in world space. Use it to bring images from different sources to a common storage order.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch reorient FILE... [-l LAYOUT] [-o FILE...] [--transform [FILE...]]
    ```

!!! note "reorient is not orient"
    `reorient` assumes that the orientation matrix of the input is **correct**. It changes the data array and the matrix together, so that the image is unchanged in world space. [`orient`](orient.md) does the opposite: it assumes the matrix is wrong and overwrites it, leaving the data array as it is.

## Description

A volume is stored as an array of voxels, and its orientation matrix tells where each voxel lies in world space. The same anatomy can be stored in many different ways: with the first array axis running from left to right (`R`) or from right to left (`L`), with the second axis running front to back or back to front, and so on. The *layout* of a volume is the name of this storage order, a permutation of three letters, where each letter says towards which direction the voxel index increases: `R` (left to Right) or `L` (right to Left), `A` (posterior to Anterior) or `P` (anterior to Posterior), `S` (inferior to Superior) or `I` (superior to Inferior). A volume in `RAS` layout has axes that run towards the right, the front and the top, so that the rotation part of its orientation matrix is (close to) a diagonal matrix with positive entries.

`reorient` reads the orientation matrix, works out the layout it describes, and then transposes and flips the data array to reach the requested layout. The orientation matrix is rewritten with its columns permuted and sign-flipped, and its translation adjusted, so that the voxel-to-world mapping of every voxel is preserved. If you view the input and the output in a viewer that honours the header, they overlap exactly; only the order of the numbers in the file differs.

Voxel values are not interpolated, so nothing is lost: the output has the same data type and the same values as the input. A permutation of axes can change the shape (a volume of $256 \times 128 \times 64$ voxels in `RAS` becomes $128 \times 64 \times 256$ in `ASR`, for instance). Volumes with more than three dimensions, such as time series or multi-channel images, are handled too: only the first three (spatial) dimensions are permuted and flipped; the others stay in place.

The header is otherwise copied from the input file, so other metadata (voxel units, description and so on) is preserved as far as the output format allows.

### Why use it?

Many tools assume, or run faster with, a particular storage order, and some ignore the orientation matrix altogether. Reorienting a collection of images to the same layout makes them directly comparable voxel by voxel when they cover the same field of view. It is also the safe way to normalise the layout: unlike [`orient`](orient.md), it never moves anything in world space.

## Arguments

| Name   | Description                                                                |
| ------ | -------------------------------------------------------------------------- |
| `FILE` | One or more input volumes. Any format NITorch reads and writes (see [file formats](../../conventions.md#file-formats)). |

## Options

| Option               | Value       | Description                                                                                                                                             | Default                            |
| -------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `-l`, `--layout`     | `LAYOUT`    | Target layout: a permutation of three letters from `R`/`L`, `A`/`P`, `S`/`I`, such as `RAS`, `LAS`, `LPS` or `RSA`.                                      | `RAS`                              |
| `-o`, `--output`     | `FILE...`   | Output filenames, one per input. [Templates](../../conventions.md#output-filenames) are accepted; `{layout}` expands to the name of the layout.         | `{dir}{sep}{base}.{layout}{ext}`   |
| `--transform`        | `[FILE...]` | Intended to write the (identity) transformation between the input and output spaces. **Currently not usable**: see below.                              | none                               |

If fewer output names than inputs are given, the last one is repeated (and would be overwritten), so give a template when you process several files.

!!! warning "The transform option does not work at present"
    The parser registers the short flag for `--transform` under a mistyped name, so `-t` is rejected as an unknown tag. With the long form `--transform`, the image is written and then the command fails with an error message, because the transform filename is not handled correctly. Avoid the option for now. The reorientation itself is unaffected.

## Examples

Store a T1-weighted image in `RAS` layout. The output is `T1w.RAS.nii.gz`, next to the input:

```shell
nitorch reorient T1w.nii.gz
```

Store an image in `LAS` layout, in which the first axis runs towards the left, and choose the output name:

```shell
nitorch reorient sub-01_T1w.nii.gz -l LAS -o sub-01_T1w_las.nii.gz
```

Bring a set of images to the same layout, writing each into a `ras` folder that already exists:

```shell
nitorch reorient sub-01_T1w.nii.gz sub-02_T1w.nii.gz -l RAS -o 'ras/{base}{ext}'
```

The template has no `{dir}` field, so each output goes to `ras/` relative to the current directory, under the name of its input. A single template is reused for every input.

## See also

- [`orient`](orient.md) overwrites the orientation matrix and leaves the data alone.
- [`info`](../images/info.md) prints the orientation matrix and layout of a file.
- [`reslice`](reslice.md) resamples a volume onto an arbitrary grid.
