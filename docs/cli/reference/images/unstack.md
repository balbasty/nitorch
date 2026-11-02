---
icon: octicons/terminal-24
---

# unstack

Split a volume into one file per slice (or per volume, or per channel) along
one dimension. Use it to turn a 3D image into a series of 2D slices, or a 4D
series into separate 3D volumes.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch unstack FILE... [-d DIM] [-o FILE...] [--transform [FILE...]]
    ```

## Description

`unstack` writes every position along one dimension of the input as a separate
file. A volume with 16 slices along that dimension gives 16 files. The pieces
keep the number of dimensions of the input, with a length of 1 along the
dimension that was split: unstacking a volume of shape `(20, 24, 16)` along its
last axis gives 16 volumes of shape `(20, 24, 1)`. The data type, the format
and the header fields of the input are copied to each piece, and the values are
not altered.

When the split dimension is a spatial axis, the orientation matrix of each
piece is updated so that the slice keeps its place in world space: its
translation is moved forward by `k` voxels along the axis for the slice of index
`k` (counting from 0), and the directions and voxel size are unchanged. For a
2 mm volume whose last axis starts at `z = -30` mm, the 16th slice (index 15)
has `z = -30 + 15 * 2 = 0` mm.

`unstack` is the special case of [`chunk`](chunk.md) with `-n 1`. Both write
the same files, except for the limitations of the dimension option explained
below.

### The dimension

`-d` gives the dimension to split, counting from 0: `0`, `1` and `2` are the
spatial axes, and `3` is the first extra dimension of a 4D volume. A negative
value counts from the end of the array, and the default, `-1`, is the last
dimension: the third axis of a 3D volume, or the extra dimension of a 4D
volume.

!!! warning "The option `-d` does not work, and 4D volumes get a wrong orientation matrix"
    In the current version, `-d` with any value stops with the error
    `'>' not supported between instances of 'str' and 'int'`, because the value is
    never converted to a number. Only the default dimension, the last one, can
    be used. To split another dimension, use [`chunk`](chunk.md) with `-n 1`,
    for example `nitorch chunk T1w.nii.gz -n 1 -d 0`.

    Moreover, the default dimension is only handled correctly for 3D volumes.
    When you unstack a 4D volume, the data of each piece is right, but the
    orientation matrix is shifted along the third spatial axis by the index of
    the piece, as if the third axis had been split. Check the result with
    [`info`](info.md) before you use it, and restore the orientation matrix if
    it matters (for example with [`orient`](../spatial/orient.md)).

### Output names

The output name is a template. In addition to `{dir}`, `{sep}`, `{base}` and
`{ext}` (see [Output filenames](../../conventions.md#output-filenames)), it can
contain `{i}`, the number of the piece, **counting from 1**, without padding
zeros. The default, `{dir}{sep}{base}.{i}{ext}`, turns `sub-01_T1w.nii.gz`
into `sub-01_T1w.1.nii.gz`, `sub-01_T1w.2.nii.gz`, and so on.

Give one template per input file. A template without `{i}` writes every piece
to the same file, and only the last one remains. Because `{i}` is not zero
padded, `10` sorts before `2` in an alphabetical listing.

### Transformation files

With `--transform`, `unstack` also writes one `.lta` file per piece (see
[Spatial transformations](../../conventions.md#spatial-transformations)). Each
file holds the identity matrix, since unstacking does not move anything in world
space, and no volume geometry. The default name is `{dir}{sep}{base}_2_{i}.lta`,
which you get by giving `--transform` without a value; if you give a name, it
is a template that can contain `{dir}`, `{sep}`, `{base}`, `{ext}` and `{i}`.

!!! note "The short option `-t` does not work"
    The built-in help lists `-t` for `--transform`, but the command only
    recognises the long form: `-t` stops with *Unknown tag -t*. Write
    `--transform`.

## Arguments

| Name   | Description                                          |
| ------ | ---------------------------------------------------- |
| `FILE` | One or more input volumes. Any format NITorch reads, see [File formats](../../conventions.md#file-formats). |

## Options

| Option                         | Value        | Description                                                                  | Default                        |
| ------------------------------ | ------------ | ---------------------------------------------------------------------------- | ------------------------------ |
| `-d`, `--dim`, `--dimension`   | `DIM`        | Dimension to split, counting from 0. A negative value counts from the end. Currently fails with any value, see [above](#the-dimension). | `-1` (last dimension) |
| `-o`, `--output`               | `FILE...`    | Output name template, one per input. Placeholders: `{dir}`, `{sep}`, `{base}`, `{ext}`, `{i}`. | `{dir}{sep}{base}.{i}{ext}` |
| `--transform`                  | `[FILE...]`  | Write one `.lta` file per piece, one template per input. [See above](#transformation-files). | none; `{dir}{sep}{base}_2_{i}.lta` if given with no value |
| `-h`, `--help`                 |              | Print the built-in help and exit.                                            |                                |

## Examples

Split a 3D volume into axial slices. The results are written to
`T1w.1.nii.gz`, `T1w.2.nii.gz`, and so on, one per index of the last axis:

```shell
nitorch unstack T1w.nii.gz
```

Choose the names and the directory of the slices, which must already exist:

```shell
nitorch unstack sub-01_T1w.nii.gz -o 'slices{sep}{base}_z{i}{ext}'
```

Unstack several volumes in one command, writing one set of files per input:

```shell
nitorch unstack sub-01_T1w.nii.gz sub-02_T1w.nii.gz
```

Also write the transformation file of each slice:

```shell
nitorch unstack T1w.nii.gz -o 'T1w_z{i}.nii.gz' --transform 'T1w_z{i}.lta'
```

Split along the first axis, using `chunk` because `unstack` cannot do it:

```shell
nitorch chunk T1w.nii.gz -n 1 -d 0
```

## See also

- [`chunk`](chunk.md) splits a volume into pieces thicker than one slice, along
  any dimension.
- [`extract_patches`](extract_patches.md) cuts a volume into a grid of patches.
