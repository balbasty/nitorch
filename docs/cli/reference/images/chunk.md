---
icon: octicons/terminal-24
---

# chunk

Split a volume into consecutive chunks of a given thickness along one
dimension, and save each chunk as its own file. Use it to cut a large volume
into slabs that are easier to process, or to feed a model that expects
slices of a fixed depth.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch chunk FILE... [-n SIZE] [-d DIM] [-o FILE...] [-t [FILE...]]
    ```

## Description

`chunk` divides the array of each input volume along dimension `DIM` into
pieces of `SIZE` voxels, starting at the first one. The last chunk holds the
remainder when the length of that dimension is not a multiple of `SIZE`, so a
volume of 16 slices cut with `-n 7` gives chunks of 7, 7 and 2 slices. A
volume of length `L` along the dimension produces `ceil(L / SIZE)` chunks.

Every chunk is written as a volume of the same number of dimensions as the
input, with only the length along `DIM` reduced. The data type, the format and
the header fields of the input are copied to each chunk, and the values are not
altered. The orientation matrix of each chunk is updated with the same rule
as in [`crop`](crop.md), described next.

### The orientation matrix of the chunks

When `DIM` is one of the spatial axes (0, 1 or 2), each chunk is a block of
the original volume that has kept its place in world space. The orientation
matrix of the chunk number `i` has the directions and voxel size of the input,
and its translation is moved forward by `(i - 1) * SIZE` voxels along the
split axis. Chunks of a volume therefore overlay the original, and one another,
without gaps. For a 2 mm volume whose third axis starts at `z = -30` mm, the
second chunk of `-n 8 -d 2` starts at `z = -30 + 8 * 2 = -14` mm.
When `DIM` is an extra (non-spatial) dimension, such as the time or echo axis
of a 4D volume, every chunk keeps the orientation matrix of the input.

`DIM` counts the dimensions of the array from 0, so `0`, `1` and `2` are the
three spatial axes and `3` would be the first extra dimension of a 4D volume.
A negative `DIM` counts from the end, and `-1`, the default, is the last
dimension. A negative `DIM` is converted to its positive equivalent before
the orientation matrix is computed: for a 4D volume, `-1` is the extra
dimension and `-2` is the third spatial axis.

### Output names

The output name is a template. In addition to `{dir}`, `{sep}`, `{base}` and
`{ext}` (see [Output filenames](../../conventions.md#output-filenames)), it can
contain `{i}`, the number of the chunk, **counting from 1**, without padding
zeros. The default, `{dir}{sep}{base}.{i}{ext}`, turns `sub-01_T1w.nii.gz` into
`sub-01_T1w.1.nii.gz`, `sub-01_T1w.2.nii.gz`, and so on.

Give one template per input file. A template without `{i}` writes every chunk
to the same file, and only the last chunk remains. When the output contains
many chunks, remember that `{i}` is not zero padded, so `10` sorts before `2`
in an alphabetical listing.

### Transformation files

With `--transform`, `chunk` also writes one `.lta` file per chunk (see
[Spatial transformations](../../conventions.md#spatial-transformations)). Each
file holds the identity matrix, since chunking does not move anything in world
space, and no volume geometry. The default name is `{dir}{sep}{base}_2_{i}.lta`,
which you get by giving `--transform` without a value; if you give a name, it
is a template that can contain `{dir}`, `{sep}`, `{base}`, `{ext}` and `{i}`.

## Arguments

| Name   | Description                                          |
| ------ | ---------------------------------------------------- |
| `FILE` | One or more input volumes. Any format NITorch reads, see [File formats](../../conventions.md#file-formats). |

## Options

| Option                            | Value        | Description                                                                 | Default                        |
| --------------------------------- | ------------ | --------------------------------------------------------------------------- | ------------------------------ |
| `-n`, `--chunk`                   | `SIZE`       | Number of voxels along `DIM` in each chunk. The last one may be smaller.    | `1`                            |
| `-d`, `--dim`, `--dimension`      | `DIM`        | Dimension to split, counting from 0. A negative value counts from the end.  | `-1` (last dimension)          |
| `-o`, `--output`                  | `FILE...`    | Output name template, one per input. Placeholders: `{dir}`, `{sep}`, `{base}`, `{ext}`, `{i}`. | `{dir}{sep}{base}.{i}{ext}` |
| `-t`, `--transform`               | `[FILE...]`  | Write one `.lta` file per chunk, one template per input. [See above](#transformation-files). | none; `{dir}{sep}{base}_2_{i}.lta` if given with no value |
| `-h`, `--help`                    |              | Print the built-in help and exit.                                           |                                |

## Examples

Cut a volume into slabs of 32 slices along its third axis. The results are
written to `T1w.1.nii.gz`, `T1w.2.nii.gz`, and so on:

```shell
nitorch chunk T1w.nii.gz -n 32 -d 2
```

Cut it along the first axis instead, into blocks of 64 voxels, and choose the
names, which here include the name of the input:

```shell
nitorch chunk sub-01_T1w.nii.gz -n 64 -d 0 -o 'slabs{sep}{base}_slab{i}{ext}'
```

Split several volumes with the same settings, writing one set of chunks per
input:

```shell
nitorch chunk sub-01_T1w.nii.gz sub-02_T1w.nii.gz -n 50 -d 2
```

Also save the transformation of each chunk, with names chosen by you:

```shell
nitorch chunk T1w.nii.gz -n 32 -d 2 -o 'T1w_z{i}.nii.gz' --transform 'T1w_z{i}.lta'
```

## See also

- [`unstack`](unstack.md) splits a volume into single slices; it is the
  special case `-n 1`.
- [`extract_patches`](extract_patches.md) cuts a volume into a grid of patches
  along all axes at once.
- [`crop`](crop.md) extracts a single region of a volume.
