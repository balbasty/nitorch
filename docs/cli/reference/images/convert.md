---
icon: octicons/terminal-24
---

# convert

Convert volumes from one file format to another, change their data type, or
overwrite some of their header fields. Use it to turn a NIfTI file into an MGZ
file (or the other way round), to store a float image as 8-bit integers, or to
read a TIFF stack into a neuroimaging format.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch convert FILE... [-dt DTYPE] [-c CASTING] [-m KEY=VAL...]
                            [-f FORMAT] [-o FILE...]
    ```

## Description

For each input volume, `convert` reads the voxel values as they are stored on
disk, and writes them to a new file with the requested format and data type.
The orientation matrix and all the other header fields of the input (voxel
size, units, intensity scaling, repetition time, and so on) are copied to the
output, except the ones you override with `--metadata`. If you ask for
nothing, the output is a faithful copy of the input.

The output format is chosen from the extension of the output file name, and
the data type from `-dt`. The remaining options control how the values are
converted, which header fields are changed, and where the result is written.

!!! warning "By default the input file is overwritten"
    The default output name is `{dir}{sep}{base}{ext}`, which is the name of
    the input. Unless you give `-o`, or `-f` with a format that differs from
    the input's, `convert` replaces the input file with the converted one. This
    differs from most other commands, which write a new file next to the
    input. Always pass `-o` when you want to keep the original.

### Output names and formats

The output name is a template, with the usual placeholders `{dir}`, `{sep}`,
`{base}` and `{ext}` (see [Output filenames](../../conventions.md#output-filenames)).
`{ext}` is the extension of the input file, unless you set `-f`, in which case
it is the extension that corresponds to the format you named. `-f` accepts
`nii`, `nii.gz`, `mgh`, `mgz` (with or without a leading dot), and `nifti` as a
synonym for `nii`. The following two commands therefore write `T1w.mgz` next to
`T1w.nii.gz`:

```shell
nitorch convert T1w.nii.gz -f mgz
nitorch convert T1w.nii.gz -o '{dir}{sep}{base}.mgz'
```

When you give an explicit `-o`, the format of the output is decided by the
extension of the name you wrote, so `-f` is not needed to select it. This
extension also decides whether the data type is adapted to the limited set of
types that MGH supports (see [Data types](#data-types)).

NIfTI and MGH are the formats that can be written. TIFF files can be read, but
not written. Whenever the output is a `.nii`, `.nii.gz`, `.mgh` or `.mgz` file,
volumes with fewer than three dimensions are given trailing singleton
dimensions so that they are stored as 3D volumes.

### Data types

`-dt` sets the type in which the values are stored on disk. It takes the name
of a type: `uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `uint64`,
`int64`, `float16`, `float32`, `float64`, among others. Without it, the data
type of the input is kept.

NIfTI can store most of these types (the exact set depends on `nibabel`). MGH only supports `uint8`, `int16`,
`int32` and `float32` (and `uint16` with recent versions of `nibabel`). When
the output file name ends in `.mgh` or `.mgz`, `convert` replaces the type you
asked for with the first of these types that can hold all its values without
loss: `int8` becomes `int16`, for example. A type that no MGH type can hold,
such as `float64`, is not adapted, and the command stops with an error from
`nibabel`.

### Casting

`-c` controls what happens to the values when they are converted to the new
data type. The default, `unsafe`, applies the plain conversion that a numerical
library would, whereas the two `rescale` modes first scale the values to fit
the range of the output type.

`unsafe`
:   Convert without checking. Floating-point values are truncated towards zero
    (`2.7` becomes `2` and `-1.6` becomes `-1`), and integers that do not fit
    wrap around: a value of `-5` stored as `uint8` becomes `251`, and `1000`
    becomes `232`. This is the right choice when the values are known to fit,
    for example when going from `uint8` to `float32` or from `int16` to
    `int32`.

`rescale`
:   Map the minimum of the data to the smallest value of the output type, and
    the maximum of the data to its largest value. This only has an effect when
    the output type is an integer type. It is meant for storing a float image
    as integers without clipping or wrapping.

`rescale_zero`
:   Like `rescale`, but the scale factor is chosen so that zero stays zero.
    With data that is positive, the maximum is mapped to the largest value of
    the output type. Do not use it for data that contains negative values when
    the output type is unsigned: all values then collapse to zero.

NIfTI headers can carry a linear intensity transform, a `slope` and an
intercept `inter`. When you use a `rescale` mode to write a NIfTI file, the
command stores the rescaled integers and adjusts the slope and the intercept
so that readers that honour them (such as [`info -s`](info.md#intensity-statistics))
recover the original intensities, up to the quantisation of the integer type. MGH headers have no such transform, so in an
MGH file the rescaled values really are different from the original ones.

The other casting rules of NumPy, `no`, `equiv`, `safe` and `same_kind`, are
passed through to the conversion. They refuse conversions that could lose
information; when they do, the command stops with a Python error message
instead of writing the file.

### Overriding header fields

`-m` sets header fields in the output. Each value is a `KEY=VAL` pair, written
without whitespace. The keys are the names that [`info`](info.md#description)
prints. Values are read as text and converted by the header writer; matrices
and tuples are written as Python literals, and need quoting in the shell:

```shell
nitorch convert T1w.nii.gz -o T1w_scaled.nii.gz \
    -m 'affine=[[3,0,0,0],[0,3,0,0],[0,0,3,0],[0,0,0,1]]'
```

Fields that you do not set are copied from the input. `voxel_size` rescales
the orientation matrix of the output (the input's, or the one given with
`affine`) so that its voxels have the requested size, e.g.
`-m 'voxel_size=[1,1,1]'`. `tr` (or `time_step`) sets the time step of a 4D
series, in seconds by default, or in milliseconds with an `ms` suffix
(`tr=2500ms`); it is only stored when the output has a fourth dimension (MGH
files always store it, in milliseconds). `slope` and `inter` set the intensity
transform of a NIfTI file. The key and the value are split at the first `=`,
so the value itself cannot contain one.

## Arguments

| Name   | Description                                                                  |
| ------ | ---------------------------------------------------------------------------- |
| `FILE` | One or more input volumes: NIfTI, MGH or TIFF. See [File formats](../../conventions.md#file-formats). |

## Options

| Option                | Value             | Description                                                                                 | Default                 |
| --------------------- | ----------------- | ------------------------------------------------------------------------------------------- | ----------------------- |
| `-dt`, `--dtype`      | `DTYPE`           | Data type of the output, such as `uint8` or `float32`. [See below](#data-types).           | that of the input       |
| `-c`, `--casting`     | `CASTING`         | How values are converted: `unsafe`, `rescale` or `rescale_zero`. [See below](#casting).    | `unsafe`                |
| `-m`, `--metadata`    | `KEY=VAL...`      | Header fields to set in the output. [See below](#overriding-header-fields).                |                         |
| `-f`, `--format`      | `FORMAT`          | Output format (`nii`, `nii.gz`, `mgh`, `mgz`). Sets the `{ext}` placeholder.                | extension of the input  |
| `-o`, `--output`      | `FILE...`         | Output name template, one per input. Placeholders: `{dir}`, `{sep}`, `{base}`, `{ext}`.     | `{dir}{sep}{base}{ext}` |
| `-h`, `--help`        |                   | Print the built-in help and exit.                                                           |                         |

If you give fewer output names than input files, the last name is reused for
the remaining inputs, so give one name per input, or a template with
`{base}`. `--meta` is accepted as a synonym of `--metadata`.

## Examples

Convert a NIfTI volume to a compressed MGH file next to it:

```shell
nitorch convert T1w.nii.gz -f mgz
```

Convert several volumes to uncompressed NIfTI in another directory, which must
already exist:

```shell
nitorch convert sub-01_T1w.mgz sub-01_FLAIR.mgz -o 'nifti{sep}{base}.nii'
```

Store a floating-point image as 16-bit integers, keeping the dynamic range in
the file header, and leave the original untouched:

```shell
nitorch convert sub-01_T1w.nii.gz -dt int16 -c rescale -o sub-01_T1w_int16.nii.gz
```

Write an 8-bit MGZ file from a NIfTI image whose values are all positive:

```shell
nitorch convert sub-01_T1w.nii.gz -dt uint8 -c rescale_zero -f mgz -o sub-01_T1w_u8.mgz
```

Convert a volume to single-precision floats, in place (the original file is
replaced by the converted one):

```shell
nitorch convert T1w.nii.gz -dt float32
```

## See also

- [`info`](info.md) to look at the data type, format and header of a file.
- [`reorient`](../spatial/reorient.md) and [`orient`](../spatial/orient.md) to
  change the orientation of a volume.
