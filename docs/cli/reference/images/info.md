---
icon: octicons/terminal-24
---

# info

Print the shape, orientation and header metadata of one or more volumes, and
optionally their intensity statistics. Use it to check what a file contains
before you feed it to another command.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch info FILE... [-m FIELD...] [-s [BOOL]]
    ```

## Description

`info` opens each input file, reads its header and prints one `name : value`
line per field, with the names padded so that the colons line up. When you
give several files, their reports are separated by a line of dashes. The
command only reads the files; it never modifies them.

Which fields are printed depends on `-m`:

- **Without `-m`**, you get a standard report. It starts with `filename`,
  `shape`, `layout` and `center`, followed by every metadata field that the
  file actually defines. Fields that the file format does not define (for
  example the echo time of a NIfTI file) are left out.
- **With `-m`**, only the fields you name are printed, in the order you list
  them, and a requested field that the file does not define is printed as
  `None`. The file name, shape, layout and centre are not part of the report
  in this mode, and asking for them prints `None`. The data type, `dtype`, is
  always printed after the fields you asked for, whether you named it or not.

The four leading fields of the standard report are computed rather than read
from the header:

`shape`
:   The size of the array along each dimension, including the extra
    dimensions of 4D volumes.

`layout`
:   The order and direction of the stored axes, such as `RAS` or `LAS`. It is
    derived from the orientation matrix. See
    [Layouts](../../conventions.md#layouts).

`center`
:   The position, in millimetres in world (RAS) space, of the voxel at index
    `(shape - 1) / 2` along each spatial axis, that is, the centre of the
    field of view. It is obtained by applying the orientation matrix to that
    (possibly fractional) index.

The metadata fields that can be requested with `-m` are listed below. What a
file provides depends on its format: for instance, `slope` and `inter` exist
in NIfTI headers, and `te`, `ti` and `fa` in MGH headers.

| Field             | Content                                                             |
| ----------------- | ------------------------------------------------------------------- |
| `format`          | Name of the file format (`Nifti1`, `MGH`, ...).                     |
| `affine`          | Orientation matrix (voxel to world, in millimetres).                |
| `dtype`           | Data type stored on disk.                                           |
| `voxel_size`      | Voxel size along the spatial axes.                                  |
| `voxel_size_unit` | Unit of the voxel size.                                             |
| `slope`, `inter`  | Linear intensity scaling applied to the stored values (NIfTI).      |
| `time_step`, `tr` | Time between volumes of a 4D series.                                |
| `time_offset`     | Constant time shift applied to a series (NIfTI).                    |
| `te`, `ti`, `fa`  | Echo time, inversion time and flip angle (MGH, or parsed from the NIfTI description field). |

The generic metadata also defines `intent` and `slice_order`, but NITorch does
not currently read them from NIfTI or MGH headers, so they are always printed
as `None`. Some fields come with a companion unit field, such as
`time_step_unit`, which is printed next to them.

### Intensity statistics

With `-s`, `info` also loads the whole volume and prints the minimum, maximum
and mean intensity. The values are the *scaled* intensities, that is, after
the `slope` and `inter` of the header have been applied. For a volume with
more than three dimensions, each statistic is computed separately for every
position along the extra dimensions and printed as a list, so that a 4D
time series gives one value per volume. Computing statistics reads all the
voxels, which is slower than reading the header alone; the statistics are
shown in the standard report only, not when you use `-m`.

`-s` takes an optional boolean, so `-s`, `-s 1`, `-s true` and `-s yes` turn
statistics on, and `-s 0`, `-s false` and `-s no` turn them off. Because the
option swallows the next argument when that is not a tag, put it after the file
names. See
[Flags, optional values and booleans](../../conventions.md#flags-optional-values-and-booleans).

## Arguments

| Name   | Description                                              |
| ------ | -------------------------------------------------------- |
| `FILE` | One or more volumes. Any format NITorch reads, see [File formats](../../conventions.md#file-formats). |

## Options

| Option           | Value        | Description                                                              | Default |
| ---------------- | ------------ | ------------------------------------------------------------------------ | ------- |
| `-m`, `--meta`   | `FIELD...`   | Print only these metadata fields, instead of the standard report.        | standard report |
| `-s`, `--stat`   | `[BOOL]`     | Compute and print the minimum, maximum and mean intensity.               | off     |
| `-h`, `--help`   |              | Print the built-in help and exit.                                        |         |

## Examples

Print the standard report for a volume:

```shell
nitorch info T1w.nii.gz
```

```text
filename        : T1w.nii.gz
shape           : (20, 24, 16)
layout          : RAS
center          : (9.0, 3.0, -15.0) mm (RAS)
format          : Nifti1
affine          : [[  2.   0.   0. -10.]
                   [  0.   2.   0. -20.]
                   [  0.   0.   2. -30.]
                   [  0.   0.   0.   1.]]
dtype           : int16
voxel_size      : (np.float32(2.0), np.float32(2.0), np.float32(2.0))
voxel_size_unit : unknown
slope           : 1.0
inter           : 0.0
time_offset     : 0.0
```

Add intensity statistics (the file name comes before the option):

```shell
nitorch info T1w.nii.gz -s
```

Print only the orientation matrix and the voxel size:

```shell
nitorch info T1w.nii.gz -m affine voxel_size
```

Compare the shapes and orientations of several images; the reports are
separated by a line of dashes:

```shell
nitorch info sub-01_T1w.nii.gz sub-01_FLAIR.nii.gz
```

Check the repetition time of a 4D series:

```shell
nitorch info bold.nii.gz -m tr
```

## See also

- [`convert`](convert.md) to change the format or data type of a volume.
- [`crop`](crop.md), [`pad`](pad.md) to change the field of view.
- [Layouts](../../conventions.md#layouts) for the meaning of `RAS`, `LAS`, and so on.
