---
icon: octicons/book-24
---

# Common conventions

Every `nitorch` command is written by hand around the same small set of ideas:
how arguments are split into options and values, which file formats are read
and written, how output files are named, how spatial transformations are
stored, and how a command is told to use a GPU. This page describes those
ideas once, so that the command pages can stay short and link here.

## Running commands

`nitorch` is a single program that dispatches to a subcommand:

```shell
nitorch COMMAND [ARGUMENTS...]
```

Running `nitorch` with no argument (or with `-h` or `--help`) lists every
available command. There are three equivalent ways to get the help text of a
command:

```shell
nitorch help crop
nitorch crop -h
nitorch crop --help
```

The same program can be started through the Python interpreter, which is
useful when the `nitorch` script is not on your `PATH` (for instance in a
virtual environment that is not activated):

```shell
python -m nitorch crop -h
```

Both forms run the same code. Each command is a separate process and starts
by importing PyTorch, which takes a moment even for a small image.

A few things are worth knowing about how commands behave.

- **There is no `--version` flag.** Print the version of the installed
  package from Python instead:
  `python -c "import nitorch; print(nitorch.__version__)"`.
- **The built-in help is a summary.** It is terse, and for commands with many
  options it can lag behind the code. The [reference pages](reference/index.md)
  describe what the commands actually do.
- **Some commands have several levels of help.** `nitorch register -h 2` and
  `nitorch register -h 3` print progressively more detail than `-h`.
- **A start-up warning is normal without the compiled backend.** Unless the
  [compiled backend](../install/compiled.md) is installed, every command
  prints `nitorch uses its non-compiled backend (TS). Some algorithms may be
  slow.` on standard error. The warning is harmless. Set the environment
  variable `NI_CHECK_BACKEND=0` to silence it.
- **Exit status is not a reliable success signal.** An unknown command makes
  `nitorch` exit with status 1. However, when most commands fail to parse
  their arguments or hit an error while working, they print an `[ERROR]`
  message on standard error and still exit with status 0. In scripts, check
  that the output files exist rather than relying on the exit status alone.

!!! note "`register2`"
    `register2` appears in the list of commands. It is a hidden, backward
    compatible alias of [`register`](reference/registration/register.md), kept
    so that older scripts continue to work.

## How options are parsed

NITorch does not use `argparse`. Each command has its own parser, and two
parser families exist, but they follow the same rules, which are close to the
ones you know from other command-line tools, with a few differences that are
worth learning.

### Files first, then options

Most commands take a list of input files followed by options:

```shell
nitorch crop T1w.nii.gz T2w.nii.gz -s 64 64 64 -o T1w_small.nii.gz T2w_small.nii.gz
```

The input files (the *positional* arguments) must come **before** the first
option. After that point, every argument that does not start a new option is
taken as a value of the option before it. A file name placed after `-s 64 64
64` would be read as a fourth size, and fail to convert to a number.

### An option takes all the values that follow it

An option (a *tag*) is followed by as many values as it accepts, up to the
next tag. Lists are written as separate, space-separated arguments, never with
commas and never with `=`:

```shell
nitorch crop T1w.nii.gz -s 64 64 64      # three values for -s
nitorch crop T1w.nii.gz --size 64 64 64  # the long spelling of the same option
```

The following spellings are **not** understood:

- `--size=64`: there is no `--tag=value` form; the tag would be rejected as
  unknown.
- `-s64`, or several short flags glued together (`-sc`): a tag is matched as
  a whole word.
- Abbreviated long options (`--si` for `--size`): tags must be spelled out in
  full.

Short tags may have more than one letter (`-dt`, `-vx`, `-il`); these are
still single-dash tags and must be typed as shown on the command's page.
Where a command accepts both a short and a long spelling, they are
interchangeable.

### What counts as a tag, and what counts as a value

An argument is a tag when it starts with a dash and its second character is
**not** a digit. Everything else is a value. This has three consequences.

- **Negative numbers are values.** `-2.5`, `-10` and `-1e-3` can be written
  as is wherever a number is expected, for example `-c -12.4 6.4 3.2`.
- **`-.5` is not a number to the parser.** A dash followed by a dot starts a
  tag, and the command reports `Unknown tag -.5`. Write `-0.5` instead.
- **A file name that starts with a dash and a letter cannot be used directly.**
  Prefix it with a directory to hide the dash, as in `./-odd-name.nii.gz`.

A lone `-` is a value.

### Keywords among numbers

Some options take a list of numbers optionally followed by a keyword that says
how to interpret them. The keyword is recognised by its text, wherever it
appears among the values, and is usually written last:

```shell
nitorch crop T1w.nii.gz -s 5 5 2 ras -c 12.4 -6.4 3.2 ras
```

The keywords (`vox`, `ras`, `mm`, `%`, ...) are described in
[Coordinates and units](#coordinates-and-units).

### Flags, optional values and booleans

An option that takes no value is a *flag*: its presence turns the feature on.
Some flags also accept an optional value, either a boolean or a level. For
example, `nitorch info -s` computes statistics, and `nitorch info -s false`
leaves them off.

Write booleans as `true` or `false`, or as `1` or `0`; these spellings are
accepted wherever a command takes a boolean. Some commands also accept `yes`
and `no`, or the initials `t`, `y`, `n`, but not all of them do (`nitorch
info -s no` is an error), so prefer the first two spellings. Take care with
flags that accept an optional value: a flag swallows the next argument if that
argument is not a tag. If you put an input file after such a flag, the flag
tries to read it as its value. Give the flag an explicit value, or put it
after the files and before another tag, where there is nothing for it to
swallow.

### Groups

Commands that need structured input (a registration made of several loss
terms, a quantitative fit made of several contrasts) divide their arguments
into *groups*. Depending on the command, a group is opened by an option such
as `-c` (`estatics`, `greeq`) or by a word that starts with `@`, as in
`@loss`, `@affine` and `@nonlin` for [`register`](reference/registration/register.md).
Nested groups start with `@@`. Everything between the opening word and the
next group belongs to the group. Each command's page explains its own groups.

### Errors

When a command cannot make sense of its arguments it prints its full help
text, followed by a line of the form

```text
[ERROR] Unknown tag --size=64
```

on standard error. Errors raised while the command works (a missing file, a
value of the wrong type) are reported in the same `[ERROR] ...` form by most
commands, and as a Python traceback by the others. See [Running
commands](#running-commands) about the exit status.

## File formats

NITorch reads and writes volumes through its own abstraction layer
(`nitorch.io`), which delegates the parsing of each format to a specialised
library. The format of a file to **read** is found from its extension; if that
fails, NITorch tries the other readers in turn, so a file with a missing or
unusual extension is often still read correctly. The format of a file to
**write** is chosen from the extension of the output name. If the extension is
not recognised, the format of the input file (the `like` file) is used, and
failing that NIfTI-1.

### Volumes

| Format       | Extensions            | Read | Write | Needs        |
| ------------ | --------------------- | :--: | :---: | ------------ |
| NIfTI-1, -2  | `.nii`, `.nii.gz`     | yes  | yes   | `nibabel`    |
| MGH / MGZ    | `.mgh`, `.mgz`        | yes  | yes   | `nibabel`    |
| TIFF         | `.tif`, `.tiff`       | yes  | see note | `tifffile` |

The `nibabel` and `tifffile` packages are optional dependencies; see
[Installation](../install/index.md#optional-dependencies). Without `nibabel`,
NITorch cannot read neuroimaging volumes at all, so for command-line use you
should always install at least the `io` extra.

NIfTI and MGH (with optional gzip compression, as in `.nii.gz`) are the
formats NITorch is developed and tested with. `nibabel` recognises other
formats as well (Analyze, MINC, PAR/REC, and so on); NITorch may open some of
them, but they are not supported. A TIFF writer exists in the library, but the
[`convert`](reference/images/convert.md) command documents TIFF as an input
format only.

When a command writes a modified copy of an input volume (cropping, smoothing,
resampling, and so on), it copies the metadata of the input, such as the data
type, intensity scaling and units, and replaces the fields that the operation
changes, namely the shape and the orientation matrix. Use
[`info`](reference/images/info.md) to inspect the result and
[`convert`](reference/images/convert.md) to change format or data type.

Volumes are three-dimensional. Further dimensions (echoes, contrasts, the
three components of a displacement field, ...) are treated as channels by the
commands that support them, and are stored after the three spatial ones.

### Transformations

| Format                       | Extensions | Read | Write | Notes                                |
| ---------------------------- | ---------- | :--: | :---: | ------------------------------------ |
| FreeSurfer linear transform  | `.lta`     | yes  | yes   | The default for affine matrices.     |
| FreeSurfer `register.dat`    | `.dat`     | yes  | yes   | tkregister-style registration files. |

Dense fields (displacement and velocity fields) are not stored in a
dedicated format: they are ordinary volumes, in any of the volume formats
above. See [Spatial transformations](#spatial-transformations).

### Streamlines

[`reslice`](reference/spatial/reslice.md) can also apply transformations to
streamlines stored in TrackVis `.trk` files.

## Output filenames

Commands that write a modified copy of an input build the output name from
the input name and write it **next to the input**, usually with a suffix such
as `.crop` or `.pad`, so that the input is left untouched.

!!! warning "`convert` is the exception"
    The default output template of [`convert`](reference/images/convert.md)
    is `{dir}{sep}{base}{ext}`, which is the input name itself. Unless you
    change the format (`-f`) or give an output name (`-o`), `convert`
    overwrites its input.

Options such as `-o`, `--output` or `-t`, `--transform` let you
choose another name, and the name you give is a *template*: it may contain
placeholders in braces, which the command replaces for each input file.

Four placeholders are available in nearly every command:

| Placeholder | Replaced by                                                                   |
| ----------- | ----------------------------------------------------------------------------- |
| `{dir}`     | The directory of the input file (`.` when the input name has no directory).   |
| `{sep}`     | The path separator of your operating system (`/` on Linux and macOS).         |
| `{base}`    | The file name without its extension.                                          |
| `{ext}`     | The extension, including the leading dot and any compression suffix.          |

A compression suffix is treated as part of the extension, so that the input
`data/sub-01_T1w.nii.gz` gives `{dir}` = `data`, `{base}` = `sub-01_T1w` and
`{ext}` = `.nii.gz`. The default template of [`crop`](reference/images/crop.md),
`{dir}{sep}{base}.crop{ext}`, therefore produces `data/sub-01_T1w.crop.nii.gz`.
You can use the placeholders to put results in another directory, to change
the suffix, or to change the format by changing the extension:

```shell
nitorch crop sub-01_T1w.nii.gz -s 96 96 96 -o 'derivatives{sep}{base}_roi.mgz'
```

Quote templates that contain braces, so that your shell does not interpret
them, and in particular so that `bash` does not try to expand `{a,b}`-style
patterns.

Some commands define additional placeholders:

| Placeholder  | Commands                                         | Replaced by                                                |
| ------------ | ------------------------------------------------ | ---------------------------------------------------------- |
| `{i}`        | `chunk`, `unstack`                               | The index of the piece, counting from 1.                   |
| `{i}`, `{j}`, `{k}` | `extract_patches`                         | The index of the patch along each axis, counting from 1.   |
| `{n}`        | `epic`                                           | The index of the echo, counting from 0.                    |
| `{layout}`   | `orient`, `reorient`                             | The orientation of the output, for example `RAS`.          |
| `{label}`    | `affopt`, `vopt`                                 | The label of the image in the groupwise model.             |
| `{name}`     | `register` and its wrappers                      | The name of the transformation model.                      |
| `{space}`    | `uniseg`                                         | The space of the output: `nat`, `mni` or `wrp`.            |

Not every command supports every placeholder. Each command page lists the
placeholders that it accepts and the default template of each output. A few
points are worth remembering:

- A template without any placeholder is used as is. When you give a single
  literal name to a command that processes several inputs, every result is
  written to the same file, and only the last one survives. Use `{base}` (or
  give one name per input) to avoid this.
- A literal brace in a file name must be doubled (`{{` and `}}`).
- Because results are written next to the inputs by default, the input
  directory must be writable. Use `{dir}` and a fixed directory in the
  template, or the output directory option of the command where there is one,
  to write elsewhere.
- A few commands do not supply every one of the four common placeholders;
  [`meanspace`](reference/spatial/meanspace.md), for instance, has no `{sep}`,
  and its templates use a literal `/`.

## Spatial transformations

Many commands produce, consume or combine spatial transformations: registration
commands estimate them, [`reslice`](reference/spatial/reslice.md) applies them,
[`compose`](reference/spatial/compose.md) chains them. Three kinds exist, and
every command that takes a transformation tells you which kinds it accepts.

### Linear transformations

A linear (affine) transformation is a 4&times;4 matrix that acts on
homogeneous world coordinates `[x, y, z, 1]`, in millimetres, in RAS space (see
[Coordinates and units](#coordinates-and-units)). It combines rotations,
translations, zooms and shears.

Matrices are stored as FreeSurfer `.lta` files, or less often as `.dat`
files. Whatever the type stored in an LTA file (voxel-to-voxel,
RAS-to-RAS, ...), NITorch converts it to a RAS-to-RAS matrix when it reads it,
using the orientation matrices stored in the file. Most NITorch commands write
RAS-to-RAS transforms. The LTA file also records the shape and orientation
matrix of the source and destination images; [`crop`](reference/images/crop.md)
uses them when you give it a transform file instead of a size.

**Direction.** Transformations are applied the way an image is resampled: for
each voxel of the *output* image, a transformation says where to look in the
*input* image. A matrix therefore maps coordinates of the output (target) space
to coordinates of the input space. For registration this means that the
matrix written by [`register`](reference/registration/register.md) and its
wrappers maps world coordinates of the **fixed** image onto world coordinates of
the **moving** image. It is the matrix to pass to `reslice -l` to resample the
moving image into the space of the fixed image.

### Displacement fields

A displacement field assigns a vector to each voxel of a lattice. It is stored
as a volume whose extra dimension holds the three components (along x, y and
z), and the orientation matrix of the file describes the lattice. Applying
the field to an image means sampling the image at `x + d(x)`, where `x` is a
location of the lattice and `d` is the field.

The **unit** of the displacements is declared by the command that reads the
field, typically with an option named `-u`/`--unit`:

- `vox` (the default in [`reslice`](reference/spatial/reslice.md) and
  [`compose`](reference/spatial/compose.md)): displacements are measured in
  voxels of the field's own lattice.
- `mm`: displacements are measured in millimetres of world space.

### Velocity fields

A velocity field is a displacement-like field that is *integrated* to obtain
a transformation. NITorch commands use **stationary velocity fields**: the
transformation is the exponential of the field, computed by scaling and
squaring, and its inverse is obtained by negating the velocity. Velocities are
expressed in voxels of the field's lattice. Registrations that use geodesic
shooting also write a small JSON file with the regularisation parameters next
to the velocity field; commands that read the field use that file, when it is
present, to integrate it correctly. [`vexp`](reference/spatial/vexp.md)
converts a velocity field into a displacement (or transformation) field.

Because a velocity field encodes a transformation that is invertible by
construction, the default non-linear model of
[`register`](reference/registration/register.md) is a velocity field rather
than a displacement field.

### Combining transformations

Commands that accept several transformations, such as `reslice` and `compose`,
take them as a sequence of options (`-l` for a linear transformation, `-d` for a
displacement field, `-v` for a velocity field). Each one can be inverted by
inserting an `i` (`-il`, `-id`, `-iv`) or appending `-inverse` to the long
form. The transformations are applied in the order that the reslicing formula
implies: for an output voxel at world position `p` in the target space, the
last transformation on the command line is applied first, and the first one
last, and the result is a position in the input space. See the pages of
[`reslice`](reference/spatial/reslice.md) and
[`compose`](reference/spatial/compose.md) for the exact formula and for the
`-t`, `--target` option that selects the output space.

## Coordinates and units

### Voxel and world coordinates

Every volume carries an **orientation matrix** (also called the vox2ras or
voxel-to-world matrix): a 4&times;4 affine matrix that maps voxel indices to
world coordinates. Voxel indices start at zero, and the first voxel is centred
on index 0. The world space is **RAS+**, in millimetres: the first world axis
increases towards the Right, the second towards the Anterior and the third
towards the Superior. This is the convention of NIfTI and of `nibabel`. The
columns of the orientation matrix are ordered like the dimensions of the data.

[`info`](reference/images/info.md) prints, among other things, the centre of
the volume in world coordinates (`mm (RAS)`) and its layout.

### Layouts

The **layout** of a volume is the order and direction in which its axes are
stored on disk. It is written as three letters, one per stored axis, naming
the direction towards which that axis points: `R` or `L` (towards the right or
the left), `A` or `P` (anterior or posterior), `S` or `I` (superior or
inferior). A volume stored with the first axis running left to right, the
second posterior to anterior and the third inferior to superior has layout
`RAS`. Commands that act on voxel
axes, such as [`crop`](reference/images/crop.md), [`pad`](reference/images/pad.md)
or [`orient`](reference/spatial/orient.md), pay attention to the layout.

### Space keywords

Commands that take sizes or positions can usually be told in which space they
are given, with one of these keywords placed after the numbers:

| Keyword | Meaning                                                                                      |
| ------- | -------------------------------------------------------------------------------------------- |
| `vox`   | Voxel units: indices and counts along the stored axes of the volume.                         |
| `ras`   | World units: millimetres along the world axes (R, A, S), whatever the layout of the volume.  |

For example, `-s 64 64 64` crops a cube of 64 voxels, whereas `-s 40 40 40 ras`
crops a cube of 40 mm (the number of voxels then depends on the voxel size).
The default is `vox`. A position given with `ras` is a point in world space,
and is converted to voxel coordinates with the inverse of the orientation
matrix; a size given with `ras` is expressed along the world axes and is
re-ordered to match the stored axes.

### Physical units and percentages

Quantities that are lengths in the physical sense, such as the full width at
half maximum of a smoothing kernel or a voxel size, are accepted with a unit
keyword, most often `mm`, and in [`smooth`](reference/images/smooth.md)
`vox` too. Some commands also accept `%`, to say that a value is a
percentage of a reference quantity; for example, the voxel size of a mean
space can be given as a percentage of the mean voxel size of the inputs
([`meanspace`](reference/spatial/meanspace.md)). The unit that applies when you
omit the keyword differs between commands and is stated on each command page.

## Hardware

NITorch commands run on the CPU by default (`denoise_mri` is the exception: it
defaults to the GPU, and needs `-cpu` to run without one). Most commands that
do heavy computation can use a CUDA GPU through PyTorch:

| Option         | Meaning                                                                     |
| -------------- | --------------------------------------------------------------------------- |
| `--cpu`        | Run on the CPU.                                                             |
| `--cpu N`      | Run on the CPU with `N` threads. Only where the command's help shows `--cpu [THREADS]`. |
| `--gpu`        | Run on the GPU.                                                             |
| `--gpu ID`     | Run on the GPU with index `ID` (`0` is the first one).                      |

The double-dash spellings `--cpu` and `--gpu` are accepted by every command
that has a hardware option. Some commands, namely `reslice`, `compose`, `vexp`,
`pool`, `inpaint`, `denoise_mri` and `autoreg`, accept the single-dash
spellings `-cpu` and `-gpu` as well, but in these commands `--cpu` takes no
number of threads. Commands whose help shows `--cpu [THREADS]` (for example
`register`, `denoise`, `meanspace` and `uniseg`) pass the number on to
PyTorch, which then uses that many CPU threads.

Which GPU is device `0` is decided by CUDA, so you can use the environment
variable `CUDA_VISIBLE_DEVICES` to restrict the GPUs that are visible, as with
any PyTorch program. Commands that do not accept these options, such as
[`info`](reference/images/info.md) or [`crop`](reference/images/crop.md), do
their little work on the CPU.

!!! note "Without a usable GPU"
    Many commands notice that CUDA is not available and switch to the CPU,
    printing a warning if you asked for the GPU. Others do not check, and
    PyTorch stops with an error. If you see a CUDA error, run the command
    again without `--gpu`.

## Environment variables

A few environment variables change how NITorch itself behaves, independently
of the command.

| Variable                   | Effect                                                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `NI_CHECK_BACKEND`         | Set to `0` to silence the start-up warning about the non-compiled backend.                                         |
| `NI_COMPILED_BACKEND`      | Choose the implementation of the resampling routines at run time: `C`, `TS` or `MONAI`. See [Compiled backend](../install/compiled.md). |
| `NI_SHOW_COMPILED_BACKEND` | Set to `1` to print which implementation is loaded.                                                                 |
