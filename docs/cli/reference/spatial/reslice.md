---
icon: octicons/terminal-24
---

# reslice

Resample one or more volumes onto a new grid, optionally through a chain of spatial transformations: affine matrices, dense displacement fields and diffeomorphic velocity fields. Use it to bring an image into the space of another one after registration, to change the resolution of an image to a given voxel size, or to apply a transformation computed elsewhere.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch reslice FILE... [TRF...] [-t FILE] [-vx VX...] [-o FILE...]
                    [-i ORDER] [-b BND] [-p [BOOL]] [-x] [-c CHANNEL...]
                    [-k [SIZE...]] [-dt DTYPE] [--log] [--logit [implicit]]
                    [--clip] [--cpu | --gpu [ID]] [-q]

    TRF is any of:
        {-l|-il|-l2|-il2} FILE
        {-d|-id} FILE [-u {vox,mm}]
        {-v|-iv|-v2|-iv2} FILE [JSON] [T0 T1]
    ```

## Description

For every output voxel, `reslice` computes where it lies in world space, moves that point through the transformations you list, finds the corresponding position in the input image, and interpolates the input there. This is the standard way of resampling an image: the output is *pulled* from the input, so there are no holes. The output grid (shape and orientation matrix) comes from a **target image** (`-t`), from a requested **voxel size** (`-vx`), or from the input itself.

All input files come first on the command line, before any tag. They can be volumes or streamline files (`.trk`), see [streamlines](#streamlines). Then come the transformations and the other options, in any order, but the *order of the transformations matters*.

The command does two kinds of work, and it helps to know which one you are asking for:

- If the chain contains only linear transformations and you give neither a target nor a voxel size, the data are **not resampled at all**. The command writes a copy of the input with a new orientation matrix, which is exact and instantaneous.
- In every other case, the data are **interpolated** onto the output grid.

### How transformations are chained

Each transformation maps *coordinates of the output space to coordinates of the input space*. This is the direction that resampling needs: for every point of the output grid, it says where to look in the input. Keep this in mind when you read a transformation file: if a registration reports a matrix that maps moving to fixed coordinates, the matrix you need to resample the moving image onto the fixed one is its **inverse**.

If you list transformations $T_1, T_2, \dots, T_n$ in this order on the command line, the value of the output image at voxel $x$ is

$$
\mathrm{out}(x) = \mathrm{in}\!\left(M_\text{in}^{-1}\; T_1 \circ T_2 \circ \dots \circ T_n \,(M_\text{out}\, x)\right),
$$

where $M_\text{out}$ is the orientation matrix of the output grid and $M_\text{in}$ that of the input image. So:

1. the voxel $x$ is converted to world coordinates with the orientation matrix of the output grid;
2. the **last** transformation on the command line is applied first, then the one before it, and so on up to the **first** one;
3. the resulting world position is converted to a voxel position in the input with the inverse orientation matrix of the input, and the input is interpolated there.

Put another way: list the transformations in order *from the input image towards the output space*. The first transformation you write sits next to the input image, and the last one sits next to the target grid.

Say a registration of `mov.nii.gz` to `fix.nii.gz` produced an affine matrix `affine.lta` that maps moving coordinates to fixed coordinates, and a velocity field `velocity.nii.gz` defined in the space of the fixed image. To resample the moving image onto the fixed one:

```shell
nitorch reslice mov.nii.gz -il affine.lta -v velocity.nii.gz -t fix.nii.gz -o mov_in_fix.nii.gz
```

Reading from the target side: a point of the fixed grid is moved by the exponentiated velocity field (the last transformation, applied first), then by the inverse of `affine.lta` (the first, applied last), which lands it in moving space. Reading the other way, the first transformation listed (`-il affine.lta`) is attached to the moving image, the second (`-v velocity.nii.gz`) to the fixed grid.

**Inverting a chain.** The inverse of $T_1 \circ \dots \circ T_n$ is $T_n^{-1} \circ \dots \circ T_1^{-1}$: reverse the order of the transformations and toggle `-i` on each of them. To resample the fixed image onto the moving one with the same registration:

```shell
nitorch reslice fix.nii.gz -iv velocity.nii.gz -l affine.lta -t mov.nii.gz -o fix_in_mov.nii.gz
```

**Consecutive linear transformations** are multiplied into a single matrix before use, in the order you wrote them: `-l A.lta -l B.lta` applies $B$ first, then $A$, like the matrix product $AB$.

### Transformation types

| Type            | Tag                         | Takes          | What the file holds                                                                                          |
| --------------- | --------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------ |
| Linear          | `-l`, `--linear`            | `FILE`         | An affine matrix mapping world coordinates (mm) of the output space to world coordinates of the input space.   |
| Displacement    | `-d`, `--displacement`      | `FILE`         | A dense field of displacements: the point $p$ is sent to $p + u(p)$.                                          |
| Velocity        | `-v`, `--velocity`          | `FILE [JSON] [T0 T1]` | A velocity field $v$; the transformation used is the part of its path between times `T0` and `T1` (by default 0 and 1), which is $\exp(v)$ for a stationary field. |

Every type has an **inverse** form and, for linear and velocity types, a **square** form:

| Modifier  | Short form                   | Long form                                      | Effect                                                                                       |
| --------- | ---------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Inverse   | `-il`, `-id`, `-iv`          | `--linear-inverse`, `--displacement-inverse`, `--velocity-inverse` | Use the inverse transformation.                                                |
| Square    | `-l2`, `-v2`                 | `--linear-square`, `--velocity-square`         | The file holds the *square* of the transformation to apply: its square root is applied.       |
| Both      | `-il2`, `-iv2`               | `--linear-inverse-square`, `--linear-square-inverse` (and the same for `--velocity`) | The inverse of the square root.                            |

The square forms let you apply half of a known transformation, for instance to move two images half-way towards each other. They do not exist for displacement fields: `-d2` and `-id2` (and their long forms) are rejected with an error. The tags are matched exactly, so `-il` (inverse linear) is not confused with `-i` (interpolation), nor `-v` (velocity) with `-vx` (voxel size).

#### Linear transformations

The file is read with the transform readers of NITorch, which understand FreeSurfer `.lta` files and `.dat` register files (see [spatial transformations](../../conventions.md#spatial-transformations)). The matrix is used as a map between world (RAS) coordinates in millimetres. The inverse is computed with a matrix inverse, and the square root through the matrix logarithm, which requires a matrix that has a real logarithm (no reflection).

#### Displacement fields

The file is a volume with three components (the shifts along the three axes) in the last dimension, defined on its own grid with its own orientation matrix. At each point the field is evaluated by linear interpolation at the position of the point in the field's voxel grid, and the shift is added to the point. Outside the field of view of the file, the field is extended periodically.

By default the three components are in **voxels** of the field's own grid. With `-u mm`, they are millimetres in world space and are converted by the orientation matrix of the field. The unit option is written right after the file name of the field, for example `-d disp.nii.gz -u mm`.

The inverse (`-id`) of a displacement field has no closed form. The field is resampled to the voxel size of the output grid and inverted numerically by a regularised least-squares scheme, so the inverse is approximate, in particular where the field folds. If the displacement was written by `register @nonlin smalldef --inversion newton` and its `.json` sidecar is next to it, the inverse is refined by regularised Gauss-Newton steps, as `register` did.

#### Velocity fields

The file is a volume with three components in the last dimension, expressed in voxels of its own grid. It is exponentiated by scaling and squaring ([see `vexp`](vexp.md#description)) with 8 steps, to a displacement field defined on the grid of the velocity file, and then applied like a displacement field. The inverse (`-iv`) is the exponential of $-v$, which is exact up to the accuracy of scaling and squaring, unlike the inverse of a displacement field. `-v2` halves $v$ before exponentiating.

A velocity file can be followed by the name of a **JSON file**, as written by [`register`](../registration/register.md#outputs). Its `model` entry says what the velocity is:

- With `"model": "shoot"`, or with no `model` entry at all, the field is an *initial velocity* and is integrated by geodesic shooting instead of by scaling and squaring. The regularisation stored in the file (`absolute`, `membrane`, `bending`, `lame` and `factor`, plus the number of `steps` and the `fast` flag) is passed to the shooting routine, together with the voxel size of the velocity file.
- With `"model": "svf"`, the field is a stationary velocity field, exponentiated by scaling and squaring.

The `inversion` entry (`default`, `push` or `newton`, as in [`register --inversion`](../registration/register.md#choosing-how-the-warp-is-inverted-with-inversion)) selects how inverted maps are computed. A JSON file without it behaves as before, with `default`. For a stationary velocity field, the `bound` entry (as in [`register @nonlin --bound`](../registration/register.md#choosing-the-boundary-condition-of-the-field-with-bound)) is the boundary condition used to exponentiate and invert it; a JSON file without it uses `dft`, as before. The same holds for the inversion (`-id`) of a `smalldef` displacement field with its sidecar.

If you give no JSON file and a file with the same name as the velocity field and the extension `.json` exists, it is used automatically, whether it describes a shooting velocity (it contains a `bending` entry, or `"model": "shoot"`) or a stationary one (`"model": "svf"`). Otherwise the field is a stationary velocity field.

**Time bounds.** A velocity field describes a path of transformations indexed by time. By convention, the moving image sits at time 0 and the fixed image at time 1. The tag `-v FILE T0 T1` applies the part of that path that goes from time `T0` to time `T1`: it turns an image at time `T0` into the image at time `T1`. The bounds default to 0 and 1, so `-v FILE 0 1` is the same as `-v FILE`. For a stationary velocity field, the transformation is exactly $\exp\big((T_1 - T_0)\,v\big)$. For a shooting velocity (one with a JSON file), the field is integrated by geodesic shooting over the interval from `T0` to `T1`. Bounds outside $[0, 1]$ are allowed and extrapolate the path, with the same loss of accuracy as for larger velocities.

The inversion method is applied to each map as it is built, exactly as the model builds it at $t=1$, and before the maps are composed; it is not applied to the composed interval map. For a shooting velocity, only the forward map $\varphi_{T_1}$ is inverted (the inverse is integrated directly). For a stationary velocity field, the maps of negative time, the exponentials of $-tv$, are the ones that are inverted. With `default`, nothing changes.

The bounds follow the velocity file, and the JSON file if there is one, which must come first. After the velocity file, a number is read as a time bound and anything else as the JSON file. Give either no bounds or exactly two. The `.json` file next to the velocity field is still found automatically when you give bounds without a JSON file.

The other velocity tags are shorthands for particular bounds, and they combine with explicit ones:

| Tag             | Same as          |
| --------------- | ---------------- |
| `-v2`           | `-v FILE 0 0.5`  |
| `-iv`           | `-v FILE 1 0`    |
| `-iv2`          | `-v FILE 0.5 0`  |

Tags that invert or halve the transformation act on the bounds you give: `-v2 FILE 0 0.4` halves them, and `-iv FILE 0 0.4` swaps them, which is `-v FILE 0.4 0`. For a shooting velocity, `-v FILE 0 -1` is not the same as `-iv`: a geodesic run backwards is not the inverse of the forward map.

**Moving both images to the midpoint.** To bring the moving and the fixed image half-way towards each other along the path, take the moving image from time 0 to time 0.5 and the fixed image from time 1 to time 0.5, both on the grid of the fixed image:

```shell
nitorch reslice mov.nii.gz -il affine.lta -v velocity.nii.gz 0 0.5 -t fix.nii.gz -o mov_mid.nii.gz
nitorch reslice fix.nii.gz -v velocity.nii.gz 1 0.5 -t fix.nii.gz -o fix_mid.nii.gz
```

For a shooting velocity, the two images agree at the midpoint up to the accuracy of the shooting inverse. Use `-v FILE 1 0.5` for the fixed image rather than `-iv2`: with shooting, `-iv2` is only approximately the second half of the path, whereas for stationary fields the two are identical.

### The output grid

| Situation                                   | Output grid                                                                                                                                         |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-t FILE`                                   | Shape and orientation matrix of the target. The first three dimensions of the target are used; its voxel values are never read.                        |
| `-t FILE` and `-vx`                         | The target, resized to the voxel size `-vx`: the centre of the first voxel stays where it is, and the shape is $\lfloor (N-1) r \rfloor + 1$ voxels along an axis of $N$ voxels, where $r$ is the ratio of old to new voxel size, so that no new voxel lies beyond the old field of view. |
| No target, `-vx`                            | Same as above, starting from the grid of the input.                                                                                                  |
| No target, only linear transformations      | No resampling. The data are copied unchanged and the orientation matrix of the input $M_\text{in}$ is replaced by $T^{-1} M_\text{in}$, where $T$ is the product of the linear transformations. |
| No target, at least one non-linear one      | The grid of the input, resampled through the chain. If the chain starts with a linear transformation $T_1$, the grid is the input grid moved by $T_1^{-1}$. |

`-vx` takes one voxel size in millimetres for all axes, or three values. Only three dimensions are supported for the grid.

The output has as many channels as the input (a fourth dimension), or as many as selected with `-c`; every channel is resampled with the same grid. Channels are selected with single indices, Python-style ranges `start:stop:step`, or a mix of both (see [`-c`](#output-and-intensities)).

### Interpolation, boundaries and extrapolation

`-i` sets the order of the B-spline used for interpolation: 0 is nearest neighbour, 1 is trilinear (the default), 2 and above are higher-order splines. With `-p yes` (the default), the spline pre-filter is applied, so that orders above 1 interpolate the data exactly. Use `-p no` to skip it.

`-b` sets the boundary condition, which describes how the input is extended beyond its edges: `dct2` (the default, mirror about the outer edge of the border voxel), `dct1`, `dst1`, `dst2`, `dft` (circular), `replicate` or `zero`, with the aliases listed in [`resize`](resize.md#boundary-conditions). It takes effect near the edges of the input and in the pre-filter.

Output voxels that fall outside the field of view of the input are set to **zero**, whatever the boundary condition. Add `-x` (extrapolate) to fill them instead according to the boundary condition, which with the default `dct2` mirrors the image outwards.

**Label images.** `-i l` reads the input as integer labels. Each label is resampled separately by linear interpolation of its indicator function, and every output voxel receives the label with the highest value. No new label values appear, and no pre-filter is applied. The output data type is that of the input. Be careful that the value is a single letter: any word beginning with `l`, including `linear`, selects label mode.

### Output type and units

Intensities are resampled in floating point and written with the **on-disk data type of the input**, unless you choose another with `-dt` (a name such as `float32`, `int16` or `uint8`). The conversion is a plain cast without rescaling, so when the output type is an integer type, interpolated values are truncated. Spatial units are unchanged: the orientation matrix is in millimetres, like the matrix of the input.

Three options act on the intensities before and after interpolation:

- `--clip` clips the resampled values to the range of finite values of the input. It removes the overshoots of higher-order splines.
- `--log` interpolates the logarithm of the intensities, after clamping them to a small positive floor, and exponentiates the result. It keeps strictly positive images positive.
- `--logit` (optionally followed by `implicit`) is a niche variant intended for probability maps: channels are interpolated as log-probabilities and mapped back with a softmax across channels. With `implicit`, the input is taken to omit its last class.

### Streamlines

Streamline files (`.trk`) can be given as inputs, alone or together with volumes, and they are moved by the same transformations in the same direction as the images. A point of the input streamlines is mapped to the output space by the inverse of the chain, so that tracts and images in the same input space end up in the same output space. No target or output grid is involved: `-t`, `-vx` and the interpolation options do not apply to them. Their default output name is the same as for volumes.

## Arguments

| Name   | Description                                                                                  |
| ------ | -------------------------------------------------------------------------------------------- |
| `FILE` | One or more volumes or streamline files, listed before any tag. Any format NITorch reads (see [file formats](../../conventions.md#file-formats)). |

!!! warning "Put all the input files first"
    Once the first tag has been read, a stray argument that does not belong to a tag is an error. Values that follow a velocity field are read as part of that tag: numbers are its two time bounds, and anything else is its JSON file.

## Options

### Transformations

| Option                                                 | Value             | Description                                                                                         | Default |
| ------------------------------------------------------ | ----------------- | --------------------------------------------------------------------------------------------------- | ------- |
| `-l`, `--linear`                                       | `FILE`            | Linear transformation. [See above](#linear-transformations).                                         |         |
| `-d`, `--displacement`                                 | `FILE`            | Displacement field. [See above](#displacement-fields).                                               |         |
| `-v`, `--velocity`                                     | `FILE [JSON] [T0 T1]` | Velocity field, with an optional JSON file of shooting parameters and optional time bounds. [See above](#velocity-fields).    |         |
| `-il`, `-id`, `-iv`, `--linear-inverse`, `--displacement-inverse`, `--velocity-inverse` | same as the tag without `i` | The inverse transformation.                             |         |
| `-l2`, `-v2`, `--linear-square`, `--velocity-square`, and their inverses | same              | The file holds the square of the transformation (for velocities, a shorthand for time bounds). [See above](#transformation-types).                 |         |
| `-u`, `--unit`                                         | `{vox,mm}`        | After `-d`/`-id` only: unit of the displacements.                                                    | `vox`   |
| `-n`, `--order`                                        | `ORDER`           | After `-d`/`-id` only: order of the splines that encode the field.                                   | `1`     |

### Output grid

| Option                  | Value        | Description                                                                                                      | Default                  |
| ----------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `-t`, `--target`        | `FILE`       | Image that defines the output grid. [See above](#the-output-grid).                                                | see the table above      |
| `-vx`, `--voxel-size`   | `VX...`      | Voxel size of the output grid, in mm: one value or three.                                                         | from the target or input |

### Output and intensities

| Option                                  | Value            | Description                                                                                                                  | Default                              |
| --------------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `-o`, `--output`                        | `FILE...`        | Output filenames, one per input. [Templates](../../conventions.md#output-filenames) with `{dir}`, `{sep}`, `{base}` and `{ext}`. | `{dir}{sep}{base}.resliced{ext}`             |
| `-i`, `-inter`, `--interpolation`       | `ORDER` or `l`   | Interpolation order (integer, 1 by default), or `l` for label images. [See above](#interpolation-boundaries-and-extrapolation). | `1`                                  |
| `-b`, `-bnd`, `--bound`                 | `BND`            | Boundary condition: `dct2`, `dct1`, `dst1`, `dst2`, `dft`, `replicate`, `zero` and aliases.                                  | `dct2`                               |
| `-p`, `--prefilter`                     | `[BOOL]`         | Apply the spline pre-filter: `yes`, `no` (also `true`, `false`, `1`, `0`). The value is optional: a bare `-p` means `yes`.                            | `yes`                                |
| `-x`, `-ex`, `--extrapolate`            |                  | Fill voxels outside the field of view of the input using the boundary condition instead of zeros.                            | off                                  |
| `-c`, `--channels`                      | `CHANNEL...`     | Channels (last dimension) to reslice, counted from 0: single indices and/or Python-style ranges `start:stop:step` (`0:2`, `:2`, `1:`, `::2`, negative bounds allowed). The output keeps only these.                           | all channels                         |
| `-dt`, `--dtype`                        | `DTYPE`          | Data type of the output file, such as `float32`, `int16` or `uint8`.                                                         | that of the input                    |
| `--clip`                                |                  | Clip the output to the range of the input.                                                                                    | off                                  |
| `--log`                                 |                  | Interpolate the logarithm of the intensities.                                                                                | off                                  |
| `--logit`                               | `[implicit]`     | Interpolation in logit space, for probability maps (softmax across channels).                                                              | off                                  |
| `-q`, `--quiet`                         |                  | Do not print the name of each file as it is processed.                                                                       | verbose                              |

### Performance and hardware

| Option                  | Value         | Description                                                                                                                | Default          |
| ----------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| `-k`, `--chunk`         | `[SIZE...]`   | Compute the output in blocks of this size (in voxels; one value or three) to limit memory use. Without a value: 128. Approximate: see [known limitations](#known-limitations). | whole volume     |
| `-cpu`, `--cpu`         |               | Use the CPU.                                                                                                               | CPU              |
| `-gpu`, `--gpu`         | `[ID]`        | Use a GPU, optionally the one with this index (see [hardware](../../conventions.md#hardware)).                              | off              |

### Known limitations

With `-k`, the blocks are laid out over the output grid. Two caveats apply:

- The `--clip` range and the floor used by `--log` and `--logit` are computed for each block rather than over the whole image.
- The spline pre-filter, for orders 2 and above, runs on each block separately, so the result differs slightly from an unchunked run. With order 1 (the default) it is exact.

`-k` thus trades exactness for memory: use it when the volume does not fit in memory, and expect small differences from an unchunked run whenever one of these caveats applies.

## Examples

Move an image by a rigid transformation, without any interpolation. Only the orientation matrix of the output changes. The matrix in `shift.lta` maps points of the output space to points of the input space:

```shell
nitorch reslice sub-01_T1w.nii.gz -l shift.lta -o sub-01_T1w_moved.nii.gz
```

Resample a moving image onto a fixed image using an affine matrix from a registration. If `mov_to_fix.lta` maps moving coordinates to fixed coordinates, use its inverse:

```shell
nitorch reslice sub-01_FLAIR.nii.gz -il mov_to_fix.lta -t sub-01_T1w.nii.gz \
    -o sub-01_FLAIR_in_T1w.nii.gz
```

Resample to 1 mm isotropic voxels with cubic splines, keeping the field of view of the input:

```shell
nitorch reslice sub-01_T1w.nii.gz -vx 1 -i 3 -o sub-01_T1w_1mm.nii.gz
```

Combine an affine and a velocity field, as written by a diffeomorphic registration, to resample the moving image onto the fixed one. The velocity field is applied to the points of the fixed grid before the affine matrix:

```shell
nitorch reslice mov.nii.gz -il affine.lta -v velocity.nii.gz -t fix.nii.gz -o mov_in_fix.nii.gz
```

Resample the fixed image into moving space with the same registration: the order of the transformations is reversed and each one is inverted:

```shell
nitorch reslice fix.nii.gz -iv velocity.nii.gz -l affine.lta -t mov.nii.gz -o fix_in_mov.nii.gz
```

Carry a segmentation along with the same transformations. The labels are interpolated as labels, so no new values are created, and the output type stays that of the input:

```shell
nitorch reslice mov_aseg.nii.gz -il affine.lta -v velocity.nii.gz -t fix.nii.gz \
    -i l -o mov_aseg_in_fix.nii.gz
```

Resample two images onto the same target in one command. The names of the outputs are given in the order of the inputs:

```shell
nitorch reslice sub-01_T1w.nii.gz sub-01_T2w.nii.gz -t template.nii.gz \
    -o sub-01_T1w_in_template.nii.gz sub-01_T2w_in_template.nii.gz
```

Warp an image with a displacement field expressed in voxels of its own grid. The option `-x` fills the part of the target that lies outside the input using the boundary condition, instead of leaving zeros:

```shell
nitorch reslice sub-01_T1w.nii.gz -d disp.nii.gz -t template.nii.gz -x -o sub-01_T1w_warped.nii.gz
```

Move tractography streamlines into the space of the fixed image, with the same command-line transformations as for the image:

```shell
nitorch reslice tracts.trk -il affine.lta -o tracts_in_fix.trk
```

## See also

- [`compose`](compose.md) combines a chain of transformations into one linear transformation or one displacement field, using the same conventions.
- [`vexp`](vexp.md) exponentiates a velocity field into a displacement field.
- [`resize`](resize.md) changes the resolution of an image on a grid derived from its own.
- [`meanspace`](meanspace.md) computes a common target space for a set of images, and can call `reslice` for you.
- [`orient`](orient.md) and [`reorient`](reorient.md) change the orientation matrix and the storage order of an image.
- [Spatial transformations](../../conventions.md#spatial-transformations) in the CLI conventions.
