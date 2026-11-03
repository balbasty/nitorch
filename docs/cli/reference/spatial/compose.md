---
icon: octicons/terminal-24
---

# compose

Combine a chain of spatial transformations (affine matrices, displacement fields, velocity fields, and their inverses) into a single transformation. Use it to invert a transformation, to merge an affine and a non-linear registration into one file, or to prepare a transformation that you will apply to many images.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch compose TRF... [-t FILE] [-o FILE] [-ou {vox,mm}] [-cpu | -gpu [ID]]

    TRF is any of:
        {-l|-il} FILE
        {-d|-id} FILE [-u {vox,mm}]
        {-v|-iv|-v2|-iv2} FILE [JSON]
    ```

## Description

`compose` uses the same description of a transformation chain as [`reslice`](reslice.md), and instead of resampling an image, it writes the resulting transformation. There are no input images: the command line is a list of transformations, optionally followed by a target and an output name.

The result is one of two things:

- If **all transformations are linear** and you give no target, the result is a single **linear transformation**, written as an `.lta` file: the product of the matrices, after any inversion.
- Otherwise the result is a **dense displacement field**, defined on the voxel grid of a *target* image, and written as a volume with three components in its last dimension.

### How the chain is read

As in `reslice`, each transformation maps *coordinates of the output space to coordinates of the input space*, and if you list $T_1, T_2, \dots, T_n$ in this order, the composed transformation is

$$
T = T_1 \circ T_2 \circ \dots \circ T_n ,
$$

so that the **last** transformation on the command line is applied **first** to a point, and the first one last. List the transformations in order from the "input" side towards the "output" side, exactly as you would for `reslice`. [The detailed rules, the transformation types and the direction conventions are explained in `reslice`](reslice.md#how-transformations-are-chained); they apply here without change.

Velocity fields are handled as in [`reslice`](reslice.md#velocity-fields). A velocity file can be followed by a **JSON file** of shooting parameters. If there is one, or if a file with the same name as the velocity field and the extension `.json` exists and contains a `bending` entry (as written by `register`), the field is an *initial velocity* and is integrated by geodesic shooting. Otherwise it is a stationary velocity field, exponentiated by scaling and squaring with 8 steps. The square forms `-v2` and `-iv2` halve the velocity first, which gives the geodesic midpoint when shooting.

Consecutive linear transformations are multiplied together first, in the order you wrote them.

To **invert** a composition, reverse the order and toggle `-i` on every transformation. For instance, the inverse of `-il affine.lta -v velocity.nii.gz` is `-iv velocity.nii.gz -l affine.lta`. A single transformation is inverted by writing it with its `-i` tag: `compose -il affine.lta` writes the inverse matrix.

### The target grid

A dense field needs a grid on which it is defined. It is the grid of `-t FILE` if you give one. Otherwise, it is the grid (shape and orientation matrix) of the file of the **last non-linear transformation on the command line**, that is, the one that is applied first. If all transformations are linear, there is no grid, and a matrix is written, unless you give `-t`, in which case a displacement field on the target grid is written.

At every voxel $x$ of the target grid, with $M$ the orientation matrix of the target, the position $y = T(Mx)$ is computed. The displacement written at $x$ is then $y - Mx$ in millimetres (`-ou mm`), or $M^{-1}y - x$ in voxels of the target grid (`-ou vox`, the default). The output file carries the orientation matrix of the target. In voxel units, it is a displacement field that can be given straight back to `reslice` or `compose` as a `-d` field, together with the same target, and it then reproduces the chain: see the last example.

A displacement written in millimetres is a correct world-space displacement. To reuse it in `reslice` or `compose`, give it with `-d FILE -u mm`.

### Output files

The default output name is `composed{ext}`: `composed.lta` for a matrix, `composed.nii.gz` for a field, in the current directory. If you give `-o`, the name is used as it is. Its extension must suit the content: `.lta` for a matrix, a volume extension such as `.nii.gz` for a field. A name with the field `{ext}` in it is filled with `.lta` or `.nii.gz` as appropriate.

Fields are written in floating point (32 bits). Matrices are written as FreeSurfer linear transform arrays.

## Arguments

`compose` takes no positional arguments. The first argument on the command line must be a transformation tag; a value that does not belong to a tag is an error.

## Options

### Transformations

| Option                              | Value                | Description                                                                                         | Default |
| ----------------------------------- | -------------------- | --------------------------------------------------------------------------------------------------- | ------- |
| `-l`, `--linear`                    | `FILE`               | Linear transformation (an `.lta` or `.dat` matrix file).                                             |         |
| `-d`, `--displacement`              | `FILE`               | Dense displacement field.                                                                            |         |
| `-v`, `--velocity`                  | `FILE [JSON]`        | Velocity field. With a JSON file of shooting parameters (given, or found next to the field), it is integrated by geodesic shooting; otherwise it is exponentiated by scaling and squaring. |         |
| `-il`, `-id`, `-iv`                 | `FILE`               | The inverse of a linear transformation, of a displacement field, of a velocity field (`-iv` also takes an optional `JSON`). |         |
| `-l2`, `-v2`                        | `FILE`               | The file holds the square of the transformation: its square root is used. `-v2` also takes an optional `JSON`. |         |
| `-il2`, `-iv2`                      | `FILE`               | The inverse of the square root.                                                                      |         |
| `-u`, `--unit`                      | `{vox,mm}`           | After `-d`/`-id` only: unit of the displacements. Voxels of the field's own grid, or millimetres.                                    | `vox`   |
| `-n`, `--order`                     | `ORDER`              | After `-d`/`-id` only: order of the splines that encode the field (`0` to `7`; `1` is linear). | `1`     |

At least one transformation is required. The inverse forms also have long names: `--linear-inverse`, `--displacement-inverse` and `--velocity-inverse`; the square forms have `--linear-square` and `--velocity-square`.

### Output

| Option                  | Value        | Description                                                                                             | Default                                                        |
| ----------------------- | ------------ | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `-t`, `--target`        | `FILE`       | Image that defines the grid on which a displacement field is written. Only its shape and orientation matrix are used. | grid of the last non-linear transformation; none if all are linear |
| `-o`, `--output`        | `FILE`       | Output file.                                                                                             | `composed{ext}`                                                |
| `-ou`, `--output-unit`  | `{vox,mm}`   | Unit of the output displacement field: voxels of the target grid (any value starting with `v`) or millimetres (any other value). Ignored for a matrix. | `vox`                                                          |
| `-cpu`, `--cpu`         |              | Use the CPU.                                                                                            | CPU                                                            |
| `-gpu`, `--gpu`         | `[ID]`       | Use a GPU, optionally the one with this index (see [hardware](../../conventions.md#hardware)).         | off                                                            |

## Examples

Combine two rigid registrations into one matrix. The matrix in `b.lta` is applied to a point first, then the one in `a.lta`:

```shell
nitorch compose -l a.lta -l b.lta -o a_b.lta
```

Invert an affine matrix:

```shell
nitorch compose -il affine.lta -o affine_inverse.lta
```

Compute the inverse of a diffeomorphism from its velocity field. The field is defined on the grid of the velocity file, and written in voxel units:

```shell
nitorch compose -iv velocity.nii.gz -o inverse_disp.nii.gz
```

If `velocity.nii.gz` was written by `register` with its `velocity.json` next to it, the commands above integrate the velocity by geodesic shooting. To use other shooting parameters, give a JSON file right after the velocity file. To compute the displacement to the midpoint of the geodesic instead of the full transformation, use `-v2`:

```shell
nitorch compose -v2 velocity.nii.gz params.json -o half_disp.nii.gz
```

Merge an affine and a non-linear registration into a single displacement field on the fixed grid, expressed in millimetres. The velocity field is applied to points of the grid first, then the inverse affine:

```shell
nitorch compose -il affine.lta -v velocity.nii.gz -t fix.nii.gz -ou mm -o mov_to_fix_mm.nii.gz
```

Write the same transformation in voxel units, and use it to resample the moving image. Resampling with the single field is equivalent to resampling through the chain, and is faster when many images share the transformation:

```shell
nitorch compose -il affine.lta -v velocity.nii.gz -t fix.nii.gz -o mov_to_fix.nii.gz
nitorch reslice mov.nii.gz -d mov_to_fix.nii.gz -t fix.nii.gz -o mov_in_fix.nii.gz
```

Express a matrix as a displacement field on the grid of an image, for use in a tool that only accepts fields:

```shell
nitorch compose -l affine.lta -t fix.nii.gz -o affine_as_disp.nii.gz
```

## See also

- [`reslice`](reslice.md) applies a chain of transformations to images and explains the conventions in full.
- [`vexp`](vexp.md) exponentiates a velocity field on its own.
- [Spatial transformations](../../conventions.md#spatial-transformations) in the CLI conventions.
