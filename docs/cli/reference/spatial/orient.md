---
icon: octicons/terminal-24
---

# orient

Overwrite the orientation matrix stored in the header of a volume, leaving the voxel data untouched. Use it to repair a file whose header is wrong, or to give an image the orientation, voxel size or field-of-view centre of another one.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch orient FILE... [-a AFFINE...] [-l LAYOUT] [-v VX...] [-c CENTRE...]
                           [-k FILE] [-o FILE...] [-t [FILE...]]
    ```

!!! warning "This overwrites the orientation matrix"
    `orient` assumes that the orientation matrix in the input file is **incorrect**. The original voxel-to-world mapping is replaced, not preserved, so the image ends up at a different place in world space. If the header is correct and you only want to change how the data are stored on disk, use [`reorient`](reorient.md) instead.

## Description

A volume file stores an array of voxels and an *orientation matrix* (also called the vox2ras matrix): a $4 \times 4$ affine matrix that maps voxel indices to world (RAS) coordinates. `orient` computes a new orientation matrix, writes it into a copy of the file, and copies the data array as it is. No voxel is moved, resampled or re-typed: the output has the same shape, the same data and the same data type as the input.

The new matrix is built in four steps, and each step is controlled by one option:

1. **Base matrix** (`-a`). The starting point. By default, the orientation matrix of the reference image (see `-k`), which is the input itself when no reference is given.
2. **Layout** (`-l`). The direction in which each voxel axis runs through world space, written as a permutation of three letters. Each letter says towards which direction the voxel index *increases*: `R` for left to Right, `L` for right to Left, `A` for posterior to Anterior, `P` for anterior to Posterior, `S` for inferior to Superior and `I` for superior to Inferior. `RAS` means that the first array axis runs towards the right, the second towards the front and the third towards the top. `LAS`, `LPS`, `RSA`... are other valid layouts.
3. **Voxel size** (`-v`), in millimetres. The values are given in the order of the array axes.
4. **Centre** (`-c`). The world coordinates, in millimetres, of the centre of the field of view.

Steps 2 to 4 are always applied to the matrix chosen in step 1. Because they all default to `like`, the layout, voxel size and centre of the reference image are imposed on a matrix given with `-a`, unless you override them. The final matrix therefore always has the layout, voxel size and centre determined by `-l`, `-v` and `-c`.

Each of `-a`, `-l`, `-v` and `-c` accepts either explicit values or one of three keywords:

| Keyword    | Meaning                                                                                      |
| ---------- | -------------------------------------------------------------------------------------------- |
| `self`     | Take the value from the input file.                                                           |
| `like`     | Take the value from the reference file given with `-k` (the input file if there is none). This is the default for all four options. |
| `standard` | Use a standard value: identity matrix for `-a`, `RAS` for `-l`, 1 mm for `-v`, 0 for `-c`.   |

With no `-k`, `like` and `self` are the same thing, so that `nitorch orient FILE -l RAS` changes the layout only and keeps the voxel size and centre of `FILE`.

### The transform file

With `-t`, `orient` also writes a linear transformation file (`.lta`) for each output. It contains the matrix $M_\text{new} M_\text{old}^{-1}$, that is, the transformation that takes world coordinates under the old orientation matrix to world coordinates under the new one. It is labelled as a world-to-world (RAS-to-RAS) affine, so FreeSurfer tools can read it. It lets you track how far the image was moved. Give `-t` alone to write it under the default name, or follow it with filenames to choose where it is written. Without `-t`, no transformation file is written.

## Arguments

| Name   | Description                                                                |
| ------ | -------------------------------------------------------------------------- |
| `FILE` | One or more input volumes. Any format NITorch reads and writes (see [file formats](../../conventions.md#file-formats)). |

## Options

| Option                | Value             | Description                                                                                                                                         | Default                                       |
| --------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| `-a`, `--affine`      | `AFFINE...`       | Base orientation matrix: `self`, `like`, `standard`, or 12 numbers (the first three rows, row by row) or 16 numbers (all four rows, row by row).    | `like`                                        |
| `-l`, `--layout`      | `LAYOUT`          | Target layout, such as `RAS` or `LPS`, or `self`, `like`, `standard` (= `RAS`).                                                                     | `like`                                        |
| `-v`, `--voxel-size`  | `VX...`           | Target voxel size in mm: one value for an isotropic voxel, or one per array axis; or `self`, `like`, `standard` (= 1).                              | `like`                                        |
| `-c`, `--center`      | `CENTRE...`       | World coordinates (mm) of the centre of the field of view: one value, or one per axis; or `self`, `like`, `standard` (= 0). Note the American spelling of the flag. | `like`                                        |
| `-k`, `--like`        | `FILE`            | Reference volume that provides every value set to `like`.                                                                                            | the input file                                |
| `-o`, `--output`      | `FILE...`         | Output filenames, one per input. [Templates](../../conventions.md#output-filenames) are accepted; `{layout}` expands to the name of the output layout, e.g. `RAS`. | `{dir}{sep}{base}.{layout}{ext}`              |
| `-t`, `--transform`   | `[FILE...]`       | Write the transformation files, one per input. The templates accept `{dir}`, `{sep}`, `{base}` and `{layout}`, but not `{ext}`. Without a value, the default name is used. | not written; `{dir}{sep}{base}_to_{layout}.lta` if `-t` has no value |

If fewer output names than inputs are given, the last one is repeated (and would be overwritten), so give a template when you process several files. Negative numbers are accepted as values; see [how options are parsed](../../conventions.md#how-options-are-parsed).

## Examples

Give an image with a corrupt header a clean RAS orientation, 2 mm isotropic voxels and a field of view centred on the world origin:

```shell
nitorch orient broken.nii.gz -a standard -l RAS -v 2 -c 0 -o fixed.nii.gz
```

Change only the voxel size, because the scanner header reports 1 mm slices where the true thickness is 3 mm. The layout, orientation and field-of-view centre of the input are kept:

```shell
nitorch orient sub-01_bold.nii.gz -v 3.5 3.5 3 -o sub-01_bold_vx.nii.gz
```

Declare that an image is stored in LPS order (the first two axes run towards the left and the back), without touching the data:

```shell
nitorch orient sub-01_T1w.nii.gz -l LPS -o sub-01_T1w_lps.nii.gz
```

Copy the whole orientation (layout, voxel size, rotation and field-of-view centre) of one image onto another that is known to be on the same voxel grid but has lost its header:

```shell
nitorch orient mask.nii.gz -k T1w.nii.gz -o mask_oriented.nii.gz
```

Process several files with a template and choose where the transformation files go:

```shell
nitorch orient sub-01_T2w.nii.gz sub-02_T2w.nii.gz -l RAS \
    -o '{dir}{sep}{base}.fixed{ext}' -t '{dir}{sep}{base}_header.lta'
```

## See also

- [`reorient`](reorient.md) permutes and flips the data and keeps the world mapping.
- [`info`](../images/info.md) prints the orientation matrix and layout of a file.
- [`resize`](resize.md) and [`reslice`](reslice.md) resample the data onto a new grid.
