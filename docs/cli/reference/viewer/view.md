---
icon: octicons/terminal-24
---

# view

Open one or more volumes in an interactive window with three orthogonal views.
Use it for a quick look at an image, or to compare several images in the same
coordinate system.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch view FILE...
    ```

## Description

`view` opens a Matplotlib window and blocks until you close it. Each volume gets
its own block of three slice views, the axial, sagittal and coronal planes through a
common cursor, drawn as a red cross-hair. With the default layout the axial view is
at the top left, the sagittal view at the top right and the coronal view at the
bottom left. When you give several files, the blocks are arranged in a grid whose
shape is chosen to fill the window. The command has no options besides `-h`; all
the display settings are fixed, and everything else is done with the mouse.

### Common space

All images are shown in the same *visualisation space*, which is world space
(RAS: right, anterior and superior) sampled on a regular grid. The voxel size of that
grid is the smallest voxel size found in any of the images, so that no image is
shown at a lower resolution than it was acquired. Each image is resampled into this
space with linear interpolation, using its own orientation matrix. This has two
consequences:

- images with different orientations, voxel sizes or fields of view line up
  anatomically, provided that their orientation matrices are correct, which makes
  `view` convenient to check a registration or a reslicing result;
- the cursor and the field of view are shared by all blocks. Moving the cursor
  in one image moves it, at the same world position, in all the others.

The field of view starts as the bounding box of all the images. The cursor starts at
its centre.

### Intensities

Images are shown in grey levels. For each image, the intensities of the three
slices on display are clipped between their 5th and 95th percentiles and rescaled
to the display range. The percentiles are recomputed from the slices on display, so
the contrast adapts as you move the cursor or zoom. There is no option to fix the window or to change the
colour map from the command line.

### Mouse controls

| Action                                          | Effect                                                               |
| ----------------------------------------------- | -------------------------------------------------------------------- |
| Left click or drag in any view                  | Move the cursor. The other views follow.                             |
| Right-button drag, or Shift + left-button drag  | Pan: move the field of view.                                         |
| Scroll wheel                                    | Zoom about the position of the pointer. Scrolling up zooms in and scrolling down zooms out, by a factor of 1.5 per step. |

The display is redrawn at most 25 times per second; when the pointer moves faster
than that, intermediate positions are skipped.

## Arguments

| Name   | Description                                                                                                                               |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `FILE` | One or more image files, in any [format NITorch reads](../../conventions.md#file-formats). They are loaded fully into memory. |

## Options

| Option         | Description                       |
| -------------- | --------------------------------- |
| `-h`, `--help` | Print the built-in help and exit. |

## Requirements

`view` needs [Matplotlib](https://matplotlib.org), which is an optional
dependency of NITorch. If it is missing, the command stops with
`[ERROR] Matplotlib not available`. Install it with `pip install matplotlib`, or
install NITorch with its `plot` extra (`pip install "nitorch[plot]"`).
It also needs a Matplotlib backend that can open windows. On a remote machine
you will need a way to display graphics, such as X forwarding, or you must select
an interactive backend yourself.

The viewer is meant for 3D volumes with a single value per voxel. Images with an
extra channel or time dimension are not specifically supported.

## Examples

Look at a single volume:

```shell
nitorch view T1w.nii.gz
```

Compare two images that should be aligned, such as a scan and its
registered, resliced counterpart:

```shell
nitorch view fixed.nii.gz moving_to_fixed.nii.gz
```

View an anatomical scan next to the corresponding label map:

```shell
nitorch view sub-01_T1w.nii.gz sub-01_seg.nii.gz
```

## See also

- [`info`](../images/info.md) — print the shape, voxel size and orientation of an image.
- [`reslice`](../spatial/reslice.md) — resample an image onto the grid of another.
