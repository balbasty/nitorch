---
icon: octicons/terminal-24
---

# distmap

Compute the distance transform of a mask or a label map: the distance from
each voxel of an object to the closest voxel outside it, or the other way
around. Use it to build distance or signed distance maps of a segmentation,
for example as a shape representation, as a weight map, or to grow a region by
a given margin.

=== ":octicons-terminal-24: Synopsis"

    ```shell
    nitorch distmap FILE... [-l LABEL... | -t THR] [-i] [-s]
                            [-m METRIC] [-u UNIT] [-o FILE]
                            [--cpu [N] | --gpu [ID]]
    ```

## Description

`distmap` first turns each input volume into a binary object. By default, the
object is made of all voxels whose value is above zero; `-t` changes this
threshold, and `-l` selects a list of labels instead. All other voxels make up
the background.

It then computes, for each voxel of the object, the distance to the closest
voxel of the background. Background voxels get a distance of zero. This is
the same convention as `scipy.ndimage.distance_transform_edt`. With `-i`, the
roles are swapped: each background voxel gets its distance to the closest
voxel of the object, and the object is zero.

With `-s`, both distances are computed and combined into a signed distance
map, positive inside the object and negative outside it. With `-s -i`, the
signs are reversed: negative inside, positive outside. Distances are measured
between voxel centres, so the voxels on either side of the boundary have a
distance of one voxel, not zero.

Only the first three dimensions of a file are spatial; any further dimension,
such as the channels of a 4D file, is treated as a set of independent volumes.

The output has the same grid as the input (same shape and orientation matrix)
and is always written as `float32`. If the object or the background is empty,
the distances it would have measured are infinite, and a warning is printed.

### Metric and units

`-m l2` (the default) computes the exact Euclidean distance; `-m l1` computes
the L1 ("taxicab" or "city block") distance, the length of the shortest path
that moves along the voxel axes.

With `-u mm` (the default), distances are in millimetres, using the voxel size
read from the orientation matrix of each file, so anisotropic voxels are taken
into account. With `-u vox`, distances are in voxels and the voxel size is
ignored.

Both metrics use the algorithm of Felzenszwalb & Huttenlocher, "Distance
Transforms of Sampled Functions", *Theory of Computing* (2012), and run on
the CPU or on a GPU.

## Arguments

| Name   | Description                                                                                                  |
| ------ | ------------------------------------------------------------------------------------------------------------ |
| `FILE` | One or more masks or label maps, in any [format NITorch reads](../../conventions.md#file-formats). They must come before the options. |

## Options

| Option                | Value              | Description                                                                                             | Default                           |
| --------------------- | ------------------ | ------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `-l`, `--labels`      | `LABEL...`         | Labels that make up the object. Takes precedence over `-t`.                                             | all voxels above the threshold    |
| `-t`, `--threshold`   | `THR`              | Voxels whose value is strictly above `THR` make up the object.                                          | `0`                               |
| `-i`, `--invert`      |                    | Swap object and background. [See above](#description).                                                  |                                   |
| `-s`, `--signed`      |                    | Signed distance: positive inside the object, negative outside it (reversed with `-i`).                  |                                   |
| `-m`, `--metric`      | `METRIC`           | `l2` (Euclidean) or `l1` (taxicab). [See above](#metric-and-units).                                     | `l2`                              |
| `-u`, `--unit`        | `UNIT`             | Unit of the distances: `mm` or `vox`. [See above](#metric-and-units).                                   | `mm`                              |
| `-o`, `--output`      | `FILE`             | Output file name or template. Only one value is accepted; it is reused for every input.                 | `{dir}{sep}{base}.distmap{ext}`   |
| `--cpu`               | `[N]`              | Compute on the CPU. `N` is the number of threads; 0 (the default) keeps the PyTorch default.            | CPU                               |
| `--gpu`               | `[ID]`             | Compute on a GPU. `ID` is the index of the device. Falls back silently to the CPU if CUDA is not available. |                               |
| `-h`, `--help`        |                    | Print the built-in help and exit.                                                                       |                                   |

The output is a template with the placeholders `{dir}`, `{base}`, `{ext}` and
`{sep}`; see [Output filenames](../../conventions.md#output-filenames) and,
for devices, [Hardware](../../conventions.md#hardware).

## Examples

Distance (in mm) from each voxel of a brain mask to the edge of the mask. The
result is `brain_mask.distmap.nii.gz`:

```shell
nitorch distmap brain_mask.nii.gz
```

Distance from every voxel outside the hippocampi (labels 17 and 53 of a
FreeSurfer `aseg`) to the closest hippocampal voxel:

```shell
nitorch distmap aseg.nii.gz -l 17 53 -i -o hippo_dist.nii.gz
```

Signed distance map of a probabilistic segmentation, thresholded at 0.5, in
voxels:

```shell
nitorch distmap c1T1w.nii.gz -t 0.5 -s -u vox -o c1T1w.sdf.nii.gz
```

## See also

- [`metrics`](metrics.md) — includes the Hausdorff distance between two
  segmentations.
- [`smooth`](smooth.md) — blur a volume.
