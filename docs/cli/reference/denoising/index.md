---
icon: octicons/sparkle-fill-24
---

# Denoising & segmentation

These commands clean up and segment MR images. Two of them remove noise from
an image with a total-variation prior, which smooths regions while keeping their
edges; the third segments brain tissue with a model that also corrects the bias
field of the scanner and aligns the image to a template.

Several contrasts of the same subject can be denoised together so that they
share their edges. The default template of the segmentation is a brain atlas,
so it expects a head image; it can combine several contrasts.

| Command | Summary |
| ------- | ------- |
| [`denoise`](denoise.md) | Total-variation denoising of one or several images, with the noise level and the regularisation under your control. |
| [`denoise_mri`](denoise_mri.md) | Joint total-variation denoising of co-registered MR sequences, with noise level and regularisation estimated from the images. |
| [`uniseg`](uniseg.md) | Unified Segmentation, as in SPM12: tissue classes, bias field and template alignment, estimated together. |

## Choosing between the two denoisers

Use [`denoise`](denoise.md) when you want to choose the noise level and the
amount of smoothing yourself, or when your images are not MR volumes. Use
[`denoise_mri`](denoise_mri.md) when you have MR sequences and prefer
parameters derived from the image intensities; note that it assumes a GPU unless
you pass `-cpu`. The page of `denoise_mri` compares the two in detail.

## See also

- [Registration](../registration/index.md): bring several contrasts into the
  same space before denoising them jointly.
- [Images](../images/index.md): smoothing, inpainting and other image
  operations.
