from nitorch import spatial
from . import objects


class Whole(objects.Image, objects.AffineModel):

    def slice_to(self, stack, cache_result=False, recompute=True):
        aff = self.exp(cache_result=cache_result, recompute=recompute)
        if recompute or not hasattr(self, '_sliced'):
            aff = spatial.affine_matmul(aff, self.affine)
            aff_reorient = spatial.affine_reorient(self.affine, self.shape, stack.layout)
            aff = spatial.affine_lmdiv(aff_reorient, aff)
            aff = spatial.affine_grid(aff, self.shape)
            sliced = spatial.grid_pull(self.dat, aff, bound=self.bound,
                                       extrapolate=self.extrapolate)
            fwhm = [0] * self.dim
            fwhm[-1] = stack.slice_width / spatial.voxel_size(aff_reorient)[-1]
            sliced = spatial.smooth(sliced, fwhm, dim=self.dim, bound=self.bound)
            slices = []
            for stack_slice in stack.slices:
                aff = spatial.affine_matmul(stack.affine, )
                aff = spatial.affine_lmdiv(aff_reorient, )
        if cache_result:
            self._sliced = sliced
        return sliced



class Slice(objects.Image):

    def __init__(self, *args, thickness=None, **kwargs):
        super().__init__(*args, **kwargs)


class SliceStack:

    def __init__(self, slices):
        self.slices = slices

    def __iter__(self):
        for slice in self.slices:
            yield slice