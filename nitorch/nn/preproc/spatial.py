from nitorch import spatial
from ..base import Module


class Reorient(Module):
    def __init__(self, target='RAS'):
        super().__init__()
        self.target = target

    def forward(self, affine, *images):
        new_images = []
        for image in images:
            _, image = spatial.affine_reorient(affine, image, self.target)
            new_images.append(image)
        return new_images[0] if len(images) == 1 else tuple(new_images)


