from .gui import ImageViewer

def view(files):
    """Interactive viewer for volumetric images.

    Parameters
    ----------
    files : list[str]
        Inputs images.

    """
    ImageViewer(files)