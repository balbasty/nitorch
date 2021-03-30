from nitorch.plot import ImageViewer
from nitorch.core.optionals import try_import
plt = try_import('matplotlib.pyplot', _as=True)

def view(files):
    """Interactive viewer for volumetric images.

    Parameters
    ----------
    files : list[str]
        Inputs images.

    """
    if plt is None:
        raise ImportError('Matplotlib not available')

    ImageViewer(files)
    plt.show(block=True)