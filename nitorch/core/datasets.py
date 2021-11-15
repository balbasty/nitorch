"""For querying NITorch data."""
import os
from tempfile import gettempdir
from .optionals import try_import
wget = try_import('wget')
appdirs = try_import('appdirs')

# Download NITorch data to OS cache
cache_dir = appdirs.user_cache_dir('nitorch') if appdirs else gettempdir()

# NITorch data dictionary.
# Keys are data names, values are list of FigShare URL and filename.
data = {}
data['atlas_t1'] = ['https://ndownloader.figshare.com/files/25595000', 'mb_avg218T1.nii.gz']
data['atlas_t2'] = ['https://ndownloader.figshare.com/files/25595003', 'mb_avg218T2.nii.gz']
data['atlas_pd'] = ['https://ndownloader.figshare.com/files/25594997', 'mb_avg218PD.nii.gz']
data['atlas_t1_mni'] = ['https://ndownloader.figshare.com/files/25438340', 'mb_mni_avg218T1.nii.gz']
data['atlas_t2_mni'] = ['https://ndownloader.figshare.com/files/25438343', 'mb_mni_avg218T2.nii.gz']
data['atlas_pd_mni'] = ['https://ndownloader.figshare.com/files/25438337', 'mb_mni_avg218PD.nii.gz']


def fetch_data(name, dir_download=None, speak=False):
    """Get nitorch package data.

    Parameters
    ----------
    name : str
        Name of nitorch data, available are:
        * atlas_t1: MRI T1w intensity atlas, 1 mm resolution.
        * atlas_t2: MRI T2w intensity atlas, 1 mm resolution.
        * atlas_pd: MRI PDw intensity atlas, 1 mm resolution.
        * atlas_t1_mni: MRI T1w intensity atlas, in MNI space, 1 mm resolution.
        * atlas_t2_mni: MRI T2w intensity atlas, in MNI space, 1 mm resolution.
        * atlas_pd_mni: MRI PDw intensity atlas, in MNI space, 1 mm resolution.
    dir_download : str, optional
        Directory where to download data. Uses OS cache folder by default.
    speak : bool, default=False
        Print download progress.
    Returns
    ----------
    pth_data : str
        Absolute path to requested nitorch data.
    """
    dir_download = dir_download or cache_dir or '.'
    fname = data[name][1]
    pth_data = os.path.join(dir_download, fname)
    if not os.path.exists(pth_data):        
        if not os.path.exists(dir_download):
            os.makedirs(dir_download, exist_ok=True)        
        url = data[name][0]        
        pth_data = download(url, fname=pth_data, verbose=speak)

    return pth_data


def bar(url, fname):
    def _bar(current, total, width=80):
        print("Downloading %s to %s | Progress: %d%% (%d/%d bytes)"
              % (url, fname, current / total * 100, current, total))
    return _bar


def download(url, fname=None, verbose=False):
    """Download data from URL."""
    if wget is None:
        raise ImportError('wget not available')
    my_bar = bar(url, fname) if verbose else None
    pth_data = wget.download(url, fname, bar=my_bar)
    
    return pth_data
