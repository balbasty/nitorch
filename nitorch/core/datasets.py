"""For querying NITorch data."""
import os
import wget
import pathlib
import appdirs


# Download NITorch data to OS cache
dir_os_cache = pathlib.Path(appdirs.user_cache_dir('nitorch'))

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
    '''Get nitorch package data.

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

    '''
    if dir_download is None:
        dir_download = dir_os_cache
    pth_data = os.path.join(dir_download, data[name][1])
    if not os.path.exists(pth_data):
        _downloader(name, dir_download=dir_download, speak=speak)

    return pth_data


def _downloader(name, dir_download=None, speak=False):
    '''Download NITorch data.
    '''
    if not os.path.exists(dir_download):
        os.makedirs(dir_download, exist_ok=True)
    # Get download options
    url = data[name][0]
    fname = data[name][1]
    pth_data = os.path.join(dir_download, fname)
    bar = None
    if speak:
        def bar(current, total, width=80):
            print("Downloading %s to %s. Progress: %d%% (%d/%d bytes)" % (fname, dir_download, current / total * 100, current, total))
    if not os.path.exists(pth_data):
        # Download data
        wget.download(url, pth_data, bar=bar)
