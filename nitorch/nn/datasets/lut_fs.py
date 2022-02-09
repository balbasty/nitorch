"""
FreeSurfer color table
"""
import os
from nitorch.core.optionals import try_import
wget = try_import('wget')
appdirs = try_import('appdirs')


def wget_check():
    if not wget:
        raise ImportError('wget needed to download dataset')
    return wget


_fs_lut_url = 'https://raw.githubusercontent.com/freesurfer/freesurfer/dev/freeview/resource/FreeSurferColorLUT.txt'
_datadir = appdirs.user_cache_dir('nitorch') if appdirs else '.'


def read_lut_fs(fname=None):
    """Read a freesurfer lookup table

    Parameters
    ----------
    fname : str, optional
        Path to the LUT file.
        By default, the full FreeSurfer LUT is used (FreeSurferColorLUT.txt)

    Returns
    -------
    lut : dict[int -> str]
        Map from (integer) ID to (string) label

    """
    if fname is None:
        fname = os.path.join(_datadir, 'FreeSurferColorLUT.txt')
        if not os.path.exists(fname):
            os.makedirs(_datadir, exist_ok=True)
            wget_check().download(_fs_lut_url, fname)
    if not os.path.exists(fname):
        raise ValueError('File does not exist (or could not be downloaded)')
    lut = {}
    with open(fname) as f:
        for line in f:
            if '#' in line:
                line = line[:line.index('#')]
            line = line.split()
            if not line:  # empty line
                continue
            if len(line) == 1:
                import warnings
                warnings.warn('Weird line: ' + str(line))
            lut[int(line[0])] = line[1]
    return lut


# full LUT
lut_fs = read_lut_fs()

# Subparts of the LUT
lut_fs_brain = {k: v for k, v in lut_fs.items() if k < 100 or 155 <= k <= 158 or k == 192}
lut_fs_abnormality = {k: v for k, v in lut_fs.items() if 100 <= k <= 117}
lut_fs_head = {k: v for k, v in lut_fs.items() if 118 <= k <= 154}
lut_fs_baby = {k: v for k, v in lut_fs.items() if 159 <= k <= 169 or k == 176}
lut_fs_brainstem = {k: v for k, v in lut_fs.items() if 170 <= k <= 179 and k != 176}
lut_fs_hippocampus = {k: v for k, v in lut_fs.items() if 193 <= k <= 246}
lut_fs_cc = {k: v for k, v in lut_fs.items() if 250 <= k <= 255}
lut_fs_extra = {k: v for k, v in lut_fs.items() if 257 <= k <= 266}  # head/eye/sinus
lut_fs_lymph = {k: v for k, v in lut_fs.items() if 331 <= k <= 359}
lut_fs_cortical = {k: v for k, v in lut_fs.items() if 400 <= k <= 439}
lut_fs_hippocampus_hires = {k: v for k, v in lut_fs.items() if 500 <= k <= 558}
lut_fs_cbm_suit = {k: v for k, v in lut_fs.items() if 601 <= k <= 691}
lut_fs_fsl = {k: v for k, v in lut_fs.items() if 701 <= k <= 703}
lut_fs_hypothalamus = {k: v for k, v in lut_fs.items() if 801 <= k <= 810}

# Below is the color table for the cortical labels of the seg volume
# created by mri_aparc2aseg in which the aseg cortex label is replaced
# by the labels in the aparc. It also supports wm labels that will
# eventually be created by mri_aparc2aseg. Otherwise, the aseg labels
# do not change from above. The cortical lables are the same as in
# colortable_desikan_killiany.txt, except that left hemisphere has
# 1000 added to the index and the right has 2000 added.  The label
# names are also prepended with ctx-lh or ctx-rh. The white matter
# labels are the same as in colortable_desikan_killiany.txt, except
# that left hemisphere has 3000 added to the index and the right has
# 4000 added. The label names are also prepended with wm-lh or wm-rh.
# Centrum semiovale is also labled with 5001 (left) and 5002 (right).
# Even further below are the color tables for aparc.a2005s and aparc.a2009s.
lut_fs_aparc2aseg = {k: v for k, v in lut_fs.items()
                     if 1000 <= k < 1100 or 2000 <= k < 2100
                     or 3000 <= k < 3100 or 4000 <= k < 4100}

# Below is the color table for a lobar parcellation obtained from running:
# mri_annotation2label --subject subject --hemi lh --lobesStrict lobes
# mri_annotation2label --subject subject --hemi rh --lobesStrict lobes
# mri_aparc2aseg --s subject --rip-unknown --volmask \
#   --o aparc.lobes.mgz --annot lobes \
#   --base-offset 300 [--base-offset must be last arg]
lut_fs_lobar_gm = {k: v for k, v in lut_fs.items()
                   if 1300 <= k < 1400 or 2300 <= k < 2400}

# Below is the color table for a lobar white matter parcellation
#  obtained from running:
# mri_annotation2label --subject subject --hemi lh --lobesStrict lobes
# mri_annotation2label --subject subject --hemi rh --lobesStrict lobes
# mri_aparc2aseg --s subject --labelwm --hypo-as-wm --rip-unknown \
#   --volmask --o wmparc.lobes.mgz --ctxseg aparc+aseg.mgz \
#   --annot lobes --base-offset 200 [--base-offset must be last ar
lut_fs_lobar_wm = {k: v for k, v in lut_fs.items()
                   if 3200 <= k < 3300 or 4200 <= k < 4300}

# Below is the color table for the cortical labels of the seg volume
# created by mri_aparc2aseg (with --a2005s flag) in which the aseg
# cortex label is replaced by the labels in the aparc.a2005s. The
# cortical labels are the same as in Simple_surface_labels2005.txt,
# except that left hemisphere has 1100 added to the index and the
# right has 2100 added.  The label names are also prepended with
# ctx-lh or ctx-rh.  The aparc.a2009s labels are further below
lut_fs_aparc2aseg2005 = {k: v for k, v in lut_fs.items()
                         if 1100 <= k < 5100 and not
                         (1300 <= k < 1400 or 2300 <= k < 2400 or
                          3200 <= k < 3300 or 4200 <= k < 4300)}

# uncommented -> wild guess
lut_fs_dmri_spinal = {k: v for k, v in lut_fs.items() if 6000 <= k < 7000}
lut_fs_nuclei = {k: v for k, v in lut_fs.items() if 7000 <= k < 8100}

# Below is the color table for white-matter pathways produced by dmri_paths
lut_fs_dmri_path = {k: v for k, v in lut_fs.items() if 5100 <= k < 6000}

# Labels for thalamus parcellation using histological atlas (Iglesias et al.)
lut_fs_thalamus_histo = {k: v for k, v in lut_fs.items() if 8000 <= k < 9000}

# Labels for thalamus parcellation using probabilistic tractography. See:
# Functional--Anatomical Validation and Individual Variation of Diffusion
# Tractography-based Segmentation of the Human Thalamus; Cerebral Cortex
# January 2005;15:31--39, doi:10.1093/cercor/bhh105, Advance Access
# publication July 6, 2004
lut_fs_thalamus_heidi = {k: v for k, v in lut_fs.items() if 8000 <= k < 9000}

# Below is the color table for the cortical labels of the seg volume
# created by mri_aparc2aseg (with --a2009s flag) in which the aseg
# cortex label is replaced by the labels in the aparc.a2009s. The
# cortical labels are the same as in Simple_surface_labels2009.txt,
# except that left hemisphere has 11100 added to the index and the
# right has 12100 added.  The label names are also prepended with
# ctx_lh_, ctx_rh_, wm_lh_ and wm_rh_ (note usage of _ instead of -
# to differentiate from a2005s labels).
lut_fs_aparc2aseg2009 = {k: v for k, v in lut_fs.items() if 11100 <= k < 15100}


def _make_lut_synthseg():
    sub_indices = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26,
                   28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]
    return {k: lut_fs[k] for k in sub_indices}


lut_synthseg = _make_lut_synthseg()


lut_fs35 = {  # used in OASIS preprocessed (learn2reg 2021)
    0:     'Unknown',
    1:     'Left-Cerebral-White-Matter',
    2:     'Left-Cerebral-Cortex',
    3:     'Left-Lateral-Ventricle',
    4:     'Left-Inf-Lat-Ventricle',
    5:     'Left-Cerebellum-White-Matter',
    6:     'Left-Cerebellum-Cortex',
    7:     'Left-Thalamus',
    8:     'Left-Caudate',
    9:     'Left-Putamen',
    10:    'Left-Pallidum',
    11:    '3rd-Ventricle',
    12:    '4th-Ventricle',
    13:    'Brain-Stem',
    14:    'Left-Hippocampus',
    15:    'Left-Amygdala',
    16:    'Left-Accumbens',
    17:    'Left-Ventral-DC',
    18:    'Left-Vessel',
    19:    'Left-Choroid-Plexus',
    20:    'Right-Cerebral-White-Matter',
    21:    'Right-Cerebral-Cortex',
    22:    'Right-Lateral-Ventricle',
    23:    'Right-Inf-Lat-Ventricle',
    24:    'Right-Cerebellum-White-Matter',
    25:    'Right-Cerebellum-Cortex',
    26:    'Right-Thalamus',
    27:    'Right-Caudate',
    28:    'Right-Putamen',
    29:    'Right-Pallidum',
    30:    'Right-Hippocampus',
    31:    'Right-Amygdala',
    32:    'Right-Accumbens',
    33:    'Right-Ventral-DC',
    34:    'Right-Vessel',
    35:    'Right-Choroid-Plexus',
}


lut_fs24 = {  # used in OASIS preprocessed (learn2reg 2021)
    0:     'Unknown',
    1:     'Left-Cerebral-White-Matter',
    2:     'Left-Cerebral-Cortex',
    3:     'Left-Lateral-Ventricle',
    4:     'Left-Inf-Lat-Ventricle',
    5:     'Left-Thalamus',
    6:     'Left-Caudate',
    7:     'Left-Putamen',
    8:     'Left-Pallidum',
    9:     '3rd-Ventricle',
    10:    'Brain-Stem',
    11:    'Left-Hippocampus',
    12:    'Left-Ventral-DC',
    13:    'Left-Choroid-Plexus',
    14:    'Right-Cerebral-White-Matter',
    15:    'Right-Cerebral-Cortex',
    16:    'Right-Lateral-Ventricle',
    17:    'Right-Inf-Lat-Ventricle',
    18:    'Right-Thalamus',
    19:    'Right-Caudate',
    20:    'Right-Putamen',
    21:    'Right-Pallidum',
    22:    'Right-Hippocampus',
    23:    'Right-Ventral-DC',
    24:    'Right-Choroid-Plexus',
}


def _make_fs35_hierarchy(fs35=dict(lut_fs35)):
    hierarchy = dict()
    hierarchy['Left'] = [v for v in fs35.values() if v.startswith('Left')]
    hierarchy['Right'] = [v for v in fs35.values() if v.startswith('Right')]
    hierarchy['Central'] = [v for v in fs35.values() if not v.startswith('Left')
                            and not v.startswith('Right')]
    for v in fs35.values():
        if v.startswith('Left'):
            basename = v[5:]
            hierarchy[basename] = ['Left-' + basename, 'Right-' + basename]

    hierarchy['CSF'] = ['Lateral-Ventricle', 'Inf-Lat-Ventricle',
                        '3rd-Ventricle', '4th-Ventricle']
    hierarchy['Dorsal-Striatum'] = ['Caudate', 'Putamen']
    hierarchy['Ventral-Striatum'] = ['Accumbens']
    hierarchy['Striatum'] = ['Dorsal-Striatum', 'Ventral-Striatum']
    hierarchy['Basal-Ganglia'] = ['Striatum', 'Pallidum', 'Ventral-DC']
    hierarchy['Sub-Cortical'] = ['Basal-Ganglia', 'Thalamus',
                                 'Hippocampus', 'Amygdala']
    hierarchy['Cerebellum'] = ['Cerebellum-Cortex', 'Cerebellum-White-Matter ']
    return hierarchy


fs35_hierarchy = _make_fs35_hierarchy()
fs24_hierarchy = _make_fs35_hierarchy(lut_fs24)
synthseg_hierarchy = _make_fs35_hierarchy(lut_synthseg)
