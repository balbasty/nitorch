"""Generic metadata"""

keys = [
    'format',        # name of the file format
    'affine',        # orientation matrix (tensor)
    'dtype',         # numpy data type
    'voxel_size',    # voxel size, in mm
    'intent',        # nii: type of data/maps stored in the file
    'slope',         # nii: intensity transform
    'inter',         # nii: intensity transform
    'slice_order',   # nii: slice acquisition order
    'time_step',     # nii: time between volumes in the time direction (sec)
    'time_offset',   # nii: constant time shift applied to all volumes (sec)
    'tr',            # mgh: repetition time, in seconds
    'te',            # mgh: echo time, in second
    'ti',            # mgh: inversion time, in second
    'fa',            # mgh: flip angle, in degrees
]
