"""Generic metadata"""

keys = [
    'format',        # name of the file format
    'affine',        # orientation matrix (tensor)
    'dtype',         # numpy data type
    'voxel_size',    # voxel size, in voxel_size_unit
    'voxel_size_unit',  # voxel size unit, default=mm
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


def doc(key):
    """Print information on a metadata attribute."""
    key = key.lower()
    if key == 'format':
        _doc = 'Name of the file format'
    elif key == 'affine':
        _doc = """Affine orientation matrix. 
        
This matrix is used to map voxel coordinates to "world" 
or "lab" coordinates, generally expressed in millimeters.

An affine matrix possess a linear component `L` and a 
translation `t`, such that a point is transformed using 
the relation `new_p = L @ p + t`.

In order to express an affine transform as a matrix-vector 
product, affine spaces use homogeneous coordinates,
which have an additional 1 appended. E.g., a 3D point 
would be written `p = [x, y, z, 1]`.

Similarly, the last row of an affine matrix is made of 
zeros and a one:
`A = [[l11, l12, l13, t1],
      [l21, l22, l23, t2],
      [l31, l32, l33, t3],
      [0,   0,   0,   1]]`
such that `new_p = A @ p`.
"""
    elif key == 'dtype':
        _doc = """On-disk data type. We encode it using numpy dtypes 
as they have much more flexibility than torch dtypes
or even string representations."""
    elif key == 'voxel_size':
        _doc = 'The size of the voxels that encode spatial dimensions, in mm.'
    elif key == 'intent':
        _doc = 'The intent of the data stored in the file.'
    elif key == 'slope':
        _doc = '''A multiplicative factor that can be applied to the raw
values stored on disk to map to a quantitative unit.'''
    elif key == 'inter':
        _doc = '''An additive shift that can be applied to the raw
values stored on disk to map to a quantitative unit.'''
    elif key == 'tr':
        _doc = '''The repetition time, which is the time between two 
pulses in MRI. In sec.'''
    elif key == 'te':
        _doc = '''The echo time, which is the time between two echoes
in MRI. In sec.'''
    elif key == 'ti':
        _doc = '''The inversion time, which is the time between the 
excitation and inversion pulses in MRI. In sec.'''
    elif key == 'fa':
        _doc = '''The nominal flip angle, which is the angle by which the 
effective magnetization is tipped in MRI. In deg.'''
    else:
        raise ValueError('Unknown attribute {}'.format(key))
    print(_doc)
