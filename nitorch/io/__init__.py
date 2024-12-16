"""
Reading and writing arrays
==========================

NITorch implements readers and writers for a number of neuroimaging
formats as well as more classical imaging formats. Under the hood, it
uses widely used packages such as nibabel. However, instead of exposing
these packages directly, it defines a unique abstraction layer that
allows the user to only manipulate a finite set of methods.

Quick load
----------

The first way to load data is to simply use the methods `load` and
`loadf`:
```python
>>> from nitorch import io
>>> dat = io.load('path/to/my/nifti_file.nii.gz')
>>> type(dat), dat.dtype
(torch.Tensor, torch.int16)
>>> datf = io.loadf('path/to/my/nifti_file.nii.gz')
>>> type(dat)
(torch.Tensor, torch.float32)
```
The difference between `load` and `loadf` is that the former loads the
raw data, while the latter applies any intensity transform that may be
stored in the file and converts the array to a floating point type.
Both functions accept arguments `dtype` (to specify the returned data
type), `device` (to either load the array on CPU or GPU) and `numpy`
(to return a `numpy` array rather than a `torch` tensor).

If the user wishes to extract metadata as well, the argument `attributes`
must be used to list fields of interest. We use a generic dictionary
of attributes to hide as much as possible implementation-specific
wording. The list of generic attributes can be found in
`nitorch.io.metadata.keys`:
```python
>>> from nitorch.io.metadata import keys, doc
>>> keys
['format',
 'affine',
 'dtype',
 'voxel_size',
 'intent',
 'slope',
 'inter',
 'slice_order',
 'time_step',
 'time_offset',
 'tr',
 'te',
 'ti',
 'fa']
>>> doc('affine')
Affine orientation matrix.

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
```

Furthermore, smart preprocessing can be applied. When the data is stored
using an integer data type, there is uncertainty about the original value
of a voxel (_e.g._, all floating point values in the range [3, 4) map to
the integer value 3). If the option `rand=True` is used, values are
sampled uniformly in this range after being loaded.

It is also possible to clip values that outside of a range defined by
percentiles, using the option `clip=(lower, upper)`. This allows outlier
values to be discarded even before the data is used. We typically use
this scheme before performing registration. Values `lower` and `upper`
should lie in the range [0, 1].

File mapping
------------

When memory is scarse, it is possible that frequent reads into the same
file are required, or that only part of the data must be loaded in
memory. To this purppose, NIBabel (which was a massive inspiration for
our abstraction layer) returns a `SpatialImage` object on `load` which
does not load the array in memory (yet) but can be symbolically `sliced`.
When the actual loading function is called on a sliced object (`get_fdata`),
only the data that is covered by the slice is loaded, saving time and
memory. We implement an almost identical protocol, except that it is
only exposed to the user when the function `map` is used:
```python
>>> file = io.map('path/to/my/nifti_file.nii.gz')
>>> file
BabelArray(shape=(256, 192, 128), dtype=int16)
```
Objects returned by `map` are all instances of `nitorch.io.MappedArray`.
These objects can be sliced, permuted, (un)squeezed, splitted, etc.
```python
>>> file[:, :, 0]
BabelArray(shape=(256, 192), dtype=int16)
>>> file[:, :, None, :]
BabelArray(shape=(256, 192, 1, 128), dtype=int16)
>>> file.permute([1, 2, 0])
BabelArray(shape=(192, 128, 256), dtype=int16)
>>> file.unsqueeze(-1)
BabelArray(shape=(256, 192, 128, 1), dtype=int16)
>>> file[:, :, :3].unbind(dim=-1)
[BabelArray(shape=(260, 311), dtype=float32),
 BabelArray(shape=(260, 311), dtype=float32),
 BabelArray(shape=(260, 311), dtype=float32)]
```

It is also possible to symbolically concatenate several mapped arrays:
```python
>>> file1 = io.map('path/to/my/nifti_file.nii.gz')
>>> file1
BabelArray(shape=(256, 192, 128), dtype=int16)
>>> file2 = io.map('path/to/my/other/nifti_file.nii.gz')
>>> file2
BabelArray(shape=(256, 192, 64), dtype=float32)
>>> file_cat = io.cat((file1, file2), dim=-1)
CatArray(shape=(256, 192, 192), dtype=(float32, float32))
```

Methods `data` and `fdata` correspond to functions `load` and `loadf`
seen previously. However, only the subarray described by the current
slice is effectively loaded in memory.
```python
>>> dat = file[:, :, 0].data()
>>> type(dat)
torch.Tensor
>>> dat.shape
torch.Size([256, 192])
```

Partial writing
---------------

Conversely to NIBabel, NITorch allows data to be partially _written_ to disk:
```python
>>> import torch
>>> slice = file[:, :, 0]
>>> new_slice = torch.randint(0, 5000, slice.shape)
>>> slice.set_data(new_slice)
```
However, not all formats implement this feature. Currently, this has been
tested for nifti and MGH formats. Note that if the file is compressed
(_e.g._, gziped nifti), true partial writing is not possible and the
snippet above would effectively trigger a full load of the file to memory
followed by a full write to disk.

Creating a new file
-------------------

To write data into a new file, the functions `save` and `savef` should be
used. Like with `load` and `loadf`, the difference between `save` and
`savef` is linked to the (inverse) scaling that should be applied before
writing the data to disk. If `save` is used, the input array is written
as raw data (even though a slope and intercept can be defined). If `savef`
is used, the data is unscaled before being written as raw data to disk.
```python
>>> from nitorch.io import save, savef
>>> save(my_tensor, 'path/to/my/new/nifti_file.nii.gz')
>>> savef(my_float_tensor, 'path/to/my/new/nifti_file.nii.gz')
```
Generic metadata (as defined in `nitorch.io.metadata`) can be provided
as additional keyword arguments:
```python
>>> save(my_tensor, 'path/to/my/new/nifti_file.nii.gz',
>>>      affine=my_affine, dtype='int32')
```
Furthermore, the path to a template file -- or a `MappedArray` instance
-- can be provided as well and used to guess non-provided metadata.
It will also be used to choose the output file format, unless it can be
guessed from the file extension (e.g., '.nii.gz').
```python
>>> save(my_tensor, 'path/to/my/new/nifti_file.nii.gz',
>>>      like='path/to/my/old/nifti_file.nii.gz',
>>>      affine=my_affine, dtype='int32')
```
Finally, a `MappedArray` instance, rather than a tensor, can hold the
data to write. In this case, the instance attributes will define the
output metadata (affine, etc) unless they are overridden by
keyword attributes.
```python
>>> file = map('path/to/my/nifti_file.nii.gz')
>>> slice = file[:, :, 0]
>>> save(slice, 'path/to/my/new/nifti_file.nii.gz')
```

"""

from . import loadsave
from . import volumes
from . import metadata
from . import optionals
from . import readers
from . import streamlines
from . import transforms
from . import utils
from . import writers

from .volumes import MappedArray, CatArray, cat, stack
from .transforms import MappedAffine
from .streamlines import MappedStreamlines
from .loadsave import map, load, loadf, save, savef
