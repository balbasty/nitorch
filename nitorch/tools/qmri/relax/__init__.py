"""Fitting of MR relaxometry models

Methods
-------
estatics
    Estimate the apparent relaxation time R2* from multi-echo GRE images.
    Also generate GRE images free of any R2* bias ("GRE intercepts").
vfa
    Estimate the (R2*-biased) proton density A, longitudinal relaxation
    rate R1 and magnetisation transfer saturation MTsat from variable
    flip angle GRE images.
    If "GRE-intercepts" are provided instead of single-echo GRE images,
    the proton-density is bias-free.
greeq
    Estimate jointly the apparent relaxation time R2*, the longitudinal
    relaxation rate R1, the magnetisation transfer saturation MTsat and
    the unbiased proton density PD from variable flip angle multi-echo
    GRE images.
gre
    Simulate Gradient-Echo images from pre-computed parameter maps.

Examples
--------
```python
>> from nitorch.tools.qmri import relax, io
>> from glob import glob
>> import os
>>
>> # Map Gradient Echo data
>> # (assumes that sequence parameters can be read from these files)
>> t1w = io.GradientEchoMulti.from_fnames(sorted(glob('t1w/*.nii')))
>> pdw = io.GradientEchoMulti.from_fnames(sorted(glob('pdw/*.nii')))
>> mtw = io.GradientEchoMulti.from_fnames(sorted(glob('mtw/*.nii')))
>>
>> # Map pre-computed field maps
>> transmit = io.PrecomputedFieldMap('b1p.nii', magnitude='b1p_mag.nii.gz')
>> receive = [io.PrecomputedFieldMap('b1m_pd.nii', magnitude='b1m_pd_mag.nii'),
>>            io.PrecomputedFieldMap('b1m_t1.nii', magnitude='b1m_t1_mag.nii'),
>>            io.PrecomputedFieldMap('b1m_mt.nii', magnitude='b1m_mt_mag.nii')]
>>
>> # ESTATICS + Rational approximations
>> te0, r2s = relax.estatics([t1w, pdw, mtw], mode='loglin')
>> pd, r1, mt = relax.vfa(te0, transmit, receive)
>>
>> # JTV-ESTATICS + Rational approximations
>> te0, r2s = relax.estatics([t1w, pdw, mtw])
>> pd, r1, mt = relax.vfa(te0, transmit, receive)
>>
>> # GREEQ
>> pd, r1, r2s, mt = relax.greeq(te0, transmit, receive)
```
"""

from ._estatics import estatics, ESTATICSOptions
from ._mpm import greeq, GREEQOptions
from ._mpm import vfa, VFAOptions
from ._mpm import gre
