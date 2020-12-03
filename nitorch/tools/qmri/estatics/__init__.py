"""
The ESTATICS model applies to multi-echo spoiled gradient-echo
acquisitions. It assumes that all echo series possess the same R2*
decay, but different initial signals (at TE=0).

This module implements two different ESTATICS fits:
- `loglin` fits the model using least-squares in the log domain:
    `log(x[contrast, echo]) = inter[contrast] - te[echo] * decay`
- `nonlin` fits the regularized model in the observed domain:
    `x[contrast, echo] = exp(inter[contrast] - te[echo] * decay) + noise[contrast]`

References
----------
..[1] "Estimating the apparent transverse relaxation time (R2*) from
       images with different contrasts (ESTATICS) reduces motion artifacts"
      N Weiskopf, MF Callaghan, O Josephs, A Lutti, S Mohammadi.
      Front Neurosci. 2014. https://doi.org/10.3389/fnins.2014.00278
..[2] "Joint Total Variation ESTATICS for Robust Multi-parameter Mapping"
      Y Balbastre, M Brudfors, M Azzarito, C Lambert, MF Callaghan, J Ashburner.
      MICCAI 2020. https://doi.org/10.1007/978-3-030-59713-9_6

"""


from ._loglin import loglin
from ._nonlin import nonlin
from ._param import ParameterMap, ParameterMaps
from ._options import *
