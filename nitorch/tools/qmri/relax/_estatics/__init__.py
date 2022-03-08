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
# from ._nonlin_new import nonlin
from ._options import ESTATICSOptions
from ._param import ESTATICSParameterMaps


def estatics(data, dist, opt=None, **kwopt):
    """Fit the ESTATICS model to multi-echo Gradient-Echo data.

    Parameters
    ----------
    data : sequence[GradientEchoMulti]
        Observed GRE data.

    dist : sequence[Optional[ParameterizedDistortion]], optional
        Pre-computed distortion fields

    opt : ESTATICSOptions, optional
        {'model':    'nonlin',                   # 'loglin' (= ESTATICS) or 'nonlin' (=JTV-ESTATICS)
         'preproc': {'register':      True},     # Co-register contrasts
         'optim':   {'max_iter_rls':  10,        # Max reweighting iterations
                     'max_iter_gn':   5,         # Max Gauss-Newton iterations
                     'max_iter_cg':   32,        # Max Conjugate-Gradient iterations
                     'tolerance_rls': 1e-05,     # Tolerance for early stopping (RLS)
                     'tolerance_gn':  1e-05,         ""
                     'tolerance_cg':  1e-03},        ""
         'backend': {'dtype':  torch.float32,    # Data type
                     'device': 'cpu'},           # Device
         'penalty': {'norm':    'jtv',           # Type of penalty: {'tkh', 'tv', 'jtv', None}
                     'factor':  {'r1':  10,      # Penalty factor per (log) map
                                 'pd':  10,
                                 'r2s': 2,
                                 'mt':  2}},
         'verbose': 1}

        Note that if 'mode' == 'loglin', regularization will *not* be used.

    Returns
    -------
    intecepts : sequence[GradientEcho]
        Echo series extrapolated to TE=0
    decay : estatics.ParameterMap
        R2* decay map
    distortions : sequence[ParameterizedDistortion], if opt.distortion.enable
        B0-induced distortion fields

    References
    ----------
    ..[1] Weiskopf et al., "Estimating the apparent transverse relaxation
          time (R2*) from images with different contrasts (ESTATICS)
          reduces motion artifacts", Front Neurosci. (2014)
          https://doi.org/10.3389/fnins.2014.00278
    ..[2] Balbastre et al., "Joint Total Variation ESTATICS for Robust
          Multi-parameter Mapping", MICCAI (2020)
          https://doi.org/10.1007/978-3-030-59713-9_6

    """
    opt = ESTATICSOptions().update(opt, **kwopt)
    if opt.model.lower() == 'loglin':
        return loglin(data, opt)
    else:
        return nonlin(data, dist, opt)
