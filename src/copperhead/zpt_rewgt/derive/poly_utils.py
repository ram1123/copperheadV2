"""Numerically-stable polynomial fitting helpers for the Z-pT SF fits.

A plain monomial fit y = sum_k c_k * x**k over a pT range that does not
start at x=0 (e.g. the mid-range [xmin1, xmax1] piece, typically
~[10-20, 80-120] GeV) is severely ill-conditioned: the regressor columns
[1, x, x^2, ..., x^n] are nearly linearly dependent over any bounded range
away from the origin, so the fit's Hessian/covariance matrix is close to
singular. MIGRAD/HESSE then report individual coefficient uncertainties
that are enormous and strongly anti-correlated even though the *curve*
(the well-constrained linear combination of those coefficients) is fit
perfectly well and visually stable.

To avoid this we fit in a centered/scaled Chebyshev basis
(t = rescale(x, xmin, xmax) -> [-1, 1], well-conditioned by construction),
then algebraically convert the result - both central values and the full
covariance matrix - back to ordinary monomial-in-x coefficients. Consumers
(build_final_piecewise_coefficients, copperhead_processor.py's
getZptWgts_3region) keep reading plain "sum coeff * x**order" coefficients,
so the on-disk YAML format and stage-1 evaluation code are unchanged.
"""
import numpy as np
import ROOT
from numpy.polynomial import chebyshev as _cheb
from numpy.polynomial import polynomial as _poly


def _t_of_x(x, xmin, xmax):
    x0 = 0.5 * (xmin + xmax)
    s = 0.5 * (xmax - xmin)
    return (x - x0) / s


def eval_chebyshev(coeffs, x, xmin, xmax):
    """Evaluate sum_k coeffs[k] * T_k(t), t = rescale(x, xmin, xmax) -> [-1, 1]."""
    t = _t_of_x(x, xmin, xmax)
    if len(coeffs) == 1:
        return coeffs[0]
    t_km2, t_km1 = 1.0, t
    total = coeffs[0] * t_km2 + coeffs[1] * t_km1
    for k in range(2, len(coeffs)):
        t_k = 2.0 * t * t_km1 - t_km2
        total += coeffs[k] * t_k
        t_km2, t_km1 = t_km1, t_k
    return total


def cheb_to_monomial_x(cheb_coeffs, xmin, xmax):
    """Convert Chebyshev coefficients (basis T_k(t), t=(x-x0)/s) to plain
    monomial-in-x coefficients c_k such that sum_k c_k * x**k is identical
    to sum_k cheb_coeffs[k] * T_k(t) everywhere (exact polynomial identity,
    not a numerical refit)."""
    x0 = 0.5 * (xmin + xmax)
    s = 0.5 * (xmax - xmin)
    cheb_coeffs = np.asarray(cheb_coeffs, dtype=float)
    b = _cheb.cheb2poly(cheb_coeffs)  # sum_j b[j] * t**j; cheb2poly trims trailing zeros,
                                       # so len(b) can be < len(cheb_coeffs) - size the
                                       # output to the requested (fixed) coefficient count.
    t_of_x = np.array([-x0 / s, 1.0 / s])  # t = t_of_x[0] + t_of_x[1]*x
    c = np.zeros(len(cheb_coeffs))
    for j, bj in enumerate(b):
        term = _poly.polypow(t_of_x, j) * bj
        c[: len(term)] += term
    return c


def build_cheb_to_monomial_matrix(order, xmin, xmax):
    """Linear map M such that monomial_coeffs = M @ cheb_coeffs, and therefore
    Cov_monomial = M @ Cov_cheb @ M.T."""
    n = order + 1
    mat = np.zeros((n, n))
    for k in range(n):
        e_k = np.zeros(n)
        e_k[k] = 1.0
        mat[:, k] = cheb_to_monomial_x(e_k, xmin, xmax)
    return mat


def fit_chebyshev_poly(hist, order, xmin, xmax, name, fit_opts="S R Q"):
    """Fit `hist` on [xmin, xmax] with a degree-`order` polynomial written in
    a centered/scaled Chebyshev basis (well-conditioned MINUIT problem), then
    convert the result back to ordinary monomial-in-x coefficients + full
    covariance.

    Uses a proper chi-square fit against the histogram's own (Gaussian,
    error-propagated) bin errors by default - *not* the Poisson
    log-likelihood option "L", which is only meaningful for raw counts and
    is not appropriate for an already-computed Data/MC ratio histogram.

    Returns a dict:
      tf1          - fitted TF1 (Chebyshev parametrization; same curve as
                     the monomial form - use for Eval/Draw/pulls/chi2/ndf)
      fit_result   - TFitResultPtr
      coeffs_cheb  - Chebyshev coefficients as fit (np.ndarray)
      coeffs_x     - equivalent monomial-in-x coefficients (np.ndarray)
      errors_x     - monomial-in-x coefficient uncertainties, sqrt(diag(cov)) (np.ndarray)
      cov_x        - full monomial-in-x covariance matrix (np.ndarray)
    """
    npar = order + 1

    # Seed MIGRAD with a cheap, well-conditioned weighted least-squares
    # Chebyshev fit of the bin centers/contents in range, instead of
    # starting MINUIT blind from all-zeros.
    xs, ys, ws = [], [], []
    for ib in range(1, hist.GetNbinsX() + 1):
        xc = hist.GetBinCenter(ib)
        if xc < xmin or xc > xmax:
            continue
        err = hist.GetBinError(ib)
        if err <= 0:
            continue
        xs.append(xc)
        ys.append(hist.GetBinContent(ib))
        ws.append(1.0 / err)

    if len(xs) > npar:
        t_pts = _t_of_x(np.array(xs), xmin, xmax)
        seed = _cheb.chebfit(t_pts, np.array(ys), order, w=np.array(ws))
    else:
        seed = np.zeros(npar)

    def _func(x, par):
        return eval_chebyshev([par[i] for i in range(npar)], x[0], xmin, xmax)

    tf1 = ROOT.TF1(name, _func, xmin, xmax, npar)
    for i in range(npar):
        tf1.SetParameter(i, float(seed[i]))

    fit_result = hist.Fit(tf1, fit_opts, "", xmin, xmax)
    if fit_result and int(fit_result.Status()) != 0:
        # one retry, re-seeded from the (non-converged) first attempt
        fit_result = hist.Fit(tf1, fit_opts, "", xmin, xmax)

    coeffs_cheb = np.array([tf1.GetParameter(i) for i in range(npar)])
    cov_cheb = np.zeros((npar, npar))
    if fit_result and int(fit_result.Status()) == 0:
        cov_matrix = fit_result.GetCovarianceMatrix()
        for i in range(npar):
            for j in range(npar):
                cov_cheb[i, j] = cov_matrix(i, j)
    else:
        for i in range(npar):
            cov_cheb[i, i] = tf1.GetParError(i) ** 2

    mat = build_cheb_to_monomial_matrix(order, xmin, xmax)
    coeffs_x = mat @ coeffs_cheb
    cov_x = mat @ cov_cheb @ mat.T
    errors_x = np.sqrt(np.clip(np.diag(cov_x), 0.0, None))

    return {
        "tf1": tf1,
        "fit_result": fit_result,
        "coeffs_cheb": coeffs_cheb,
        # Plain Python floats, not numpy.float64: these end up in a dict that
        # gets handed to OmegaConf.create() for the final YAML, which rejects
        # numpy scalar types ("Value 'float64' is not a supported primitive
        # type").
        "coeffs_x": [float(v) for v in coeffs_x],
        "errors_x": [float(v) for v in errors_x],
        "cov_x": cov_x,
    }
