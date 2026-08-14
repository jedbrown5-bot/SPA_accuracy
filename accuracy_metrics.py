"""Pure accuracy-assessment maths for the Week 10 confusion-matrix explorer.

No Streamlit dependency, so it can be unit-tested on its own.
Convention throughout: matrix rows = MAP (classification), columns = REFERENCE.
M[i, j] = number of sample points the map called class i and the reference called j.

The area estimation follows Olofsson et al. (2014), "Good practices for estimating
area and assessing accuracy of land change", Remote Sensing of Environment 148.
Strata are the map classes, with user-supplied area weights W_i (mapped area
proportion of class i, summing to 1).
"""
import numpy as np


def _f(M):
    return np.asarray(M, float)


def overall_accuracy(M):
    M = _f(M)
    return np.trace(M) / M.sum()


def users_accuracy(M):
    """UA_i = n_ii / row_total_i  (1 - commission error). Returns array over classes."""
    M = _f(M)
    rt = M.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ua = np.where(rt > 0, np.diag(M) / rt, 0.0)
    return ua


def producers_accuracy(M):
    """PA_j = n_jj / col_total_j  (1 - omission error). Returns array over classes."""
    M = _f(M)
    ct = M.sum(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        pa = np.where(ct > 0, np.diag(M) / ct, 0.0)
    return pa


def kappa(M):
    M = _f(M)
    N = M.sum()
    po = np.trace(M) / N
    pe = (M.sum(1) * M.sum(0)).sum() / N**2
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0


def disagreement(M):
    """Pontius & Millones (2011). Returns (agreement, quantity, allocation), summing to 1."""
    M = _f(M)
    N = M.sum()
    oa = np.trace(M) / N
    r = M.sum(1) / N
    c = M.sum(0) / N
    q = 0.5 * np.abs(r - c).sum()
    a = (1 - oa) - q
    return oa, q, a


def olofsson_area(M, weights, total_area):
    """Error-adjusted area estimate with 95% CI per reference class (Olofsson 2014).

    weights: mapped area proportion of each MAP class (strata), summing to 1.
    total_area: total mapped area (any unit); results come back in the same unit.

    Returns a dict with:
      p_area   estimated area proportion per reference class
      se       standard error of each proportion
      area     estimated area per reference class
      ci95     half-width of the 95% CI on area per reference class
      oa_w     area-weighted overall accuracy
      ua       user's accuracy per class (= q_ii, stratified)
      pa       producer's accuracy per class (adjusted)
    """
    M = _f(M)
    W = _f(weights)
    n = M.shape[0]
    ni = M.sum(1)                                   # row (map) sample totals
    with np.errstate(divide='ignore', invalid='ignore'):
        q = np.where(ni[:, None] > 0, M / ni[:, None], 0.0)   # q_ij = n_ij / n_i+
    # estimated area proportion of each cell: p_ij = W_i * q_ij
    p = W[:, None] * q
    p_area = p.sum(0)                               # p_+j  (reference class proportions)

    # standard error of p_+j  (Olofsson 2014, eq. 10)
    se = np.zeros(n)
    for j in range(n):
        s = 0.0
        for i in range(n):
            if ni[i] > 1:
                s += W[i]**2 * q[i, j] * (1 - q[i, j]) / (ni[i] - 1)
        se[j] = np.sqrt(s)

    area = p_area * total_area
    ci95 = 1.96 * se * total_area

    oa_w = np.sum([W[i] * q[i, i] for i in range(n)])
    ua = np.diag(q)                                 # user's accuracy = q_ii
    with np.errstate(divide='ignore', invalid='ignore'):
        pa = np.where(p_area > 0, np.diag(p) / p_area, 0.0)   # producer's (adjusted)

    return dict(p_area=p_area, se=se, area=area, ci95=ci95,
                oa_w=oa_w, ua=ua, pa=pa)


# --------------------------------------------------------------- self test
if __name__ == '__main__':
    SYN = [[34, 1, 2], [1, 28, 5], [2, 3, 24]]
    HAST = [[47, 0, 0, 1, 0], [1, 78, 6, 4, 1], [0, 11, 34, 2, 1],
            [1, 5, 1, 41, 6], [0, 1, 0, 5, 28]]

    print('SYN  OA', round(overall_accuracy(SYN), 3),
          'kappa', round(kappa(SYN), 3),
          'disagg', tuple(round(x, 3) for x in disagreement(SYN)))
    print('HAST OA', round(overall_accuracy(HAST), 3),
          'kappa', round(kappa(HAST), 3),
          'disagg', tuple(round(x, 3) for x in disagreement(HAST)))
    print('SYN  PA', np.round(producers_accuracy(SYN), 3),
          'UA', np.round(users_accuracy(SYN), 3))
    print('HAST PA', np.round(producers_accuracy(HAST), 3),
          'UA', np.round(users_accuracy(HAST), 3))
    W = np.array([.10, .46, .13, .23, .08])
    r = olofsson_area(HAST, W, 40000)
    print('HAST area-weighted OA', round(r['oa_w'], 3))
    for k, name in enumerate(['Water', 'Forest', 'Burnt', 'Pasture', 'Urban']):
        print(f"  {name:8s} area {r['area'][k]:8.0f} +/- {r['ci95'][k]:6.0f} ha")
