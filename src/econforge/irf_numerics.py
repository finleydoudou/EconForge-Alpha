"""Native float64 finite-horizon propagation and separate componentwise verification."""
import warnings
from econforge.irf_contract import need, real_matrix, summary, canonical_json


def matmul(np, left, right, operation, horizon, observe):
    need(type(left) is np.ndarray and type(right) is np.ndarray
         and left.dtype == right.dtype == np.dtype('float64') and left.ndim == right.ndim == 2
         and left.shape[1] == right.shape[0] and bool(np.isfinite(left).all())
         and bool(np.isfinite(right).all()), 'NUMERICAL_NONFINITE',
         'Finite float64 compatible matrix operands required', 'backend')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with np.errstate(over='raise', invalid='raise', divide='raise', under='warn'):
            value = left @ right
    real_matrix(np, value, (left.shape[0], right.shape[1]), computed=True)
    underflow = []
    for w in caught:
        need(issubclass(w.category, RuntimeWarning) and 'underflow' in str(w.message).lower(),
             'BACKEND_WARNING', str(w.message), 'backend')
        underflow.append(str(w.message))
    if bool(np.any((np.abs(value) > 0) & (np.abs(value) < np.finfo(np.float64).tiny))):
        underflow.append('Returned finite subnormal entries detected')
    if underflow and observe is not None:
        observe(operation, horizon, '; '.join(underflow))
    return value


def stored(np, values, scale, operation, horizon, observe):
    # Diagonal multiplication once, in admitted partition row order.
    return matmul(np, np.diag(scale), values, operation, horizon, observe)


def propagate(np, G, H, Ls, V, N, scales, observe):
    """The only production response recurrence; caller has already passed resource gates."""
    ns, r = Ls.shape[0], V.shape[1]; nj = G.shape[0]
    Z = np.empty((N, ns, r), dtype=np.float64)
    J = np.empty((N, nj, r), dtype=np.float64)
    Z[0] = matmul(np, Ls, V, 'impact L_s V', 0, observe)
    for h in range(N):
        if h:
            Z[h] = matmul(np, H, Z[h - 1], 'state H Z', h, observe)
        J[h] = matmul(np, G, Z[h], 'control G Z', h, observe)
    SZ = SJ = None
    if scales is not None:
        SZ = np.empty_like(Z); SJ = np.empty_like(J)
        for h in range(N):
            SZ[h] = stored(np, Z[h], scales[0], 'stored T_s Z', h, observe)
            SJ[h] = stored(np, J[h], scales[1], 'stored T_j J', h, observe)
    return Z, J, SZ, SJ


def verify(np, products, G, H, Ls, Lj, V, axes, impulse_ids, scales, gate, start, observe):
    """Every actual response component is checked, including both impact expressions."""
    Z, J, SZ, SJ = products; N, ns, r = Z.shape; nj = G.shape[0]
    for value, shape in ((Z, (N, ns, r)), (J, (N, nj, r))):
        need(type(value) is np.ndarray and value.dtype == np.dtype('float64')
             and value.shape == shape and bool(np.isfinite(value).all()),
             'NUMERICAL_NONFINITE', 'Finite unsqueezed response tensor required', 'backend')
    def compare(name, left, right, rows, h, code):
        start(name)
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            passed, record, detail = summary(np, left, right, rows, impulse_ids, h)
        gate(name, passed, record, detail, code)
    start('B.impact')
    compare('B.impact', Z[0], matmul(np, Ls, V, 'verify impact L_s V', 0, observe),
            axes['state_order'], 0, 'IMPACT_CHECK_FAILED')
    need(bool(np.all(Z[0, :axes['n_k']] == 0.0)), 'IMPACT_CHECK_FAILED',
         'Predetermined capital impact must be exactly zero', 'check')
    compare('B.impact', J[0], matmul(np, Lj, V, 'verify impact L_j V', 0, observe),
            axes['control_order'], 0, 'IMPACT_CHECK_FAILED')
    for h in range(N):
        start('B.controls')
        compare('B.controls', J[h], matmul(np, G, Z[h], 'verify control G Z', h, observe),
                axes['control_order'], h, 'CONTROL_CHECK_FAILED')
        if h:
            start('B.recurrence')
            compare('B.recurrence', Z[h], matmul(np, H, Z[h - 1], 'verify recurrence H Z', h, observe),
                    axes['state_order'], h, 'RECURRENCE_FAILED')
        if scales is not None:
            start('B.coordinates')
            need(SZ is not None and SJ is not None and SZ.shape == Z.shape and SJ.shape == J.shape,
                 'COORDINATE_UNSUPPORTED', 'Both stored tensors must have exact native shapes', 'check')
            compare('B.coordinates', SZ[h], stored(np, Z[h], scales[0], 'verify stored state', h, observe),
                    axes['state_order'], h, 'CONTROL_CHECK_FAILED')
            compare('B.coordinates', SJ[h], stored(np, J[h], scales[1], 'verify stored control', h, observe),
                    axes['control_order'], h, 'CONTROL_CHECK_FAILED')
    if N == 1:
        start('B.recurrence'); gate('B.recurrence', True, None, None, None, vacuous=True)
    if scales is None:
        need(SZ is None and SJ is None, 'COORDINATE_UNSUPPORTED', 'Unrequested stored output forbidden', 'check')
        start('B.coordinates'); gate('B.coordinates', True, None, None, None, vacuous=True)
