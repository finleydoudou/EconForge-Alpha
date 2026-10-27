"""Frozen complex128 solve-only conditional-mean recovery and component checks."""
import math

from econforge.gh_admission import GHError, need

ATOL, RTOL = 1e-10, 1e-8
REAL_ATOL, REAL_RTOL = 1e-12, 1e-10
WARN, REFUSE = 1e8, 1e10


def finite(np, value, shape, label, *, complex_required=False):
    need(type(value) is np.ndarray and value.shape == shape
         and value.dtype == np.dtype('complex128' if complex_required else 'float64')
         and bool(np.isfinite(value).all()), 'NUMERICAL_OUTPUT_INVALID',
         'Finite exact dtype/shape required: '+label, numerical=True)
    return value


def wire_array(np, value):
    return np.stack((value.real, value.imag), axis=-1).tolist() if np.iscomplexobj(value) else value.tolist()


def component_check(np, left, right, row_labels, column_labels, *, two_term=False,
                    atol=ATOL, rtol=RTOL, magnitude=None):
    """Every component is scored; zero-size comparisons are explicitly vacuous."""
    need(left.shape == right.shape == (len(row_labels), len(column_labels))
         and bool(np.isfinite(left).all()) and bool(np.isfinite(right).all()),
         'RESIDUAL_NONFINITE_OR_SHAPE', 'Finite correctly labeled residual terms required', numerical=True)
    error = np.abs(left-right)
    ref = np.abs(left)+np.abs(right) if two_term else np.abs(right)
    if magnitude is not None:
        ref = magnitude
    bound = atol+rtol*ref
    need(bool(np.isfinite(error).all()) and bool(np.isfinite(bound).all()),
         'RESIDUAL_NONFINITE', 'Residual errors/bounds must be representable', numerical=True)
    out = dict(passed=bool(np.all(error <= bound)), shape=list(error.shape), vacuous=error.size == 0,
               rule='two_term' if two_term else 'declared_reference', atol=atol, rtol=rtol,
               left=wire_array(np,left), right=wire_array(np,right),
               absolute_errors=error.tolist(), allowed_bounds=bound.tolist(),
               max_absolute_error=None, max_allowed_bound=None, max_error_over_bound=None, worst=None)
    if error.size:
        ratios = error/bound
        need(bool(np.isfinite(ratios).all()), 'RESIDUAL_NONFINITE', 'Finite discrepancy ratios required', numerical=True)
        i,j = np.unravel_index(int(np.argmax(ratios)), ratios.shape)
        relative = float(error[i,j])/float(abs(right[i,j])) if right[i,j] != 0 else None
        if relative is not None and not math.isfinite(relative):
            relative = None
        out.update(max_absolute_error=float(np.max(error)), max_allowed_bound=float(np.max(bound)),
                   max_error_over_bound=float(np.max(ratios)),
                   worst=dict(row=row_labels[i], state=column_labels[j], row_index=int(i), column_index=int(j),
                              absolute_error=float(error[i,j]), allowed_bound=float(bound[i,j]),
                              relative_error=relative))
    return out


def require_check(diagnostics, name, value, code):
    diagnostics[name] = value
    if not value['passed']:
        raise GHError(code, 'Componentwise check failed: '+name, numerical=True)


def condition_gate(sigma_max, sigma_min, *, graph):
    """Named threshold policy, separately testable at exact binary64 endpoints."""
    need(math.isfinite(sigma_max) and math.isfinite(sigma_min) and 0 <= sigma_min <= sigma_max,
         'CONDITION_ESTIMATE_INVALID', 'Finite nonnegative singular values required', numerical=True)
    if graph:
        tau = max(1e-12, 1e-10*sigma_max)
        rank_pass = sigma_min > 10*tau
        if not rank_pass:
            return dict(status='REFUSED', reason='RANK_UNRESOLVED', tau_rank=tau,
                        sigma_max=sigma_max, sigma_min=sigma_min, kappa_2=None, metric=None)
    else:
        tau = None
        need(sigma_min > 0, 'STABLE_BLOCK_SINGULAR', 'Stable Schur block is singular', numerical=True)
    kappa = sigma_max/sigma_min
    metric = max(kappa, 1/sigma_min) if graph else kappa
    need(math.isfinite(kappa) and math.isfinite(metric), 'CONDITION_ESTIMATE_INVALID',
         'Condition metric must be finite and representable', numerical=True)
    status = 'REFUSED' if metric >= REFUSE else 'WARN' if metric >= WARN else 'PASS'
    return dict(status=status, reason=('GRAPH_CONDITION_REFUSED' if graph else 'STABLE_BLOCK_CONDITION_REFUSED')
                if status == 'REFUSED' else None, sigma_max=sigma_max, sigma_min=sigma_min,
                tau_rank=tau, kappa_2=kappa, metric=metric, warning_threshold=WARN, refusal_threshold=REFUSE)


def conditioning(np, linalg, X, SA, diagnostics):
    for name, matrix, graph in (('X_conditioning', X, True), ('stable_block_conditioning', SA, False)):
        if not len(X):
            diagnostics[name] = dict(status='PASS', vacuous=True, sigma_min=None, sigma_max=None,
                                     tau_rank=None, kappa_2=None, metric=None)
            continue
        singular = linalg.svdvals(matrix, check_finite=True)
        finite(np,singular,(len(X),),name)
        need(bool((singular >= 0).all()), 'CONDITION_ESTIMATE_INVALID', 'Nonnegative singular values required', numerical=True)
        try:
            d = condition_gate(float(np.max(singular)), float(np.min(singular)), graph=graph)
        except GHError as exc:
            diagnostics[name] = dict(status='FAIL', reason=exc.code, singular_values=singular.tolist())
            raise
        d.update(vacuous=False, singular_values=singular.tolist())
        diagnostics[name] = d
        if d['status'] == 'REFUSED':
            raise GHError(d['reason'], 'Conservative numerical admission refused: '+name,
                          refused=True, numerical=True)


def solve_candidates(np, linalg, X, Y, SA, SB):
    """SA M=SB; X.T G.T=Y.T; X.T H.T=(X M).T. No conjugate transpose."""
    ns, nj = len(X), len(Y)
    if ns == 0:
        return (np.empty((nj,0),dtype=np.complex128), np.empty((0,0),dtype=np.complex128),
                np.empty((0,0),dtype=np.complex128))
    M = linalg.solve_triangular(SA, SB, lower=False, trans=0, unit_diagonal=False,
                               overwrite_b=False, check_finite=True)
    finite(np,M,(ns,ns),'M',complex_required=True)
    G = (linalg.solve(X.T, Y.T, assume_a='gen', overwrite_a=False, overwrite_b=False,
                      check_finite=True).T if nj else np.empty((0,ns),dtype=np.complex128))
    finite(np,G,(nj,ns),'G',complex_required=True)
    XM = X@M
    finite(np,XM,(ns,ns),'X M',complex_required=True)
    H = linalg.solve(X.T, XM.T, assume_a='gen', overwrite_a=False, overwrite_b=False, check_finite=True).T
    finite(np,H,(ns,ns),'H',complex_required=True)
    return G,H,M


def embedding(np, G, state_rows, control_rows, n):
    ns = len(state_rows)
    need(G.shape == (len(control_rows),ns) and sorted(state_rows+control_rows) == list(range(n)),
         'PARTITION_MISMATCH', 'Complete disjoint explicit row selectors required')
    E = np.zeros((n,ns),dtype=G.dtype)
    E[state_rows,:] = np.eye(ns)
    E[control_rows,:] = G
    return E


def recover_complex(a, diagnostics):
    np,linalg = a['np'],a['linalg']
    ns,nj = a['n_state'],a['n_jump']
    SB,SA,_,_,_,Z = a['output']
    X,Y = Z[a['state_rows'],:ns], Z[a['control_rows'],:ns]
    SA,SB = SA[:ns,:ns],SB[:ns,:ns]
    conditioning(np,linalg,X,SA,diagnostics)
    c = max(float(np.linalg.norm(a['A'],ord=np.inf)), float(np.linalg.norm(a['B'],ord=np.inf)))
    need(math.isfinite(c) and c > 0, 'PENCIL_SCALE_UNREPRESENTABLE',
         'Positive representable infinity row-sum scale required', numerical=True)
    diagnostics['common_infinity_scale'] = c
    diagnostics['solve_execution_status'] = 'FAILED'
    G,H,M = solve_candidates(np,linalg,X,Y,SA,SB)
    diagnostics['solve_execution_status'] = 'COMPLETED'
    states = a['metadata']['state_variable_order']; controls = a['metadata']['control_variable_order']
    stable_labels = ['stable_coordinate_'+str(i) for i in range(ns)]
    for name,left,right,rows,cols,code in (
        ('stable_solve', (SA/c)@M, SB/c, stable_labels, stable_labels, 'STABLE_SOLVE_RESIDUAL_FAILED'),
        ('graph_solve', G@X, Y, controls, stable_labels, 'GRAPH_SOLVE_RESIDUAL_FAILED'),
        ('state_solve', H@X, X@M, states, stable_labels, 'STATE_SOLVE_RESIDUAL_FAILED')):
        require_check(diagnostics,name,component_check(np,left,right,rows,cols),code)
    return G,H


def real_candidates(a, G, H, diagnostics):
    np = a['np']; states = a['metadata']['state_variable_order']; controls = a['metadata']['control_variable_order']
    for name,value,rows in (('G_realness',G,controls),('H_realness',H,states)):
        d = component_check(np,value.imag,np.zeros_like(value.real),rows,states,
                            atol=REAL_ATOL,rtol=REAL_RTOL,magnitude=np.abs(value.real))
        d['rule'] = 'imaginary_magnitude_with_real_reference'
        require_check(diagnostics,name,d,'IMAGINARY_COMPONENT_INVALID')
    # Both gates have passed; only now may either real matrix be produced.
    return G.real.copy(),H.real.copy()


def policy_check(a, G, H, diagnostics, name='final_policy'):
    np = a['np']
    E = embedding(np,G,a['state_rows'],a['control_rows'],len(a['A']))
    d = component_check(np,a['A']@E@H,a['B']@E,a['metadata']['equation_order'],
                        a['metadata']['state_variable_order'],two_term=True)
    require_check(diagnostics,name,d,'POLICY_RESIDUAL_FAILED')
    return E
