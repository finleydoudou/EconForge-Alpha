"""Native A1 innovation loadings and original pathwise coefficient checks."""
import json

from econforge.innovation_contract import need, real_matrix, component
from econforge.gh_numerics import embedding


def recover_candidates(a):
    np,axes = a['np'],a['axes']
    ns,nj,m,nk = [axes[k] for k in ('n_state','n_jump','n_innovation','n_k')]
    Ls = np.zeros((ns,m),dtype=np.float64)
    zi = [axes['variable_order'].index(z) for z in axes['partitions']['z']]
    for i,position in enumerate(zi):
        Ls[nk+i,i] = 1.0/a['T'][position]
    Lj = a['G']@Ls
    real_matrix(np,Ls,(ns,m),computed=True); real_matrix(np,Lj,(nj,m),computed=True)
    return Ls,Lj


def verify_loadings(a, certificate, Ls, Lj):
    """Yield all measured gates; conditional rows are diagnostics, never zeros."""
    np,axes = a['np'],a['axes']
    ns,nj,m,nk = [axes[k] for k in ('n_state','n_jump','n_innovation','n_k')]
    real_matrix(np,Ls,(ns,m),computed=True); real_matrix(np,Lj,(nj,m),computed=True)
    states,controls,shocks = axes['state_order'],axes['control_order'],axes['innovation_order']
    physical = Ls[:nk,:]
    passed,record = component(np,physical,np.zeros((nk,m)),states[:nk],shocks)
    # The physical-state restriction is exact structural zero, beyond tolerance.
    yield 'physical_zero',bool(np.all(physical == 0)) and passed,record,'LOADING_CHECK_FAILED'
    zindices = [axes['variable_order'].index(z) for z in axes['partitions']['z']]
    passed,record = component(np,a['T'][zindices,None]*Ls[nk:,:],np.eye(m),states[nk:],shocks)
    yield 'exogenous_loading',passed,record,'LOADING_CHECK_FAILED'
    passed,record = component(np,Lj,a['G']@Ls,controls,shocks)
    yield 'jump_loading',passed,record,'LOADING_CHECK_FAILED'
    E = embedding(np,a['G'],a['state_rows'],a['control_rows'],len(axes['variable_order']))
    original = a['raw_plus']@(a['T'][:,None]*E)@Ls
    jp = real_matrix(np,certificate['J_u_plus'],(len(axes['equation_order']),m))
    path = [i for i,row in enumerate(certificate['rows']) if row['kind'] in ('k_transition','z_process')]
    conditional = [i for i,row in enumerate(certificate['rows']) if row['kind'] == 'conditional']
    U,V = original[path,:],jp[path,:]
    passed,record = component(np,U/a['S'][path,None],-V/a['S'][path,None],
                              [axes['equation_order'][i] for i in path],shocks)
    # Frozen Diagnostic union has no raw-array kind. Preserve original terms in
    # an event's finite JSON message, alongside the full normalized components.
    raw = dict(row_labels=[axes['equation_order'][i] for i in path],innovation_labels=shocks,
               shape=[len(path),m],raw_lead_term=U.tolist(),raw_innovation_term=V.tolist(),
               raw_residual=(U+V).tolist(),S=a['S'][path].tolist())
    yield 'pathwise_raw',True,dict(stage='A1.pathwise_coefficients',reason='ORIGINAL_RAW_TERMS',
        message=json.dumps(raw,allow_nan=False,separators=(',',':')),exception_type=None),None
    yield 'pathwise_coefficients',passed,record,'PATHWISE_CHECK_FAILED'
    coefficients = original[conditional,:]
    real_matrix(np,coefficients,(len(conditional),m),computed=True)
    yield 'conditional_semantics',True,dict(row_labels=[axes['equation_order'][i] for i in conditional],
        innovation_labels=shocks,shape=[len(conditional),m],coefficients=coefficients.tolist(),
        meaning='REALIZED_FORECAST_ERROR_NOT_PATHWISE_ZERO_REQUIREMENT'),None
