"""Original admitted AST incidence certificate; no AD, FD or general algebra."""
import json

from econforge.innovation_contract import need, sha_bytes
from econforge.p35_admission import digest
from econforge.p35_structure import _terms, validate_structure
from econforge.model import expand_aliases
from econforge.gh_source import affine


def ast_record(node):
    return dict(kind=node.kind,value=node.value,offset=node.offset,
                source=node.source,start=node.start,end=node.end,
                children=[ast_record(child) for child in node.children])


def occurrences(node):
    # Keep duplicates, zero multipliers, exact dates and original traversal order.
    result = [[str(node.value),node.offset,node.start,node.end]] if node.kind == 'symbol' else []
    for child in node.children:
        result.extend(occurrences(child))
    return result


def axes_coordinates(a):
    model,md = a['model'],a['metadata']
    source = json.loads(model.source_json)
    innovations = source['innovations']
    nk,nz,nj = [len(md['partitions'][role]) for role in ('k','z','j')]
    axes = dict(n_k=nk,n_z=nz,n_state=nk+nz,n_jump=nj,n_innovation=nz,
        equation_order=md['equation_order'],variable_order=md['variable_order'],
        state_order=md['state_variable_order'],control_order=md['control_variable_order'],
        innovation_order=[s['id'] for s in innovations],partitions=md['partitions'],
        permutations=md['permutations'])
    variables = {v['id']:v for v in source['variables']}
    ordered = [variables[name] for name in axes['variable_order']]
    coordinates = dict(convention=md['coordinate_convention'],S=a['S'].tolist(),T=a['T'].tolist(),
        innovation_scales=[float(s['scale']) for s in innovations],
        variable_units=[v['unit']['label'] for v in ordered],
        innovation_units=[s['unit']['label'] for s in innovations],
        representations=[v['representation'] for v in ordered],
        local_coordinates=[v['coordinate'] for v in ordered],reference=list(model.reference),
        reference_sha256=a['bundle']['bindings']['reference_hash'],frequency=model.frequency)
    return axes,coordinates


def source_certificate(a, axes, coordinates):
    """Derive coefficients solely from retained original source, before any L."""
    np,model = a['np'],a['model']
    validate_structure(model)  # preserve accepted narrow grammar
    registry = {s.name:s for s in model.symbols}
    aliases = dict(model.aliases)
    n,m = len(axes['variable_order']),axes['n_innovation']
    shocks,zs = axes['innovation_order'],axes['partitions']['z']
    jp,j0,R = np.zeros((n,m)),np.zeros((n,m)),np.zeros((m,m))
    intercepts = np.zeros(m)
    records = []
    rows = [e for e in model.equations if e.kind != 'identity']
    need([e.equation_id for e in rows] == axes['equation_order'],
         'INVALID_AXES','Original canonical row labels differ')
    for i,row in enumerate(rows):
        tree = expand_aliases(row.tree,aliases,registry)
        incidence = occurrences(tree)
        shockrefs = [x for x in incidence if x[0] in shocks]
        terms = _terms(tree)
        process_index = None
        if row.kind in ('k_transition','z_process'):
            original_leads = [x for x in incidence if registry[x[0]].role in ('k','z','j') and x[1] == 1]
            original_target = [(sign,node) for sign,node in terms
                               if node.kind == 'symbol' and node.value == row.target and node.offset == 1]
            need(len(original_leads) == 1 and original_leads[0][:2] == [row.target,1]
                 and len(original_target) == 1 and original_target[0][0] == 1,
                 'SOURCE_UNSUPPORTED','Every pathwise transition requires one exact additive target lead','unsupported')
        if row.kind in ('k_transition','conditional'):
            need(not shockrefs,'SOURCE_UNSUPPORTED','Forbidden direct innovation incidence','unsupported')
        elif row.kind == 'z_process':
            process_index = zs.index(row.target)
            shock = shocks[process_index]
            need(len(shockrefs) == 1 and shockrefs[0][:2] == [shock,1],
                 'SOURCE_UNSUPPORTED','Unique next-date innovation required; no cancellation','unsupported')
            selected = [(sign,node) for sign,node in terms
                        if node.kind == 'symbol' and node.value == shock and node.offset == 1]
            need(len(selected) == 1 and selected[0][0] == -1,
                 'SOURCE_UNSUPPORTED','Exact original minus innovation term required','unsupported')
            leads = [x for x in incidence if registry[x[0]].role in ('k','z','j') and x[1] == 1]
            target = [(sign,node) for sign,node in terms
                      if node.kind == 'symbol' and node.value == row.target and node.offset == 1]
            need(len(leads) == 1 and leads[0][:2] == [row.target,1]
                 and len(target) == 1 and target[0][0] == 1,
                 'SOURCE_UNSUPPORTED','Unique additive positive target lead required','unsupported')
            jp[i,process_index] = float(selected[0][0])
            constant = 0.0
            coefficients = np.zeros(n)
            for sign,term in terms:
                if term is selected[0][1] or term is target[0][1]:
                    continue
                c,v = affine(np,term,model,allowed=zs)
                constant += sign*c
                coefficients += sign*v
            intercepts[process_index] = -constant
            R[process_index,:] = [-coefficients[axes['variable_order'].index(z)] for z in zs]
        else:
            need(False,'SOURCE_UNSUPPORTED','Unsupported canonical row kind','unsupported')
        records.append(dict(equation_id=row.equation_id,kind=row.kind,target=row.target,
            expression=row.expression,expression_sha256=sha_bytes(row.expression.encode('utf-8')),
            source_span=[row.tree.source,row.tree.start,row.tree.end],
            expanded_ast_sha256=digest(ast_record(tree)),occurrences=incidence,
            innovation_coefficients=jp[i].tolist(),row_scale=float(row.scale),process_index=process_index))
    need(bool(np.isfinite(R).all()) and bool(np.isfinite(intercepts).all()),
         'NUMERICAL_NONFINITE','Source coefficients are not representable')
    return dict(schema='p39-source-coefficients',revision=1,original_model_sha256=model.digest,
        canonical_model_sha256=digest(json.loads(model.source_json)),
        reference_sha256=coordinates['reference_sha256'],p34_result_sha256=digest(a['bundle']['p34_result']),
        coordinate_sha256=digest(coordinates),axes_sha256=digest(axes),rows=records,
        J_u_plus=jp.tolist(),J_u_current=j0.tolist(),R=R.tolist(),
        process_means=[float(dict(model.process_means)[z]) for z in zs],
        process_intercepts=intercepts.tolist(),structural_zero_mask=(jp == 0).tolist(),
        derivation='ORIGINAL_ADMITTED_AST_INCIDENCE_V1')
