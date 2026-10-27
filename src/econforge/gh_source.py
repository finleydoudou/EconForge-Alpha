"""Original P3.4 row checks and bounded affine identity/process verification.

This is source coefficient algebra for the frozen affine subset, never AD/FD.
Nonlinear reporting observables remain metadata and are not recovered.
"""
import hashlib
import json
import math

from econforge.gh_admission import GHError, need
from econforge.gh_numerics import component_check, require_check, embedding, wire_array
from econforge.model import expand_aliases
from econforge.model_ast import evaluate_ast, references
from econforge.qz_admission import same
from econforge.p35_structure import _terms

NUMERIC_MAP_FIELDS = {'stored_coefficients','native_coefficients','constant','reference_value'}
MAP_FIELDS = NUMERIC_MAP_FIELDS | {'id','kind','source_expression','source_span','source_sha256',
    'model_hash','reference_hash','dependency_order','slot_order','row_scale','original_occurrences','map_provenance'}


def affine(np, tree, model, *, allowed=None, alias_coefficients=None):
    """Syntactic affine algebra; dependency flags survive zero cancellation."""
    names = list(model.variable_order); registry = {s.name:s for s in model.symbols}
    parameters = {(k,0):v for k,v in model.parameters}
    allowed = set(names) if allowed is None else set(allowed)
    n = len(names)

    def visit(node):
        refs = references(node)
        if all(registry[name].role == 'parameter' for name,_ in refs):
            return evaluate_ast(node,registry,parameters),np.zeros(n),False
        if node.kind == 'symbol':
            name = str(node.value)
            need(node.offset == 0, 'IDENTITY_TIMING_UNSUPPORTED', 'Required affine maps are current-date only', refused=True)
            if alias_coefficients is not None and name in alias_coefficients:
                c,v = alias_coefficients[name]
                return c,v.copy(),True
            need(name in allowed and name in names, 'AFFINE_SYMBOL_UNSUPPORTED',
                 'Affine source uses a forbidden variable/innovation', refused=True)
            value = np.zeros(n); value[names.index(name)] = 1.0
            return 0.0,value,True
        parts = [visit(c) for c in node.children]
        if node.kind == 'unary':
            sign = 1 if node.value == '+' else -1
            c,v,d = parts[0]; return sign*c,sign*v,d
        if node.kind == 'binary':
            c,v,d = parts[0]; f,w,e = parts[1]
            if node.value in ('+','-'):
                sign = 1 if node.value == '+' else -1
                return c+sign*f,v+sign*w,d or e
            if node.value == '*' and not (d and e):
                return c*f,v*f+w*c,d or e
            if node.value == '/' and not e:
                need(math.isfinite(f) and f != 0, 'AFFINE_DENOMINATOR_INVALID', 'Finite nonzero constant denominator required')
                return c/f,v/f,d
        raise GHError('IDENTITY_GRAMMAR_UNSUPPORTED', 'Required expression exceeds the frozen affine subset', refused=True)

    c,v,d = visit(tree)
    need(math.isfinite(c) and bool(np.isfinite(v).all()), 'AFFINE_COEFFICIENT_NONFINITE', 'Finite source coefficients required')
    return c,v


def required_aliases(model):
    aliases = dict(model.aliases); order = []
    def add(name):
        if name in order:return
        for dependency,_ in references(aliases[name]):
            if dependency in aliases:add(dependency)
        order.append(name)
    for row in model.equations:
        if row.kind == 'identity':
            for name,_ in references(row.tree):
                if name in aliases:add(name)
    return order


def source_metadata(model, reference_hash, name, kind, dependency_order, row_scale):
    """Exact source identity of a map; callers still supply separate coefficients."""
    source = json.loads(model.source_json)
    if kind == 'alias':
        item = next(x for x in source['aliases'] if x['id'] == name)
        tree = dict(model.aliases)[name]
    else:
        item = next(x for x in source['equations'] if x['id'] == name)
        tree = next(e.tree for e in model.equations if e.equation_id == name)
    occurrences = []
    aliases = dict(model.aliases)
    if kind == 'alias':
        def scan(node, shift, equation):
            if node.kind == 'symbol':
                symbol = str(node.value); offset = node.offset+shift
                if symbol == name:occurrences.append([equation,offset])
                if symbol in aliases:scan(aliases[symbol],offset,equation)
            else:
                for child in node.children:scan(child,shift,equation)
        for equation in model.equations:scan(equation.tree,0,equation.equation_id)
    return dict(id=name,kind=kind,source_expression=item['expression'],
                source_span=[tree.source,tree.start,tree.end],
                source_sha256=hashlib.sha256(item['expression'].encode()).hexdigest(),
                model_hash=model.digest,reference_hash=reference_hash,dependency_order=list(dependency_order),
                slot_order=[[0,x] for x in model.variable_order],row_scale=row_scale,
                original_occurrences=sorted([list(x) for x in set(map(tuple,occurrences))]),
                map_provenance='original_affine_source_coefficients_v1')


def validate_map(a, record, metadata, constant, coefficients, diagnostics, name):
    np = a['np']; reference = np.array(a['bundle']['p34_result']['reference'])
    need(type(record) is dict and set(record) == MAP_FIELDS, 'IDENTITY_MAP_MALFORMED', 'Complete explicit affine map required')
    need(all(same(record[k],metadata[k]) for k in metadata), 'IDENTITY_MAP_PROVENANCE_MISMATCH',
         'Source spans, binding, dependency/slot order or scale differs')
    for k in ('constant','reference_value','row_scale'):
        need(type(record[k]) is float and math.isfinite(record[k]), 'IDENTITY_MAP_MALFORMED', 'Finite binary64 map scalar required')
    need(record['row_scale'] > 0, 'IDENTITY_MAP_MALFORMED', 'Positive explicit map scale required')
    actual = []
    expected = []
    for k,v in (('stored_coefficients',coefficients),('native_coefficients',coefficients*a['T'])):
        x = record[k]
        need(type(x) is list and len(x) == len(reference)
             and all(type(y) is float and math.isfinite(y) for y in x),
             'IDENTITY_MAP_MALFORMED', 'Finite coefficient array with complete variable axes required')
        actual.extend(x); expected.extend(v.tolist())
    actual.extend((record['constant'],record['reference_value']))
    expected.extend((constant,float(constant+coefficients@reference)))
    labels = ['stored:'+x for x in a['model'].variable_order]+['native:'+x for x in a['model'].variable_order]+['constant','reference_value']
    d = component_check(np,np.array([actual]),np.array([expected]),[name],labels)
    diagnostics[name] = d
    need(d['passed'], 'IDENTITY_COEFFICIENT_MISMATCH', 'Supplied affine map differs from source coefficients', refused=True)
    chart = component_check(np,np.array([record['native_coefficients']]),
                            np.array([record['stored_coefficients']])*a['T'],[name],list(a['model'].variable_order))
    diagnostics[name+':chart'] = chart
    need(chart['passed'], 'IDENTITY_CHART_MISMATCH', 'Stored/native maps must use T exactly once', refused=True)
    return record['constant'],np.array(record['stored_coefficients'])


def prepare(a, diagnostics):
    """Validate every required map and original process before recovery starts."""
    np,model = a['np'],a['model']; aliases = dict(model.aliases)
    registry = {s.name:s for s in model.symbols}
    order = required_aliases(model)
    alias_records = a['bundle']['alias_maps']; identity_records = a['bundle']['identity_maps']
    identities = [e for e in model.equations if e.kind == 'identity']
    need(len(alias_records) == len(order) and len(identity_records) == len(identities)
         and all(type(r) is dict for r in alias_records+identity_records)
         and [r['id'] for r in alias_records if 'id' in r] == order
         and [r['id'] for r in identity_records if 'id' in r] == [e.equation_id for e in identities],
         'IDENTITY_MAPS_MISSING', 'Every required identity and dependent alias must have an ordered map', refused=True)
    verified_aliases = {}
    reference_hash = a['bundle']['bindings']['reference_hash']
    for name,record in zip(order,alias_records):
        c,v = affine(np,expand_aliases(aliases[name],aliases,registry),model)
        scale = record.get('row_scale')
        need(type(scale) is float and math.isfinite(scale) and scale > 0,
             'IDENTITY_SCALE_MISSING', 'Alias map needs a separately retained positive scale', refused=True)
        metadata = source_metadata(model,reference_hash,name,'alias',order,scale)
        verified_aliases[name] = validate_map(a,record,metadata,c,v,diagnostics,'alias_map:'+name)
        # Independently substitute already-verified dependency maps in source order.
        cc,vv = affine(np,aliases[name],model,alias_coefficients=verified_aliases)
        d = component_check(np,np.array([[*verified_aliases[name][1],verified_aliases[name][0]]]),
                            np.array([[*vv,cc]]),[name],[*model.variable_order,'constant'])
        require_check(diagnostics,'alias_expansion:'+name,d,'ALIAS_EXPANSION_FAILED')
    prepared = []
    reference = np.array(a['bundle']['p34_result']['reference'])
    for equation,record in zip(identities,identity_records):
        c,v = affine(np,expand_aliases(equation.tree,aliases,registry),model)
        metadata = source_metadata(model,reference_hash,equation.equation_id,'identity',order,equation.scale)
        validate_map(a,record,metadata,c,v,diagnostics,'identity_map:'+equation.equation_id)
        cc,vv = affine(np,equation.tree,model,alias_coefficients=verified_aliases)
        bound = equation.scale*1e-10+equation.magnitude*1e-8
        constant_residual = float(cc+vv@reference)
        need(math.isfinite(bound) and math.isfinite(constant_residual), 'IDENTITY_REFERENCE_NONFINITE', 'Finite reference identity required')
        diagnostics['identity_reference:'+equation.equation_id] = dict(passed=abs(constant_residual)<=bound,
            raw_residual=constant_residual,allowed_bound=bound)
        need(abs(constant_residual)<=bound, 'IDENTITY_REFERENCE_FAILED', 'Original scalar identity reference failed')
        substituted = component_check(np,np.array([record['native_coefficients']]),
                                      np.array([vv*a['T']]),[equation.equation_id],list(model.variable_order))
        require_check(diagnostics,'identity_expansion:'+equation.equation_id,substituted,'IDENTITY_EXPANSION_FAILED')
        prepared.append((equation.equation_id,vv*a['T'],equation.scale))
    # Extract current-z affine terms from original direct process source syntax.
    z_names = [v.symbol.name for v in model.variables if v.symbol.role == 'z']
    process_rows = [e for e in model.equations if e.kind == 'z_process']
    stored_R = np.zeros((len(z_names),len(z_names)))
    means = dict(model.process_means)
    process_source = []
    innovations = a['metadata']['structure']['innovation_order']
    for i,(name,row,shock) in enumerate(zip(z_names,process_rows,innovations)):
        tree = expand_aliases(row.tree,aliases,registry); c = 0.0; v = np.zeros(len(model.variable_order))
        for sign,term in _terms(tree):
            if term.kind == 'symbol' and ((term.value == name and term.offset == 1)
                                          or (term.value == shock and term.offset == 1)):
                continue
            cc,vv = affine(np,term,model,allowed=z_names)
            c += sign*cc; v += sign*vv
        stored_R[i,:] = [-v[list(model.variable_order).index(z)] for z in z_names]
        process_source.append(dict(equation_id=row.equation_id,target=name,expression=row.expression,
            expression_sha256=hashlib.sha256(row.expression.encode()).hexdigest(),mean=means[name],
            stored_intercept=-c,stored_R=stored_R[i].tolist()))
    zi = [list(model.variable_order).index(z) for z in z_names]
    Rq = stored_R/a['T'][zi,None]*a['T'][None,zi] if z_names else stored_R
    need(bool(np.isfinite(Rq).all()), 'PROCESS_COEFFICIENT_NONFINITE', 'Finite native process mean coefficients required')
    diagnostics['source_maps'] = dict(status='PASS',required_aliases=order,
        required_identities=[e.equation_id for e in identities],process_source=process_source,
        native_R=Rq.tolist(),observables='PRESERVED_METADATA_NOT_RECOVERED_OR_VERIFIED')
    return prepared,Rq


def original_equations(a, G, H, diagnostics, name):
    np = a['np']; E = embedding(np,G,a['state_rows'],a['control_rows'],len(a['A']))
    TE = a['T'][:,None]*E
    U,V = a['raw_plus']@TE@H,a['raw_current']@TE
    d = component_check(np,U/a['S'][:,None],-V/a['S'][:,None],
                        a['metadata']['equation_order'],a['metadata']['state_variable_order'],two_term=True)
    d.update(raw_lead_term=wire_array(np,U),raw_current_term=wire_array(np,V),
             raw_residual=wire_array(np,U+V),normalization='both original terms divided by row S exactly once')
    require_check(diagnostics,name,d,'ORIGINAL_EQUATION_RESIDUAL_FAILED')


def identities_and_process(a, G, H, prepared, diagnostics):
    np = a['np']; identities,Rq = prepared
    E = embedding(np,G,a['state_rows'],a['control_rows'],len(a['A']))
    states = a['metadata']['state_variable_order']
    for name,Dq,scale in identities:
        value = (Dq@E/scale).reshape(1,len(states))
        d = component_check(np,value,np.zeros_like(value),[name],states)
        require_check(diagnostics,'identity_policy:'+name,d,'IDENTITY_POLICY_RESIDUAL_FAILED')
    z = a['metadata']['partitions']['z']; k = a['metadata']['partitions']['k']
    zr,kr = [states.index(s) for s in z],[states.index(s) for s in k]
    zz = H[np.ix_(zr,zr)]; zk = H[np.ix_(zr,kr)]
    require_check(diagnostics,'process_zz',component_check(np,zz,Rq,z,z),'PROCESS_MEAN_RESIDUAL_FAILED')
    require_check(diagnostics,'process_zk',component_check(np,zk,np.zeros_like(zk),z,k),'PROCESS_ENDOGENOUS_DEPENDENCY')
