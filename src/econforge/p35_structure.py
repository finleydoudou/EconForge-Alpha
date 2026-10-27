"""Conservative direct-canonical syntax admission, never differentiation.

No timing reduction, coefficient extraction, variable expansion or elimination.
"""
import json
from econforge.model import expand_aliases
from econforge.model_ast import references
from econforge.model_errors import ModelError


def _terms(node, sign=1):
    """Expose additive syntax only, retaining nonlinear subexpressions intact."""
    if node.kind == 'binary' and node.value in ('+', '-'):
        return _terms(node.children[0],sign)+_terms(node.children[1],sign if node.value=='+' else -sign)
    if node.kind == 'unary':
        return _terms(node.children[0],sign if node.value=='+' else -sign)
    return [(sign,node)]


def _affine(node, registry):
    """Syntactic degree bound in current z only; no coefficients/derivatives."""
    refs=references(node)
    if not any(registry[name].role!='parameter' for name,_ in refs):
        return 0
    if node.kind=='symbol':
        if registry[str(node.value)].role=='z' and node.offset==0:
            return 1
        raise ModelError('UNSUPPORTED_PROCESS_STRUCTURE','Process requires current z and fixed parameters only',node.source,unsupported=True)
    degrees=[_affine(c,registry) for c in node.children]
    if node.kind=='unary': return degrees[0]
    if node.kind=='binary':
        if node.value in ('+','-'): return max(degrees)
        if node.value=='*' and sum(degrees)<=1: return sum(degrees)
        if node.value=='/' and degrees[1]==0: return degrees[0]
    raise ModelError('UNSUPPORTED_PROCESS_STRUCTURE','Direct affine AR/VAR syntax required; no pointwise inference',node.source,unsupported=True)


def validate_structure(model):
    registry={s.name:s for s in model.symbols}
    rows=[e for e in model.equations if e.kind!='identity']
    innovations=json.loads(model.source_json)['innovations']
    z_names=[v.symbol.name for v in model.variables if v.symbol.role=='z']
    if len(innovations)!=len(z_names):
        raise ModelError('UNSUPPORTED_INNOVATION_STRUCTURE','One explicit physical innovation per z required',unsupported=True)
    shock_for=dict(zip(z_names,[s['id'] for s in innovations]))
    original_slots=[];static_rows=[]
    for row in rows:
        tree=expand_aliases(row.tree,dict(model.aliases),registry)
        refs=references(tree);original_slots.append([list(x) for x in refs])
        dynamic=[(name,offset) for name,offset in refs if registry[name].role in ('k','z','j')]
        leads=[x for x in dynamic if x[1]==1]
        if any(offset==-1 for _,offset in dynamic):
            code='UNSUPPORTED_STATIC_STRUCTURE' if row.kind=='conditional' and not leads else 'UNSUPPORTED_LAG_STRUCTURE'
            raise ModelError(code,'Raw lagged syntax requires separately reviewed re-expression, even if its point derivative is zero',row.equation_id,unsupported=True)
        shockrefs=[x for x in refs if registry[x[0]].role=='innovation']
        if row.kind=='conditional':
            if shockrefs:
                raise ModelError('UNSUPPORTED_INNOVATION_STRUCTURE','Current forcing must be a declared exogenous state',row.equation_id,unsupported=True)
            if not leads: static_rows.append(row.equation_id)
            continue
        # Canonical transitions explicitly contain +target(+1), exactly once.
        terms=_terms(tree)
        target=[(sign,node) for sign,node in terms if node.kind=='symbol' and node.value==row.target and node.offset==1]
        if len(target)!=1 or target[0][0]!=1 or leads!=[(row.target,1)]:
            raise ModelError('UNSUPPORTED_TRANSITION_STRUCTURE','Require direct +target(+1) with no other lead incidence',row.equation_id,unsupported=True)
        if row.kind=='k_transition':
            if shockrefs:
                raise ModelError('UNSUPPORTED_INNOVATION_STRUCTURE','Physical k transition cannot contain innovations',row.equation_id,unsupported=True)
        elif row.kind=='z_process':
            shock=shock_for[row.target]
            selected=[(sign,node) for sign,node in terms if node.kind=='symbol' and node.value==shock and node.offset==1]
            if len(selected)!=1 or selected[0][0]!=-1 or shockrefs!=[(shock,1)]:
                raise ModelError('UNSUPPORTED_INNOVATION_STRUCTURE','Require exactly -u(+1) in declared physical shock order',row.equation_id,unsupported=True)
            for term in terms:
                if term not in target and term not in selected:
                    _affine(term[1],registry)
        else:
            raise ModelError('UNSUPPORTED_ROW_STRUCTURE','Unknown canonical row kind',row.equation_id,unsupported=True)
    return {'original_slots':original_slots,'static_equation_order':static_rows,
            'static_elimination':'NOT_PERFORMED','static_variable_classification':'NOT_INFERRED',
            'innovation_order':[s['id'] for s in innovations],
            'innovation_metadata':innovations,'covariance':json.loads(model.source_json)['covariance'],
            'process_equations':[{'equation_id':e.equation_id,'expression':e.expression,'target':e.target} for e in rows if e.kind=='z_process'],
            'forcing_stability':'NOT_EVALUATED','timing_transformation':'NONE_DIRECT_CANONICAL_ONLY'}
