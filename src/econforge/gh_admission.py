"""P3.8 trusted wire admission and certificate rechecks; no QZ or AD rerun."""
import hashlib
import json
import math
import re

from econforge import bk_admission, bk_numerics, qz_admission
from econforge.matrix_assembly import _verify_p34, PERMUTATIONS
from econforge.model import parse_model
from econforge.p35_admission import canonical_json, digest, validate_admission
from econforge.p35_structure import validate_structure
from econforge.qz_admission import QZError, require, same

SPEC_SHA256 = '8ef5dd847529861780168322a1db56ffe22b6336e72ba4124abea026fc766ba5'
SPEC_IDENTITIES = dict(p38=SPEC_SHA256, p37=bk_admission.SPEC_SHA256,
                       p36=qz_admission.SPEC_SHA256, legacy=qz_admission.MATH_SHA256)
EXPECTED_FIELDS = {'schema_version', 'p37_result_sha256', 'p37_bindings',
                   'verification_bundle_sha256', 'bindings', 'specifications'}
BUNDLE_FIELDS = {'schema_version', 'model_source', 'p34_result', 'p35_result',
                 'scalar_reference_verification', 'alias_maps', 'identity_maps', 'bindings'}


class GHError(QZError):
    def __init__(self, code, message, *, refused=False, numerical=False):
        super().__init__(code, message, refused=refused)
        self.numerical = numerical


def need(condition, code, message, **kwargs):
    if not condition:
        raise GHError(code, message, **kwargs)


def _finite_number(value, *, positive=False):
    return type(value) is float and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _matching_record(record, left, right, left_classes, right_classes):
    """Validate the retained assignment itself, allowing tied-root permutations."""
    n = len(left)
    need(type(record) is dict and set(record) == {'passed','threshold','max_projective_distance','matching'}
         and record['passed'] is True and record['threshold'] == bk_numerics.ROOT_MATCH
         and _finite_number(record['max_projective_distance'])
         and record['max_projective_distance'] <= bk_numerics.ROOT_MATCH
         and type(record['matching']) is list and len(record['matching']) == n
         and all(type(i) is int for i in record['matching']) and set(record['matching']) == set(range(n)),
         'P37_MATCHING_INVALID', 'Complete passing one-to-one retained assignment required')
    p,q = bk_numerics.normalized_pairs(left),bk_numerics.normalized_pairs(right)
    need(all(abs(p[i][0]*q[j][1]-p[i][1]*q[j][0]) <= bk_numerics.ROOT_MATCH
             and left_classes[i] == right_classes[j] for i,j in enumerate(record['matching'])),
         'P37_MATCHING_INVALID', 'Retained assignment must match actual pairs and classes')


def _certificate_records(r, rechecks, ns, tau, state_rows):
    """Check retained certificate semantics; fresh SVD values need not be bitwise equal."""
    reg = r.get('regularity')
    fresh = rechecks['regularity']
    need(type(reg) is dict and reg.get('status') == 'PASS' and fresh['status'] == 'PASS'
         and reg.get('tau_reg') == fresh['tau_reg'] and reg.get('common_scale') == fresh['common_scale']
         and reg.get('claim') == fresh['claim'], 'P37_REGULARITY_INVALID', 'Regularity certificate required')
    witnesses = reg.get('witnesses')
    need(type(witnesses) is list and len(witnesses) == len(fresh['witnesses']),
         'P37_REGULARITY_INVALID', 'All prescribed witnesses required')
    for i, item in enumerate(witnesses):
        need(type(item) is dict and type(item.get('mu')) is int and item['mu'] == i
             and all(_finite_number(item.get(k)) for k in ('rho', 'sigma_min', 'sigma_max'))
             and item['sigma_min'] <= item['sigma_max']
             and item['rho'] == (item['sigma_min']/item['sigma_max'] if item['sigma_max'] else 0.0),
             'P37_REGULARITY_INVALID', 'Malformed witness diagnostics')
    winners = [x for x in witnesses if x['rho'] > reg['tau_reg']]
    need(bool(winners) and same(reg.get('witness_mu'), winners[0]['mu'])
         and reg.get('witness_rho') == winners[0]['rho']
         and fresh['witnesses'][winners[0]['mu']]['rho'] > fresh['tau_reg'],
         'P37_REGULARITY_INVALID', 'Retained witness must independently pass')
    proj = r.get('projection')
    p = rechecks['projection']
    need(type(proj) is dict and proj.get('status') == p['status'] == 'PASS'
         and same(proj.get('state_row_indices'), state_rows)
         and proj.get('tau_projection') == p['tau_projection']
         and proj.get('vacuous') is (ns == 0), 'P37_PROJECTION_INVALID', 'State projection certificate differs')
    if ns:
        need(all(_finite_number(proj.get(k), positive=True) for k in ('projection_margin', 'sigma_min', 'sigma_max'))
             and proj['sigma_min'] <= proj['sigma_max']
             and proj['projection_margin'] == proj['sigma_min']/proj['sigma_max']
             and proj['projection_margin'] > p['tau_projection'],
             'P37_PROJECTION_INVALID', 'Projection must strictly pass')
    else:
        need(all(proj.get(k) is None for k in ('projection_margin', 'sigma_min', 'sigma_max')),
             'P37_PROJECTION_INVALID', 'Empty projection has null measurements')
    d = r.get('ordered_diagnostics')
    need(type(d) is dict and set(d) == set(rechecks['ordered'])
         and d.get('classes') == rechecks['ordered']['classes']
         and same(d.get('stable_count'), ns) and d.get('stable_block_contiguous') is True
         and same(d.get('small_nonzero_beta_indices'),rechecks['ordered']['small_nonzero_beta_indices']),
         'P37_DIAGNOSTICS_INVALID', 'Complete ordered diagnostics required')
    need(type(d.get('structural')) is dict and set(d['structural']) == set(rechecks['ordered']['structural'])
         and all(_finite_number(v) and v <= tau for v in d['structural'].values()),
         'P37_DIAGNOSTICS_INVALID', 'All six retained diagnostics must pass')
    n = len(r['unordered_classes'])
    for key in ('spectrum', 'conjugacy'):
        m = d.get(key)
        need(type(m) is dict and m.get('passed') is True and m.get('threshold') == bk_numerics.ROOT_MATCH
             and _finite_number(m.get('max_projective_distance'))
             and m['max_projective_distance'] <= bk_numerics.ROOT_MATCH
             and type(m.get('matching')) is list and len(m['matching']) == n
             and all(type(i) is int for i in m['matching']) and set(m['matching']) == set(range(n)),
             'P37_DIAGNOSTICS_INVALID', 'Complete one-to-one matching diagnostics required')
    checks = r.get('checks')
    ids = ['input.admitted', 'regularity.admitted', *['ordered.'+k for k in rechecks['ordered']['structural']]]
    need(type(checks) is list and all(type(c) is dict for c in checks)
         and [c.get('check_id') for c in checks] == ids
         and all(c.get('status') == 'PASS' and type(c.get('evidence')) is list and c['evidence']
                 and all(type(e) is dict and e.get('digest') == r['metadata']['bindings']['p36_result_sha256']
                         for e in c['evidence']) for c in checks),
         'P37_CHECKS_INVALID', 'Required P3.7 evidenced checks missing')
    for c in checks[2:]:
        need(_finite_number(c.get('value')) and c['value'] <= tau and c.get('threshold') == tau,
             'P37_CHECKS_INVALID', 'Ordered check measurement/threshold invalid')


def admit(p37_result, expected, bundle, specifications):
    """Detach inputs and compare all identities to independently retained context."""
    need(bundle is not None, 'VERIFICATION_BUNDLE_MISSING', 'Separate original source bundle required', refused=True)
    need(type(p37_result) is dict, 'P37_RESULT_MISSING', 'Accepted P3.7 wire required')
    r, e, b = [json.loads(canonical_json(x)) for x in (p37_result, expected, bundle)]
    need(type(e) is dict and set(e) == EXPECTED_FIELDS and type(e['schema_version']) is int,
         'INVALID_CONTEXT', 'Exact versioned trusted context required')
    need(e['schema_version'] == 1, 'UNSUPPORTED_CONTEXT_VERSION', 'Context version 1 required', refused=True)
    for key in ('p37_result_sha256', 'verification_bundle_sha256'):
        need(type(e[key]) is str and re.fullmatch('[0-9a-f]{64}', e[key]) is not None,
             'INVALID_CONTEXT', 'Trusted SHA256 required')
    need(type(specifications) is dict and set(specifications) == set(SPEC_IDENTITIES)
         and same(e['specifications'], SPEC_IDENTITIES), 'SPEC_HASH_MISMATCH', 'All specification identities required')
    for key, value in SPEC_IDENTITIES.items():
        need(type(specifications[key]) is bytes and hashlib.sha256(specifications[key]).hexdigest() == value,
             'SPEC_HASH_MISMATCH', 'Frozen specification bytes differ: '+key)
    need(digest(r) == e['p37_result_sha256'], 'P37_RESULT_HASH_MISMATCH', 'P3.7 differs from trusted retained wire')
    need(type(b) is dict, 'BUNDLE_MALFORMED', 'Original bundle must be a finite JSON object')
    need(not (BUNDLE_FIELDS-set(b)) and all(b[k] is not None for k in
         ('model_source','p34_result','p35_result','scalar_reference_verification','alias_maps','identity_maps')),
         'ORIGINAL_CONTENT_MISSING', 'Required original content is absent', refused=True)
    if type(b['p34_result']) is dict:
        need(b['p34_result'].get('blocks') is not None and b['p34_result'].get('flattened_jacobian') is not None,
             'ORIGINAL_CONTENT_MISSING', 'Original raw derivative blocks/flattened evidence are missing', refused=True)
    need(set(b) == BUNDLE_FIELDS and type(b.get('schema_version')) is int,
         'BUNDLE_MALFORMED', 'Complete original content bundle required')
    need(b['schema_version'] == 1, 'UNSUPPORTED_BUNDLE_VERSION', 'Bundle version 1 required', refused=True)
    need(digest(b) == e['verification_bundle_sha256'], 'BUNDLE_HASH_MISMATCH', 'Bundle differs from trusted content')
    need(type(r.get('schema_version')) is int and r['schema_version'] == 2,
         'UNSUPPORTED_P37_SCHEMA', 'Only accepted P3.7 schema 2', refused=True)
    required = dict(claim='P3.7_numerical_BK_conditions_only', execution_status='COMPLETED',
                    input_admission_status='ADMITTED', ordering_execution_status='COMPLETED',
                    ordered_qz_valid=True, classification_valid=True, bk_assessment='BK_DETERMINATE',
                    model_status='NOT_EVALUATED', backend='scipy.linalg.ordqz', backend_order='STABLE_FIRST',
                    dtype='complex128', pencil='B - lambda A', boundary_count=0, unresolved_pair_count=0,
                    failure_reason=None)
    need(all(k in r and same(r[k], v) for k, v in required.items()), 'P37_NOT_ACCEPTED', 'All P3.7 admission conditions required')
    need(type(r.get('warnings')) is list and all(type(x) is str for x in r['warnings'])
         and r.get('assessment_status') == ('PASS_WITH_WARNINGS' if r['warnings'] else 'PASS'),
         'P37_NOT_ACCEPTED', 'Successful scoped aggregate required')
    md = r.get('metadata')
    need(type(md) is dict and same(md.get('bindings'), e['p37_bindings']),
         'P37_BINDINGS_MISMATCH', 'P3.7 bindings differ from independently retained context')
    upstream, _, np, linalg, A, B, sr, ns, nj, tau, versions = bk_admission.admit(
        r.get('upstream_result'), e['p37_bindings'], specifications['p37'], specifications['p36'])
    need(not any(p['pair_status'] == 'INDETERMINATE_0_OVER_0' for p in upstream['pairs']),
         'P37_SPECIAL_PAIR', 'Upstream 0/0 is forbidden')
    need(same(r.get('n_state'), ns) and same(r.get('n_jump'), nj)
         and same(r.get('stable_count'), ns) and same(r.get('outside_count'), nj),
         'P37_COUNT_MISMATCH', 'Complete complementary dimensions required')
    thresholds = dict(specification_revision='p37-bk-1', backend_versions=versions, tau_qz=tau,
                      root_band=bk_numerics.ROOT_BAND, normalized_pair_zero=bk_numerics.PAIR_ZERO,
                      root_match_atol=bk_numerics.ROOT_MATCH, state_row_indices=sr, input_dtype='float64')
    need(all(same(md.get(k), v) for k, v in thresholds.items()), 'P37_METADATA_MISMATCH', 'Pinned axes/runtime/thresholds differ')
    alpha = np.array([complex(*p['alpha']) for p in upstream['pairs']], dtype=np.complex128)
    beta = np.array([complex(*p['beta']) for p in upstream['pairs']], dtype=np.complex128)
    classes, small = bk_numerics.classify_pairs(np, A, B, alpha, beta)
    need(classes.count('STABLE') == ns and classes.count('OUTSIDE') == nj
         and same(r.get('unordered_classes'), classes) and same(md.get('small_nonzero_beta_indices'), small)
         and same(r.get('exact_infinite_count'), int(np.count_nonzero(beta == 0)))
         and same(r.get('upstream_numerical_infinite_count'), sum(p['pair_status'] == 'INFINITE' for p in upstream['pairs'])),
         'P37_CLASSIFICATION_MISMATCH', 'All admitted homogeneous classes/counts required')
    conjugacy = bk_numerics.conjugacy(list(zip(alpha, beta)), classes)
    need(type(md.get('unordered_conjugacy')) is dict and md['unordered_conjugacy'].get('passed') is True,
         'P37_DIAGNOSTICS_INVALID', 'Retained conjugacy certificate required')
    ordered = r.get('ordered_result')
    need(type(ordered) is dict and same(ordered.get('stable_block_size'), ns)
         and ordered.get('source_p36_result_sha256') == e['p37_bindings']['p36_result_sha256'],
         'P37_ORDERED_INVALID', 'Stable-first result provenance required')
    output = tuple(bk_admission.decode(np, ordered.get('ordered_'+k), (len(A),) if k in ('alpha', 'beta') else A.shape,
                                      'P37_FACTOR_INVALID') for k in ('schur_B', 'schur_A', 'alpha', 'beta', 'Q', 'Z'))
    rechecks = dict(regularity=bk_numerics.regularity(np, linalg, A, B),
                    ordered=bk_numerics.verify_order(np, A, B, output, list(zip(alpha, beta)), classes, tau),
                    projection=bk_numerics.projection(np, linalg, output[5], sr, ns), unordered_conjugacy=conjugacy)
    old_pairs = list(zip(alpha,beta)); new_pairs = list(zip(output[2],output[3]))
    old_classes,new_classes = classes,rechecks['ordered']['classes']
    _matching_record(md['unordered_conjugacy'],old_pairs,
                     [(a.conjugate(),b.conjugate()) for a,b in old_pairs],old_classes,old_classes)
    _matching_record(r.get('ordered_diagnostics',{}).get('spectrum'),old_pairs,new_pairs,old_classes,new_classes)
    _matching_record(r.get('ordered_diagnostics',{}).get('conjugacy'),new_pairs,
                     [(a.conjugate(),b.conjugate()) for a,b in new_pairs],new_classes,new_classes)
    need(ordered.get('ordered_classes') == rechecks['ordered']['classes'], 'P37_ORDERED_INVALID', 'Ordered classes differ')
    _certificate_records(r, rechecks, ns, tau, sr)
    # Validate the separately retained original assembly, without reassembling it.
    p36_context = e['p37_bindings']['p36_bindings']
    p35, _ = qz_admission.admit(b['p35_result'], p36_context, specifications['p36'])
    need(same(p35['metadata'], upstream['metadata']['input_metadata'])
         and same(p35['A'], upstream['metadata']['input_A']) and same(p35['B'], upstream['metadata']['input_B']),
         'P35_LINEAGE_MISMATCH', 'Original assembly and P3.6 input differ')
    need(type(b['model_source']) is str, 'MODEL_SOURCE_MISSING', 'Exact original source required', refused=True)
    model = parse_model(b['model_source'])
    validate_admission(model, b['p34_result'], p35['metadata']['admission_manifest'], math_spec_bytes=specifications['legacy'])
    structure = validate_structure(model)
    S, T, scalar = _verify_p34(model, b['p34_result'], structure)
    m = p35['metadata']
    need(same(structure, m['structure']) and same(S, m['S']) and same(T, m['T'])
         and same({k:list(getattr(model,k)) for k in PERMUTATIONS}, m['permutations'])
         and same(json.loads(json.dumps(scalar.to_dict(), allow_nan=False)), b['scalar_reference_verification'])
         and same(b['scalar_reference_verification'], m['scalar_reference_verification']),
         'ORIGINAL_SOURCE_MISMATCH', 'Original reference, structure, coordinates and permutations must match')
    bindings = dict(**p36_context, p36_result_sha256=e['p37_bindings']['p36_result_sha256'],
                    p37_result_sha256=e['p37_result_sha256'], p37_math_sha256=SPEC_IDENTITIES['p37'],
                    p38_math_sha256=SPEC_SHA256, branch_id=model.branch_id)
    need(same(b['bindings'], e['bindings']) and same(e['bindings'], bindings),
         'BUNDLE_BINDINGS_MISMATCH', 'Complete original lineage must match independent context')
    need(type(b['alias_maps']) is list and type(b['identity_maps']) is list,
         'IDENTITY_MAPS_MISSING', 'Explicit original identity map lists required', refused=True)
    return dict(wire=r, expected=e, bundle=b, model=model, np=np, linalg=linalg,
                A=A, B=B, output=output, state_rows=sr,
                control_rows=[m['variable_order'].index(x) for x in m['partitions']['j']],
                n_state=ns, n_jump=nj, S=np.array(S), T=np.array(T),
                raw_current=np.array(b['p34_result']['blocks'][1]['jacobian']),
                raw_plus=np.array(b['p34_result']['blocks'][2]['jacobian']),
                rechecks=rechecks, versions=versions, metadata=m)
