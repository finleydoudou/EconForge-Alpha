"""Public frozen P3.9B API. One request, one privately verified immutable JSON delivery."""
import json
import warnings
from dataclasses import dataclass

from econforge import innovation, irf_numerics
from econforge.innovation_contract import InnovationError, need, real_matrix
from econforge.irf_contract import (CLAIM, CAPABILITY, CHECK_IDS, ARRAY_NAMES, REVISION,
    SPEC_SHA256, A2_SOURCE, MAX_SERIALIZED_PAYLOAD, canonical_json, digest,
    request_shape, impulse_support, response_budget, tensor, delivery_bytes, byte_gate, fields,
    validate_delivery)
from econforge.irf_admission import admit_b
from econforge.qz_admission import QZError
from econforge.innovation_wire import evidence, strings, validate_axes_coordinates, validate_diagnostic


@dataclass(frozen=True, init=False)
class IRFResult:
    """Returned only by compute_irf; caller dictionaries cannot invoke a public publisher."""
    _delivery_json: str

    def __init__(self):
        raise TypeError('IRFResult is created only by the verified compute_irf publisher')

    def to_dict(self):
        return json.loads(self._delivery_json)['result']

    def delivery_dict(self):
        return json.loads(self._delivery_json)

    def to_json(self):
        return self._delivery_json

    def result_sha256(self):
        return digest(self.to_dict())


def result_template():
    r = innovation.result_template()
    for k in ('L_s', 'L_j', 'L_s_shape', 'L_j_shape', 'source_certificate', 'forcing_certification'):
        r.pop(k)
    r.update(schema='p39-linear-irf-result', capability=CAPABILITY, requested_claim=CLAIM,
        required_checks=list(CHECK_IDS), checks=[dict(check_id=k, status='NOT_RUN', evidence=[],
            reason=None, diagnostic_ids=[], vacuous=False) for k in CHECK_IDS], request=None,
        horizon_indices=None, impulse_ids=None, state_irf=None, control_irf=None,
        state_irf_shape=None, control_irf_shape=None, stored_state_irf=None, stored_control_irf=None,
        law_result_sha256=None, upstream_model_status=None,
        limitations=['Verified finite-horizon first-order linear responses about the selected deterministic reference only',
            'The accepted local causal uniformly L2-bounded law and its conditional assumptions are inherited premises',
            'B does not issue a new uniqueness certificate; recurrence consistency is not a global forward-error guarantee',
            'Only original diagonal covariance and explicit physical_vector or coordinate_sd impulses are supported',
            'Stored-coordinate deviations use T once; nonlinear levels, display aliases and annualization are unsupported',
            'Hashes provide integrity, not authorship; future consumers must re-admit against separate expected identities',
            'Finite underflow observations are incomplete; no warning does not prove absence of BLAS underflow',
            'Independent B acceptance, Windows/Linux, Dynare, EF-REF-1 and external scientific validation remain NOT_RUN',
            'Simulation, estimation, higher order, unrestricted covariance, P3.10 and release validation are excluded'])
    return r


def check(r, name, status='PASS', diagnostic_id=None, reason=None, vacuous=False):
    row = r['checks'][CHECK_IDS.index(name)]
    if diagnostic_id is not None:
        row['diagnostic_ids'].append(diagnostic_id)
    row.update(status=status, reason=reason, vacuous=vacuous,
        evidence=[dict(kind='executed_B_gate', location='inline:diagnostics.' + diagnostic_id
            if diagnostic_id else 'private:verified_' + name, digest=r['trust_sha256'])])
    if vacuous:
        row['evidence'].append(dict(kind='explicit_vacuous_definition',
            location='inline:axes,request; N=1 zero transitions, empty control axis, or unrequested stored outputs as applicable',
            digest=r['trust_sha256']))


def suppress(r):
    r.update(product_valid=False, claim=None, **dict.fromkeys(ARRAY_NAMES))


def fail(r, stage, exc):
    if isinstance(exc, InnovationError):
        code, origin = exc.code, exc.kind
    elif isinstance(exc, QZError):
        numeric=getattr(exc,'numerical',False)
        code = 'DEPENDENCY_MISSING' if 'BACKEND_VERSION' in exc.code else 'UPSTREAM_GATE_UNRESOLVED'
        origin = 'missing' if code == 'DEPENDENCY_MISSING' else 'uncertain' if numeric and exc.refused else 'check' if numeric else 'unsupported' if exc.refused else 'input'
    elif isinstance(exc, ImportError):
        code, origin = 'DEPENDENCY_MISSING', 'missing'
    elif isinstance(exc, (ValueError, TypeError)) and r['computation_status'] == 'NOT_RUN':
        code, origin = 'INVALID_INPUT', 'input'
    elif isinstance(exc, (FloatingPointError, OverflowError)):
        code, origin = 'NUMERICAL_NONFINITE', 'backend'
    else:
        code, origin = 'BACKEND_EXCEPTION', 'backend'
    refused = origin in ('missing', 'unsupported', 'request_refusal', 'byte_refusal', 'uncertain')
    if stage == 'B.serialization' and code != 'BYTE_LIMIT':
        code, origin, refused = 'SERIALIZATION_FAILED', 'backend', False
    reason = 'P39.B.' + code
    suppress(r)
    r.update(execution_status='COMPLETED' if origin in ('check', 'byte_refusal', 'uncertain') else 'NOT_RUN' if refused else 'FAILED',
        assessment_status='REFUSED' if refused else 'FAIL', failure_reason=reason, failure_reasons=[reason],
        model_status='UNSUPPORTED' if origin == 'unsupported' else
            'NUMERICALLY_UNRESOLVED' if origin in ('check', 'backend', 'uncertain') else 'NOT_EVALUATED')
    if origin == 'backend' and stage not in ('B.serialization', 'B.resources_actual', 'B.atomicity') and r['computation_status'] != 'NOT_RUN':
        r['computation_status'] = 'FAILED'
    check(r, stage, 'UNSUPPORTED' if refused else 'FAIL', reason=reason)
    # Failure delivery is intentionally small; response summaries may themselves exceed the budget.
    r['diagnostics'] = [dict(id='B.failure', kind='event', values=dict(stage=stage, reason=reason,
        message=str(exc)[:1024], exception_type=type(exc).__name__))]


def preflight_bytes(r, resource):
    """Bound all response floats, brackets, known escaped metadata and bounded diagnostics."""
    q, a = r['request'], r['axes']; N, columns = q['N'], r['impulse_ids']
    label_bytes = max(len(canonical_json(a['state_order'])), len(canonical_json(a['control_order'])))
    column_bytes = len(canonical_json(columns))
    longest_label = max([len(canonical_json(x)) for x in a['state_order'] + a['control_order'] + columns] or [0])
    summaries = 2 * N + 1 + (2 * N if q['stored_deviations'] else 0)
    operations = 4 * N + 1 + (4 * N if q['stored_deviations'] else 0)
    # A summary plus its worst-component event, IDs/evidence and numeric text bounds.
    diagnostics_bound = summaries * (2048 + label_bytes + column_bytes + 2 * longest_label)
    # At most one bounded underflow event+warning per matrix operation.
    diagnostics_bound += operations * 1536
    delimiters = sum(2 + 2 * x['shape'][0] + 2 * x['shape'][0] * x['shape'][1]
                     + x['elements'] for x in resource['counted_arrays'])
    known = len(canonical_json(r).encode('utf-8')) + len(canonical_json(resource).encode('utf-8'))
    estimate = known + 32 * resource['response_elements'] + delimiters + diagnostics_bound + 4096
    need(estimate <= MAX_SERIALIZED_PAYLOAD, 'BYTE_LIMIT', 'Conservative complete-delivery preflight exceeds 64 MiB', 'request_refusal')
    resource['estimated_bytes'] = estimate
    return estimate


def validate_wire(r, verified_wire, identity_wire=None):
    fields(r, set(result_template()), 'SERIALIZATION_FAILED')
    need(r['schema'] == 'p39-linear-irf-result' and type(r['revision']) is int and r['revision'] == 1
         and type(r['schema_version']) is int and r['schema_version'] == 2
         and r['capability'] == CAPABILITY and r['requested_claim'] == CLAIM,
         'SERIALIZATION_FAILED', 'Exact B literals required', 'backend')
    need(type(r['product_valid']) is bool and r['required_checks'] == CHECK_IDS
         and type(r['checks']) is list and [x['check_id'] for x in r['checks']] == CHECK_IDS,
         'SERIALIZATION_FAILED', 'Complete B check registry required', 'backend')
    if r['product_valid']:
        need(verified_wire is not None and canonical_json(r) == verified_wire,
             'SERIALIZATION_FAILED', 'Publication differs from independently verified immutable content', 'backend')
        need(identity_wire is not None, 'SERIALIZATION_FAILED', 'Private precomputation identities required', 'backend')
        identity = json.loads(identity_wire)
        for key in ('provenance', 'axes', 'coordinates', 'request', 'backend'):
            need(canonical_json(r[key]) == canonical_json(identity[key]), 'SERIALIZATION_FAILED',
                 'Published ' + key + ' differs from independently admitted precomputation content', 'backend')
        validate_axes_coordinates(r['axes'], r['coordinates'])
        need(r['claim'] == CLAIM and r['model_status'] == 'NOT_EVALUATED'
             and r['upstream_model_status'] == 'UNIQUE_LOCAL_SOLUTION'
             and r['upstream_source_commit'] == A2_SOURCE and r['upstream_stage'] == 'P3.9A2'
             and r['execution_status'] == r['computation_status'] == 'COMPLETED'
             and r['assessment_status'] == ('PASS_WITH_WARNINGS' if r['warnings'] else 'PASS')
             and r['input_admission_status'] == 'ADMITTED' and r['failure_reason'] is None
             and r['failure_reasons'] == [] and r['specification_revision'] == REVISION
             and r['specification_sha256'] == SPEC_SHA256 and r['backend'] is not None
             and r['trust_sha256'] == digest(r['provenance'])
             and r['law_result_sha256'] == r['upstream_result_sha256'] == r['provenance']['upstream']['result_sha256'],
             'SERIALIZATION_FAILED', 'Positive B truth table or source binding invalid', 'backend')
        need(all(x['status'] == 'PASS' and x['evidence'] for x in r['checks'][:-1])
             and r['checks'][-1]['status'] == ('WARN' if r['warnings'] else 'PASS'),
             'SERIALIZATION_FAILED', 'Every correctness obligation must pass', 'backend')
        a, q = r['axes'], r['request']; N, count = q['N'], len(r['impulse_ids'])
        need(type(r['horizon_indices']) is list and all(type(x) is int for x in r['horizon_indices'])
             and all(type(r[k]) is list and all(type(x) is int for x in r[k])
                     for k in ('state_irf_shape','control_irf_shape'))
             and r['horizon_indices'] == list(range(N)) and r['impulse_ids'] == [p['id'] for p in q['impulses']]
             and r['state_irf_shape'] == [N, a['n_state'], count]
             and r['control_irf_shape'] == [N, a['n_jump'], count],
             'SERIALIZATION_FAILED', 'Horizon, impulse or tensor axes differ', 'backend')
        import numpy as np
        actual = tuple(None if r[k] is None else tensor(np, r[k],
            r['state_irf_shape'] if 'state' in k else r['control_irf_shape']) for k in ARRAY_NAMES)
        need(actual[0] is not None and actual[1] is not None
             and ((actual[2] is not None and actual[3] is not None) if q['stored_deviations']
                  else actual[2] is None and actual[3] is None),
             'SERIALIZATION_FAILED', 'All requested products must be jointly present', 'backend')
        law = identity['law']; ns, nj, m = a['n_state'], a['n_jump'], a['n_innovation']
        G, H, Ls, Lj = [real_matrix(np, law[k], shape) for k, shape in
            [('G', (nj, ns)), ('H', (ns, ns)), ('L_s', (ns, m)), ('L_j', (nj, m))]]
        V = real_matrix(np, q['V'], (m, count)); positions = {x: i for i, x in enumerate(a['variable_order'])}
        scales = None if not q['stored_deviations'] else tuple(np.array(
            [r['coordinates']['T'][positions[x]] for x in a[key]], dtype=np.float64)
            for key in ('state_order', 'control_order'))
        recorded = [d for d in r['diagnostics'] if d['kind'] == 'component_summary']; observed = []; worst_events=[]
        def final_component(name, passed, record, detail, code, **kwargs):
            need(passed, 'SERIALIZATION_FAILED', 'Final published recurrence/impact/control failed', 'backend')
            if record is not None:
                observed.append(record)
            if detail is not None:
                worst_events.append(dict(stage=name,reason='COMPONENTWISE_MAXIMUM',message=canonical_json(detail),exception_type=None))
        irf_numerics.verify(np, actual, G, H, Ls, Lj, V, a, r['impulse_ids'], scales,
                            final_component, lambda name: None, None)
        need(canonical_json([d['values'] for d in recorded]) == canonical_json(observed), 'SERIALIZATION_FAILED',
             'Published component summaries differ from actual complete component checks', 'backend')
        need(canonical_json([d['values'] for d in r['diagnostics'] if d['kind']=='event'
             and d['values']['reason']=='COMPONENTWISE_MAXIMUM']) == canonical_json(worst_events),
             'SERIALIZATION_FAILED','Worst component discrepancies and tolerance bounds differ from actual measurements','backend')
        resources = [d['values'] for d in r['diagnostics'] if d['kind'] == 'resource']
        need(canonical_json(resources) == canonical_json([identity['resource']]), 'SERIALIZATION_FAILED',
             'Resource accounting differs from independent preallocation record', 'backend')
    else:
        need(r['claim'] is None and all(r[k] is None for k in ARRAY_NAMES)
             and r['assessment_status'] not in ('PASS', 'PASS_WITH_WARNINGS'),
             'SERIALIZATION_FAILED', 'Failed or unrun batch must suppress all responses', 'backend')
    need(r['independent_validation_status'] == r['external_validation_status'] == 'NOT_RUN'
         and r['acceptance_evidence'] == [] and r['capability_maturity'] == 'IMPLEMENTED',
         'SERIALIZATION_FAILED', 'Ordinary B cannot manufacture acceptance', 'backend')
    need(all(r[k] in ('COMPLETED','FAILED','NOT_RUN','SKIPPED') for k in ('execution_status','computation_status'))
         and r['assessment_status'] in ('PASS','PASS_WITH_WARNINGS','FAIL','REFUSED','NOT_RUN','NOT_IMPLEMENTED')
         and r['model_status'] in ('NOT_EVALUATED','NUMERICALLY_UNRESOLVED','UNSUPPORTED')
         and r['input_admission_status'] in ('ADMITTED','NOT_ADMITTED'),
         'SERIALIZATION_FAILED','Frozen execution/check/model vocabulary required','backend')
    diagnostics = r['diagnostics']; dids = {d['id'] for d in diagnostics}
    need(len(dids) == len(diagnostics), 'SERIALIZATION_FAILED', 'Unique diagnostic IDs required', 'backend')
    for row in r['checks']:
        fields(row, {'check_id', 'status', 'evidence', 'reason', 'diagnostic_ids', 'vacuous'}, 'SERIALIZATION_FAILED')
        evidence(row['evidence']); strings(row['diagnostic_ids'], unique=True)
        need(type(row['vacuous']) is bool and set(row['diagnostic_ids']) <= dids
             and row['status'] in ('PASS','WARN','FAIL','UNSUPPORTED','NOT_IMPLEMENTED','NOT_RUN')
             and (row['reason'] is None or type(row['reason']) is str)
             and (row['status'] != 'NOT_RUN' or not row['diagnostic_ids'] and not row['evidence'] and not row['vacuous']),
             'SERIALIZATION_FAILED', 'Malformed check dimensions/diagnostic binding', 'backend')
    for d in diagnostics:
        fields(d, {'id','kind','values'}, 'SERIALIZATION_FAILED')
        if d['kind'] == 'event':
            validate_diagnostic(d)
        elif d['kind'] == 'underflow':
            v=d['values'];fields(v, {'operation','horizon','array','indices','context','observability'}, 'SERIALIZATION_FAILED')
            need((v['horizon'] is None or type(v['horizon']) is int and v['horizon'] >= 0)
                 and type(v['indices']) is list and all(type(x) is list and all(type(i) is int for i in x) for x in v['indices'])
                 and all(type(v[k]) is str for k in ('operation','array','context','observability')),
                 'SERIALIZATION_FAILED','Underflow diagnostic malformed','backend')
        else:
            need(d['kind'] in ('resource','component_summary'), 'SERIALIZATION_FAILED',
                 'Unauthorized response diagnostic kind', 'backend')
    for w in r['warnings']:
        fields(w, {'code', 'check_id', 'message', 'evidence'}, 'SERIALIZATION_FAILED'); evidence(w['evidence'])
        need(w['code'] in ('FINITE_UNDERFLOW', 'INHERITED_ACCEPTED_WARNING') and w['check_id'] in CHECK_IDS
             and type(w['message']) is str, 'SERIALIZATION_FAILED', 'Unauthorized warning', 'backend')


def _publish(candidate, verified_wire, identity_wire=None):
    """Fixed bytes are checked and emitted; no mutable validation object is an output source."""
    view = None
    try:
        wire, _ = delivery_bytes(candidate)
        view = validate_delivery(wire)
        validate_wire(view['result'], verified_wire, identity_wire)
        need(canonical_json(view) == wire, 'SERIALIZATION_FAILED', 'Validation-to-publication substitution detected', 'backend')
        result = object.__new__(IRFResult)
        object.__setattr__(result, '_delivery_json', wire)
        return result
    except Exception:
        suppress(candidate)
        if view is not None:
            suppress(view['result'])
        raise


def compute_irf(expected, content, request, *, a1_result=None):
    """Consume independently enrolled B trust and frozen content; never enroll a candidate.

    a1_result resolves the complete predecessor whose digest is already bound by
    the accepted A2 result. No upstream certification, recovery or solver runs.
    """
    r = result_template(); current = ['B.trust']
    def start(name):
        current[0] = name
    def measured(name, passed, record, detail, code, vacuous=False):
        did = None
        if record is not None:
            did = name + '.' + str(len(r['diagnostics']))
            r['diagnostics'].append(dict(id=did, kind='component_summary', values=record))
            if detail is not None:
                r['diagnostics'].append(dict(id=did + '.worst', kind='event', values=dict(
                    stage=name, reason='COMPONENTWISE_MAXIMUM', message=canonical_json(detail), exception_type=None)))
        check(r, name, 'PASS' if passed else 'FAIL', did, vacuous=vacuous or bool(record and record['vacuous']))
        need(passed, code, 'Required component verification rejected: ' + name, 'check')
    def observe(operation, horizon, context):
        did = 'B.underflow.' + str(len(r['diagnostics']))
        r['diagnostics'].append(dict(id=did, kind='underflow', values=dict(operation=operation, horizon=horizon,
            array='operation_result', indices=[], context=context[:256],
            observability='Returned subnormals and backend warnings only; BLAS intermediate and rounded-to-zero underflow may be unobservable')))
        r['warnings'].append(dict(code='FINITE_UNDERFLOW', check_id='B.finite_output',
            message=operation + ': finite underflow/subnormal context',
            evidence=[dict(kind='observed_finite_underflow', location='inline:diagnostics.' + did, digest=None)]))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            t, c, a2, a = admit_b(expected, content, a1_result)
        for w in caught:
            need(issubclass(w.category, RuntimeWarning) and 'underflow' in str(w.message).lower(),
                 'BACKEND_WARNING', str(w.message), 'backend')
            observe('accepted upstream recheck', None, str(w.message))
        r.update(provenance=json.loads(canonical_json(t)), trust_sha256=digest(t),
            upstream_result_sha256=t['upstream']['result_sha256'], law_result_sha256=t['upstream']['result_sha256'],
            upstream_stage='P3.9A2', upstream_source_commit=A2_SOURCE, upstream_model_status='UNIQUE_LOCAL_SOLUTION',
            specification_sha256=SPEC_SHA256, axes=json.loads(canonical_json(a['axes'])),
            coordinates=json.loads(canonical_json(a['coordinates'])), backend=innovation.backend_record(a))
        r['backend']['methods'] = ['accepted A2 full-digest admission', 'retained P3.8 certificate rechecks',
            'numpy.matmul float64 repeated finite-horizon propagation', 'componentwise two-term verification',
            'canonical JSON private publication with fixed-point byte receipt']
        for w in a2['warnings']:
            r['warnings'].append(dict(code='INHERITED_ACCEPTED_WARNING', check_id='B.accepted_A2',
                message=w['message'], evidence=w['evidence']))
        check(r, 'B.trust'); start('B.accepted_A2'); check(r, 'B.accepted_A2')
        np = a['np']; axes = r['axes']; ns, nj, m = axes['n_state'], axes['n_jump'], axes['n_innovation']
        law = a2['law']; G = real_matrix(np, law['G'], (nj, ns)); H = real_matrix(np, law['H'], (ns, ns))
        Ls = real_matrix(np, law['L_s'], (ns, m)); Lj = real_matrix(np, law['L_j'], (nj, m))
        start('B.request'); q, ids = request_shape(request, axes, r['coordinates'], t)
        r.update(request=q, impulse_ids=ids, horizon_indices=list(range(q['N'])),
            state_irf_shape=[q['N'], ns, len(ids)], control_irf_shape=[q['N'], nj, len(ids)])
        check(r, 'B.request'); start('B.impulse_support'); impulse_support(q, axes, law['covariance'])
        check(r, 'B.impulse_support', vacuous=m == 0)
        start('B.resources_preflight')
        resource = response_budget(q['N'], ns, nj, len(ids), q['stored_deviations'])
        preflight_bytes(r, resource)
        r['diagnostics'].append(dict(id='B.resource', kind='resource', values=resource))
        check(r, 'B.resources_preflight', diagnostic_id='B.resource')
        identity_wire = canonical_json(dict(provenance=t, axes=a['axes'], coordinates=a['coordinates'],
            request=q, backend=r['backend'], law=law, resource=resource))
        r['input_admission_status'] = 'ADMITTED'
        if m == 0:
            r.update(execution_status='SKIPPED', assessment_status='NOT_RUN')
            r['diagnostics'].append(dict(id='B.no_innovations', kind='event', values=dict(
                stage='B.request', reason='NO_INNOVATIONS', message='m=r=0; explicit empty shapes, no numerical response product', exception_type=None)))
            return _publish(r, None)
        V = real_matrix(np, q['V'], (m, len(ids)))
        indices = {name: i for i, name in enumerate(axes['variable_order'])}
        scales = None if not q['stored_deviations'] else tuple(np.array(
            [r['coordinates']['T'][indices[x]] for x in axes[key]], dtype=np.float64)
            for key in ('state_order', 'control_order'))
        start('B.impact'); r['computation_status'] = 'FAILED'
        products = irf_numerics.propagate(np, G, H, Ls, V, q['N'], scales, observe)
        r['computation_status'] = 'COMPLETED'
        irf_numerics.verify(np, products, G, H, Ls, Lj, V, axes, ids, scales, measured, start, observe)
        start('B.finite_output')
        arrays = dict(zip(ARRAY_NAMES, [None if x is None else x.tolist() for x in products]))
        # Verify the actual detached JSON arrays, not only intermediate ndarray aliases.
        wire_products = tuple(None if arrays[k] is None else tensor(np, arrays[k],
            r['state_irf_shape'] if 'state' in k else r['control_irf_shape']) for k in ARRAY_NAMES)
        irf_numerics.verify(np, wire_products, G, H, Ls, Lj, V, axes, ids, scales,
            lambda name, passed, record, detail, code, **kw: need(passed, code, 'Actual publication array failed ' + name, 'check'),
            start, None)
        check(r, 'B.finite_output')
        start('B.serialization'); check(r, 'B.serialization')
        check(r, 'B.resources_actual', diagnostic_id='B.resource'); check(r, 'B.atomicity')
        check(r, 'B.warnings', 'WARN' if r['warnings'] else 'PASS')
        r.update(**arrays, product_valid=True, claim=CLAIM, execution_status='COMPLETED',
            assessment_status='PASS_WITH_WARNINGS' if r['warnings'] else 'PASS')
        # Freeze independently admitted metadata and fully checked actual response bytes.
        verified_wire = canonical_json(r)
        start('B.resources_actual')
        _, receipt = delivery_bytes(r); byte_gate(receipt['total_bytes'])
        need(receipt['total_bytes'] <= resource['estimated_bytes'], 'SERIALIZATION_FAILED',
             'Actual delivery exceeded its preflight bound', 'backend')
        start('B.serialization')
        return _publish(r, verified_wire, identity_wire)
    except Exception as exc:
        fail(r, current[0], exc)
        try:
            return _publish(r, None)
        except Exception:
            # An oversized/malformed failure retains neither request nor provenance nor arrays.
            safe = result_template(); safe.update(execution_status=r['execution_status'],
                computation_status=r['computation_status'], assessment_status=r['assessment_status'],
                model_status=r['model_status'], failure_reason=r['failure_reason'], failure_reasons=r['failure_reasons'])
            check(safe, current[0], 'UNSUPPORTED' if r['assessment_status'] == 'REFUSED' else 'FAIL', reason=r['failure_reason'])
            safe['diagnostics'] = r['diagnostics']
            return _publish(safe, None)
