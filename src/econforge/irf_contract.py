"""Frozen P3.9B requests, component summaries and aggregate resource accounting."""
import json
import math

from econforge.innovation_contract import need, REVISION, SPEC_SHA256, real_matrix
from econforge.p35_admission import canonical_json, digest

CLAIM = 'VERIFIED_FINITE_HORIZON_LINEAR_IRF'
CAPABILITY = 'P3.9B_FINITE_HORIZON_IRF'
A2_SOURCE = 'f5a02493785e03135e8d1578745274ad70e35b57'
A2_ACCEPTANCE_SHA256 = '68c4861ac0be406acbd26685cd2b10ad3e57e63f086c19450bb54dbf382c12c8'
CHECK_IDS = ['B.' + x for x in ('trust', 'accepted_A2', 'request', 'impulse_support',
    'resources_preflight', 'impact', 'recurrence', 'controls', 'coordinates',
    'finite_output', 'serialization', 'resources_actual', 'atomicity', 'warnings')]
REQUEST_FIELDS = {'schema', 'revision', 'N', 'innovation_order', 'impulses', 'V',
    'V_shape', 'stored_deviations', 'impulse_date', 'frequency', 'reference_sha256',
    'covariance_source_sha256', 'law_result_sha256'}
IMPULSE_FIELDS = {'id', 'mode', 'coordinate', 'amplitude', 'physical_vector', 'units'}
ARRAY_NAMES = ('state_irf', 'control_irf', 'stored_state_irf', 'stored_control_irf')
MAX_RESPONSE_ELEMENTS = 1_000_000
MAX_SERIALIZED_PAYLOAD = 67_108_864


def fields(obj, keys, code='INVALID_INPUT'):
    need(type(obj) is dict and set(obj) == set(keys), code, 'Exact revision-1 fields required')


def float_vector(value, n, code='INVALID_IMPULSE'):
    need(type(value) is list and len(value) == n and all(type(x) is float and math.isfinite(x)
         for x in value), code, 'Finite JSON floats in exact declared dimensions required')


def tensor(np, value, shape):
    """Validate every JSON float and every empty-axis bracket before reshaping."""
    def walk(v, dims):
        if not dims:
            need(type(v) is float and math.isfinite(v), 'SERIALIZATION_FAILED',
                 'Response entries must be finite JSON floats', 'backend')
        else:
            need(type(v) is list and len(v) == dims[0], 'SERIALIZATION_FAILED',
                 'Explicit unsqueezed response axes required', 'backend')
            for child in v:
                walk(child, dims[1:])
    walk(value, shape)
    return np.array(value, dtype=np.float64).reshape(shape)


def request_shape(request, axes, coords, trust):
    need(request is not None, 'INVALID_IMPULSE', 'Explicit shock request required')
    fields(request, REQUEST_FIELDS)
    need(type(request['N']) is int and 1 <= request['N'] <= 1000, 'INVALID_HORIZON',
         'N must be an integer from 1 through 1000, excluding bool/nonfinite values')
    q = json.loads(canonical_json(request))
    need(q['schema'] == 'p39-irf-request' and type(q['revision']) is int and q['revision'] == 1,
         'SOURCE_UNSUPPORTED', 'Only frozen IRF request revision 1', 'request_refusal')
    need(type(q['N']) is int and 1 <= q['N'] <= 1000, 'INVALID_HORIZON',
         'N must be an integer from 1 through 1000, excluding bool')
    need(type(q['stored_deviations']) is bool and type(q['impulse_date']) is int,
         'INVALID_INPUT', 'Explicit boolean stored flag and integer economic date required')
    need(q['innovation_order'] == axes['innovation_order'] and type(q['innovation_order']) is list,
         'IMPULSE_ORDER_INVALID', 'Exact source innovation order required')
    need(q['frequency'] == coords['frequency'] and q['reference_sha256'] == trust['reference_sha256']
         and q['covariance_source_sha256'] == trust['covariance_source_sha256'],
         'SOURCE_MISMATCH', 'Request source, baseline, frequency or covariance identity differs')
    need(q['law_result_sha256'] == trust['upstream']['result_sha256'], 'LAW_DIGEST_STALE',
         'Request must name the independently admitted complete A2 law')
    m = axes['n_innovation']
    need(type(q['impulses']) is list, 'INVALID_IMPULSE', 'Ordered explicit impulse records required')
    r = len(q['impulses'])
    need((m == 0 and r == 0) or (m > 0 and 1 <= r <= m), 'INVALID_IMPULSE',
         'Batch size must be 1..m; m=0 admits r=0 only')
    need(type(q['V_shape']) is list and len(q['V_shape']) == 2
         and all(type(x) is int for x in q['V_shape']) and q['V_shape'] == [m, r],
         'INVALID_IMPULSE', 'Explicit V shape must equal (m,r)')
    ids = []
    for impulse in q['impulses']:
        fields(impulse, IMPULSE_FIELDS, 'INVALID_IMPULSE')
        name = impulse['id']
        need(type(name) is str and bool(name.strip()) and name not in ids,
             'INVALID_IMPULSE', 'Unique nonempty requested impulse IDs required')
        ids.append(name)
        need(impulse['mode'] in ('physical_vector', 'coordinate_sd'), 'SHOCK_MODE_UNSUPPORTED',
             'Only physical_vector and coordinate_sd are admitted', 'request_refusal')
        float_vector(impulse['physical_vector'], m)
        need(impulse['units'] == coords['innovation_units'] and type(impulse['units']) is list,
             'INVALID_IMPULSE', 'Physical innovation units must match the source exactly')
        if impulse['mode'] == 'physical_vector':
            need(impulse['coordinate'] is None and impulse['amplitude'] is None,
                 'INVALID_IMPULSE', 'Physical vector itself specifies the amplitude')
        else:
            need(type(impulse['coordinate']) is str and impulse['coordinate'] in axes['innovation_order']
                 and type(impulse['amplitude']) is float and math.isfinite(impulse['amplitude']),
                 'INVALID_IMPULSE', 'Coordinate-sd requires a source coordinate and finite float amplitude')
    need(type(q['V']) is list and len(q['V']) == m, 'INVALID_IMPULSE', 'V row count differs')
    for i, row in enumerate(q['V']):
        float_vector(row, r)
        need(row == [p['physical_vector'][i] for p in q['impulses']], 'INVALID_IMPULSE',
             'V must exactly equal the ordered full physical impulse columns')
    return q, ids


def impulse_support(q, axes, covariance):
    fields(covariance, {'kind', 'values'}, 'INVALID_COVARIANCE')
    need(covariance['kind'] == 'diagonal', 'SOURCE_UNSUPPORTED',
         'Only original diagonal covariance', 'request_refusal')
    variances = covariance['values']; m = axes['n_innovation']
    need(type(variances) is list and len(variances) == m and all(type(v) in (int, float)
         and math.isfinite(v) and v >= 0 for v in variances), 'INVALID_COVARIANCE',
         'Original finite nonnegative variances required')
    for p in q['impulses']:
        vector = p['physical_vector']
        need(all(v != 0 or vector[i] == 0.0 for i, v in enumerate(variances)),
             'INACTIVE_SUPPORT', 'Inactive coordinates require EXACT zero impulses', 'request_refusal')
        if p['mode'] == 'coordinate_sd':
            j = axes['innovation_order'].index(p['coordinate'])
            need(variances[j] > 0, 'INACTIVE_SUPPORT', 'Zero-variance sd normalization refused', 'request_refusal')
            need(all(x == 0.0 for i, x in enumerate(vector) if i != j),
                 'INVALID_IMPULSE', 'Coordinate-sd sets all other coordinates exactly zero')
            expected = p['amplitude'] * math.sqrt(variances[j])
            need(math.isfinite(expected), 'INVALID_IMPULSE', 'Normalized physical impulse must be finite')
            need(abs(vector[j] - expected) <= 1e-10 + 1e-8 * abs(expected),
                 'INVALID_IMPULSE', 'Coordinate-sd physical magnitude differs; normalization occurs once')


def response_budget(N, ns, nj, r, stored):
    """Python arbitrary-precision integers; inclusive aggregate count, before allocation."""
    need(all(type(x) is int and x >= 0 for x in (N, ns, nj, r)) and type(stored) is bool,
         'INVALID_INPUT', 'Integer resource dimensions required')
    counted = [dict(name=name, shape=[N, size, r], elements=N * size * r)
               for name, size in [('state_irf', ns), ('control_irf', nj)]
               + ([('stored_state_irf', ns), ('stored_control_irf', nj)] if stored else [])]
    total = sum(x['elements'] for x in counted)
    need(total <= MAX_RESPONSE_ELEMENTS, 'ELEMENT_LIMIT', 'Aggregate response element ceiling exceeded', 'request_refusal')
    return dict(response_elements=total, element_limit=MAX_RESPONSE_ELEMENTS,
        estimated_bytes=0, actual_bytes=None, byte_limit=MAX_SERIALIZED_PAYLOAD,
        counted_arrays=counted, metadata_elements_included=False)


def summary(np, left, right, rows, columns, horizon):
    """Check ALL components; retain scalar maxima instead of response-valued copies."""
    shape = (len(rows), len(columns))
    real_matrix(np, left, shape, computed=True); real_matrix(np, right, shape, computed=True)
    # Scale before addition, so a large finite response need not overflow its tolerance.
    error = np.abs(left - right)
    bound = 1e-10 + 1e-8 * np.abs(left) + 1e-8 * np.abs(right)
    need(bool(np.isfinite(error).all()) and bool(np.isfinite(bound).all()),
         'NUMERICAL_NONFINITE', 'Nonfinite component error or bound', 'backend')
    ratios = error / bound
    worst = list(map(int, np.unravel_index(int(np.argmax(ratios)), shape))) if ratios.size else None
    result = dict(shape=list(shape), horizon=horizon, row_labels=list(rows), column_labels=list(columns),
        checked_components=int(error.size), max_absolute_error=float(np.max(error)) if error.size else None,
        max_error_over_bound=float(np.max(ratios)) if ratios.size else None, worst_index=worst,
        atol=1e-10, rtol=1e-8, rule='two_term', vacuous=not bool(error.size))
    detail = None if worst is None else dict(horizon=horizon, variable=rows[worst[0]],
        impulse=columns[worst[1]], absolute_discrepancy=float(error[tuple(worst)]),
        actual_tolerance_bound=float(bound[tuple(worst)]), validation_status='PASS' if bool(np.all(error <= bound)) else 'FAIL')
    return bool(np.all(error <= bound)), result, detail


def delivery_bytes(result):
    """Canonical complete delivery and exact receipt fixed point, with final remeasurement."""
    inner = canonical_json(result).encode('utf-8')
    receipt = dict(result_bytes=len(inner), receipt_bytes=0, total_bytes=0)
    for _ in range(32):
        record = canonical_json(receipt).encode('utf-8')
        total = len(b'{"receipt":') + len(record) + len(b',"result":') + len(inner) + 1
        revised = dict(result_bytes=len(inner), receipt_bytes=len(record), total_bytes=total)
        if revised == receipt:
            wire = b'{"receipt":' + record + b',"result":' + inner + b'}'
            need(len(wire) == receipt['total_bytes'] and len(record) == receipt['receipt_bytes'],
                 'SERIALIZATION_FAILED', 'Actual delivery size differs from receipt', 'backend')
            return wire.decode('utf-8'), receipt
        receipt = revised
    need(False, 'SERIALIZATION_FAILED', 'Receipt digit-length fixed point failed', 'backend')


def byte_gate(total):
    need(type(total) is int and 0 <= total <= MAX_SERIALIZED_PAYLOAD,
         'BYTE_LIMIT', 'Complete canonical delivery exceeds 64 MiB', 'byte_refusal')


def validate_delivery(wire):
    """Measure the exact immutable publication, independently of serializer metadata."""
    need(type(wire) is str, 'SERIALIZATION_FAILED',
         'Publication must be an immutable JSON string', 'backend')
    actual_bytes = len(wire.encode('utf-8'))
    byte_gate(actual_bytes)
    view = json.loads(wire)
    fields(view, {'result', 'receipt'}, 'SERIALIZATION_FAILED')
    receipt = view['receipt']
    fields(receipt, {'result_bytes', 'receipt_bytes', 'total_bytes'}, 'SERIALIZATION_FAILED')
    need(all(type(x) is int and x >= 0 for x in receipt.values()),
         'SERIALIZATION_FAILED', 'Receipt counts must be nonnegative integers', 'backend')
    measured = dict(result_bytes=len(canonical_json(view['result']).encode('utf-8')),
        receipt_bytes=len(canonical_json(receipt).encode('utf-8')), total_bytes=actual_bytes)
    need(receipt == measured, 'SERIALIZATION_FAILED',
         'Receipt differs from independently measured final publication bytes', 'backend')
    need(canonical_json(view) == wire, 'SERIALIZATION_FAILED',
         'Final publication must use the frozen canonical JSON encoding', 'backend')
    return view
