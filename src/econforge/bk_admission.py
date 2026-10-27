"""P3.7 admission of immutable trusted P3.6 evidence; no assembly or QZ rerun."""
import hashlib
import json
import math
import re

from econforge.p35_admission import canonical_json, digest, parse_manifest, MATH_SHA256
from econforge.qz_admission import SPEC_SHA256 as UPSTREAM_SPEC_SHA256, CONTEXT_FIELDS, BINDINGS, require, same
from econforge import qz_backend

SPEC_SHA256 = "813630dd67c2eead2328c489cc60998a27dc50391f2760c82456f9fc6080bfac"
EXPECTED_FIELDS = {'p36_result_sha256', 'p36_bindings', 'upstream_math_sha256', 'math_sha256'}


def decode(np, value, shape, code):
    require(type(value) is list, code, "Complex wire array required")
    raw = np.asarray(value)
    require(raw.shape == (*shape, 2) and raw.dtype.kind == 'f' and bool(np.isfinite(raw).all()),
            code, "Finite binary64 real/imag array with exact dimensions required")
    # Arithmetic construction can erase signed zero in either component,
    # changing canonical upstream pair evidence despite identical mathematics.
    decoded = np.empty(shape, dtype=np.complex128)
    decoded.real = raw[..., 0]
    decoded.imag = raw[..., 1]
    return decoded


def admit(p36_result, expected, math_spec_bytes, upstream_math_spec_bytes):
    require(type(p36_result) is dict, "MISSING_P36_RESULT", "Complete P3.6 wire snapshot required")
    r = json.loads(canonical_json(p36_result))
    e = json.loads(canonical_json(expected))
    require(type(e) is dict and set(e) == EXPECTED_FIELDS, "INVALID_CONTEXT", "Exact independent expected identity fields required")
    for key in ('p36_result_sha256', 'upstream_math_sha256', 'math_sha256'):
        require(type(e[key]) is str and re.fullmatch('[0-9a-f]{64}', e[key]) is not None, "INVALID_CONTEXT", "SHA256 expected")
    require(type(math_spec_bytes) is bytes and hashlib.sha256(math_spec_bytes).hexdigest() == e['math_sha256'] == SPEC_SHA256,
            "P37_MATH_SPEC_MISMATCH", "Current specification identity differs")
    require(type(upstream_math_spec_bytes) is bytes and hashlib.sha256(upstream_math_spec_bytes).hexdigest() == e['upstream_math_sha256'] == UPSTREAM_SPEC_SHA256,
            "P36_MATH_SPEC_MISMATCH", "Historical P3.6 specification identity differs")
    require(digest(r) == e['p36_result_sha256'], "P36_RESULT_HASH_MISMATCH", "Supplied snapshot differs from independently retained identity")
    required = dict(schema_version=2, claim='P3.6_verified_unordered_complex_QZ',
                    execution_status='COMPLETED', qz_execution_status='COMPLETED',
                    input_admission_status='ADMITTED', qz_valid=True,
                    pencil_regularity='NOT_EVALUATED', model_status='NOT_EVALUATED',
                    backend='scipy.linalg.qz', dtype='complex128', pencil='B - lambda A',
                    backend_order='NON_SEMANTIC_UNSORTED', failure_reason=None)
    require(all(k in r and same(r[k], v) for k, v in required.items()), "P36_NOT_ACCEPTED", "Accepted P3.6 result required")
    md = r.get('metadata')
    require(type(md) is dict, "INVALID_METADATA", "Upstream metadata missing")
    ctx = md.get('bindings')
    require(type(ctx) is dict and set(ctx) == CONTEXT_FIELDS and same(ctx, e['p36_bindings']),
            "P36_BINDINGS_MISMATCH", "Expected upstream bindings differ")
    require(ctx['math_sha256'] == UPSTREAM_SPEC_SHA256 and ctx['legacy_math_sha256'] == MATH_SHA256,
            "P36_MATH_SPEC_MISMATCH", "Upstream bound revisions differ")
    for key in CONTEXT_FIELDS - {'model_id', 'equation_order', 'variable_order', 'partitions'}:
        require(type(ctx[key]) is str and re.fullmatch('[0-9a-f]{64}', ctx[key]) is not None, "INVALID_CONTEXT", "Invalid upstream digest")
    m = md.get('input_metadata')
    require(type(m) is dict, "INVALID_METADATA", "P3.5 metadata missing")
    manifest = parse_manifest(m.get('admission_manifest')).to_dict()
    require(type(m.get('bindings')) is dict, "INVALID_METADATA", "P3.5 identity bindings missing")
    for key in BINDINGS:
        require(ctx[key] == m['bindings'].get(key) == manifest.get(key), "UPSTREAM_BINDING_MISMATCH", "P3.5 binding differs: " + key)
    require(manifest['math_sha256'] == m['bindings'].get('math_sha256') == MATH_SHA256
            and manifest['model_id'] == ctx['model_id'], "UPSTREAM_BINDING_MISMATCH", "P3.5 model/spec identity differs")
    cols, rows, parts = ctx['variable_order'], ctx['equation_order'], ctx['partitions']
    require(type(cols) is list and type(rows) is list and 0 < len(cols) == len(rows) <= 256,
            "DIMENSION_MISMATCH", "Bounded square axes required")
    n = len(cols)
    for key, axis in (('variable_order', cols), ('equation_order', rows)):
        require(all(type(s) is str and s for s in axis) and len(set(axis)) == n
                and same(axis, m.get(key)) and same(axis, manifest[key]), "AXIS_MISMATCH", "Explicit unique axes required")
    require(type(parts) is dict and set(parts) == {'k', 'z', 'j'} and all(type(v) is list for v in parts.values()),
            "PARTITION_MISMATCH", "Explicit k/z/j lists required")
    require(same(parts, m.get('partitions')) and same(parts, manifest['partitions'])
            and parts['k'] + parts['z'] + parts['j'] == cols, "PARTITION_MISMATCH", "Complete canonical partition order required")
    ns, nj = len(parts['k']) + len(parts['z']), len(parts['j'])
    require(all(same(m.get(k), v) for k, v in dict(n_s=ns, n_j=nj, n_k=len(parts['k']), n_z=len(parts['z']), shape=[n, n]).items())
            and m.get('state_variable_order') == parts['k'] + parts['z'] and m.get('control_variable_order') == parts['j'],
            "PARTITION_MISMATCH", "Dimension/state-control metadata differs")
    state_rows = [cols.index(s) for s in parts['k'] + parts['z']]
    coord = m.get('coordinate_convention')
    require(type(coord) is dict and digest(coord) == ctx['coordinate_sha256'], "COORDINATE_MISMATCH", "Coordinate signature differs")
    for field, ckey in (('S', 'row_scales_S'), ('T', 'coordinate_diagonal_T')):
        v = m.get(field)
        require(type(v) is list and len(v) == n and all(type(x) is float and math.isfinite(x) and x > 0 for x in v)
                and same(v, coord.get(ckey)), "COORDINATE_MISMATCH", "S/T convention differs")
    structure = m.get('structure')
    require(type(structure) is dict, "INVALID_METADATA", "Structural provenance missing")
    from econforge.bk_numerics import BKError
    if (m.get('timing_reduction') != 'NOT_PERFORMED' or m.get('dimension_expansion') != 'NOT_PERFORMED'
            or structure.get('timing_transformation') != 'NONE_DIRECT_CANONICAL_ONLY'
            or structure.get('static_elimination') != 'NOT_PERFORMED'):
        raise BKError('UNSUPPORTED_STATIC_STRUCTURE', 'Only unchanged direct canonical structure is admitted', refused=True)
    require(m.get('system') == 'A E_t q(t+1) = B q(t)'
            and m.get('orientation') == 'equation rows; variable columns; no transpose', "PENCIL_CONVENTION_MISMATCH", "Forward canonical pencil required")
    np, linalg, versions = qz_backend.runtime()
    require(md.get('backend_versions') == versions and md.get('specification_revision') == 'p36-qz-1'
            and same(md.get('n'), n) and md.get('input_dtype') == 'float64', "P36_METADATA_MISMATCH", "P3.6 runtime/spec dimensions differ")
    matrices = []
    for key in ('input_A', 'input_B'):
        wire = md.get(key)
        require(type(wire) is list and len(wire) == n and all(type(row) is list and len(row) == n for row in wire), "MATRIX_SHAPE_MISMATCH", "Exact matrix dimensions required")
        require(all(type(x) is float and math.isfinite(x) for row in wire for x in row), "MATRIX_DTYPE_OR_NONFINITE", "Finite binary64 matrices required")
        matrices.append(np.array(wire, dtype=np.float64))
    A, B = matrices
    scale, tau, pair_tau = qz_backend.thresholds(np, A, B)
    require(md.get('pencil_scale') == scale and md.get('tau_qz') == tau and md.get('tau_pair') == pair_tau,
            "P36_THRESHOLD_MISMATCH", "Frozen upstream thresholds differ")
    SB, SA, Q, Z = [decode(np, r.get(k), (n, n), 'P36_FACTOR_INVALID') for k in ('schur_B', 'schur_A', 'Q', 'Z')]
    d = qz_backend.verify(np, A, B, (SB, SA, Q, Z), tau)
    require(all(v <= tau for v in d.values()), "P36_VERIFICATION_FAILED", "Supplied decomposition failed independent recheck")
    require(type(r.get('diagnostics')) is dict and set(r['diagnostics']) == set(d)
            and all(type(v) is float and math.isfinite(v) and 0 <= v <= tau for v in r['diagnostics'].values()),
            "P36_DIAGNOSTICS_INVALID", "Upstream diagnostics missing/invalid")
    pairs = qz_backend.pair_records(np, SB, SA, pair_tau)
    require(same(r.get('pairs'), pairs), "P36_PAIRS_MISMATCH", "Wire pairs/labels must match factor diagonals")
    complete = all(p['pair_status'] == 'FINITE' for p in pairs)
    require(r.get('scalar_spectrum_complete') is complete and r.get('assessment_status') == ('PASS' if complete else 'PASS_WITH_WARNINGS'),
            "P36_NOT_ACCEPTED", "Upstream scalar completeness/status differs")
    checks = r.get('checks')
    require(type(checks) is list and [c.get('check_id') for c in checks] == ['input.admitted', 'backend.admitted', *d]
            and all(c.get('status') == 'PASS' and c.get('evidence')
                    and all(v.get('digest') == ctx['p35_result_sha256'] for v in c['evidence']) for c in checks),
            "P36_CHECKS_INVALID", "Required upstream checks/evidence missing")
    return r, e, np, linalg, A, B, state_rows, ns, nj, tau, versions
