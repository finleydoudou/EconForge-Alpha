"""Strict B admission of an independently accepted, separately enrolled complete A2 law."""
import json
from econforge.irf_contract import A2_SOURCE, A2_ACCEPTANCE_SHA256, fields
from econforge.innovation_contract import InnovationError, need, TRUST_FIELDS, CONTENT_FIELDS, SPECIFICATIONS
from econforge.stochastic_contract import A1_SOURCE, A1_ACCEPTANCE_SHA256, ACCEPTANCE_SHA256
from econforge.stochastic_admission import admit_a2, record_bytes_sha256
from econforge.innovation_admission import bound_digest, recheck_p38
from econforge.stochastic_wire import validate_wire as validate_A2
from econforge.stochastic import result_template as a2_template
from econforge.p35_admission import canonical_json, digest


def admit_b(expected, content, a1_result):
    need(expected is not None and content is not None, 'TRUST_MISSING',
         'Separately retained B enrollment and complete detached content required', 'missing')
    for value, keys in ((expected, TRUST_FIELDS), (content, CONTENT_FIELDS)):
        need(type(value) is dict, 'INVALID_INPUT', 'Trust/content must be plain objects')
        need(not (keys - set(value)) and all(value[k] is not None for k in keys),
             'TRUST_MISSING', 'Required B trust/content absent', 'missing')
        fields(value, keys)
    t, c = [json.loads(canonical_json(x)) for x in (expected, content)]
    need(t['schema'] == 'p39-stochastic-trust' and t['stage'] == 'P3.9B'
         and type(t['revision']) is int and t['revision'] == 1
         and c['schema'] == 'p39-stochastic-content' and type(c['revision']) is int and c['revision'] == 1,
         'SOURCE_UNSUPPORTED', 'Frozen B trust/content revision 1 required', 'unsupported')
    fields(t['upstream'], {'stage', 'source_commit', 'result_sha256', 'acceptance_sha256'})
    u = t['upstream']
    need(u['stage'] == 'P3.9A2' and u['source_commit'] == A2_SOURCE
         and u['acceptance_sha256'] == A2_ACCEPTANCE_SHA256, 'ACCEPTANCE_MISSING',
         'Only exact independently accepted A2 implementation admitted', 'missing')
    need(u['result_sha256'] is not None and u['result_sha256'] != '', 'TRUST_MISSING',
         'Separately retained expected complete A2 digest required', 'missing')
    need(type(c['upstream_result']) is dict, 'LAW_MISSING', 'Complete accepted A2 result missing', 'missing')
    # No candidate provenance is trusted until its complete bytes match the separate store.
    bound_digest(c['upstream_result'], u['result_sha256'], 'accepted complete A2 law')
    fields(t['enrollment'], {'authority', 'record_id', 'enrolled_at_utc', 'source_location'})
    need(all(type(x) is str and bool(x.strip()) for x in t['enrollment'].values()),
         'TRUST_MISSING', 'Explicit separately retained enrollment provenance required', 'missing')
    fields(c['acceptance_records'], {ACCEPTANCE_SHA256, A1_ACCEPTANCE_SHA256, A2_ACCEPTANCE_SHA256})
    for h, text in c['acceptance_records'].items():
        need(type(text) is str and record_bytes_sha256(text) == h, 'DIGEST_MISMATCH',
             'Retained original independent ACCEPT report bytes differ')
    a2 = c['upstream_result']; t2 = a2.get('provenance')
    need(a2.get('schema') == 'p39-stochastic-law-result' and type(a2.get('revision')) is int
         and a2['revision'] == 1 and a2.get('capability') == 'P3.9A2_STOCHASTIC_LAW',
         'SOURCE_UNSUPPORTED', 'Unsupported upstream schema/capability is not an A2 law', 'unsupported')
    expected_fields = set(a2_template())
    need(not (set(a2) - expected_fields), 'SOURCE_UNSUPPORTED',
         'Unknown top-level A2 fields are unsupported in frozen revision 1', 'unsupported')
    fields(a2, expected_fields)
    need(a2['requested_claim'] == 'CERTIFIED_FIRST_ORDER_CAUSAL_L2_STOCHASTIC_LAW'
         and a2['claim'] in (None, 'CERTIFIED_FIRST_ORDER_CAUSAL_L2_STOCHASTIC_LAW'),
         'SOURCE_UNSUPPORTED', 'Unsupported A2 scientific claim', 'unsupported')
    need(type(t2) is dict, 'TRUST_MISSING', 'Complete certified A2 provenance required', 'missing')
    fields(t2, TRUST_FIELDS)
    shared = TRUST_FIELDS - {'stage', 'upstream', 'acceptance_records', 'enrollment'}
    need(all(canonical_json(t[k]) == canonical_json(t2[k]) for k in shared),
         'SOURCE_MISMATCH', 'B/A2 original source, coordinates or historical identities differ')
    expected_records = list(t2['acceptance_records'])
    matching = [x for x in t['acceptance_records'] if type(x) is dict
                and x.get('stage') == 'P3.9A2' and x.get('source_commit') == A2_SOURCE
                and x.get('record_sha256') == A2_ACCEPTANCE_SHA256]
    need(len(matching) == 1, 'ACCEPTANCE_MISSING', 'Genuine exact-source A2 acceptance entry required', 'missing')
    fields(matching[0], {'stage', 'source_commit', 'record_sha256', 'location'})
    need(type(matching[0]['location']) is str and bool(matching[0]['location'].strip()),
         'ACCEPTANCE_MISSING', 'Retained A2 report location required', 'missing')
    need(t['acceptance_records'] == expected_records + matching,
         'ACCEPTANCE_MISSING', 'Complete ordered original/A1/A2 acceptance chain required', 'missing')
    need(t['specifications'] == SPECIFICATIONS, 'SPEC_HASH_MISMATCH', 'Frozen specification identities differ')
    # The digest-bound record is retained evidence, not a newly recomputed A2 certificate.
    trusted = {k: a2[k] for k in ('law', 'axes', 'coordinates', 'checks', 'diagnostics', 'warnings', 'backend')}
    trusted['expected'] = t2
    try:
        validate_A2(a2, trusted=trusted, field_names=set(a2_template()))
    except InnovationError as exc:
        raise InnovationError('INVALID_INPUT', 'Malformed digest-bound A2 certificate: ' + str(exc)) from exc
    need(a2['product_valid'] and a2['law_valid'] and a2['law'] is not None,
         'LAW_MISSING', 'A successful complete A2 law is mandatory', 'missing')
    need(a1_result is not None, 'TRUST_MISSING', 'Full retained A1 predecessor must resolve separately', 'missing')
    # Resolve the A2 predecessor without modifying either frozen detached-content schema.
    c2 = json.loads(canonical_json(c)); c2['upstream_result'] = a1_result
    del c2['acceptance_records'][A2_ACCEPTANCE_SHA256]
    _, _, a1, t1, c1, specs = admit_a2(t2, c2)
    a = recheck_p38(t1, c1, specs)
    need(a2['axes'] == a['axes'] and a2['coordinates'] == a['coordinates'],
         'INVALID_AXES', 'A2 actual axes/chart differ from original admitted source')
    law = a2['law']
    need(law['G'] == a['G'].tolist() and law['H'] == a['H'].tolist()
         and law['L_s'] == a1['L_s'] and law['L_j'] == a1['L_j'],
         'SOURCE_MISMATCH', 'A2 matrices must retain exact accepted G/H and A1 loadings')
    original = json.loads(a['model'].source_json)
    need(law['covariance'] == original['covariance']
         and digest(law['covariance']) == t['covariance_source_sha256'],
         'INVALID_COVARIANCE', 'Only original source-bound covariance admitted')
    need(law['source_certificate_sha256'] == t['source_certificate_sha256'],
         'DIGEST_MISMATCH', 'Source coefficient certificate identity differs')
    return t, c, a2, a
