"""Detached, independently expected A1 content and accepted P3.8 rechecks."""
import json
import math
import re

from econforge import gh_admission, gh_numerics, gh_source
from econforge.bk_admission import decode
from econforge.innovation_contract import (
    need, sha_bytes, real_matrix, TRUST_FIELDS, CONTENT_FIELDS, SPECIFICATIONS,
    ASSUMPTIONS, P38_SOURCE, P38_ORIGINAL, ACCEPTANCE_SHA256)
from econforge.innovation_source import axes_coordinates, source_certificate
from econforge.p35_admission import canonical_json, digest
from econforge.qz_admission import same


def require_fields(value, fields, label):
    need(type(value) is dict and set(value) == set(fields),'INVALID_INPUT',
         'Exact fields required for '+label)


def bound_digest(value, expected, label):
    need(type(expected) is str and re.fullmatch('[0-9a-f]{64}',expected) is not None,
         'INVALID_INPUT','Expected SHA256 malformed: '+label)
    need(digest(value) == expected,'DIGEST_MISMATCH','Detached trusted content differs: '+label)


def acceptance(trust, content):
    """Closed registry of inspected original/maintenance ACCEPT records."""
    records = trust['acceptance_records']
    need(type(records) is list and bool(records),'ACCEPTANCE_MISSING',
         'Required original and maintenance implementation ACCEPT records missing','missing')
    require_fields(content['acceptance_records'],{ACCEPTANCE_SHA256},'acceptance content registry')
    text = content['acceptance_records'][ACCEPTANCE_SHA256]
    need(type(text) is str and sha_bytes(text.encode('utf-8')) == ACCEPTANCE_SHA256,
         'DIGEST_MISMATCH','Exact inspected acceptance bytes required')
    found = set()
    for record in records:
        require_fields(record,{'stage','source_commit','record_sha256','location'},'acceptance entry')
        need(record['stage'] == 'P3.8' and record['source_commit'] in (P38_SOURCE,P38_ORIGINAL)
             and record['record_sha256'] == ACCEPTANCE_SHA256
             and type(record['location']) is str and bool(record['location']),
             'ACCEPTANCE_MISSING','Record is not an enrolled ACCEPT for the exact P3.8 implementation','missing')
        need(record['source_commit'] not in found,'INVALID_INPUT','Duplicate acceptance entry')
        found.add(record['source_commit'])
    need(found == {P38_SOURCE,P38_ORIGINAL},'ACCEPTANCE_MISSING',
         'Original and maintenance acceptances both required','missing')
    require_fields(trust['upstream'],{'stage','source_commit','result_sha256','acceptance_sha256'},'upstream')
    need(same(trust['upstream'],dict(stage='P3.8',source_commit=P38_SOURCE,
         result_sha256=trust['p38_result_sha256'],acceptance_sha256=ACCEPTANCE_SHA256)),
         'DIGEST_MISMATCH','Exact accepted upstream source/result required')


def admit_trust(expected, detached):
    """No expected identity is calculated from the incoming candidate."""
    need(expected is not None and detached is not None,'TRUST_MISSING',
         'Separately retained expected identities and full detached content required','missing')
    for value,fields,label in ((expected,TRUST_FIELDS,'trust'),(detached,CONTENT_FIELDS,'content')):
        need(type(value) is dict,'INVALID_INPUT',label+' must be a plain object')
        need(not (set(fields)-set(value)),'TRUST_MISSING','Required '+label+' field absent','missing')
        require_fields(value,fields,label)
        need(all(v is not None for v in value.values()),'TRUST_MISSING','Required '+label+' content absent','missing')
    # Detach all nested values and reject NaN/Inf/non-plain objects before use.
    t,c = [json.loads(canonical_json(x)) for x in (expected,detached)]
    for key in CONTENT_FIELDS-{'schema','revision'}:
        need(type(c[key]) is dict,'INVALID_INPUT','Content object malformed: '+key)
    for value,schema in ((t,'p39-stochastic-trust'),(c,'p39-stochastic-content')):
        need(value['schema'] == schema and type(value['revision']) is int and value['revision'] == 1,
             'SOURCE_UNSUPPORTED','Unsupported stochastic trust/content schema','unsupported')
    need(t['stage'] == 'P3.9A1','SOURCE_UNSUPPORTED','This API admits A1 only','unsupported')
    require_fields(t['enrollment'],{'authority','record_id','enrolled_at_utc','source_location'},'enrollment')
    need(all(type(x) is str and bool(x.strip()) for x in t['enrollment'].values()),
         'TRUST_MISSING','Trusted-store enrollment provenance required','missing')
    need(t['p38_source_commit'] == P38_SOURCE and t['p38_original_source_commit'] == P38_ORIGINAL,
         'DIGEST_MISMATCH','Unaccepted implementation source identity')
    need(same(t['specifications'],SPECIFICATIONS),'SPEC_HASH_MISMATCH','Frozen five-spec binding differs')
    require_fields(c['specification_bytes'],SPECIFICATIONS,'specification content')
    specifications = {}
    for name,identity in SPECIFICATIONS.items():
        text = c['specification_bytes'][name]
        need(type(text) is str,'INVALID_INPUT','Specification content is exact UTF-8 source text')
        specifications[name] = text.encode('utf-8')
        need(sha_bytes(specifications[name]) == identity['sha256'],
             'SPEC_HASH_MISMATCH','Frozen specification byte mismatch: '+name)
    b = c['verification_bundle']
    need(type(b) is dict,'INVALID_INPUT','Original verification bundle malformed')
    need(all(b.get(k) is not None for k in ('model_source','p34_result','p35_result')),
         'TRUST_MISSING','Original model/derivatives/assembly content missing','missing')
    need(type(b['p34_result']) is dict and b['p34_result'].get('blocks') is not None
         and b['p34_result'].get('flattened_jacobian') is not None,
         'TRUST_MISSING','Original derivative evidence missing','missing')
    for field,key in [('p38_result','p38_result_sha256'),('p38_context','p38_context_sha256'),
                      ('verification_bundle','verification_bundle_sha256'),('p36_result','p36_result_sha256'),
                      ('p37_result','p37_result_sha256'),('source_certificate','source_certificate_sha256'),
                      ('assumptions','assumptions_sha256')]:
        bound_digest(c[field],t[key],field)
    for field in ('p34','p35'):
        bound_digest(b[field+'_result'],t[field+'_result_sha256'],field)
    need(same(c['p38_result'],c['upstream_result']),'DIGEST_MISMATCH','A1 upstream must be identical P3.8 wire')
    need(same(c['p38_result'].get('upstream_result'),c['p37_result'])
         and same(c['p37_result'].get('upstream_result'),c['p36_result']),
         'DIGEST_MISMATCH','Complete retained P3.6/P3.7 chain differs')
    acceptance(t,c)
    require_fields(c['assumptions'],set(ASSUMPTIONS)|{'source_assertion','neighborhood','declaration_authority'},'assumptions')
    need(all(same(c['assumptions'][k],v) for k,v in ASSUMPTIONS.items())
         and type(c['assumptions']['declaration_authority']) is str
         and bool(c['assumptions']['declaration_authority'].strip()),
         'SOURCE_UNSUPPORTED','Required common-information MDS/local-class declarations absent','unsupported')
    return t,c,specifications


def check_retained_component(a, name, record):
    """Recheck emitted terms, not invented unrecorded complex candidates or M."""
    np = a['np']
    need(type(record) is dict and record.get('passed') is True,'DIGEST_MISMATCH',
         'Required retained P3.8 diagnostic not passed: '+name)
    shape = record.get('shape')
    need(type(shape) is list and len(shape) == 2
         and all(type(i) is int and i >= 0 for i in shape),
         'INVALID_INPUT','Retained diagnostic shape invalid')
    shape = tuple(shape)
    def terms(value):
        # Shape is explicit even when a logical axis is empty.
        if shape[0]*shape[1] == 0:
            need(type(value) is list and len(value) == shape[0], 'INVALID_INPUT','Empty term shape')
            return np.array(value,dtype=float).reshape(shape)
        raw = np.array(value)
        if raw.shape == shape+(2,):
            return decode(np,value,shape,'P38_DIAGNOSTIC_INVALID')
        return real_matrix(np,value,shape)
    left,right = terms(record.get('left')),terms(record.get('right'))
    realness = name in ('G_realness','H_realness')
    atol,rtol = (1e-12,1e-10) if realness else (1e-10,1e-8)
    need(record.get('atol') == atol and record.get('rtol') == rtol,
         'DIGEST_MISMATCH','Retained P3.8 tolerance differs')
    error = np.abs(left-right)
    if realness:
        matrix = a['G'] if name == 'G_realness' else a['H']
        bound = atol+rtol*np.abs(matrix)
    else:
        two = name in ('original_complex','original_real','final_policy')
        need(record.get('rule') == ('two_term' if two else 'declared_reference'),
             'DIGEST_MISMATCH','Retained component rule differs')
        bound = atol+rtol*(np.abs(left)+np.abs(right) if two else np.abs(right))
    need(bool(np.isfinite(bound).all()) and bool(np.isfinite(error).all())
         and bool(np.all(error <= bound)),'UPSTREAM_GATE_UNRESOLVED',
         'Retained P3.8 term check no longer passes','uncertain')
    need(same(error.tolist(),record.get('absolute_errors'))
         and same(bound.tolist(),record.get('allowed_bounds')),
         'DIGEST_MISMATCH','Retained errors/bounds inconsistent with actual terms')
    return left,right


def recheck_p38(t,c,specifications):
    """Reuse admission/source verifiers; never recover G/H or rerun QZ/AD."""
    r,b = c['p38_result'],c['verification_bundle']
    # Validate this initial source subset explicitly, before the unchanged
    # legacy parser. This is no covariance override or new parser admission.
    need(type(b.get('model_source')) is str,'INVALID_INPUT','Original source must be exact text')
    try:
        raw_source = json.loads(b['model_source'])
    except (ValueError,TypeError,RecursionError) as exc:
        need(False,'INVALID_INPUT','Malformed original model JSON: '+type(exc).__name__)
    need(type(raw_source) is dict and type(raw_source.get('covariance')) is dict,
         'INVALID_COVARIANCE','Original covariance object required')
    covariance = raw_source['covariance']
    need(covariance.get('kind') == 'diagonal','SOURCE_UNSUPPORTED',
         'Initial A1 admits original diagonal covariance only','unsupported')
    need(set(covariance) == {'kind','values'} and type(covariance['values']) is list
         and type(raw_source.get('innovations')) is list
         and len(covariance['values']) == len(raw_source['innovations'])
         and all(type(x) in (float,int) and math.isfinite(x) and x >= 0 for x in covariance['values']),
         'INVALID_COVARIANCE','Finite nonnegative variances in exact innovation dimensions required')
    expected = dict(schema_version=2,claim='FIRST_ORDER_CONDITIONAL_MEAN_GH_VERIFIED',
        specification_revision='p38-gh-1',execution_status='COMPLETED',
        input_admission_status='ADMITTED',recovery_execution_status='COMPLETED',
        recovery_valid=True,model_status='NOT_EVALUATED',failure_reason=None,
        dtype='float64',intermediate_dtype='complex128',backend='scipy.linalg.solve_triangular + scipy.linalg.solve')
    need(type(r.get('schema_version')) is int,'INVALID_INPUT','P3.8 schema version must be an integer')
    need(r['schema_version'] == 2,'SOURCE_UNSUPPORTED','Unsupported P3.8 schema','unsupported')
    need(all(k in r and same(r[k],v) for k,v in expected.items()),
         'DIGEST_MISMATCH','Accepted, jointly valid completed P3.8 result required')
    need(type(r.get('warnings')) is list and all(type(x) is str for x in r['warnings'])
         and r.get('assessment_status') == ('PASS_WITH_WARNINGS' if r['warnings'] else 'PASS'),
         'DIGEST_MISMATCH','P3.8 aggregate/warnings invalid')
    a = gh_admission.admit(c['p37_result'],c['p38_context'],b,
                          {k:specifications[k] for k in gh_admission.SPEC_IDENTITIES})
    np,ns,nj = a['np'],a['n_state'],a['n_jump']
    axes,coords = axes_coordinates(a)
    md = r.get('metadata')
    need(type(md) is dict and same(md.get('bindings'),c['p38_context'])
         and md.get('forcing_stability') == 'NOT_EVALUATED',
         'DIGEST_MISMATCH','P3.8 context/claim boundary differs')
    for key in ('variable_order','equation_order','partitions','permutations','S','T','coordinate_convention'):
        need(same(md.get(key),a['metadata'][key]),'INVALID_AXES','P3.8 axes/chart differs: '+key)
    for key,target in [('state_variable_order',axes['state_order']),('control_variable_order',axes['control_order']),
                       ('G_shape',[nj,ns]),('H_shape',[ns,ns])]:
        need(same(md.get(key),target),'INVALID_AXES','P3.8 partition/shape differs: '+key)
    need(same(md.get('backend_versions'),a['versions'])
         and md.get('coordinate_meaning') == 'native q; stored displacement dx=T dq',
         'DIGEST_MISMATCH','Required accepted P3.8 runtime/chart semantics differ')
    a['G'],a['H'] = real_matrix(np,r['G'],(nj,ns)),real_matrix(np,r['H'],(ns,ns))
    diagnostics = r.get('diagnostics')
    need(type(diagnostics) is dict,'INVALID_INPUT','Required P3.8 diagnostics absent')
    fresh = {}
    SB,SA,_,_,_,Z = a['output']
    X,Y = Z[a['state_rows'],:ns],Z[a['control_rows'],:ns]
    gh_numerics.conditioning(np,a['linalg'],X,SA[:ns,:ns],fresh)
    for name in ('X_conditioning','stable_block_conditioning'):
        need(type(diagnostics.get(name)) is dict and diagnostics[name].get('status') in ('PASS','WARN'),
             'DIGEST_MISMATCH','Retained conditioning gate missing')
        retained = diagnostics[name]
        if ns:
            singular = retained.get('singular_values')
            need(type(singular) is list and len(singular) == ns
                 and all(type(x) is float and math.isfinite(x) and x > 0 for x in singular),
                 'INVALID_INPUT','Retained conditioning singular values invalid')
            expected_condition = gh_numerics.condition_gate(max(singular),min(singular),graph=name=='X_conditioning')
            need(all(same(retained.get(k),v) for k,v in expected_condition.items())
                 and retained.get('vacuous') is False,'DIGEST_MISMATCH','Retained condition formula/status differs')
        else:
            need(same(retained,fresh[name]),'DIGEST_MISMATCH','Empty condition diagnostic differs')
    for name in ('stable_solve','graph_solve','state_solve','original_complex',
                 'G_realness','H_realness','final_policy','original_real','process_zz','process_zk'):
        check_retained_component(a,name,diagnostics.get(name))
    scale = max(float(np.linalg.norm(a['A'],ord=np.inf)),float(np.linalg.norm(a['B'],ord=np.inf)))
    need(math.isfinite(scale) and scale > 0 and diagnostics.get('common_infinity_scale') == scale,
         'DIGEST_MISMATCH','Original P3.8 diagnostic scale differs')
    _,stable_right = check_retained_component(a,'stable_solve',diagnostics['stable_solve'])
    need(same(gh_numerics.wire_array(np,stable_right),gh_numerics.wire_array(np,SB[:ns,:ns]/scale)),
         'DIGEST_MISMATCH','Retained stable right term not bound to supplied Schur block')
    _,state_right = check_retained_component(a,'state_solve',diagnostics['state_solve'])
    _,graph_right = check_retained_component(a,'graph_solve',diagnostics['graph_solve'])
    need(same(gh_numerics.wire_array(np,graph_right),gh_numerics.wire_array(np,Y)),
         'DIGEST_MISMATCH','Retained graph right term differs from actual stable Schur columns')
    for name,left,right,rows in [('returned_graph',a['G']@X,Y,axes['control_order']),
                               ('returned_state',a['H']@X,state_right,axes['state_order'])]:
        gh_numerics.require_check(fresh,name,gh_numerics.component_check(np,left,right,rows,
            ['stable_coordinate_'+str(i) for i in range(ns)]),'P38_RETURNED_GRAPH_FAILED')
    prepared = gh_source.prepare(a,fresh)
    gh_numerics.policy_check(a,a['G'],a['H'],fresh)
    gh_source.original_equations(a,a['G'],a['H'],fresh,'original_real')
    gh_source.identities_and_process(a,a['G'],a['H'],prepared,fresh)
    checks = r.get('checks')
    expected_ids = ['input_and_original_source.admitted','X_conditioning','stable_block_conditioning',
        'stable_solve','graph_solve','state_solve','original_complex','G_realness','H_realness',
        'final_policy','original_real','process_zz','process_zk']
    need(type(checks) is list and [x.get('check_id') for x in checks] == expected_ids
         and all(x.get('status') == (diagnostics[x['check_id']]['status']
             if x['check_id'] in ('X_conditioning','stable_block_conditioning') else 'PASS')
             and x.get('evidence') for x in checks),
         'DIGEST_MISMATCH','All P3.8 evidenced gates required')
    for index,item in enumerate(checks):
        need(set(item) == {'check_id','status','evidence'} and type(item['evidence']) is list
             and bool(item['evidence']) and all(type(e) is dict and type(e.get('kind')) is str
             and type(e.get('location')) is str and e['kind'] and e['location'] for e in item['evidence']),
             'INVALID_INPUT','Malformed retained P3.8 check evidence')
        evidence_expected = ([dict(kind='trusted_P37_wire',digest=c['p38_context']['p37_result_sha256'],location='inline:P37'),
             dict(kind='original_verification_bundle',digest=t['verification_bundle_sha256'],location='inline:bundle')]
             if index == 0 else [dict(kind='computed_P38_check',location='inline:diagnostics.'+item['check_id'])])
        need(same(item['evidence'],evidence_expected),'DIGEST_MISMATCH',
             'Retained P3.8 check evidence not bound to its original terms/content')
    original_warnings=['Retained upstream: '+message for message in a['wire']['warnings']]
    for name in ('X_conditioning','stable_block_conditioning'):
        if diagnostics[name]['status'] == 'WARN':
            original_warnings.append(name+' at or above 1e8 and below 1e10')
    need(same(r['warnings'],original_warnings),'DIGEST_MISMATCH',
         'Retained P3.8 warning list is not bound to accepted upstream/conditioning evidence')
    source = json.loads(a['model'].source_json)
    expected_neighborhood = dict(variables=[dict(id=v['id'],domain=v['domain']) for v in source['variables']],
                                 parameters=[dict(id=p['id'],domain=p['domain']) for p in source['parameters']])
    need(same(c['assumptions']['source_assertion'],a['metadata']['admission_manifest'])
         and same(c['assumptions']['neighborhood'],expected_neighborhood),
         'SOURCE_MISMATCH','Original domain/local-system declarations differ')
    for key,value in [('model_id',a['model'].model_id),('branch_id',a['model'].branch_id),
                      ('frequency',a['model'].frequency),('original_model_sha256',a['model'].digest),
                      ('canonical_model_sha256',digest(source)),('reference_sha256',coords['reference_sha256']),
                      ('calibration_sha256',digest(source['parameters'])),
                      ('process_source_sha256',digest(a['metadata']['structure']['process_equations'])),
                      ('innovation_metadata_sha256',digest(source['innovations'])),
                      ('covariance_source_sha256',digest(source['covariance'])),
                      ('coordinate_sha256',digest(coords)),('axes_sha256',digest(axes))]:
        need(same(t[key],value),'INVALID_AXES' if key in ('coordinate_sha256','axes_sha256') else 'SOURCE_MISMATCH',
             'Separately expected source binding differs: '+key)
    a.update(axes=axes,coordinates=coords,p38_rechecks=fresh)
    return a
