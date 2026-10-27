"""Exact A1 wire validation, using the frozen union without new dependencies."""
import math
import re

from econforge.innovation_contract import need,CLAIM,CAPABILITY,CHECK_IDS,REVISION,TRUST_FIELDS,SPEC_SHA256
from econforge.p35_admission import canonical_json,digest

COMMON_FIELDS = set(('schema revision schema_version capability requested_claim claim execution_status computation_status '
    'assessment_status model_status input_admission_status product_valid specification_revision specification_sha256 '
    'trust_sha256 upstream_result_sha256 upstream_stage upstream_source_commit provenance axes coordinates required_checks '
    'checks diagnostics warnings failure_reason failure_reasons backend capability_maturity independent_validation_status '
    'external_validation_status acceptance_evidence limitations L_s L_j L_s_shape L_j_shape source_certificate forcing_certification').split())
AXES_FIELDS = set(('n_k n_z n_state n_jump n_innovation equation_order variable_order state_order control_order '
                   'innovation_order partitions permutations').split())
COORDINATE_FIELDS = set(('convention S T innovation_scales variable_units innovation_units representations local_coordinates '
                         'reference reference_sha256 frequency').split())
CERTIFICATE_FIELDS = set(('schema revision original_model_sha256 canonical_model_sha256 reference_sha256 p34_result_sha256 '
    'coordinate_sha256 axes_sha256 rows J_u_plus J_u_current R process_means process_intercepts structural_zero_mask derivation').split())


def fields(value,expected):
    need(type(value) is dict and set(value) == set(expected),'SERIALIZATION_FAILED','Exact frozen object fields required','backend')


def strings(value,*,unique=False,length=None):
    need(type(value) is list and all(type(x) is str for x in value)
         and (length is None or len(value) == length)
         and (not unique or len(set(value)) == len(value)),
         'SERIALIZATION_FAILED','String-list wire axes invalid','backend')


def matrix(value,shape,*,boolean=False):
    need(type(value) is list and len(value) == shape[0]
         and all(type(row) is list and len(row) == shape[1]
                 and all(type(x) is bool if boolean else type(x) is float and math.isfinite(x) for x in row)
                 for row in value),'SERIALIZATION_FAILED','Explicit finite matrix shape/dtype invalid','backend')


def evidence(values):
    need(type(values) is list,'SERIALIZATION_FAILED','Evidence must be an ordered list','backend')
    for value in values:
        fields(value,{'kind','location','digest'})
        need(type(value['kind']) is str and bool(value['kind']) and type(value['location']) is str
             and bool(value['location']) and (value['digest'] is None or type(value['digest']) is str
             and re.fullmatch('[0-9a-f]{64}',value['digest']) is not None),
             'SERIALIZATION_FAILED','Malformed evidence record','backend')


def validate_certificate(cert,axes):
    fields(cert,CERTIFICATE_FIELDS)
    need(cert['schema'] == 'p39-source-coefficients' and type(cert['revision']) is int and cert['revision'] == 1
         and cert['derivation'] == 'ORIGINAL_ADMITTED_AST_INCIDENCE_V1',
         'SERIALIZATION_FAILED','Source certificate schema/derivation differs','backend')
    n,m = len(axes['variable_order']),axes['n_innovation']
    for key in ('original_model_sha256','canonical_model_sha256','reference_sha256',
                'p34_result_sha256','coordinate_sha256','axes_sha256'):
        need(type(cert[key]) is str and re.fullmatch('[0-9a-f]{64}',cert[key]) is not None,
             'SERIALIZATION_FAILED','Source certificate identity malformed','backend')
    for key in ('J_u_plus','J_u_current'):matrix(cert[key],(n,m))
    matrix(cert['R'],(m,m));matrix(cert['structural_zero_mask'],(n,m),boolean=True)
    for key in ('process_means','process_intercepts'):
        need(type(cert[key]) is list and len(cert[key]) == m
             and all(type(x) is float and math.isfinite(x) for x in cert[key]),
             'SERIALIZATION_FAILED','Source scalar axes invalid','backend')
    need(type(cert['rows']) is list and len(cert['rows']) == n,
         'SERIALIZATION_FAILED','Source row count differs','backend')
    for i,row in enumerate(cert['rows']):
        fields(row,{'equation_id','kind','target','expression','expression_sha256','source_span',
                    'expanded_ast_sha256','occurrences','innovation_coefficients','row_scale','process_index'})
        matrix([row['innovation_coefficients']],(1,m))
        need(type(row['row_scale']) is float and row['row_scale'] > 0
             and row['kind'] in ('k_transition','z_process','conditional'),
             'SERIALIZATION_FAILED','Source row type/scale differs','backend')
        need(row['equation_id']==axes['equation_order'][i] and type(row['expression']) is str
             and (row['target'] is None or type(row['target']) is str)
             and (row['process_index'] is None or type(row['process_index']) is int
                  and 0<=row['process_index']<m),
             'SERIALIZATION_FAILED','Source row metadata invalid','backend')
        for key in ('expression_sha256','expanded_ast_sha256'):
            need(type(row[key]) is str and re.fullmatch('[0-9a-f]{64}',row[key]) is not None,
                 'SERIALIZATION_FAILED','Source row digest malformed','backend')
        span=row['source_span']
        need(type(span) is list and len(span)==3 and type(span[0]) is str
             and all(type(x) is int for x in span[1:]) and 0<=span[1]<=span[2],
             'SERIALIZATION_FAILED','Source span malformed','backend')
        need(type(row['occurrences']) is list and all(type(x) is list and len(x)==4
             and type(x[0]) is str and all(type(v) is int for v in x[1:])
             and 0<=x[2]<=x[3] for x in row['occurrences']),
             'SERIALIZATION_FAILED','Source occurrence metadata malformed','backend')


def validate_axes_coordinates(a,c):
    """Validate existing nested fields; trusted identities are checked separately."""
    ns,nj,m,nk=a['n_state'],a['n_jump'],a['n_innovation'],a['n_k']
    n=ns+nj
    fields(a['partitions'],{'k','z','j'})
    for role,length in (('k',nk),('z',m),('j',nj)):
        strings(a['partitions'][role],unique=True,length=length)
    need(a['state_order']==a['partitions']['k']+a['partitions']['z']
         and a['control_order']==a['partitions']['j']
         and set(a['state_order']+a['control_order'])==set(a['variable_order']),
         'SERIALIZATION_FAILED','Nested partition/order mismatch','backend')
    pairs=[('canonical_to_source_'+kind,'source_to_canonical_'+kind) for kind in ('variables','equations')]
    fields(a['permutations'],{key for pair in pairs for key in pair})
    for forward,reverse in pairs:
        p,q=a['permutations'][forward],a['permutations'][reverse]
        # Accepted equation maps retain reporting identities as well as model rows.
        count=n if forward.endswith('variables') else len(p) if type(p) is list else 0
        need(count>=n and all(type(v) is list and len(v)==count and all(type(x) is int for x in v)
                 and sorted(v)==list(range(count)) for v in (p,q))
             and all(q[p[i]]==i for i in range(count)),
             'SERIALIZATION_FAILED','Nested permutation metadata invalid','backend')
    for key,length in (('variable_units',n),('innovation_units',m),('representations',n),('local_coordinates',n)):
        strings(c[key],length=length)
        need(all(bool(x) for x in c[key]),'SERIALIZATION_FAILED','Empty coordinate source metadata','backend')
    convention=c['convention']
    fields(convention,{'coordinate_diagonal_T','coordinate_metadata','coordinate_normalized_definition',
                       'raw_definition','row_normalized_definition','row_scales_S'})
    need(convention['coordinate_diagonal_T']==c['T'] and convention['row_scales_S']==c['S']
         and all(type(convention[k]) is str and bool(convention[k]) for k in
                 ('coordinate_normalized_definition','raw_definition','row_normalized_definition')),
         'SERIALIZATION_FAILED','Accepted coordinate convention metadata differs','backend')
    metadata=convention['coordinate_metadata']
    need(type(metadata) is list and len(metadata)==n and all(type(row) is list and len(row)==6
         and all(type(row[j]) is str for j in (0,1,2,5)) for row in metadata)
         and [row[0] for row in metadata]==a['variable_order'],
         'SERIALIZATION_FAILED','Accepted coordinate source records malformed','backend')
    need(type(c['frequency']) is str and bool(c['frequency'])
         and type(c['reference_sha256']) is str and re.fullmatch('[0-9a-f]{64}',c['reference_sha256']) is not None,
         'SERIALIZATION_FAILED','Coordinate identity metadata invalid','backend')


def validate_metadata_binding(r,expected):
    """Expected is separately retained trust, never enrolled from this result."""
    fields(expected,TRUST_FIELDS)
    need(canonical_json(r['provenance'])==canonical_json(expected)
         and r['trust_sha256']==digest(expected)
         and r['upstream_result_sha256']==expected['p38_result_sha256']
         and r['upstream_stage']=='P3.8' and r['upstream_source_commit']==expected['p38_source_commit']
         and r['specification_sha256']==SPEC_SHA256,
         'SERIALIZATION_FAILED','Success provenance differs from independently retained trust','backend')
    need(digest(r['axes'])==expected['axes_sha256']
         and digest(r['coordinates'])==expected['coordinate_sha256'],
         'SERIALIZATION_FAILED','Nested metadata differs from trusted axes/coordinates','backend')
    # Bind the entire published certificate, including ordered source metadata,
    # to the private trust snapshot; never enroll its identity from this result.
    need(type(expected['source_certificate_sha256']) is str
         and re.fullmatch('[0-9a-f]{64}',expected['source_certificate_sha256']) is not None
         and digest(r['source_certificate'])==expected['source_certificate_sha256'],
         'SERIALIZATION_FAILED','Published source certificate differs from independently retained trust','backend')
    a=r['axes'];ns,nk=a['n_state'],a['n_k'];shocks=a['innovation_order']
    labels={'A1.physical_zero.measurement':a['state_order'][:nk],
            'A1.exogenous_loading.measurement':a['state_order'][nk:],
            'A1.jump_loading.measurement':a['control_order'],
            'A1.pathwise_coefficients.measurement':a['equation_order'][:ns]}
    for d in r['diagnostics']:
        v=d['values']
        if d['kind']=='component':
            need(d['id'] in labels and v['row_labels']==labels[d['id']] and v['column_labels']==shocks,
                 'SERIALIZATION_FAILED','Diagnostic labels/order differ from trusted axes','backend')
        elif d['kind']=='conditional_forecast_error':
            need(v['row_labels']==a['equation_order'][ns:] and v['innovation_labels']==shocks,
                 'SERIALIZATION_FAILED','Conditional diagnostic labels/order differ','backend')
        elif d['kind']=='upstream_recheck':
            need(v['result_sha256']==expected['p38_result_sha256'],
                 'SERIALIZATION_FAILED','Upstream diagnostic identity differs','backend')


def validate_diagnostic(d):
    fields(d,{'id','kind','values'})
    need(type(d['id']) is str,'SERIALIZATION_FAILED','Diagnostic ID invalid','backend')
    v=d['values'];kind=d['kind']
    if kind == 'component':
        fields(v,{'shape','row_labels','column_labels','left','right','absolute_errors','allowed_bounds',
                  'max_error_over_bound','worst_index','atol','rtol','rule','vacuous'})
        shape=v['shape']
        need(type(shape) is list and len(shape)==2 and all(type(x) is int and x>=0 for x in shape),
             'SERIALIZATION_FAILED','Diagnostic shape invalid','backend')
        strings(v['row_labels'],length=shape[0]);strings(v['column_labels'],length=shape[1])
        for name in ('left','right','absolute_errors','allowed_bounds'):matrix(v[name],shape)
        if 0 in shape:
            need(v['worst_index'] is None and v['max_error_over_bound'] is None,
                 'SERIALIZATION_FAILED','Empty diagnostic extrema must be null','backend')
        else:
            index=v['worst_index'];maximum=v['max_error_over_bound']
            need(type(index) is list and len(index)==2 and all(type(x) is int and 0<=x<shape[i]
                 for i,x in enumerate(index)) and type(maximum) is float and math.isfinite(maximum) and maximum>=0,
                 'SERIALIZATION_FAILED','Diagnostic extrema type/index invalid','backend')
        need(v['atol']==1e-10 and v['rtol']==1e-8 and v['rule']=='two_term'
             and v['vacuous'] is (0 in shape),'SERIALIZATION_FAILED','Diagnostic rule/empty convention differs','backend')
    elif kind == 'conditional_forecast_error':
        fields(v,{'row_labels','innovation_labels','shape','coefficients','meaning'})
        strings(v['row_labels']);strings(v['innovation_labels'])
        need(type(v['shape']) is list and all(type(x) is int for x in v['shape'])
             and v['shape']==[len(v['row_labels']),len(v['innovation_labels'])]
             and v['meaning']=='REALIZED_FORECAST_ERROR_NOT_PATHWISE_ZERO_REQUIREMENT',
             'SERIALIZATION_FAILED','Conditional diagnostic meaning differs','backend')
        matrix(v['coefficients'],v['shape'])
    elif kind == 'upstream_recheck':
        fields(v,{'stage','result_sha256','check_id','record','fresh_record'})
        need(v['stage']=='P3.8' and type(v['record']) is dict
             and (v['fresh_record'] is None or type(v['fresh_record']) is dict),
             'SERIALIZATION_FAILED','Retained recheck format differs','backend')
    elif kind == 'event':
        fields(v,{'stage','reason','message','exception_type'})
        need(all(type(v[k]) is str for k in ('stage','reason','message'))
             and (v['exception_type'] is None or type(v['exception_type']) is str),
             'SERIALIZATION_FAILED','Event diagnostic malformed','backend')
    elif kind == 'underflow':
        fields(v,{'operation','horizon','array','indices','context','observability'})
        need(v['horizon'] is None and type(v['indices']) is list
             and all(type(v[k]) is str for k in ('operation','array','context','observability')),
             'SERIALIZATION_FAILED','Underflow diagnostic malformed','backend')
    else:
        need(False,'SERIALIZATION_FAILED','Unauthorized diagnostic kind in A1','backend')


def validate_wire(r,*,expected=None):
    """Positive trust validation requires the existing independently enrolled context."""
    canonical_json(r)
    fields(r,COMMON_FIELDS)
    need(r['schema']=='p39-innovation-loading-result' and type(r['revision']) is int and r['revision']==1
         and type(r['schema_version']) is int and r['schema_version']==2 and r['capability']==CAPABILITY
         and r['requested_claim']==CLAIM and r['specification_revision']==REVISION
         and r['forcing_certification']=='NOT_EVALUATED',
         'SERIALIZATION_FAILED','A1 literal contract invalid','backend')
    for key in ('execution_status','computation_status'):
        need(r[key] in ('COMPLETED','FAILED','NOT_RUN','SKIPPED'),
             'SERIALIZATION_FAILED','Execution vocabulary invalid','backend')
    need(r['assessment_status'] in ('PASS','PASS_WITH_WARNINGS','FAIL','REFUSED','NOT_RUN','NOT_IMPLEMENTED')
         and r['model_status'] in ('NOT_EVALUATED','NUMERICALLY_UNRESOLVED','UNSUPPORTED')
         and r['input_admission_status'] in ('ADMITTED','NOT_ADMITTED') and type(r['product_valid']) is bool,
         'SERIALIZATION_FAILED','A1 status vocabulary invalid','backend')
    need(r['required_checks']==CHECK_IDS and type(r['checks']) is list
         and [c.get('check_id') for c in r['checks']]==CHECK_IDS,
         'SERIALIZATION_FAILED','Complete ordered check registry required','backend')
    for c in r['checks']:
        fields(c,{'check_id','status','evidence','reason','diagnostic_ids','vacuous'})
        need(c['status'] in ('PASS','WARN','FAIL','UNSUPPORTED','NOT_IMPLEMENTED','NOT_RUN')
             and type(c['vacuous']) is bool and (c['reason'] is None or type(c['reason']) is str),
             'SERIALIZATION_FAILED','Check record malformed','backend')
        evidence(c['evidence']);strings(c['diagnostic_ids'])
        need(c['status']!='PASS' or bool(c['evidence']),'SERIALIZATION_FAILED','PASS without evidence','backend')
        if c['status']=='NOT_RUN':
            need(not c['evidence'] and not c['diagnostic_ids'] and not c['vacuous'],
                 'SERIALIZATION_FAILED','Unreached check cannot contain passing evidence','backend')
    need(type(r['diagnostics']) is list and type(r['warnings']) is list,
         'SERIALIZATION_FAILED','Diagnostic/warning collections must be lists','backend')
    for d in r['diagnostics']:validate_diagnostic(d)
    need(len({d['id'] for d in r['diagnostics']})==len(r['diagnostics']),
         'SERIALIZATION_FAILED','Diagnostic IDs must be unique','backend')
    for w in r['warnings']:
        fields(w,{'code','check_id','message','evidence'});evidence(w['evidence'])
        need(w['code'] in ('FINITE_UNDERFLOW','INHERITED_ACCEPTED_WARNING'),
             'SERIALIZATION_FAILED','Unauthorized warning category','backend')
    evidence(r['acceptance_evidence']);strings(r['limitations']);strings(r['failure_reasons'],unique=True)
    need(bool(r['limitations']),'SERIALIZATION_FAILED','Scientific limitations missing','backend')
    if r['provenance'] is not None:fields(r['provenance'],TRUST_FIELDS)
    if r['backend'] is not None:
        b=r['backend'];fields(b,{'python','numpy','scipy','jax','jaxlib','os','architecture','blas','lapack',
            'thread_settings','loaded_libraries','dtype','intermediate_dtype','methods','policy'})
        need(all(type(b[k]) is str and bool(b[k]) for k in ('python','numpy','scipy','jax','jaxlib','os',
             'architecture','blas','lapack')) and b['dtype']==b['intermediate_dtype']=='float64'
             and b['policy']==REVISION and type(b['thread_settings']) is dict
             and all(type(x) is str for x in b['thread_settings'].values()),
             'SERIALIZATION_FAILED','Actual A1 runtime evidence malformed','backend')
        strings(b['loaded_libraries']);strings(b['methods'])
        need(bool(b['loaded_libraries']) and bool(b['methods']),'SERIALIZATION_FAILED','Runtime evidence missing','backend')
    need(r['independent_validation_status']==r['external_validation_status']=='NOT_RUN'
         and r['acceptance_evidence']==[] and r['capability_maturity']=='IMPLEMENTED',
         'SERIALIZATION_FAILED','Ordinary A1 request cannot invent acceptance or external validation','backend')
    if r['axes'] is not None:
        a=r['axes'];fields(a,AXES_FIELDS);fields(r['coordinates'],COORDINATE_FIELDS)
        need(all(type(a[k]) is int and a[k]>=0 for k in ('n_k','n_z','n_state','n_jump','n_innovation'))
             and a['n_state']==a['n_k']+a['n_z'] and a['n_z']==a['n_innovation'],
             'SERIALIZATION_FAILED','Axis dimensions invalid','backend')
        n=a['n_state']+a['n_jump'];m=a['n_innovation']
        need(0<n<=256,'SERIALIZATION_FAILED','Source dimensions outside admitted range','backend')
        for key,length in [('equation_order',n),('variable_order',n),('state_order',a['n_state']),
                           ('control_order',a['n_jump']),('innovation_order',m)]:strings(a[key],unique=True,length=length)
        validate_axes_coordinates(a,r['coordinates'])
        for key,length in [('S',n),('T',n),('innovation_scales',m),('reference',n)]:
            values=r['coordinates'][key]
            need(type(values) is list and len(values)==length
                 and all(type(x) is float and math.isfinite(x) and (key=='reference' or x>0) for x in values),
                 'SERIALIZATION_FAILED','Coordinate numeric axes invalid','backend')
        need(r['L_s_shape']==[a['n_state'],m] and r['L_j_shape']==[a['n_jump'],m],
             'SERIALIZATION_FAILED','Loading shapes do not match admitted axes','backend')
        if r['source_certificate'] is not None:validate_certificate(r['source_certificate'],a)
        if r['product_valid']:
            matrix(r['L_s'],r['L_s_shape']);matrix(r['L_j'],r['L_j_shape'])
    if r['product_valid']:
        validate_metadata_binding(r,expected)
        need(r['claim']==CLAIM and r['assessment_status']==('PASS_WITH_WARNINGS' if r['warnings'] else 'PASS')
             and r['execution_status']==r['computation_status']=='COMPLETED'
             and r['input_admission_status']=='ADMITTED' and r['model_status']=='NOT_EVALUATED'
             and r['axes'] is not None and r['source_certificate'] is not None and r['backend'] is not None
             and r['failure_reason'] is None and not r['failure_reasons']
             and all(c['status']=='PASS' for c in r['checks'][:-1])
             and r['checks'][-1]['status']==('WARN' if r['warnings'] else 'PASS'),
             'SERIALIZATION_FAILED','Positive-product truth table invalid','backend')
    else:
        need(r['claim'] is None and r['L_s'] is None and r['L_j'] is None
             and r['assessment_status'] not in ('PASS','PASS_WITH_WARNINGS'),
             'SERIALIZATION_FAILED','Failed A1 product must be atomically suppressed','backend')
