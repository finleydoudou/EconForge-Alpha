"""Frozen A2 wire validation and immutable-buffer publication; no new schema fields."""
import json,math
from econforge.stochastic_contract import (need,CLAIM,CAPABILITY,CHECK_IDS,LAW_FIELDS,
    REVISION,SPEC_SHA256,canonical_json,digest,SOLUTION_CLASS,EQUATIONS,TRUST_FIELDS)
from econforge.innovation_wire import (fields,matrix,strings,evidence,validate_diagnostic,
    validate_axes_coordinates,AXES_FIELDS,COORDINATE_FIELDS)
from econforge.stochastic_forcing import LYAP_ARRAYS,LYAP_SCALARS


def complex_matrix(value,shape):
    need(type(value) is list and len(value)==shape[0]
         and all(type(row) is list and len(row)==shape[1] and all(type(x) is list and len(x)==2
             and all(type(v) is float and math.isfinite(v) for v in x) for x in row) for row in value),
         'SERIALIZATION_FAILED','Finite complex-pair matrix shape invalid','backend')


def validate_forcing_diagnostic(d,m,positive):
    v=d['values']
    if d['kind']=='schur':
        fields(v,{'R_q','F','U','roots','classes','distance_inner','distance_outer','stable_margin',
            'boundary_distance','reconstruction','unitarity','triangularity','tau','stable_count',
            'outside_count','boundary_count','vacuous','source_certificate_sha256'})
        matrix(v['R_q'],(m,m))
        for key in ('F','U'):
            if v[key] is not None:complex_matrix(v[key],(m,m))
        need(type(v['roots']) is list and len(v['roots']) in (0,m) and all(type(x) is list and len(x)==2
             and all(type(z) is float and math.isfinite(z) for z in x) for x in v['roots']),
             'SERIALIZATION_FAILED','Schur root list invalid','backend')
        strings(v['classes']);need(all(x in ('STABLE','OUTSIDE','BOUNDARY') for x in v['classes']) and len(v['classes'])==len(v['roots']),
             'SERIALIZATION_FAILED','Schur classes/multiplicity invalid','backend')
        for key in ('distance_inner','distance_outer','stable_margin','boundary_distance'):
            need(type(v[key]) is list and len(v[key])==len(v['roots']) and all(type(x) is float and math.isfinite(x) for x in v[key]),'SERIALIZATION_FAILED','Root-distance list invalid','backend')
        for key in ('reconstruction','unitarity','triangularity','tau'):
            need(v[key] is None or type(v[key]) is float and math.isfinite(v[key]) and v[key]>=0,'SERIALIZATION_FAILED','Schur diagnostic scalar invalid','backend')
        need(all(type(v[k]) is int and v[k]>=0 for k in ('stable_count','outside_count','boundary_count'))
             and v['vacuous'] is (m==0),'SERIALIZATION_FAILED','Schur counts/vacuity invalid','backend')
        if positive:need(v['stable_count']==m and v['outside_count']==v['boundary_count']==0 and len(v['roots'])==m,'SERIALIZATION_FAILED','Positive law requires all source roots stable','backend')
    else:
        fields(v,set(LYAP_ARRAYS)|set(LYAP_SCALARS)|{'vacuous','method'})
        need(v['vacuous'] is (m==0) and v['method']=='bilinear','SERIALIZATION_FAILED','Lyapunov method/vacuity invalid','backend')
        for key in LYAP_ARRAYS:
            if v[key] is None:
                need(not positive,'SERIALIZATION_FAILED','Unreached witness array in positive law','backend');continue
            if key in ('P_eigenvalues','D_eigenvalues'):
                need(type(v[key]) is list and len(v[key])==m and all(type(x) is float and math.isfinite(x) for x in v[key]),'SERIALIZATION_FAILED','Witness eigenvalue list invalid','backend')
            else:complex_matrix(v[key],(m,m))
        for key in LYAP_SCALARS:
            need(v[key] is None or type(v[key]) is float and math.isfinite(v[key]),'SERIALIZATION_FAILED','Finite witness scalar required','backend')
            if positive:need((v[key] is None)==(m==0),'SERIALIZATION_FAILED','Witness reached/null policy differs','backend')


def validate_wire(r,*,trusted,field_names):
    canonical_json(r);fields(r,field_names)
    need(r['schema']=='p39-stochastic-law-result' and type(r['revision']) is int and r['revision']==1
         and type(r['schema_version']) is int and r['schema_version']==2 and r['capability']==CAPABILITY
         and r['requested_claim']==CLAIM and r['specification_revision']==REVISION and r['solution_class']==SOLUTION_CLASS,
         'SERIALIZATION_FAILED','Exact frozen A2 literal contract required','backend')
    need(type(r['product_valid']) is bool and type(r['law_valid']) is bool and r['law_valid']==r['product_valid'],
         'SERIALIZATION_FAILED','Joint law-valid flag differs','backend')
    need(r['execution_status'] in ('COMPLETED','FAILED','NOT_RUN','SKIPPED') and r['computation_status'] in ('COMPLETED','FAILED','NOT_RUN','SKIPPED')
         and r['assessment_status'] in ('PASS','PASS_WITH_WARNINGS','FAIL','REFUSED','NOT_RUN','NOT_IMPLEMENTED')
         and r['model_status'] in ('NOT_EVALUATED','NUMERICALLY_UNRESOLVED','UNSUPPORTED','UNIQUE_LOCAL_SOLUTION')
         and r['input_admission_status'] in ('ADMITTED','NOT_ADMITTED') and r['forcing_status'] in ('NOT_EVALUATED','PASS','REFUSED','FAIL'),
         'SERIALIZATION_FAILED','Frozen status vocabulary invalid','backend')
    need(r['required_checks']==CHECK_IDS and type(r['checks']) is list and [x.get('check_id') for x in r['checks']]==CHECK_IDS,
         'SERIALIZATION_FAILED','Complete ordered A2 registry required','backend')
    diagnostics=r['diagnostics'];need(type(diagnostics) is list and len({x['id'] for x in diagnostics})==len(diagnostics),
         'SERIALIZATION_FAILED','Unique diagnostic IDs required','backend')
    dids={d['id'] for d in diagnostics}
    for check in r['checks']:
        fields(check,{'check_id','status','evidence','reason','diagnostic_ids','vacuous'});evidence(check['evidence']);strings(check['diagnostic_ids'])
        need(check['status'] in ('PASS','WARN','FAIL','UNSUPPORTED','NOT_IMPLEMENTED','NOT_RUN') and type(check['vacuous']) is bool
             and (check['reason'] is None or type(check['reason']) is str) and set(check['diagnostic_ids'])<=dids,
             'SERIALIZATION_FAILED','Malformed A2 check evidence','backend')
        need(check['status']!='PASS' or bool(check['evidence']),'SERIALIZATION_FAILED','PASS requires individual evidence','backend')
        need(check['status']!='NOT_RUN' or not check['evidence'] and not check['diagnostic_ids'] and not check['vacuous'],
             'SERIALIZATION_FAILED','Unreached check has fabricated evidence','backend')
    need(r['failure_reason'] is None or type(r['failure_reason']) is str,'SERIALIZATION_FAILED','Failure reason type invalid','backend')
    if r['provenance'] is not None:fields(r['provenance'],TRUST_FIELDS)
    if r['backend'] is not None:
        b=r['backend'];fields(b,{'python','numpy','scipy','jax','jaxlib','os','architecture','blas','lapack','thread_settings','loaded_libraries','dtype','intermediate_dtype','methods','policy'})
        need(all(type(b[k]) is str and bool(b[k]) for k in ('python','numpy','scipy','jax','jaxlib','os','architecture','blas','lapack'))
             and b['dtype']=='float64' and b['intermediate_dtype']=='complex128' and b['policy']==REVISION
             and type(b['thread_settings']) is dict and all(type(k) is str and type(v) is str for k,v in b['thread_settings'].items()),
             'SERIALIZATION_FAILED','Actual A2 runtime evidence malformed','backend')
        strings(b['loaded_libraries']);strings(b['methods'])
        need(bool(b['loaded_libraries']) and bool(b['methods']),'SERIALIZATION_FAILED','Runtime evidence missing','backend')
    if r['axes'] is not None:
        fields(r['axes'],AXES_FIELDS);fields(r['coordinates'],COORDINATE_FIELDS);validate_axes_coordinates(r['axes'],r['coordinates'])
        a=r['axes'];n=a['n_state']+a['n_jump']
        need(all(type(a[k]) is int and a[k]>=0 for k in ('n_k','n_z','n_state','n_jump','n_innovation'))
             and a['n_state']==a['n_k']+a['n_z'] and a['n_z']==a['n_innovation'] and 0<n<=256,
             'SERIALIZATION_FAILED','Admitted dimensions invalid','backend')
        for key,length in [('equation_order',n),('variable_order',n),('state_order',a['n_state']),('control_order',a['n_jump']),('innovation_order',a['n_innovation'])]:strings(a[key],unique=True,length=length)
        for key,length in [('S',n),('T',n),('innovation_scales',a['n_innovation']),('reference',n)]:
            values=r['coordinates'][key]
            need(type(values) is list and len(values)==length and all(type(x) is float and math.isfinite(x) and (key=='reference' or x>0) for x in values),'SERIALIZATION_FAILED','Coordinate numeric axes invalid','backend')

    m=r['axes']['n_innovation'] if r['axes'] is not None else 0
    for d in diagnostics:
        fields(d,{'id','kind','values'})
        if d['kind'] in ('schur','lyapunov'):validate_forcing_diagnostic(d,m,r['product_valid'])
        elif d['kind']=='upstream_recheck':
            v=d['values'];fields(v,{'stage','result_sha256','check_id','record','fresh_record'})
            need(v['stage'] in ('SOURCE','P3.4','P3.5','P3.6','P3.7','P3.8','P3.9A1') and type(v['record']) is dict
                 and (v['fresh_record'] is None or type(v['fresh_record']) is dict),
                 'SERIALIZATION_FAILED','Complete upstream recheck format required','backend')
        else:validate_diagnostic(d)
    need(type(r['warnings']) is list,'SERIALIZATION_FAILED','Warning list required','backend')
    for w in r['warnings']:
        fields(w,{'code','check_id','message','evidence'});evidence(w['evidence'])
        need(w['code'] in ('FINITE_UNDERFLOW','INHERITED_ACCEPTED_WARNING') and w['check_id'] in CHECK_IDS and type(w['message']) is str,'SERIALIZATION_FAILED','Unauthorized warning category','backend')
    strings(r['limitations']);strings(r['failure_reasons'],unique=True);evidence(r['acceptance_evidence'])
    need(bool(r['limitations']) and r['independent_validation_status']==r['external_validation_status']=='NOT_RUN'
         and r['acceptance_evidence']==[] and r['capability_maturity']=='IMPLEMENTED','SERIALIZATION_FAILED','Ordinary A2 cannot manufacture acceptance','backend')
    if r['product_valid']:
        need(r['backend'] is not None and r['axes'] is not None and trusted is not None and canonical_json(r['provenance'])==canonical_json(trusted['expected']) and r['trust_sha256']==digest(trusted['expected'])
             and r['specification_sha256']==SPEC_SHA256 and r['upstream_stage']=='P3.9A1'
             and r['upstream_source_commit']==trusted['expected']['upstream']['source_commit']
             and r['upstream_result_sha256']==r['A1_result_sha256']==trusted['expected']['upstream']['result_sha256'],
             'SERIALIZATION_FAILED','A2 published provenance differs from independent admitted trust','backend')
        for key in ('law','axes','coordinates','checks','diagnostics','warnings','backend'):
            need(canonical_json(r[key])==canonical_json(trusted[key]),'SERIALIZATION_FAILED','Published '+key+' differs from verified private snapshot','backend')
        fields(r['law'],LAW_FIELDS);law=r['law'];fields(law['shapes'],{'G','H','L_s','L_j'})
        need(law['equations']==EQUATIONS and law['source_certificate_sha256']==trusted['expected']['source_certificate_sha256'],
             'SERIALIZATION_FAILED','Frozen law equation/source binding invalid','backend')
        ns,nj=r['axes']['n_state'],r['axes']['n_jump'];shapes={'G':[nj,ns],'H':[ns,ns],'L_s':[ns,m],'L_j':[nj,m]}
        need(law['shapes']==shapes,'SERIALIZATION_FAILED','Law matrix shape metadata differs','backend')
        for key,shape in shapes.items():matrix(law[key],shape)
        matrix(law['R_q'],(m,m));need(set(law['forcing_evidence_ids'])<=dids,'SERIALIZATION_FAILED','Forcing diagnostic reference missing','backend')
        need(all(x['status']=='PASS' for x in r['checks'][:-1]) and r['checks'][-1]['status']==('WARN' if r['warnings'] else 'PASS')
             and r['claim']==CLAIM and r['model_status']=='UNIQUE_LOCAL_SOLUTION' and r['forcing_status']=='PASS'
             and r['input_admission_status']=='ADMITTED' and r['execution_status']==r['computation_status']=='COMPLETED'
             and r['assessment_status']==('PASS_WITH_WARNINGS' if r['warnings'] else 'PASS') and r['failure_reason'] is None
             and not r['failure_reasons'],'SERIALIZATION_FAILED','Complete positive A2 truth table invalid','backend')
    else:
        need(r['law'] is None and r['claim'] is None and not r['law_valid'] and r['assessment_status'] not in ('PASS','PASS_WITH_WARNINGS')
             and r['model_status']!='UNIQUE_LOCAL_SOLUTION','SERIALIZATION_FAILED','Failed law must be atomically suppressed','backend')
