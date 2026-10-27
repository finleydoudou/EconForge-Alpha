"""Individually evidenced frozen A0–A6 and A7 state obligations; no upstream solve."""
import json
from econforge.stochastic_contract import need,real_matrix,canonical_json,digest,ASSUMPTIONS
from econforge.innovation_source import source_certificate
from econforge.innovation_loading import verify_loadings
from econforge.matrix_assembly import _verify_p34
from econforge.p35_structure import validate_structure
from econforge.gh_numerics import component_check,policy_check
from econforge.gh_source import prepare,original_equations,identities_and_process
from econforge.qz_admission import same


def upstream(r,gate,identifier,record,fresh,condition=True,*,stage='P3.8',result_digest=None,vacuous=False):
    did=identifier+'.measurement'
    r['diagnostics'].append(dict(id=did,kind='upstream_recheck',values=dict(
        stage=stage,result_sha256=result_digest or r['provenance']['p38_result_sha256'],
        check_id=identifier,record=record,fresh_record=fresh)))
    gate(identifier,condition,diagnostic_id=did,vacuous=vacuous)


def verify_source(a,t,c,a1,r,gate,start):
    np=a['np'];axes=a['axes'];model=a['model'];md=a['metadata'];b=a['bundle']
    ns,nj,nk,m=[axes[k] for k in ('n_state','n_jump','n_k','n_innovation')]
    start('A2.A0.model');structure=validate_structure(model)
    upstream(r,gate,'A2.A0.model',{'source_hash':t['original_model_sha256']},
        {'source_hash':model.digest,'canonical':digest(json.loads(model.source_json))},
        model.digest==t['original_model_sha256'] and digest(json.loads(model.source_json))==t['canonical_model_sha256'] and same(structure,md['structure']),stage='SOURCE')
    start('A2.A0.reference');S,T,scalar=_verify_p34(model,b['p34_result'],structure)
    scalar_wire=json.loads(json.dumps(scalar.to_dict(),allow_nan=False))
    upstream(r,gate,'A2.A0.reference',b['scalar_reference_verification'],scalar_wire,
        same(scalar_wire,b['scalar_reference_verification']) and scalar_wire['evaluation_valid'] and scalar_wire['residuals_within_tolerance'] and model.branch_id==t['branch_id'],stage='P3.4',result_digest=t['p34_result_sha256'])
    blocks=b['p34_result']['blocks']
    zero_lag=all(x==0.0 for row in blocks[0]['jacobian'] for x in row)
    upstream(r,gate,'A2.A0.derivatives',b['p34_result'],{'S':S,'T':T,'raw_lag_exact_zero':zero_lag},
        b['p34_result']['derivative_valid'] and zero_lag and [x['name'] for x in blocks]==['J_minus','J_current','J_plus'] and same(S,md['S']) and same(T,md['T']),stage='P3.4',result_digest=t['p34_result_sha256'])
    assumptions=c['assumptions'];source=json.loads(model.source_json)
    neighborhood=dict(variables=[dict(id=x['id'],domain=x['domain']) for x in source['variables']],parameters=[dict(id=x['id'],domain=x['domain']) for x in source['parameters']])
    upstream(r,gate,'A2.A0.assumptions',assumptions,{'neighborhood':neighborhood,'source_assertion':md['admission_manifest']},
        all(same(assumptions[k],v) for k,v in ASSUMPTIONS.items()) and same(assumptions['neighborhood'],neighborhood) and same(assumptions['source_assertion'],md['admission_manifest']),stage='SOURCE')
    covariance=source['covariance'];covok=(covariance['kind']=='diagonal' and len(covariance['values'])==m and all(type(v) in (float,int) and np.isfinite(v) and v>=0 for v in covariance['values']))
    upstream(r,gate,'A2.A0.covariance',covariance,{'source_digest':digest(covariance),'dimension':m},covok and digest(covariance)==t['covariance_source_sha256'],stage='SOURCE',vacuous=m==0)
    p37=c['p37_result'];re=a['rechecks'];fresh=a['p38_rechecks'];tau=p37['metadata']['tau_qz']
    upstream(r,gate,'A2.A1.lineage',p37['metadata']['bindings'],a['expected']['p37_bindings'],
        digest(p37)==t['p37_result_sha256'] and digest(c['p36_result'])==t['p36_result_sha256'] and digest(b['p35_result'])==t['p35_result_sha256'],stage='P3.7',result_digest=t['p37_result_sha256'])
    upstream(r,gate,'A2.A1.pencil',{'orientation':md['orientation'],'matrix_provenance':md['matrix_provenance']},
        {'A':a['A'].tolist(),'B':a['B'].tolist(),'pencil':p37['pencil']},p37['pencil']=='B - lambda A' and same(a['A'].tolist(),b['p35_result']['A']) and same(a['B'].tolist(),b['p35_result']['B']),stage='P3.5',result_digest=t['p35_result_sha256'])
    reg=re['regularity'];upstream(r,gate,'A2.A1.regularity',p37['regularity'],reg,
        reg['status']=='PASS' and any(x['rho']>reg['tau_reg'] for x in reg['witnesses']),stage='P3.7',result_digest=t['p37_result_sha256'])
    structural=re['ordered']['structural'];six={'reconstruction_A','reconstruction_B','unitarity_Q','unitarity_Z','triangularity_A','triangularity_B'}
    upstream(r,gate,'A2.A1.factors',p37['ordered_diagnostics']['structural'],structural,
        set(structural)==six and all(np.isfinite(v) and v<=tau for v in structural.values()),stage='P3.7',result_digest=t['p37_result_sha256'])
    ordered=re['ordered'];upstream(r,gate,'A2.A1.roots',p37['ordered_diagnostics'],ordered,
        ordered['spectrum']['passed'] and ordered['conjugacy']['passed'] and re['unordered_conjugacy']['passed'] and len(ordered['classes'])==ns+nj and ordered['classes'].count('STABLE')==ns and ordered['classes'].count('OUTSIDE')==nj,stage='P3.7',result_digest=t['p37_result_sha256'])
    upstream(r,gate,'A2.A1.algebraic',{'exact_infinite_count':p37['exact_infinite_count'],'pairs':c['p36_result']['pairs']},
        {'regularity':reg['status'],'classes':ordered['classes']},p37['unresolved_pair_count']==0 and not any(x['pair_status']=='INDETERMINATE_0_OVER_0' for x in c['p36_result']['pairs']),stage='P3.7',result_digest=t['p37_result_sha256'])
    upstream(r,gate,'A2.A2.dimensions',{'n_state':p37['n_state'],'n_jump':p37['n_jump']},
        {'stable':ordered['classes'].count('STABLE'),'outside':ordered['classes'].count('OUTSIDE')},p37['stable_count']==ns and p37['outside_count']==nj and ordered['stable_count']==ns,stage='P3.7',result_digest=t['p37_result_sha256'])
    projection=re['projection'];condition=fresh['X_conditioning'];full_projection=projection['status']=='PASS' and condition['status'] in ('PASS','WARN')
    upstream(r,gate,'A2.A2.projection',p37['projection'],{'relative':projection,'full_absolute_graph':condition},full_projection,stage='P3.8',vacuous=ns==0)
    upstream(r,gate,'A2.A2.free_directions',{'declared_prescribed_states':axes['state_order']},
        {'stable_dimension':ordered['stable_count'],'full_state_projection':full_projection},ordered['stable_count']==ns and full_projection,stage='P3.7',result_digest=t['p37_result_sha256'])
    upstream(r,gate,'A2.A2.causality',assumptions,{'state_order':axes['state_order'],'control_order':axes['control_order'],'post_surprise_graph':True},
        assumptions['initial_states']=='ARBITRARY_ADMISSIBLE_FINITE_NEARBY' and assumptions['information']=='COMMON_POST_SURPRISE' and axes['state_order']==axes['partitions']['k']+axes['partitions']['z'] and full_projection,stage='SOURCE')
    upstream(r,gate,'A2.A3.p38',c['p38_result']['metadata'],{'G':a['G'].tolist(),'H':a['H'].tolist()},
        c['p38_result']['recovery_valid'] and a['G'].shape==(nj,ns) and a['H'].shape==(ns,ns) and bool(np.isfinite(a['G']).all()) and bool(np.isfinite(a['H']).all()))
    start('A2.A3.recovery')
    from econforge.innovation_admission import check_retained_component
    retained={}
    for key in ('stable_solve','graph_solve','state_solve'):
        check_retained_component(a,key,c['p38_result']['diagnostics'][key]);retained[key]=c['p38_result']['diagnostics'][key]
    upstream(r,gate,'A2.A3.recovery',retained,{'returned_graph':fresh['returned_graph'],'returned_state':fresh['returned_state'],'stable_block':fresh['stable_block_conditioning']},fresh['returned_graph']['passed'] and fresh['returned_state']['passed'] and fresh['stable_block_conditioning']['status'] in ('PASS','WARN'),vacuous=ns==0)
    for key in ('G_realness','H_realness'):check_retained_component(a,key,c['p38_result']['diagnostics'][key])
    upstream(r,gate,'A2.A3.realness',{k:c['p38_result']['diagnostics'][k] for k in ('G_realness','H_realness')},
        {'real_float64':a['G'].dtype==a['H'].dtype==np.dtype('float64')},a['G'].dtype==a['H'].dtype==np.dtype('float64'))
    start('A2.A3.policy');d={};policy_check(a,a['G'],a['H'],d)
    upstream(r,gate,'A2.A3.policy',c['p38_result']['diagnostics']['final_policy'],d['final_policy'],d['final_policy']['passed'])
    start('A2.A3.identities');prepared=prepare(a,d);identities_and_process(a,a['G'],a['H'],prepared,d)
    identity_keys=[k for k in d if k.startswith(('identity_','alias_'))]
    upstream(r,gate,'A2.A3.identities',{'alias_maps':b['alias_maps'],'identity_maps':b['identity_maps']},
        {k:d[k] for k in identity_keys},all(d[k]['passed'] for k in identity_keys),vacuous=not identity_keys)
    start('A2.A4.source_process');cert=source_certificate(a,axes,a['coordinates'])
    need(same(cert,c['source_certificate']) and digest(cert)==t['source_certificate_sha256'],
        'COEFFICIENT_MISMATCH','Fresh original AST certificate differs from independent trust')
    R=real_matrix(np,cert['R'],(m,m));zi=[axes['variable_order'].index(z) for z in axes['partitions']['z']]
    Rq=R/a['T'][zi,None]*a['T'][None,zi] if m else R.copy()
    need(bool(np.isfinite(Rq).all()),'NUMERICAL_NONFINITE','Native original process nonfinite','backend')
    upstream(r,gate,'A2.A4.source_process',c['source_certificate'],cert,
        same(Rq.tolist(),fresh['source_maps']['native_R']) and len(cert['process_means'])==m,stage='SOURCE',result_digest=t['source_certificate_sha256'],vacuous=m==0)
    means=np.array(cert['process_means'],dtype=np.float64);intercepts=np.array(cert['process_intercepts'],dtype=np.float64)
    source_rows=[row for row in model.equations if row.kind=='z_process'];reference_terms=intercepts+R@means-means
    bounds=np.array([row.scale*1e-10+row.magnitude*1e-8 for row in source_rows])
    need(bool(np.isfinite(reference_terms).all()) and bool(np.isfinite(bounds).all()),'NUMERICAL_NONFINITE','Process reference diagnostic nonfinite','backend')
    upstream(r,gate,'A2.A4.reference_process',{'means':cert['process_means'],'intercepts':cert['process_intercepts']},
        {'residual':reference_terms.tolist(),'allowed_bound':bounds.tolist()},bool(np.all(np.abs(reference_terms)<=bounds)),stage='SOURCE',vacuous=m==0)
    for name,record in [('mean_zz',d['process_zz']),('mean_zk',d['process_zk'])]:
        upstream(r,gate,'A2.A4.'+name,c['p38_result']['diagnostics']['process_zz' if name=='mean_zz' else 'process_zk'],record,record['passed'],vacuous=record['vacuous'])
    upstream(r,gate,'A2.A4.source_coefficients',c['source_certificate'],cert,digest(cert)==t['source_certificate_sha256'],stage='SOURCE',result_digest=t['source_certificate_sha256'],vacuous=m==0)
    Ls=real_matrix(np,a1['L_s'],(ns,m));Lj=real_matrix(np,a1['L_j'],(nj,m))
    measured={};raw=None
    for name,passed,record,code in verify_loadings(a,cert,Ls,Lj):
        if name=='pathwise_raw':raw=record;continue
        measured[name]=(passed,record,code)
    for key,name in [('physical_zero','physical_timing'),('exogenous_loading','exogenous_loading'),('jump_loading','jump_loading')]:
        passed,record,code=measured[key];did='A2.A5.'+name+'.measurement'
        r['diagnostics'].append(dict(id=did,kind='component',values=record));gate('A2.A5.'+name,passed,diagnostic_id=did,reason=code or 'LOADING_CHECK_FAILED',vacuous=record['vacuous'])
    upstream(r,gate,'A2.A5.axes_units',{'axes':a1['axes'],'coordinates':a1['coordinates']},
        {'axes':axes,'coordinates':a['coordinates']},same(axes,a1['axes']) and same(a['coordinates'],a1['coordinates']) and digest(axes)==t['axes_sha256'] and digest(a['coordinates'])==t['coordinate_sha256'],stage='P3.9A1',result_digest=t['upstream']['result_sha256'])
    upstream(r,gate,'A2.A5.a1',{'checks':a1['checks'],'source':t['upstream']['source_commit']},
        {'complete_wire_sha256':digest(a1),'joint_loadings':a1['product_valid']},digest(a1)==t['upstream']['result_sha256'] and all(x['status']=='PASS' for x in a1['checks'][:-1]) and a1['checks'][-1]['status'] in ('PASS','WARN'),stage='P3.9A1',result_digest=t['upstream']['result_sha256'])
    start('A2.A6.mean_rows');original_equations(a,a['G'],a['H'],d,'A2.original_mean')
    upstream(r,gate,'A2.A6.mean_rows',c['p38_result']['diagnostics']['original_real'],d['A2.original_mean'],d['A2.original_mean']['passed'])
    for key,name,kind in [('pathwise_coefficients','pathwise_rows','component'),('conditional_semantics','conditional_rows','conditional_forecast_error')]:
        passed,record,code=measured[key];did='A2.A6.'+name+'.measurement';r['diagnostics'].append(dict(id=did,kind=kind,values=record));gate('A2.A6.'+name,passed,diagnostic_id=did,reason=code or 'CONDITIONAL_SEMANTICS_INVALID',vacuous=not bool(record.get('shape',[1,1])[0]*m))
    r['diagnostics'].append(dict(id='A2.original_raw_innovation_terms',kind='event',values=raw))
    # These three A7 obligations remain separately evidenced, not inferred from forcing.
    upstream(r,gate,'A2.A7.covariance',covariance,{'PSD_diagonal':covok,'source_digest':digest(covariance)},covok and digest(covariance)==t['covariance_source_sha256'],stage='SOURCE',vacuous=m==0)
    upstream(r,gate,'A2.A7.state_stability',{'stable_solve':retained['stable_solve'],'state_solve':retained['state_solve']},
        {'full_projection':condition,'actual_HX':fresh['returned_state'],'stable_classes':ordered['classes'][:ns]},full_projection and fresh['returned_state']['passed'] and all(k=='STABLE' for k in ordered['classes'][:ns]) and fresh['stable_block_conditioning']['status'] in ('PASS','WARN'),vacuous=ns==0)
    upstream(r,gate,'A2.A7.bounded_class',assumptions,{'class':ASSUMPTIONS['solution_class'],'covariance_finite':covok,'common_information':assumptions['information']},assumptions['solution_class']==ASSUMPTIONS['solution_class'] and assumptions['innovation']==ASSUMPTIONS['innovation'] and covok,stage='SOURCE')
    upstream(r,gate,'A2.A7.stationarity_scope',{'initial_states':assumptions['initial_states']},
        {'stationarity':'Stationary initialization gives second-order stationarity; finite initial conditions may be transient; constant covariance does not imply strict stationarity'},assumptions['initial_states']=='ARBITRARY_ADMISSIBLE_FINITE_NEARBY',stage='SOURCE')
    return cert,Rq,Ls,Lj,covariance
