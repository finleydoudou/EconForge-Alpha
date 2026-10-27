"""Public frozen P3.9A2 complete first-order stochastic-law certification API."""
from dataclasses import dataclass
import json,math,warnings,importlib.metadata
from econforge import innovation
from econforge.stochastic_contract import *
from econforge.stochastic_admission import admit_a2
from econforge.innovation_admission import recheck_p38
from econforge.stochastic_verification import verify_source
from econforge.stochastic_forcing import forcing
from econforge.stochastic_wire import validate_wire
from econforge.qz_admission import QZError
from econforge.model_errors import ModelError


@dataclass(frozen=True)
class StochasticLawResult:
    wire_json:str
    def to_dict(self):return json.loads(self.wire_json)


def result_template():
    r=innovation.result_template()
    for k in ('L_s','L_j','L_s_shape','L_j_shape','source_certificate','forcing_certification'):r.pop(k)
    r.update(schema='p39-stochastic-law-result',capability=CAPABILITY,requested_claim=CLAIM,
        law=None,law_valid=False,forcing_status='NOT_EVALUATED',solution_class=SOLUTION_CLASS,A1_result_sha256=None,
        required_checks=list(CHECK_IDS),checks=[dict(check_id=k,status='NOT_RUN',evidence=[],reason=None,diagnostic_ids=[],vacuous=False) for k in CHECK_IDS],
        limitations=['Conditional local first-order causal uniformly L2-bounded law for the admitted canonical source',
            'Declared common-information MDS, finite covariance, smooth local class and nearby finite prescribed states are conditional assumptions',
            'Numerical Schur/Lyapunov sufficient admission is not interval proof or global nonlinear uniqueness',
            'Original source/derivatives share accepted upstream lineage; this is not external scientific validation',
            'Stationary initialization gives second-order stationarity; finite initial states may be transient; no strict stationarity claim',
            'Hashes bind integrity, not authorship; consumers must re-admit full result identity against separate trust',
            'IRFs, simulation, estimation, higher order, general covariance and release validation are excluded'])
    return r


def record_check(r,name,status='PASS',*,diagnostic_id=None,reason=None,vacuous=False):
    c=next(x for x in r['checks'] if x['check_id']==name)
    c.update(status=status,reason=reason,vacuous=vacuous,diagnostic_ids=[] if diagnostic_id is None else [diagnostic_id],
        evidence=[dict(kind='executed_A2_obligation',location='inline:diagnostics.'+diagnostic_id if diagnostic_id else 'private:verified_'+name,
            digest=r['trust_sha256'])])
    if vacuous:c['evidence'].append(dict(kind='explicit_empty_dimension_check',location='inline:axes; defined empty '+name,digest=r['trust_sha256']))


def suppress(r):
    r.update(law=None,law_valid=False,product_valid=False,claim=None)
    if r['model_status']=='UNIQUE_LOCAL_SOLUTION':r['model_status']='NUMERICALLY_UNRESOLVED'


def _serialize_result(candidate,verified):
    """Internal publisher. Verified is private computation context, never candidate enrollment."""
    r=candidate
    try:
        expected_wire=canonical_json(verified);private=json.loads(expected_wire)
        wire=canonical_json(candidate);r=json.loads(wire)
        validate_wire(r,trusted=private,field_names=set(result_template()))
        need(canonical_json(r)==wire and canonical_json(private)==expected_wire,'SERIALIZATION_FAILED',
             'A2 publication snapshot changed during verification','backend')
        return StochasticLawResult(wire)
    except Exception:
        suppress(candidate);suppress(r);raise


def fail(r,stage,exc):
    if isinstance(exc,InnovationError):code,origin=exc.code,exc.kind
    elif isinstance(exc,QZError):
        code='DEPENDENCY_MISSING' if 'BACKEND_VERSION' in exc.code else 'UPSTREAM_GATE_UNRESOLVED' if getattr(exc,'numerical',False) and exc.refused else 'SOURCE_MISMATCH'
        origin='missing' if code=='DEPENDENCY_MISSING' else 'uncertain' if getattr(exc,'numerical',False) and exc.refused else 'check' if getattr(exc,'numerical',False) else 'unsupported' if exc.refused else 'input'
    elif isinstance(exc,(ImportError,importlib.metadata.PackageNotFoundError)):code,origin='DEPENDENCY_MISSING','missing'
    elif isinstance(exc,ModelError):code,origin=('SOURCE_UNSUPPORTED','unsupported') if exc.diagnostic.unsupported else ('INVALID_INPUT','input')
    elif isinstance(exc,(FloatingPointError,OverflowError)):code,origin='NUMERICAL_NONFINITE','backend'
    else:code,origin='BACKEND_EXCEPTION','backend'
    if stage=='A2.serialization':code,origin='SERIALIZATION_FAILED','backend'
    refused=origin in ('missing','unsupported','uncertain','outside');reason='P39.A2.'+code
    suppress(r)
    r.update(execution_status='COMPLETED' if origin in ('check','uncertain','outside') else 'NOT_RUN' if refused else 'FAILED',
        assessment_status='REFUSED' if refused else 'FAIL',model_status='UNSUPPORTED' if origin in ('unsupported','outside') else 'NUMERICALLY_UNRESOLVED' if origin in ('check','uncertain','backend') else 'NOT_EVALUATED',failure_reason=reason,failure_reasons=[reason])
    if origin in ('check','uncertain','outside') and r['computation_status']!='NOT_RUN':r['computation_status']='COMPLETED'
    if origin=='backend' and stage!='A2.serialization' and r['computation_status']!='NOT_RUN':r['computation_status']='FAILED'
    if '.schur.' in stage or '.lyapunov.' in stage or stage=='A2.A7.forcing':r['forcing_status']='REFUSED' if refused else 'FAIL'
    record_check(r,stage,'UNSUPPORTED' if refused else 'FAIL',reason=reason)
    r['diagnostics'].append(dict(id='A2.failure',kind='event',values=dict(stage=stage,reason=reason,message=str(exc),exception_type=type(exc).__name__)))


def certify_law(expected,content):
    """Admit separately enrolled frozen trust/content and certify; never enroll incoming candidates."""
    r=result_template();current=['A2.A8.trust'];verified=None
    def start(name):current[0]=name
    def gate(name,passed=True,*,diagnostic_id=None,reason='SOURCE_MISMATCH',kind='check',vacuous=False):
        start(name);record_check(r,name,'PASS' if passed else 'FAIL',diagnostic_id=diagnostic_id,
            reason=None if passed else 'P39.A2.'+reason,vacuous=vacuous)
        need(passed,reason,'Required A2 check rejected: '+name,kind)
    try:
        t,c,a1,t1,c1,specs=admit_a2(expected,content)
        independent=json.loads(canonical_json(t))
        r.update(provenance=json.loads(canonical_json(t)),trust_sha256=digest(t),specification_sha256=SPEC_SHA256,
            upstream_stage='P3.9A1',upstream_source_commit=A1_SOURCE,upstream_result_sha256=t['upstream']['result_sha256'],A1_result_sha256=t['upstream']['result_sha256'])
        start('A2.A3.p38');r['computation_status']='FAILED'
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always');a=recheck_p38(t1,c1,specs)
        for w in observed:
            need(issubclass(w.category,RuntimeWarning) and 'underflow' in str(w.message).lower(),'BACKEND_WARNING',str(w.message),'backend')
            r['warnings'].append(dict(code='FINITE_UNDERFLOW',check_id='A2.A3.p38',message=str(w.message),evidence=[dict(kind='observed_runtime_warning',location='A2.P38_reverification',digest=None)]))
        r['backend']=innovation.backend_record(a);r['backend']['intermediate_dtype']='complex128'
        r.update(axes=json.loads(canonical_json(a['axes'])),coordinates=json.loads(canonical_json(a['coordinates'])))
        np=a['np']
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            with np.errstate(over='raise',invalid='raise',divide='raise',under='warn'):
                cert,Rq,Ls,Lj,covariance=verify_source(a,independent,c,a1,r,gate,start)
        for w in observed:
            need(issubclass(w.category,RuntimeWarning) and 'underflow' in str(w.message).lower(),'BACKEND_WARNING',str(w.message),'backend')
            r['warnings'].append(dict(code='FINITE_UNDERFLOW',check_id='A2.A6.pathwise_rows',message=str(w.message),evidence=[dict(kind='observed_runtime_warning',location='A2.original_source_verification',digest=None)]))
        r['input_admission_status']='ADMITTED';r['computation_status']='COMPLETED'
        start('A2.schur.quality');r['computation_status']='FAILED'
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            with np.errstate(over='raise',invalid='raise',divide='raise',under='warn'):
                forcing(np,a['linalg'],Rq,independent['source_certificate_sha256'],gate,start,r['diagnostics'])
        need(not observed,'BACKEND_WARNING','Forcing numerical warning: '+'; '.join(str(w.message) for w in observed),'backend')
        r['computation_status']='COMPLETED';r['forcing_status']='PASS'
        if len(Rq):r['backend']['methods']+=['scipy.linalg.schur complex unsorted','scipy.linalg.svdvals finite','scipy.linalg.solve_discrete_lyapunov adjoint bilinear','scipy.linalg.eigh evd checked Hermitian witnesses']
        gate('A2.A7.forcing',all(x['status']=='PASS' for x in r['checks'] if x['check_id'] in FORCING_CHECKS),diagnostic_id='A2.source_lyapunov',vacuous=len(Rq)==0)
        gate('A2.A8.trust',digest(c['upstream_result'])==independent['upstream']['result_sha256'] and independent['upstream']['source_commit']==A1_SOURCE)
        gate('A2.A8.specifications',independent['specifications']==SPECIFICATIONS)
        substantive=[x for x in r['checks'] if x['check_id'] not in ('A2.A8.checks','A2.A8.atomicity','A2.A8.claims','A2.serialization','A2.warnings')]
        gate('A2.A8.checks',all(x['status']=='PASS' and x['evidence'] for x in substantive))
        law=dict(G=a['G'].tolist(),H=a['H'].tolist(),L_s=Ls.tolist(),L_j=Lj.tolist(),
            shapes=dict(G=[a['n_jump'],a['n_state']],H=[a['n_state'],a['n_state']],L_s=list(Ls.shape),L_j=list(Lj.shape)),
            R_q=Rq.tolist(),covariance=covariance,equations=list(EQUATIONS),source_certificate_sha256=independent['source_certificate_sha256'],forcing_evidence_ids=['A2.source_schur','A2.source_lyapunov'])
        gate('A2.A8.atomicity',r['law'] is None and not r['law_valid'] and all(k in law for k in LAW_FIELDS))
        gate('A2.A8.claims',r['claim'] is None and r['forcing_status']=='PASS' and c['assumptions']['solution_class']==SOLUTION_CLASS)
        # Retain only advisory warnings authorized by the accepted A1 producer.
        for w in a1['warnings']:
            r['warnings'].append(dict(code=w['code'],check_id='A2.A5.a1',message=w['message'],evidence=w['evidence']))
        gate('A2.serialization')
        record_check(r,'A2.warnings','WARN' if r['warnings'] else 'PASS')
        verified={k:json.loads(canonical_json(r[k])) for k in ('axes','coordinates','checks','diagnostics','warnings','backend')}
        verified.update(expected=independent,law=json.loads(canonical_json(law)))
        r.update(law=json.loads(canonical_json(law)),law_valid=True,product_valid=True,claim=CLAIM,
            execution_status='COMPLETED',assessment_status='PASS_WITH_WARNINGS' if r['warnings'] else 'PASS',model_status='UNIQUE_LOCAL_SOLUTION')
        return _serialize_result(r,verified)
    except Exception as exc:
        fail(r,current[0],exc)
        try:return _serialize_result(r,None)
        except Exception as serialization_error:
            safe=result_template();reasons=list(dict.fromkeys(r['failure_reasons']+['P39.A2.SERIALIZATION_FAILED']))
            decisive=r['failure_reason'] if r['assessment_status']=='FAIL' else 'P39.A2.SERIALIZATION_FAILED'
            safe.update(execution_status='FAILED',computation_status=r['computation_status'],assessment_status='FAIL',failure_reason=decisive,failure_reasons=reasons)
            record_check(safe,'A2.serialization','FAIL',reason=decisive)
            safe['diagnostics'] += [d for d in r['diagnostics'] if d.get('id')=='A2.failure']
            safe['diagnostics'].append(dict(id='A2.serialization.failure',kind='event',values=dict(stage='A2.serialization',reason=decisive,message=str(serialization_error),exception_type=type(serialization_error).__name__)))
            return StochasticLawResult(canonical_json(safe))
