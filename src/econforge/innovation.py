"""P3.9A1 public API: source-verified native loadings, atomic and fail closed."""
from dataclasses import dataclass
import contextlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import warnings

from econforge.innovation_contract import (
    InnovationError, need, CLAIM, CAPABILITY, REVISION, SPEC_SHA256, CHECK_IDS,
    real_matrix, sha_bytes)
from econforge.innovation_admission import admit_trust, recheck_p38
from econforge.innovation_source import source_certificate
from econforge import innovation_loading
from econforge.p35_admission import canonical_json, digest
from econforge.model_errors import ModelError
from econforge.qz_admission import QZError, same
from econforge.innovation_wire import validate_wire


@dataclass(frozen=True)
class InnovationResult:
    wire_json: str

    def to_dict(self):
        return json.loads(self.wire_json)


def result_template():
    return dict(schema='p39-innovation-loading-result',revision=1,schema_version=2,
        capability=CAPABILITY,requested_claim=CLAIM,claim=None,
        execution_status='NOT_RUN',computation_status='NOT_RUN',assessment_status='NOT_RUN',
        model_status='NOT_EVALUATED',input_admission_status='NOT_ADMITTED',product_valid=False,
        specification_revision=REVISION,specification_sha256=None,trust_sha256=None,
        upstream_result_sha256=None,upstream_stage=None,upstream_source_commit=None,
        provenance=None,axes=None,coordinates=None,required_checks=list(CHECK_IDS),
        checks=[dict(check_id=k,status='NOT_RUN',evidence=[],reason=None,diagnostic_ids=[],vacuous=False)
                for k in CHECK_IDS],diagnostics=[],warnings=[],failure_reason=None,failure_reasons=[],
        backend=None,capability_maturity='IMPLEMENTED',independent_validation_status='NOT_RUN',
        external_validation_status='NOT_RUN',acceptance_evidence=[],
        limitations=['Source-verified local first-order native innovation loadings only',
            'Declared common information and MDS assumptions are conditional, not empirically proven',
            'Original raw-row verification shares the admitted model and P3.4 derivative lineage',
            'Trusted digests provide integrity, not signatures or scientific acceptance',
            'Forcing stability and complete stochastic-law certification remain deferred to A2',
            'IRFs, simulation, general covariance, external and release validation are not provided'],
        L_s=None,L_j=None,L_s_shape=None,L_j_shape=None,source_certificate=None,
        forcing_certification='NOT_EVALUATED')


def check(r,name,status='PASS',*,diagnostic=None,kind='component',reason=None,vacuous=False):
    identifier = 'A1.'+name
    target = next(x for x in r['checks'] if x['check_id'] == identifier)
    diagnostic_ids = list(target['diagnostic_ids'])
    if diagnostic is not None:
        did = identifier+'.measurement'
        r['diagnostics'].append(dict(id=did,kind=kind,values=diagnostic))
        diagnostic_ids.append(did)
    locations = dict(trust='inline:provenance',p38='content:p38_result and inline:upstream_rechecks',
        source='content:verification_bundle.model_source',axes_units='inline:axes,coordinates',
        covariance='content:verification_bundle.model_source.covariance',source_coefficients='inline:source_certificate',
        finite_output='private:verified_finite_float64_loading_candidates',
        serialization='private:finite_exact_shape_atomic_wire_validation',warnings='inline:warnings and executed warning capture')
    target.update(status=status,reason=reason,vacuous=vacuous,diagnostic_ids=diagnostic_ids,
        evidence=[dict(kind='executed_A1_gate',location='inline:diagnostics.'+diagnostic_ids[0]
                  if diagnostic_ids else locations.get(name,'inline:A1.failure'),digest=r['trust_sha256'])])
    if vacuous:
        target['evidence'].append(dict(kind='explicit_empty_component_definition',
            location='inline:axes and '+identifier+'.measurement; zero-component comparison',digest=r['trust_sha256']))


def backend_record(a):
    import numpy as np
    import scipy
    config = json.dumps(dict(numpy=np.show_config(mode='dicts'),scipy=scipy.show_config(mode='dicts')),
                        sort_keys=True,allow_nan=False)
    libraries = sorted({str(getattr(module,'__file__','')) for module in list(sys.modules.values())
                        if str(getattr(module,'__file__','')).endswith(('.so','.dylib','.dll','.pyd'))})
    need(bool(libraries) and bool(config),'DEPENDENCY_MISSING',
         'Actual loaded numerical library/build evidence unavailable','missing')
    return dict(python=platform.python_version(),
        **{k:importlib.metadata.version(k) for k in ('numpy','scipy','jax','jaxlib')},
        os=platform.system(),architecture=platform.machine(),blas=config,lapack=config,
        thread_settings={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS','JAX_ENABLE_X64','JAX_PLATFORMS','JAX_DISABLE_JIT') if k in os.environ},
        loaded_libraries=libraries,dtype='float64',intermediate_dtype='float64',
        methods=['original AST incidence','retained upstream certificate rechecks including scipy.linalg.svdvals',
                 'elementwise positive diagonal reciprocal','numpy.matmul'],policy=REVISION)


def fail(r,stage,exc):
    if isinstance(exc,InnovationError):
        code,origin = exc.code,exc.kind
    elif isinstance(exc,(ImportError,importlib.metadata.PackageNotFoundError)):
        code,origin = 'DEPENDENCY_MISSING','missing'
    elif isinstance(exc,ModelError):
        code = 'SOURCE_UNSUPPORTED' if exc.diagnostic.unsupported else 'INVALID_INPUT'
        origin = 'unsupported' if exc.diagnostic.unsupported else 'input'
    elif isinstance(exc,QZError):
        original = exc.code
        numeric = getattr(exc,'numerical',False)
        code = ('DEPENDENCY_MISSING' if 'BACKEND_VERSION' in original else
                'NUMERICAL_NONFINITE' if numeric and ('NONFINITE' in original or 'OUTPUT_INVALID' in original) else
                'UPSTREAM_GATE_UNRESOLVED' if numeric and exc.refused else
                'TRUST_MISSING' if 'MISSING' in original else
                'SOURCE_UNSUPPORTED' if exc.refused else 'SOURCE_MISMATCH')
        origin = ('backend' if code == 'NUMERICAL_NONFINITE' else 'uncertain' if numeric and exc.refused
                  else 'check' if numeric else 'missing' if code in ('DEPENDENCY_MISSING','TRUST_MISSING')
                  else 'unsupported' if exc.refused else 'input')
    else:
        code,origin = ('NUMERICAL_NONFINITE','backend') if isinstance(exc,FloatingPointError) else ('BACKEND_EXCEPTION','backend')
    if stage == 'serialization':
        code,origin = 'SERIALIZATION_FAILED','backend'
    reason = 'P39.A1.'+code
    refused = origin in ('missing','unsupported','uncertain')
    r.update(product_valid=False,claim=None,L_s=None,L_j=None,
        assessment_status='REFUSED' if refused else 'FAIL',failure_reason=reason,failure_reasons=[reason],
        execution_status='COMPLETED' if origin in ('check','uncertain') else 'NOT_RUN' if refused else 'FAILED',
        model_status='NUMERICALLY_UNRESOLVED' if origin in ('backend','check','uncertain') else
                     'UNSUPPORTED' if origin == 'unsupported' else 'NOT_EVALUATED')
    if origin == 'backend' and stage != 'serialization' and r['computation_status'] != 'NOT_RUN':
        r['computation_status'] = 'FAILED'
    check(r,stage,'UNSUPPORTED' if refused else 'FAIL',reason=reason)
    r['diagnostics'].append(dict(id='A1.failure',kind='event',values=dict(stage=stage,reason=reason,
        message=str(exc),exception_type=type(exc).__name__)))


def serialize_result(r,*,expected=None):
    """Verify a detached view of immutable bytes, then publish those same bytes.

    Neither the caller's candidate nor the mutable validation view is an output
    source after verification. Expected identity is separately retained; it is
    never enrolled from the candidate. All failures clear both mutable views.
    """
    candidate = r
    try:
        if not r['product_valid']:
            r.update(L_s=None,L_j=None,claim=None)
        else:
            correctness = [c for c in r['checks'] if c['check_id'] != 'A1.warnings']
            need(all(c['status'] == 'PASS' and c['evidence'] for c in correctness)
                 and r['L_s'] is not None and r['L_j'] is not None
                 and r['claim'] == CLAIM and r['model_status'] == 'NOT_EVALUATED'
                 and r['forcing_certification'] == 'NOT_EVALUATED',
                 'SERIALIZATION_FAILED','Atomic positive-product obligations incomplete','backend')
        need(set(r) == set(result_template()),'SERIALIZATION_FAILED','Exact frozen top-level fields required','backend')
        trusted_wire = canonical_json(expected)
        expected = json.loads(trusted_wire)
        wire = canonical_json(r)
        r = json.loads(wire)
        validate_wire(r,expected=expected)
        # Detect faults in the validator's mutable view as well. The emitted
        # string is already fixed, even if either view changes after this check.
        need(canonical_json(r) == wire and canonical_json(expected) == trusted_wire,
             'SERIALIZATION_FAILED','Publication snapshot changed during verification','backend')
        return InnovationResult(wire)
    except Exception:
        candidate.update(product_valid=False,L_s=None,L_j=None,claim=None)
        r.update(product_valid=False,L_s=None,L_j=None,claim=None)
        raise


def recover_loadings(expected, content):
    """Consume separately enrolled trust and full detached §20.2 content.

    specification_bytes contains exact UTF-8 text; acceptance_records maps exact
    reviewed record byte digests to their UTF-8 text. No defaults, overrides,
    raw-matrix entry point or predecessor reexecution is provided.
    """
    r = result_template(); stage = 'trust'; trusted_expected = None
    try:
        t,c,specs = admit_trust(expected,content)
        # Separate from mutable result provenance and admitted metadata objects.
        trusted_expected = json.loads(canonical_json(t))
        r.update(trust_sha256=digest(t),upstream_result_sha256=t['p38_result_sha256'],
            upstream_stage='P3.8',upstream_source_commit=t['p38_source_commit'],provenance=t,
            specification_sha256=SPEC_SHA256)
        check(r,'trust')
        stage = 'p38'
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            a = recheck_p38(t,c,specs)
        for warning in observed:
            need(issubclass(warning.category,RuntimeWarning) and 'underflow' in str(warning.message).lower(),
                 'BACKEND_WARNING','Numerical warning during P3.8 rechecks: '+str(warning.message),'backend')
            r['warnings'].append(dict(code='FINITE_UNDERFLOW',check_id='A1.p38',message=str(warning.message),
                evidence=[dict(kind='observed_runtime_warning',location='A1.p38_rechecks',digest=None)]))
        r['backend'] = backend_record(a)
        check(r,'p38')
        for name,record in a['p38_rechecks'].items():
            original_id = {'returned_graph':'graph_solve','returned_state':'state_solve'}.get(name,name)
            if original_id in c['p38_result']['diagnostics']:
                r['diagnostics'].append(dict(id='A1.p38.'+name,kind='upstream_recheck',values=dict(
                    stage='P3.8',result_sha256=t['p38_result_sha256'],check_id=original_id,
                    record=c['p38_result']['diagnostics'][original_id],fresh_record=record)))
        stage = 'source'
        check(r,'source')
        stage = 'axes_units'
        r.update(axes=a['axes'],coordinates=a['coordinates'],
                 L_s_shape=[a['n_state'],a['axes']['n_innovation']],
                 L_j_shape=[a['n_jump'],a['axes']['n_innovation']])
        check(r,'axes_units')
        stage = 'covariance'
        covariance = json.loads(a['model'].source_json)['covariance']
        need(covariance['kind'] == 'diagonal','SOURCE_UNSUPPORTED','Only original diagonal covariance','unsupported')
        check(r,'covariance',vacuous=a['axes']['n_innovation'] == 0)
        stage = 'source_coefficients'
        np = a['np']
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            certificate = source_certificate(a,a['axes'],a['coordinates'])
        need(same(certificate,c['source_certificate']),'COEFFICIENT_MISMATCH',
             'Independently source-derived certificate differs from enrolled original content')
        r['source_certificate'] = certificate
        check(r,'source_coefficients',vacuous=a['axes']['n_innovation'] == 0)
        r['input_admission_status'] = 'ADMITTED'
        stage = 'finite_output'; r['computation_status'] = 'FAILED'
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            with np.errstate(over='raise',invalid='raise',divide='raise',under='warn'):
                Ls,Lj = innovation_loading.recover_candidates(a)
        for warning in observed:
            need(issubclass(warning.category,RuntimeWarning) and 'underflow' in str(warning.message).lower(),
                 'BACKEND_WARNING',str(warning.message),'backend')
            r['warnings'].append(dict(code='FINITE_UNDERFLOW',check_id='A1.finite_output',
                message=str(warning.message),evidence=[dict(kind='observed_runtime_warning',location='A1.loading',digest=None)]))
        r['computation_status'] = 'COMPLETED'
        for name,value in (('L_s',Ls),('L_j',Lj)):
            subnormal = (np.abs(value) > 0) & (np.abs(value) < np.finfo(np.float64).tiny)
            if bool(np.any(subnormal)):
                r['warnings'].append(dict(code='FINITE_UNDERFLOW',check_id='A1.finite_output',
                    message='Detected finite subnormal entries in '+name,
                    evidence=[dict(kind='computed_float64_subnormal_mask',location='inline:'+name,digest=None)]))
                r['diagnostics'].append(dict(id='A1.subnormal.'+name,kind='underflow',values=dict(
                    operation='native loading recovery',horizon=None,array=name,
                    indices=np.argwhere(subnormal).tolist(),context='Finite nonzero values below binary64 normal range',
                    observability='Detects returned subnormals; no warning or mask proves absence of intermediate or rounded-to-zero underflow')))
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for name,passed,record,code in innovation_loading.verify_loadings(a,certificate,Ls,Lj):
                if name == 'pathwise_raw':
                    r['diagnostics'].append(dict(id='A1.original_raw_terms',kind='event',values=record))
                    continue
                stage = name
                kind = 'conditional_forecast_error' if name == 'conditional_semantics' else 'component'
                check(r,name,'PASS' if passed else 'FAIL',diagnostic=record,kind=kind,
                      vacuous=record.get('vacuous',False))
                need(passed,code or 'CONDITIONAL_SEMANTICS_INVALID','Required A1 verification rejected: '+name,'check')
        stage = 'finite_output'
        real_matrix(np,Ls,tuple(r['L_s_shape']),computed=True);real_matrix(np,Lj,tuple(r['L_j_shape']),computed=True)
        check(r,'finite_output',vacuous=a['axes']['n_innovation'] == 0)
        for message in c['p38_result']['warnings']:
            if message in ('X_conditioning at or above 1e8 and below 1e10',
                           'stable_block_conditioning at or above 1e8 and below 1e10'):
                r['warnings'].append(dict(code='INHERITED_ACCEPTED_WARNING',check_id='A1.p38',message=message,
                    evidence=[dict(kind='accepted_P38_conditioning_warning',location='content.p38_result.warnings',digest=t['p38_result_sha256'])]))
            else:
                r['diagnostics'].append(dict(id='A1.retained_upstream_note.'+str(len(r['diagnostics'])),kind='event',
                    values=dict(stage='P3.8',reason='RETAINED_UPSTREAM_SCOPE_NOTE',message=message,exception_type=None)))
        for name in ('X_conditioning','stable_block_conditioning'):
            message=name+' at or above 1e8 and below 1e10'
            if a['p38_rechecks'][name]['status']=='WARN' and not any(w['message']==message for w in r['warnings']):
                r['warnings'].append(dict(code='INHERITED_ACCEPTED_WARNING',check_id='A1.p38',message=message,
                    evidence=[dict(kind='rechecked_upstream_conditioning_warning',location='inline:A1.p38.'+name,digest=t['p38_result_sha256'])]))
        stage = 'serialization'
        # Inspect the complete detached candidate before the publication gate.
        canonical_json(dict(L_s=Ls.tolist(),L_j=Lj.tolist(),source_certificate=certificate,
                            axes=r['axes'],coordinates=r['coordinates'],backend=r['backend']))
        check(r,'serialization')
        check(r,'warnings','WARN' if r['warnings'] else 'PASS')
        r.update(L_s=Ls.tolist(),L_j=Lj.tolist(),product_valid=True,claim=CLAIM,
                 execution_status='COMPLETED',assessment_status='PASS_WITH_WARNINGS' if r['warnings'] else 'PASS')
        return serialize_result(r,expected=trusted_expected)
    except Exception as exc:
        fail(r,stage,exc)
        # Failure diagnostics themselves must stay finite and cannot leak arrays.
        try:
            return serialize_result(r,expected=trusted_expected)
        except Exception as serialization_error:
            # Last-resort failure receipt never preserves malformed products.
            safe = result_template()
            reasons=list(dict.fromkeys(r['failure_reasons']+['P39.A1.SERIALIZATION_FAILED']))
            safe.update(execution_status='FAILED',computation_status=r['computation_status'],
                assessment_status='FAIL',failure_reason=(reasons[0] if r['assessment_status']=='FAIL'
                    else 'P39.A1.SERIALIZATION_FAILED'),failure_reasons=reasons)
            check(safe,'serialization','FAIL',reason=safe['failure_reason'])
            safe['diagnostics'].append(dict(id='A1.serialization.failure',kind='event',values=dict(
                stage='serialization',reason=safe['failure_reason'],
                message=str(serialization_error),exception_type=type(serialization_error).__name__)))
            return InnovationResult(json.dumps(safe,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False))
