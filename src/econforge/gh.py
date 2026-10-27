"""Public P3.8 API: joint verified first-order conditional-mean G/H only."""
from dataclasses import dataclass
import json
import warnings

from econforge.gh_admission import admit, GHError
from econforge import gh_numerics as numeric
from econforge import gh_source as source
from econforge.model_errors import ModelError
from econforge.qz_admission import QZError


@dataclass(frozen=True)
class GHResult:
    """Immutable finite JSON snapshot; every to_dict call returns detached data."""
    wire_json: str

    def to_dict(self):
        return json.loads(self.wire_json)


def _finalize(record):
    # The joint product has exactly one serialization boundary.
    if not record['recovery_valid']:
        record['G'] = record['H'] = None
    elif record['G'] is None or record['H'] is None:
        record.update(recovery_valid=False,G=None,H=None,assessment_status='FAIL',
                      failure_reason='ATOMIC_PRODUCT_INVALID',execution_status='FAILED')
    return GHResult(json.dumps(record,allow_nan=False,separators=(',',':')))


def recover_p37(p37_result, expected, verification_bundle, *, math_spec_bytes=None,
                p37_math_spec_bytes=None, p36_math_spec_bytes=None, legacy_math_spec_bytes=None):
    """Consume retained P3.7 + independent trust context + original source bundle.

    No raw-matrix entry point, upstream rerun, stochastic loading or IRF exists.
    See docs/P3_8_API.md for exact version-1 context and bundle wire fields.
    """
    r = dict(schema_version=2,claim='FIRST_ORDER_CONDITIONAL_MEAN_GH_VERIFIED',
             specification_revision='p38-gh-1',assessment_status='FAIL',execution_status='NOT_RUN',
             input_admission_status='NOT_ADMITTED',recovery_execution_status='NOT_RUN',
             recovery_valid=False,G=None,H=None,model_status='NOT_EVALUATED',
             capability_maturity='IMPLEMENTED',independent_validation_status='NOT_RUN',
             backend='scipy.linalg.solve_triangular + scipy.linalg.solve',dtype='float64',
             intermediate_dtype='complex128',upstream_result=None,metadata={},checks=[],diagnostics={},
             warnings=[],failure_reason=None,
             limitations=['First-order local conditional mean only',
                          'Original raw-row check shares the admitted model/AD lineage',
                          'Conservative conditioning admission is not a universal forward-error guarantee',
                          'No forcing-stability or full stochastic equilibrium certification',
                          'Nonlinear observables retained as metadata; no reporting-map recovery',
                          'No innovation loadings, IRFs, simulation, estimation or higher order',
                          'Independent acceptance, external validation and release validation NOT_RUN'])
    stage = 'input_admission'
    d = r['diagnostics']
    specifications = dict(p38=math_spec_bytes,p37=p37_math_spec_bytes,
                          p36=p36_math_spec_bytes,legacy=legacy_math_spec_bytes)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            a = admit(p37_result,expected,verification_bundle,specifications)
            np = a['np']
            evidence = [dict(kind='trusted_P37_wire',digest=a['expected']['p37_result_sha256'],location='inline:P37'),
                        dict(kind='original_verification_bundle',digest=a['expected']['verification_bundle_sha256'],location='inline:bundle')]
            d['certificate_rechecks'] = a['rechecks']
            # Retain admitted upstream evidence even when a later source check refuses.
            r['upstream_result'] = a['wire']
            r['metadata']['bindings'] = a['expected']
            stage = 'source_admission'
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                prepared = source.prepare(a,d)
                r.update(input_admission_status='ADMITTED',upstream_result=a['wire'])
                m = a['metadata']
                r['metadata'] = dict(bindings=a['expected'],variable_order=m['variable_order'],
                    equation_order=m['equation_order'],state_variable_order=m['state_variable_order'],
                    control_variable_order=m['control_variable_order'],partitions=m['partitions'],
                    G_shape=[a['n_jump'],a['n_state']],H_shape=[a['n_state'],a['n_state']],
                    permutations=m['permutations'],S=m['S'],T=m['T'],
                    coordinate_convention=m['coordinate_convention'],coordinate_meaning='native q; stored displacement dx=T dq',
                    backend_versions=a['versions'],forcing_stability='NOT_EVALUATED',
                    external_validation='NOT_RUN',Dynare='NOT_RUN',EF_REF_1='NOT_RUN',release_validated=False)
                r['checks'].append(dict(check_id='input_and_original_source.admitted',status='PASS',evidence=evidence))
                stage = 'recovery'
                G,H = numeric.recover_complex(a,d)
                stage = 'original_complex'
                source.original_equations(a,G,H,d,'original_complex')
                stage = 'realness'
                G,H = numeric.real_candidates(a,G,H,d)
                numeric.finite(np,G,(a['n_jump'],a['n_state']),'returned real G')
                numeric.finite(np,H,(a['n_state'],a['n_state']),'returned real H')
                stage = 'final_policy'
                numeric.policy_check(a,G,H,d)
                stage = 'original_real'
                source.original_equations(a,G,H,d,'original_real')
                stage = 'identity_process'
                source.identities_and_process(a,G,H,prepared,d)
                r['warnings'] = ['Retained upstream: '+x for x in a['wire']['warnings']]
                for key in ('X_conditioning','stable_block_conditioning'):
                    if d[key]['status'] == 'WARN':
                        r['warnings'].append(key+' at or above 1e8 and below 1e10')
                # Publish both real matrices only after every required check.
                r.update(G=G.tolist(),H=H.tolist(),recovery_valid=True,
                         recovery_execution_status='COMPLETED',execution_status='COMPLETED',
                         assessment_status='PASS_WITH_WARNINGS' if r['warnings'] else 'PASS')
    except Exception as exc:
        model_error = isinstance(exc,ModelError)
        refused = ((isinstance(exc,QZError) and exc.refused)
                   or (model_error and exc.diagnostic.unsupported) or isinstance(exc,ImportError))
        numerical = getattr(exc,'numerical',False)
        code = exc.code if isinstance(exc,QZError) else exc.diagnostic.code if model_error else (
            'DEPENDENCY_MISSING' if isinstance(exc,ImportError) else
            'NUMERICAL_BACKEND_WARNING' if isinstance(exc,Warning) else
            'INVALID_INPUT' if stage in ('input_admission','source_admission') else 'NUMERICAL_BACKEND_EXCEPTION')
        # Invalid backend output fails before solve completion. Later executed
        # verification checks may reject/refuse a completed computation.
        solve_status = d.get('solve_execution_status','NOT_RUN')
        r.update(assessment_status='REFUSED' if refused else 'FAIL',failure_reason=code,
                 execution_status=('FAILED' if solve_status == 'FAILED'
                                   else 'COMPLETED' if isinstance(exc,GHError)
                                   and (numerical or (refused and solve_status == 'COMPLETED'))
                                   else 'NOT_RUN' if refused else 'FAILED'),
                 recovery_execution_status=solve_status)
        if numerical or stage not in ('input_admission','source_admission'):
            r['model_status'] = 'NUMERICALLY_UNRESOLVED'
        elif refused and code not in ('VERIFICATION_BUNDLE_MISSING','ORIGINAL_CONTENT_MISSING'):
            r['model_status'] = 'UNSUPPORTED'
        r['checks'].append(dict(check_id=stage,status='UNSUPPORTED' if refused else 'FAIL',
                                reason=code,message=str(exc),evidence=[]))
    # Preserve reached diagnostics and clearly mark every unreached obligation.
    required = ('X_conditioning','stable_block_conditioning','stable_solve','graph_solve','state_solve',
                'original_complex','G_realness','H_realness','final_policy','original_real','process_zz','process_zk')
    for name in required:
        value = d.get(name)
        status = ('NOT_RUN' if value is None else value.get('status',
                  'PASS' if value.get('passed') else 'FAIL'))
        if status == 'REFUSED':status = 'UNSUPPORTED'
        r['checks'].append(dict(check_id=name,status=status,
            evidence=[] if value is None else [dict(kind='computed_P38_check',location='inline:diagnostics.'+name)]))
    return _finalize(r)
