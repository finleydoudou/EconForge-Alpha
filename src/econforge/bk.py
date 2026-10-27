"""Narrow P3.7 assessment of trusted, accepted P3.6 evidence."""
from dataclasses import dataclass
import json

from econforge.bk_admission import admit
from econforge.bk_numerics import BKError
from econforge import bk_numerics as numeric
from econforge.qz_admission import QZError


@dataclass(frozen=True)
class BKResult:
    """Immutable finite JSON; to_dict returns an independent copy."""
    wire_json: str

    def to_dict(self):
        return json.loads(self.wire_json)


def assess_p36(p36_result, expected, *, math_spec_bytes, upstream_math_spec_bytes):
    """Assess numerical BK conditions; never recover a graph or solution.

    expected contains the independently retained P3.6 wire digest/bindings and
    both versioned specification identities. Hashes are integrity, not signatures.
    """
    r = dict(schema_version=2, claim='P3.7_numerical_BK_conditions_only',
             assessment_status='FAIL', execution_status='NOT_RUN',
             input_admission_status='NOT_ADMITTED', ordering_execution_status='NOT_RUN',
             ordered_qz_valid=False, classification_valid=False, bk_assessment='NOT_EVALUATED',
             model_status='NOT_EVALUATED', capability_maturity='IMPLEMENTED',
             independent_validation_status='NOT_RUN', backend='scipy.linalg.ordqz',
             backend_order='STABLE_FIRST', dtype='complex128', pencil='B - lambda A',
             stable_count=None, outside_count=None, boundary_count=None, unresolved_pair_count=None,
             exact_infinite_count=None, upstream_numerical_infinite_count=None,
             n_state=None, n_jump=None, regularity={'status': 'NOT_RUN'},
             projection={'status': 'NOT_RUN'}, unordered_classes=None, ordered_result=None,
             ordered_diagnostics=None, upstream_result=None, metadata={}, checks=[], warnings=[],
             limitations=['Numerical sufficient regularity admission, not symbolic proof',
                          'Dimensional surplus/deficit are narrow linear-system claims',
                          'Relative projection does not certify graph conditioning or policy accuracy',
                          'Upstream forcing stability remains NOT_EVALUATED',
                          'No global uniqueness, empirical validity, policy, transition or IRF claim',
                          'External scientific validation and P3.8 recovery NOT_RUN'], failure_reason=None)
    stage = 'admission'
    try:
        upstream, context, np, linalg, A, B, state_rows, ns, nj, tau, versions = admit(
            p36_result, expected, math_spec_bytes, upstream_math_spec_bytes)
        r.update(input_admission_status='ADMITTED', upstream_result=upstream, n_state=ns, n_jump=nj)
        r['metadata'] = dict(bindings=context, specification_revision='p37-bk-1',
                             backend_versions=versions, tau_qz=tau, root_band=numeric.ROOT_BAND,
                             normalized_pair_zero=numeric.PAIR_ZERO, root_match_atol=numeric.ROOT_MATCH,
                             state_row_indices=state_rows, input_dtype='float64',
                             forcing_stability=upstream['metadata']['input_metadata']['structure'].get('forcing_stability', 'NOT_EVALUATED'),
                             external_validation='NOT_RUN', EF_REF_1='NOT_RUN', Dynare='NOT_RUN')
        evidence = [{'kind': 'P36_wire_sha256', 'digest': context['p36_result_sha256'], 'location': 'inline:P36'}]
        r['checks'].append(dict(check_id='input.admitted', status='PASS', evidence=evidence))
        stage = 'special_pairs'
        pairs = upstream['pairs']
        if any(p['pair_status'] == 'INDETERMINATE_0_OVER_0' for p in pairs):
            raise BKError('INDETERMINATE_0_OVER_0', 'Upstream special pair excludes economic counting',
                          refused=True, bk_status='REFUSED_SPECIAL_PAIR')
        stage = 'regularity'
        r['regularity'] = numeric.regularity(np, linalg, A, B)
        if r['regularity']['status'] != 'PASS':
            raise BKError('REGULARITY_UNRESOLVED', 'No deterministic witness passed', refused=True,
                          bk_status='DETERMINACY_UNRESOLVED_REGULARITY')
        r['checks'].append(dict(check_id='regularity.admitted', status='PASS', evidence=evidence))
        stage = 'classification'
        alpha = np.array([complex(*p['alpha']) for p in pairs], dtype=np.complex128)
        beta = np.array([complex(*p['beta']) for p in pairs], dtype=np.complex128)
        classes, small = numeric.classify_pairs(np, A, B, alpha, beta)
        r.update(unordered_classes=classes, boundary_count=classes.count('BOUNDARY'),
                 unresolved_pair_count=classes.count('UNRESOLVED_PAIR'),
                 exact_infinite_count=int(np.count_nonzero(beta == 0)),
                 upstream_numerical_infinite_count=sum(p['pair_status'] == 'INFINITE' for p in pairs))
        r['metadata']['small_nonzero_beta_indices'] = small
        if 'UNRESOLVED_PAIR' in classes:
            raise BKError('NORMALIZED_PAIR_UNRESOLVED', 'Jointly small pair', refused=True, bk_status='REFUSED_SPECIAL_PAIR')
        if 'BOUNDARY' in classes:
            raise BKError('UNIT_BOUNDARY', 'Closed unit-circle band is unresolved', refused=True,
                          bk_status='DETERMINACY_UNRESOLVED_BOUNDARY')
        original_pairs = list(zip(alpha, beta))
        r['metadata']['unordered_conjugacy'] = numeric.conjugacy(original_pairs, classes)
        stable = classes.count('STABLE')
        r.update(stable_count=stable, outside_count=len(classes)-stable, classification_valid=True)
        stage = 'ordering'
        r['ordering_execution_status'] = 'FAILED'
        output = numeric.order(np, linalg, A, B)
        r['ordering_execution_status'] = 'COMPLETED'
        stage = 'ordered_verification'
        d = numeric.verify_order(np, A, B, output, original_pairs, classes, tau)
        r.update(ordered_diagnostics=d, ordered_qz_valid=True)
        for key, value in d['structural'].items():
            r['checks'].append(dict(check_id='ordered.'+key, status='PASS', value=value, threshold=tau, evidence=evidence))
        stage = 'projection'
        r['projection'] = numeric.projection(np, linalg, output[5], state_rows, stable)
        if stable == ns:
            if r['projection']['status'] != 'PASS':
                raise BKError('PROJECTION_UNRESOLVED', 'State projection margin does not pass', refused=True,
                              bk_status='DETERMINACY_UNRESOLVED_PROJECTION')
            numeric.require(len(classes)-stable == nj, 'COUNT_INCONSISTENT', 'Complementary jump count differs')
            result = 'BK_DETERMINATE'
        elif stable > ns:
            numeric.require(len(classes)-stable < nj, 'COUNT_INCONSISTENT', 'Complementary count differs')
            result = 'BK_INDETERMINATE'
        else:
            numeric.require(len(classes)-stable > nj, 'COUNT_INCONSISTENT', 'Complementary count differs')
            result = 'BK_NO_STABLE_SOLUTION'
        ordered = dict(source_p36_result_sha256=context['p36_result_sha256'],
                       ordered_classes=d['classes'], stable_block_size=stable)
        for key, value in zip(('ordered_schur_B', 'ordered_schur_A', 'ordered_alpha',
                               'ordered_beta', 'ordered_Q', 'ordered_Z'), output):
            ordered[key] = np.stack((value.real, value.imag), axis=-1).tolist()
        if r['upstream_numerical_infinite_count'] or small:
            r['warnings'].append('Numerical INFINITE labels and small nonzero beta were independently reclassified')
        r.update(ordered_result=ordered, bk_assessment=result, execution_status='COMPLETED',
                 assessment_status='PASS_WITH_WARNINGS' if r['warnings'] else 'PASS')
    except Exception as exc:
        refused = isinstance(exc, QZError) and exc.refused
        code = exc.code if isinstance(exc, QZError) else ('DEPENDENCY_MISSING' if isinstance(exc, ImportError)
                                                        else {'ordering': 'ORDERED_BACKEND_EXCEPTION',
                                                              'admission': 'INVALID_METADATA'}.get(stage, 'BK_NUMERICAL_EXCEPTION'))
        refused = refused or isinstance(exc, ImportError)
        bk = getattr(exc, 'bk_status', 'NOT_EVALUATED')
        r.update(assessment_status='REFUSED' if refused else 'FAIL', failure_reason=code,
                 execution_status='COMPLETED' if bk != 'NOT_EVALUATED' else ('NOT_RUN' if refused else 'FAILED'),
                 bk_assessment=bk, ordered_result=None)
        if bk != 'NOT_EVALUATED':
            r['model_status'] = 'NUMERICALLY_UNRESOLVED'
        elif refused:
            r['model_status'] = 'UNSUPPORTED'
        if stage in ('ordering', 'ordered_verification'):
            r.update(classification_valid=False, ordered_qz_valid=False, stable_count=None, outside_count=None)
        if hasattr(exc, 'diagnostics'):
            r['ordered_diagnostics'] = exc.diagnostics
        r['checks'].append(dict(check_id=stage, status='UNSUPPORTED' if refused else 'FAIL',
                                reason=code, message=str(exc), evidence=[]))
    return BKResult(json.dumps(r, allow_nan=False, separators=(',', ':')))
