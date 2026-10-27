"""Frozen P3.9A2 contract and complete §20.5/20.9 check registry."""
import math
from econforge.innovation_contract import (
    InnovationError, need, real_matrix, SPEC_SHA256, REVISION, SPECIFICATIONS,
    TRUST_FIELDS, CONTENT_FIELDS, ASSUMPTIONS, P38_SOURCE, P38_ORIGINAL,
    ACCEPTANCE_SHA256)
from econforge.p35_admission import canonical_json,digest

CLAIM='CERTIFIED_FIRST_ORDER_CAUSAL_L2_STOCHASTIC_LAW'
CAPABILITY='P3.9A2_STOCHASTIC_LAW'
A1_SOURCE='933fe322a7e68866d98531971fdc64971e168005'
A1_ACCEPTANCE_SHA256='5796a1f1e71771e8d70b652667c6064555ec8092e9231c3aba518fcbb63a67ff'
SOLUTION_CLASS=ASSUMPTIONS['solution_class']
DELTA=1e-6
EQUATIONS=['q_s,t+1=H q_s,t+L_s u_t+1','q_j,t=G q_s,t',
           'q_j,t+1=G H q_s,t+L_j u_t+1']
OBLIGATIONS=['A2.A0.model', 'A2.A0.reference', 'A2.A0.derivatives', 'A2.A0.assumptions', 'A2.A0.covariance', 'A2.A1.lineage', 'A2.A1.pencil', 'A2.A1.regularity', 'A2.A1.factors', 'A2.A1.roots', 'A2.A1.algebraic', 'A2.A2.dimensions', 'A2.A2.projection', 'A2.A2.free_directions', 'A2.A2.causality', 'A2.A3.p38', 'A2.A3.recovery', 'A2.A3.realness', 'A2.A3.policy', 'A2.A3.identities', 'A2.A4.source_process', 'A2.A4.reference_process', 'A2.A4.mean_zz', 'A2.A4.mean_zk', 'A2.A4.source_coefficients', 'A2.A5.physical_timing', 'A2.A5.exogenous_loading', 'A2.A5.jump_loading', 'A2.A5.axes_units', 'A2.A5.a1', 'A2.A6.mean_rows', 'A2.A6.pathwise_rows', 'A2.A6.conditional_rows', 'A2.A7.forcing', 'A2.A7.covariance', 'A2.A7.state_stability', 'A2.A7.bounded_class', 'A2.A7.stationarity_scope', 'A2.A8.trust', 'A2.A8.specifications', 'A2.A8.checks', 'A2.A8.atomicity', 'A2.A8.claims']
FORCING_CHECKS=['A2.schur.quality','A2.schur.roots','A2.lyapunov.transform',
 'A2.lyapunov.hermitian','A2.lyapunov.eigen_quality','A2.lyapunov.positive_P',
 'A2.lyapunov.positive_D','A2.lyapunov.cancellation','A2.lyapunov.equation']
CHECK_IDS=OBLIGATIONS+FORCING_CHECKS+['A2.serialization','A2.warnings']
LAW_FIELDS={'G','H','L_s','L_j','shapes','R_q','covariance','equations',
            'source_certificate_sha256','forcing_evidence_ids'}


def finite(np,value,shape,*,complex_required=False):
    need(type(value) is np.ndarray and value.shape==shape
         and value.dtype==np.dtype('complex128' if complex_required else 'float64')
         and bool(np.isfinite(value).all()),'NUMERICAL_NONFINITE',
         'Finite exact computed dtype/shape required','backend')
    return value


def scalar(value):
    value=float(value)
    need(math.isfinite(value),'NUMERICAL_NONFINITE','Unrepresentable numerical diagnostic','backend')
    return value


def strict_margin(value,tau):
    return math.isfinite(value) and value>tau


def quality_bound(value,tau):
    return math.isfinite(value) and value<=tau
