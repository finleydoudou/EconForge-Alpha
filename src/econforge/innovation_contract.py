"""Frozen P3.9A1 contracts. No stochastic-law or response capability."""
import hashlib
import math

from econforge.gh_admission import SPEC_IDENTITIES as HISTORICAL_SPECS
from econforge.p35_admission import canonical_json

REVISION = 'p39-stochastic-irf-1'
SPEC_SHA256 = '31db27620b2266c11490641ccd96732d9af135b42c4222a0dff3edbe8593ee12'
CLAIM = 'SOURCE_VERIFIED_NATIVE_INNOVATION_LOADINGS'
CAPABILITY = 'P3.9A1_SOURCE_LOADING'
P38_SOURCE = 'ed301ae2beb2f69dd82c742a84001c3e2d5af7af'
P38_ORIGINAL = '06d7e90222f9d2ab979f50b8e72d7896ead64cb0'
# Reviewed, retained ACCEPT content for both exact sources; never candidate supplied.
ACCEPTANCE_SHA256 = '620a271dbf3e261a3fb83dd20c3fb145f5c4b40c45f37e14b11ace9bc12433fa'
SPECIFICATIONS = {k:dict(revision=v,sha256=s) for k,v,s in (
    ('legacy','v0.1-math-freeze-1',HISTORICAL_SPECS['legacy']),
    ('p36','p36-qz-1',HISTORICAL_SPECS['p36']),
    ('p37','p37-bk-1',HISTORICAL_SPECS['p37']),
    ('p38','p38-gh-1',HISTORICAL_SPECS['p38']),('p39',REVISION,SPEC_SHA256))}
CHECK_IDS = ['A1.'+x for x in ('trust','p38','source','axes_units','covariance',
    'source_coefficients','physical_zero','exogenous_loading','jump_loading',
    'pathwise_coefficients','conditional_semantics','finite_output','serialization','warnings')]
TRUST_FIELDS = {'schema','revision','stage','p38_source_commit','p38_original_source_commit',
    'p38_result_sha256','p38_context_sha256','verification_bundle_sha256',
    'p34_result_sha256','p35_result_sha256','p36_result_sha256','p37_result_sha256',
    'model_id','branch_id','frequency','original_model_sha256','canonical_model_sha256',
    'reference_sha256','calibration_sha256','process_source_sha256','innovation_metadata_sha256',
    'covariance_source_sha256','source_certificate_sha256','coordinate_sha256','axes_sha256',
    'assumptions_sha256','specifications','upstream','acceptance_records','enrollment'}
CONTENT_FIELDS = {'schema','revision','p38_result','p38_context','verification_bundle',
    'p36_result','p37_result','upstream_result','source_certificate','assumptions',
    'acceptance_records','specification_bytes'}
ASSUMPTIONS = dict(revision=1,information='COMMON_POST_SURPRISE',
    innovation='MARTINGALE_DIFFERENCE_FINITE_CONSTANT_COVARIANCE',
    solution_class='LOCAL_FIRST_ORDER_CAUSAL_UNIFORMLY_L2_BOUNDED',
    initial_states='ARBITRARY_ADMISSIBLE_FINITE_NEARBY')


class InnovationError(Exception):
    """Origin distinguishes absent/unsupported/input/numerical/check rejection."""
    def __init__(self, code, message, kind='input'):
        super().__init__(message)
        self.code, self.kind = code, kind


def need(condition, code, message, kind='input'):
    if not condition:
        raise InnovationError(code,message,kind)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def real_matrix(np, value, shape, *, computed=False):
    kind = 'backend' if computed else 'input'
    code = 'NUMERICAL_NONFINITE' if computed else 'INVALID_INPUT'
    if computed:
        need(type(value) is np.ndarray and value.dtype == np.dtype('float64')
             and value.shape == shape and bool(np.isfinite(value).all()),code,
             'Computed matrix must have finite float64 values and exact shape',kind)
        return value
    need(type(value) is list and len(value) == shape[0]
         and all(type(row) is list and len(row) == shape[1]
                 and all(type(x) is float and math.isfinite(x) for x in row) for row in value),
         code,'Finite JSON floats and explicit matrix axes required',kind)
    return np.array(value,dtype=np.float64).reshape(shape)


def component(np, left, right, rows, columns):
    """Every frozen two-term component, including evidenced empty dimensions."""
    shape = (len(rows),len(columns))
    real_matrix(np,left,shape,computed=True); real_matrix(np,right,shape,computed=True)
    errors = np.abs(left-right)
    bounds = 1e-10+1e-8*(np.abs(left)+np.abs(right))
    need(bool(np.isfinite(errors).all()) and bool(np.isfinite(bounds).all()),
         'NUMERICAL_NONFINITE','Unrepresentable residual/bound','backend')
    ratios = errors/bounds
    worst = list(map(int,np.unravel_index(int(np.argmax(ratios)),shape))) if ratios.size else None
    return bool(np.all(errors <= bounds)),dict(shape=list(shape),row_labels=list(rows),
        column_labels=list(columns),left=left.tolist(),right=right.tolist(),
        absolute_errors=errors.tolist(),allowed_bounds=bounds.tolist(),
        max_error_over_bound=float(np.max(ratios)) if ratios.size else None,
        worst_index=worst,atol=1e-10,rtol=1e-8,rule='two_term',vacuous=not bool(errors.size))
