"""P3.5 assembly ONLY from explicitly admitted, accepted P3.4 wire evidence."""
from dataclasses import asdict, dataclass
import json
import math

from econforge.model_errors import ModelError
from econforge.p35_admission import validate_admission, canonical_json, partitions, COORDINATE_KEYS
from econforge.p35_structure import validate_structure
from econforge.residuals import evaluate_residuals

PERMUTATIONS=('source_to_canonical_variables','canonical_to_source_variables',
              'source_to_canonical_equations','canonical_to_source_equations')
BLOCK_SCOPE='all original verification rows including identities; derivative rows are equation_order'


@dataclass(frozen=True)
class MatrixAssemblyResult:
    assessment_status: str
    execution_status: str
    assembly_valid: bool
    A: tuple | None
    B: tuple | None
    input_admission_status: str
    checks: tuple
    diagnostics: tuple
    metadata: dict
    schema_version: int = 2
    claim: str = 'canonical_matrix_assembly_from_explicitly_admitted_P34_evidence'
    model_status: str = 'NOT_EVALUATED'
    independent_validation_status: str = 'NOT_RUN'
    benchmark_status: str = 'NOT_RUN'
    capability_maturity: str = 'IMPLEMENTED'

    def to_dict(self):
        record=asdict(self)
        record['failure_reason']=self.diagnostics[0]['code'] if self.diagnostics else None
        return json.loads(json.dumps(record,allow_nan=False))


def _matrix(value,rows,cols,label):
    if type(value) is not list or len(value)!=rows or any(type(row) is not list or len(row)!=cols for row in value):
        raise ModelError('P34_SHAPE_MISMATCH',label)
    if any(type(x) is not float or not math.isfinite(x) for row in value for x in row):
        raise ModelError('P34_NONFINITE_OR_DTYPE',label+' requires finite binary64 values')
    return value


def _verify_p34(model,r,structure):
    """Validate supplied numerical evidence without an AD/FD/backend import."""
    n=len(model.variable_order); meta=r['metadata']
    expected={'assessment_status':'PASS','execution_status':'COMPLETED','ad_execution_status':'COMPLETED',
              'derivative_valid':True,'input_admission_status':'ADMITTED','backend':'jax.jacfwd',
              'dtype':'float64','shape':[n,3*n],'diagnostics':[],'failure_reason':None}
    for key,value in expected.items():
        if key not in r or canonical_json(r[key])!=canonical_json(value):
            raise ModelError('P34_NOT_ACCEPTED',f'Unaccepted/malformed P3.4 field {key}')
    if type(r.get('checks')) is not list or any(type(c) is not dict for c in r['checks']) or [c.get('check_id') for c in r['checks']]!=[
            'reference.verified','backend.admitted','dynamic_derivative.point_checks'] or any(c.get('status')!='PASS' for c in r['checks']):
        raise ModelError('P34_NOT_ACCEPTED','All three P3.4 checks must explicitly PASS')
    for check in r['checks']:
        evidence=check.get('evidence')
        if type(evidence) is not list or not evidence or any(type(e) is not dict or e.get('digest')!=meta['evaluation_hash'] for e in evidence):
            raise ModelError('P34_CHECK_PROVENANCE_MISMATCH','P3.4 checks must bind the verified reference')
    scalar=evaluate_residuals(model,r['reference'])
    if not scalar.evaluation_valid or not scalar.residuals_within_tolerance:
        raise ModelError('STEADY_REFERENCE_REQUIRED','Fresh original residual and reporting identity checks must pass')
    sm=json.loads(json.dumps(dict(scalar.metadata)))
    for key in ('specification_revision','tolerance_profile','ss_resid_atol','ss_resid_rtol',
                'model_hash','evaluation_hash','model_id','branch_id','frequency','parameters',
                'coordinate_metadata','row_scales','verification_equation_order',*PERMUTATIONS):
        if canonical_json(meta.get(key))!=canonical_json(sm[key]):
            raise ModelError('P34_METADATA_MISMATCH',f'Scalar/model metadata differs: {key}')
    if canonical_json(r.get('residual_vector'))!=canonical_json(list(scalar.residual_vector)):
        raise ModelError('P34_REFERENCE_RESIDUAL_MISMATCH','Saved scalar residual differs from fresh verification')
    rows=[e for e in model.equations if e.kind!='identity']
    S=[e.scale for e in rows]
    T=[x if v.coordinate=='log_deviation' else v.scale if v.coordinate=='scaled_level' else 1.0
       for v,x in zip(model.variables,r['reference'])]
    expected_meta={'row_scales_S':S,'coordinate_diagonal_T':T,
        'raw_definition':'partial original signed integrand / partial stored w(offset)',
        'row_normalized_definition':'S^-1 J_slot','coordinate_normalized_definition':'S^-1 J_slot T',
        'row_kinds':[e.kind for e in rows],'original_slots':structure['original_slots'],
        'date_offsets':[-1,0,1],'innovations':'fixed zero; not differentiated',
        'interpretation':'untransformed original integrand; no timing reduction or nonlinear expectation',
        'actual_platform':'cpu','jit_disabled':True,'scalar_backend':'python_scalar_binary64',
        'backend_versions':[['jax','0.6.2'],['jaxlib','0.6.2'],['numpy','2.2.6'],['scipy','1.15.3']]}
    for key,value in expected_meta.items():
        if canonical_json(meta.get(key))!=canonical_json(value):
            raise ModelError('P34_COORDINATE_OR_STRUCTURE_MISMATCH',f'Inconsistent metadata {key}')
    if canonical_json(r.get('slot_order'))!=canonical_json([[offset,name] for offset in (-1,0,1) for name in model.variable_order]):
        raise ModelError('P34_SLOT_ORDER_MISMATCH','Require exact -1/0/+1 column blocks')
    blocks=r.get('blocks')
    if type(blocks) is not list or len(blocks)!=3 or any(type(b) is not dict for b in blocks):
        raise ModelError('P34_BLOCKS_MISSING','Require J_minus, J_current and J_plus')
    flat=_matrix(r.get('flattened_jacobian'),n,3*n,'flattened_jacobian')
    for block,name,offset in zip(blocks,('J_minus','J_current','J_plus'),(-1,0,1)):
        if block.get('name')!=name or type(block.get('offset')) is not int or block['offset']!=offset:
            raise ModelError('P34_BLOCK_IDENTITY_MISMATCH','Missing/swapped date block')
        expected_block={'shape':[n,n],'equation_order':list(model.equation_order),
                        'variable_order':list(model.variable_order),'row_permutation_scope':BLOCK_SCOPE,
                        **{k:list(getattr(model,k)) for k in PERMUTATIONS}}
        for key,value in expected_block.items():
            if canonical_json(block.get(key))!=canonical_json(value):
                raise ModelError('P34_BLOCK_AXES_MISMATCH',f'{name}.{key}')
        raw=_matrix(block.get('jacobian'),n,n,name)
        scaled=_matrix(block.get('row_normalized_jacobian'),n,n,name+' row normalized')
        chart=_matrix(block.get('coordinate_normalized_jacobian'),n,n,name+' coordinate normalized')
        for i in range(n):
            for j in range(n):
                if not math.isfinite(raw[i][j]/S[i]*T[j]):
                    raise ModelError('ASSEMBLY_NONFINITE','S/T transformation overflows')
                # Exact internal consistency with accepted P3.4 operation order.
                if scaled[i][j]!=raw[i][j]/S[i] or chart[i][j]!=scaled[i][j]*T[j]:
                    raise ModelError('P34_NORMALIZATION_MISMATCH',f'{name}[{i},{j}]')
                if flat[i][(offset+1)*n+j]!=raw[i][j]:
                    raise ModelError('P34_FLATTENED_MISMATCH',f'{name}[{i},{j}]')
                if offset==-1 and raw[i][j]!=0:
                    raise ModelError('P34_NONZERO_LAG','No raw lag block may be discarded')
    return S,T,scalar


def assemble_matrices(model,p34_result,manifest=None,*,math_spec_bytes):
    """A E_t q(+1)=B q; no spectral, derivative or solution capability.

    Input P3.4 results must be plain JSON (serialize `to_dict()` first). All
    refusal paths suppress both matrices. Admission is a research assertion.
    """
    checks=[];metadata={};admission='NOT_ADMITTED'
    try:
        # Validate and consume the SAME detached wire snapshot, preventing a
        # caller mutation between admission and assembly from changing inputs.
        try:
            r=json.loads(canonical_json(p34_result))
        except ModelError as exc:
            raise ModelError('MALFORMED_P34_RESULT','P3.4 evidence is not finite plain JSON') from exc
        admitted=validate_admission(model,r,manifest,math_spec_bytes=math_spec_bytes)
        admission='ADMITTED';m=admitted.to_dict()
        provenance={key:m[key] for key in ('model_hash','canonical_model_hash','reference_hash',
                    'p34_result_sha256','manifest_sha256','math_sha256','math_revision','coordinate_sha256')}
        evidence=[{'kind':'hash_bound_P35_admission','location':'inline:admission_manifest','digest':m['manifest_sha256']}]
        checks.append({'check_id':'admission.explicit','status':'PASS','summary':'Explicit caller assertion accepted; not proven linear','evidence':evidence})
        metadata={'bindings':provenance,'admission_manifest':m,'admission_claim':'EXPLICIT_RESEARCH_INPUT_ASSERTION_NOT_PROOF'}
        structure=validate_structure(model)
        S,T,scalar=_verify_p34(model,r,structure)
        checks.append({'check_id':'structure.and_P34','status':'PASS','summary':'Direct canonical structure, exact P3.4 metadata/blocks, fresh scalar baseline','evidence':evidence})
        n=len(model.variable_order)
        A=tuple(tuple(r['blocks'][2]['jacobian'][i][j]/S[i]*T[j] for j in range(n)) for i in range(n))
        B=tuple(tuple(-(r['blocks'][1]['jacobian'][i][j]/S[i]*T[j]) for j in range(n)) for i in range(n))
        p=partitions(model)
        metadata.update(equation_order=list(model.equation_order),variable_order=list(model.variable_order),
             state_variable_order=p['k']+p['z'],control_variable_order=p['j'],partitions=p,shape=[n,n],
             n_k=len(p['k']),n_z=len(p['z']),n_s=len(p['k'])+len(p['z']),n_j=len(p['j']),
             permutations={k:list(getattr(model,k)) for k in PERMUTATIONS},row_permutation_scope=BLOCK_SCOPE,
             derivative_row_source_indices=[e.source_index for e in model.equations if e.kind!='identity'],
             S=S,T=T,coordinate_convention={k:r['metadata'][k] for k in COORDINATE_KEYS},
             matrix_provenance={'A':{'block':'J_plus','offset':1,'sign':1,'formula':'S^-1 J_plus T'},
                                'B':{'block':'J_current','offset':0,'sign':-1,'formula':'-S^-1 J_current T'}},
             orientation='equation rows; variable columns; no transpose',
             system='A E_t q(t+1) = B q(t)',normalization_count={'S_left':1,'T_right':1},
             structure=structure,scalar_reference_verification=scalar.to_dict(),
             timing_reduction='NOT_PERFORMED',dimension_expansion='NOT_PERFORMED',
             solver_ready_admission='NOT_EVALUATED',external_validation='NOT_RUN',EF_REF_1='NOT_RUN',
             Dynare='NOT_RUN',release_validated=False)
        checks.append({'check_id':'assembly.finite','status':'PASS','summary':'A/B assembled from raw blocks with frozen signs and S/T exactly once','evidence':evidence})
        return MatrixAssemblyResult('PASS','COMPLETED',True,A,B,admission,tuple(checks),(),metadata)
    except ModelError as exc:
        if admission!='ADMITTED' and exc.diagnostic.code in ('MALFORMED_MANIFEST','INVALID_ASSERTION','INVALID_JSON_ARTIFACT'):
            admission='INVALID_ADMISSION'
        gate=exc.diagnostic.unsupported or exc.diagnostic.code in (
            'MISSING_MANIFEST','MISSING_ASSERTION','ASSERTION_FALSE','UNSUPPORTED_MANIFEST_VERSION',
            'UNSUPPORTED_ADMISSION_SCOPE','P34_CAPABILITY_MISMATCH')
        checks.append({'check_id':'assembly.request','status':'UNSUPPORTED' if gate else 'FAIL',
                       'summary':str(exc),'evidence':[]})
        return MatrixAssemblyResult('REFUSED' if gate else 'FAIL','NOT_RUN' if gate else 'FAILED',
                                    False,None,None,admission,tuple(checks),(exc.diagnostic.to_dict(),),metadata)
    except Exception as exc:
        diagnostic={'code':'ASSEMBLY_EXCEPTION','message':f'{type(exc).__name__}: {exc}'}
        checks.append({'check_id':'assembly.request','status':'FAIL','summary':diagnostic['message'],'evidence':[]})
        return MatrixAssemblyResult('FAIL','FAILED',False,None,None,admission,tuple(checks),(diagnostic,),metadata)
