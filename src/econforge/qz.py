"""P3.6 public API: trusted P3.5 result to verified unordered complex QZ."""
from dataclasses import dataclass
import importlib.metadata
import json

from econforge.qz_admission import QZError, admit
from econforge import qz_backend


@dataclass(frozen=True)
class QZResult:
    """Immutable finite-JSON snapshot; complex scalars use [real,imag]."""
    wire_json: str

    def to_dict(self):
        return json.loads(self.wire_json)


def decompose_p35(p35_result, expected, *, math_spec_bytes):
    """Expected identities are independently retained by the caller.

    No original model parsing, numerical assembly or differentiation occurs.
    All failed requests suppress factors and spectral outputs.
    """
    r=dict(schema_version=2,claim="P3.6_verified_unordered_complex_QZ",
           assessment_status="FAIL",execution_status="NOT_RUN",qz_execution_status="NOT_RUN",
           input_admission_status="NOT_ADMITTED",qz_valid=False,scalar_spectrum_complete=False,
           pencil_regularity="NOT_EVALUATED",model_status="NOT_EVALUATED",
           capability_maturity="IMPLEMENTED",independent_validation_status="NOT_RUN",
           backend="scipy.linalg.qz",dtype="complex128",pencil="B - lambda A",
           backend_order="NON_SEMANTIC_UNSORTED",Q=None,Z=None,schur_B=None,schur_A=None,
           pairs=None,diagnostics={},checks=[],warnings=[],metadata={},failure_reason=None)
    stage="admission"
    try:
        upstream,context=admit(p35_result,expected,math_spec_bytes)
        r["input_admission_status"]="ADMITTED"
        r["metadata"]={"bindings":context,"input_metadata":upstream["metadata"],
                       "input_A":upstream["A"],"input_B":upstream["B"],
                       "external_validation":"NOT_RUN","EF_REF_1":"NOT_RUN","Dynare":"NOT_RUN"}
        evidence=[{"kind":"P35_wire_sha256","digest":context["p35_result_sha256"],"location":"inline:P35"}]
        r["checks"].append(dict(check_id="input.admitted",status="PASS",evidence=evidence))
        stage="backend_admission"
        np,linalg,versions=qz_backend.runtime()
        r["metadata"]["backend_versions"]=versions
        r["checks"].append(dict(check_id="backend.admitted",status="PASS",evidence=evidence))
        A=np.array(upstream["A"],dtype=np.float64);B=np.array(upstream["B"],dtype=np.float64)
        stage="thresholds"
        scale,tau,pair=qz_backend.thresholds(np,A,B)
        r["metadata"].update(pencil_scale=scale,tau_qz=tau,tau_pair=pair,n=A.shape[0],
                             input_dtype="float64",specification_revision="p36-qz-1")
        stage="qz"
        r["qz_execution_status"]="FAILED"
        output=qz_backend.execute(linalg,A,B)
        r["qz_execution_status"]="COMPLETED"
        stage="verification"
        d=qz_backend.verify(np,A,B,output,tau)
        r["diagnostics"]=d
        for name,value in d.items():
            r["checks"].append(dict(check_id=name,status="PASS" if value<=tau else "FAIL",
                                    value=value,threshold=tau,evidence=evidence))
        if any(value>tau for value in d.values()):
            raise QZError("QZ_VERIFICATION_FAILED","Structural verification failed")
        schur_B,schur_A,Q,Z=output
        stage="pairs"
        pairs=qz_backend.pair_records(np,schur_B,schur_A,pair)
        for key,x in zip(("schur_B","schur_A","Q","Z"),output):
            r[key]=[[[float(z.real),float(z.imag)] for z in row] for row in x]
        r["pairs"]=pairs
        r["scalar_spectrum_complete"]=all(p["pair_status"]=="FINITE" for p in pairs)
        if not r["scalar_spectrum_complete"]:
            r["warnings"].append("Scalar spectrum incomplete; homogeneous pairs and verified decomposition retained")
        if any(p["pair_status"]=="INDETERMINATE_0_OVER_0" for p in pairs):
            r["warnings"].append("Indeterminate homogeneous pair present; unsuitable for economic classification without later policy")
        r["assessment_status"]="PASS" if r["scalar_spectrum_complete"] else "PASS_WITH_WARNINGS"
        r["execution_status"]="COMPLETED"
        r["qz_valid"]=True
    except Exception as exc:
        if isinstance(exc,QZError):
            code,refused=exc.code,exc.refused
        elif stage=="backend_admission" and isinstance(exc,(ImportError,importlib.metadata.PackageNotFoundError)):
            code,refused="DEPENDENCY_MISSING",True
        else:
            code,refused={"qz":"QZ_BACKEND_EXCEPTION","admission":"INVALID_METADATA"}.get(stage,"QZ_EXCEPTION"),False
        r["assessment_status"]="REFUSED" if refused else "FAIL"
        r["execution_status"]="NOT_RUN" if refused and r["qz_execution_status"]=="NOT_RUN" else "FAILED"
        r["failure_reason"]=code
        r["checks"].append(dict(check_id=stage,status="UNSUPPORTED" if refused else "FAIL",
                                reason=code,message=str(exc),evidence=[]))
        for key in ("Q","Z","schur_B","schur_A","pairs"):
            r[key]=None
        r["qz_valid"]=False;r["scalar_spectrum_complete"]=False
    return QZResult(json.dumps(r,allow_nan=False,separators=(",",":")))
