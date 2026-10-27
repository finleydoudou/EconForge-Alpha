"""P3.6 trusted-evidence integrity boundary; never assemble or differentiate."""
import hashlib
import json
import math
import re

from econforge.model_errors import ModelError
from econforge.p35_admission import canonical_json, digest, parse_manifest, MATH_SHA256

SPEC_REVISION = "p36-qz-1"
SPEC_SHA256 = "ab37032dc2b3088b1de94fc46152fa3a89931b06f4eb0153731c7c850a75f6cf"
BINDINGS = ("model_hash", "canonical_model_hash", "reference_hash",
            "p34_result_sha256", "manifest_sha256", "coordinate_sha256")
CONTEXT_FIELDS = {*BINDINGS, "p35_result_sha256", "math_sha256", "legacy_math_sha256",
                  "model_id", "equation_order", "variable_order", "partitions"}


class QZError(Exception):
    def __init__(self, code, message, *, refused=False):
        super().__init__(message)
        self.code = code
        self.refused = refused


def require(condition, code, message):
    if not condition:
        raise QZError(code, message)


def same(a, b):
    return canonical_json(a) == canonical_json(b)


def admit(p35_result, expected, math_spec_bytes):
    """Detach once; compare with independently retained caller identities.

    Expected identities must originate from trusted upstream evidence, not be
    refreshed from a mutated request. Hashes are not authenticated signatures.
    """
    require(type(p35_result) is dict, "MISSING_P35_RESULT", "Complete P3.5 wire result required")
    try:
        r = json.loads(canonical_json(p35_result))
        e = json.loads(canonical_json(expected))
    except ModelError as exc:
        raise QZError("INVALID_WIRE_DATA", str(exc)) from exc
    require(type(e) is dict and set(e) == CONTEXT_FIELDS, "INVALID_CONTEXT", "Exact expected identity fields required")
    for key in CONTEXT_FIELDS - {"model_id", "equation_order", "variable_order", "partitions"}:
        require(type(e[key]) is str and re.fullmatch("[0-9a-f]{64}", e[key]) is not None,
                "INVALID_CONTEXT", "Invalid digest: " + key)
    require(type(math_spec_bytes) is bytes and hashlib.sha256(math_spec_bytes).hexdigest() == SPEC_SHA256
            and e["math_sha256"] == SPEC_SHA256 and e["legacy_math_sha256"] == MATH_SHA256,
            "MATH_SPEC_MISMATCH", "Current and upstream specification bytes/identities differ")
    require(digest(r) == e["p35_result_sha256"], "P35_RESULT_HASH_MISMATCH", "Exact upstream result differs")
    accepted = dict(schema_version=2, claim="canonical_matrix_assembly_from_explicitly_admitted_P34_evidence",
                    assessment_status="PASS", execution_status="COMPLETED", assembly_valid=True,
                    input_admission_status="ADMITTED", model_status="NOT_EVALUATED",
                    failure_reason=None, diagnostics=[])
    require(all(k in r and same(r[k], v) for k,v in accepted.items()),
            "P35_NOT_ACCEPTED", "P3.5 result is not a successful assembly")
    m = r.get("metadata")
    require(type(m) is dict, "INVALID_METADATA", "P3.5 metadata missing")
    bindings = m.get("bindings")
    require(type(bindings) is dict, "INVALID_METADATA", "P3.5 bindings missing")
    try:
        manifest = parse_manifest(m.get("admission_manifest")).to_dict()
    except ModelError as exc:
        raise QZError("INVALID_P35_MANIFEST", str(exc)) from exc
    for key in BINDINGS:
        require(bindings.get(key) == e[key] == manifest.get(key),
                key.upper() + "_MISMATCH", "Upstream binding differs: " + key)
    require(bindings.get("math_sha256") == MATH_SHA256 == manifest["math_sha256"]
            and bindings.get("math_revision") == "v0.1-math-freeze-1" == manifest["math_revision"],
            "MATH_SPEC_MISMATCH", "Upstream math binding differs")
    require(type(e["model_id"]) is str and e["model_id"] == manifest["model_id"],
            "MODEL_ID_MISMATCH", "Model identity differs")
    rows, cols = m.get("equation_order"), m.get("variable_order")
    require(type(rows) is list and type(cols) is list and 0 < len(rows) == len(cols) <= 256,
            "AXIS_MISMATCH", "Bounded square canonical axes required")
    n = len(cols)
    for key, values in (("equation_order", rows), ("variable_order", cols)):
        require(all(type(x) is str and x for x in values) and len(set(values)) == n
                and same(values, e[key]) and same(values, manifest[key]),
                "AXIS_MISMATCH", "Missing, duplicated or stale axes")
    require(same(m.get("shape"), [n,n]), "SHAPE_MISMATCH", "Shape metadata differs")
    p = m.get("partitions")
    require(type(p) is dict and set(p) == {"k","z","j"}
            and all(type(v) is list for v in p.values()), "PARTITION_MISMATCH", "Explicit partitions required")
    require(same(p, e["partitions"]) and same(p, manifest["partitions"])
            and same(p["k"]+p["z"]+p["j"], cols), "PARTITION_MISMATCH", "Partition order differs")
    counts = dict(n_k=len(p["k"]), n_z=len(p["z"]), n_j=len(p["j"]), n_s=len(p["k"])+len(p["z"]))
    require(all(same(m.get(k),v) for k,v in counts.items())
            and same(m.get("state_variable_order"),p["k"]+p["z"])
            and same(m.get("control_variable_order"),p["j"]),
            "PARTITION_MISMATCH", "Partition dimensions/labels differ")
    coord = m.get("coordinate_convention")
    require(type(coord) is dict and digest(coord) == e["coordinate_sha256"],
            "COORDINATE_MISMATCH", "Coordinate provenance differs")
    for key, field in (("S", "row_scales_S"), ("T", "coordinate_diagonal_T")):
        value = m.get(key)
        require(type(value) is list and len(value) == n
                and all(type(x) is float and math.isfinite(x) and x > 0 for x in value)
                and same(value,coord.get(field)), "COORDINATE_MISMATCH", "Invalid or inconsistent S/T")
    definitions = {"raw_definition":"partial original signed integrand / partial stored w(offset)",
                   "row_normalized_definition":"S^-1 J_slot",
                   "coordinate_normalized_definition":"S^-1 J_slot T"}
    require(all(coord.get(k)==v for k,v in definitions.items()),
            "COORDINATE_MISMATCH", "Unsupported coordinate definitions")
    require(same(m.get("matrix_provenance"), {
        "A":{"block":"J_plus","offset":1,"sign":1,"formula":"S^-1 J_plus T"},
        "B":{"block":"J_current","offset":0,"sign":-1,"formula":"-S^-1 J_current T"}})
        and m.get("system")=="A E_t q(t+1) = B q(t)"
        and m.get("orientation")=="equation rows; variable columns; no transpose"
        and same(m.get("normalization_count"),{"S_left":1,"T_right":1}),
        "MATRIX_CONVENTION_MISMATCH", "P3.5 direction/normalization differs")
    structure = m.get("structure")
    require(type(structure) is dict, "INVALID_METADATA", "Structure missing")
    if (m.get("timing_reduction") != "NOT_PERFORMED" or m.get("dimension_expansion") != "NOT_PERFORMED"
            or structure.get("timing_transformation") != "NONE_DIRECT_CANONICAL_ONLY"
            or structure.get("static_elimination") != "NOT_PERFORMED"):
        raise QZError("UNSUPPORTED_STRUCTURE", "Only accepted direct canonical structure", refused=True)
    scalar = m.get("scalar_reference_verification")
    require(type(scalar) is dict and scalar.get("evaluation_valid") is True
            and scalar.get("residuals_within_tolerance") is True,
            "REFERENCE_MISMATCH", "Verified scalar reference required")
    sm = scalar.get("metadata")
    require(type(sm) is dict and sm.get("model_hash")==e["model_hash"]
            and sm.get("evaluation_hash")==e["reference_hash"] and sm.get("model_id")==e["model_id"],
            "REFERENCE_MISMATCH", "Scalar reference identity differs")
    candidate = scalar.get("candidate")
    require(type(candidate) is list and len(candidate)==n
            and all(type(x) is float and math.isfinite(x) for x in candidate)
            and scalar.get("assessment_status")=="PASS" and scalar.get("execution_status")=="COMPLETED"
            and same(scalar.get("variable_order"),cols) and same(scalar.get("equation_order"),rows),
            "REFERENCE_MISMATCH", "Scalar reference values/status/axes differ")
    reference_digest = hashlib.sha256((e["model_hash"] + json.dumps(candidate,allow_nan=False)).encode()).hexdigest()
    require(reference_digest==e["reference_hash"],"REFERENCE_MISMATCH","Candidate does not match bound reference")
    require(same(coord.get("coordinate_metadata"),sm.get("coordinate_metadata")),
            "COORDINATE_MISMATCH", "Reference coordinate metadata differs")
    permutations = m.get("permutations")
    require(type(permutations) is dict, "AXIS_MISMATCH", "Source permutations required")
    for axis,size in (("variables",n),("equations",len(sm.get("verification_equation_order",[])))):
        f = permutations.get("source_to_canonical_"+axis)
        b = permutations.get("canonical_to_source_"+axis)
        require(type(f) is list and type(b) is list and len(f)==len(b)==size
                and all(type(i) is int for i in f+b)
                and set(f)==set(b)==set(range(size))
                and all(b[f[i]]==i for i in range(size))
                and same(f,sm.get("source_to_canonical_"+axis))
                and same(b,sm.get("canonical_to_source_"+axis)),
                "AXIS_MISMATCH", "Invalid inverse source permutations")
    indices = m.get("derivative_row_source_indices")
    require(type(indices) is list and len(indices)==n
            and all(type(i) is int and 0 <= i < len(sm["verification_equation_order"]) for i in indices)
            and len(set(indices))==n, "AXIS_MISMATCH", "Invalid derivative source indices")
    forward = permutations["source_to_canonical_equations"]
    require([sm["verification_equation_order"][forward[i]] for i in indices]==rows,
            "AXIS_MISMATCH", "Derivative source indices do not identify canonical rows")
    checks = r.get("checks")
    require(type(checks) is list and all(type(c) is dict for c in checks)
            and [c.get("check_id") for c in checks]==["admission.explicit","structure.and_P34","assembly.finite"]
            and all(c.get("status")=="PASS" and type(c.get("evidence")) is list and c["evidence"]
                    and all(type(x) is dict and x.get("digest")==e["manifest_sha256"] for x in c["evidence"])
                    for c in checks), "P35_CHECKS_INVALID", "Three evidenced upstream PASS checks required")
    for key in ("A","B"):
        a = r.get(key)
        require(type(a) is list and len(a)==n and all(type(row) is list and len(row)==n for row in a),
                "MATRIX_SHAPE_MISMATCH", key+" must be square")
        require(all(type(x) is float and math.isfinite(x) for row in a for x in row),
                "MATRIX_DTYPE_OR_NONFINITE", "Require finite real binary64 wire values")
    return r, e
