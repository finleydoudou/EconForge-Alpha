"""Separate caller assertion for one exact P3.5 input state; not a proof.

No derivatives or matrices are constructed here. Legacy admission is unchanged.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import re

from econforge.model import require_valid_model
from econforge.model_errors import ModelError
from econforge.residual_policy import SPEC_REVISION

MATH_SHA256 = "3e16d20bf17a26d8fb29df96803aba774359198b1997ef4ae83a8927865e2305"
SCOPE = "P3.5_matrix_assembly_only"
CAPABILITY = "P3.4_original_integrand_date_partials/schema-2"
CLAIM = "original_integrand_date_partials_at_verified_reference_point_checks"
COORDINATE_KEYS = ("coordinate_metadata", "row_scales_S", "coordinate_diagonal_T",
                   "raw_definition", "row_normalized_definition", "coordinate_normalized_definition")
FIELDS = {"schema_version", "already_linear_for_p35", "scope", "model_id", "model_hash",
          "canonical_model_hash", "math_revision", "math_sha256", "reference_hash",
          "p34_result_sha256", "coordinate_sha256", "equation_order", "variable_order",
          "partitions", "p34_capability", "manifest_sha256"}


def canonical_json(value):
    """Version-1 wire digest: plain JSON only, sorted keys, ASCII, compact, finite.

    Arrays preserve order; int and float encodings are not interchangeable.
    This is the specified Python JSON convention, not a claim of RFC 8785.
    """
    def plain(x):
        if x is None or type(x) in (str, bool, int):
            return
        if type(x) is float and math.isfinite(x):
            return
        if type(x) is list:
            for item in x:
                plain(item)
            return
        if type(x) is dict and all(type(k) is str for k in x):
            for item in x.values():
                plain(item)
            return
        raise ModelError("INVALID_JSON_ARTIFACT", "Require plain finite JSON data")
    plain(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def partitions(model):
    return {role: [v.symbol.name for v in model.variables if v.symbol.role == role]
            for role in ("k", "z", "j")}


@dataclass(frozen=True)
class AdmissionManifest:
    """Immutable typed wire snapshot; every consumer revalidates even this type."""
    wire_json: str

    def to_dict(self):
        return json.loads(self.wire_json)


def parse_manifest(value):
    if value is None:
        raise ModelError("MISSING_MANIFEST", "Explicit caller assertion is required")
    if type(value) is AdmissionManifest:
        try:
            value = json.loads(value.wire_json)
        except (TypeError, ValueError) as exc:
            raise ModelError("MALFORMED_MANIFEST", "Invalid typed wire snapshot") from exc
    if type(value) is not dict:
        raise ModelError("MALFORMED_MANIFEST", "Manifest must be a separate JSON object")
    if "already_linear_for_p35" not in value:
        raise ModelError("MISSING_ASSERTION", "Absence never implies admission")
    if type(value["already_linear_for_p35"]) is not bool:
        raise ModelError("INVALID_ASSERTION", "Require a strict boolean; no coercion")
    if type(value.get("schema_version")) is not int:
        raise ModelError("MALFORMED_MANIFEST", "Version must be a plain integer")
    if value["schema_version"] != 1:
        raise ModelError("UNSUPPORTED_MANIFEST_VERSION", "Only version 1 is implemented")
    if set(value) != FIELDS:
        raise ModelError("MALFORMED_MANIFEST", "Manifest fields must match version 1 exactly")
    for key in FIELDS - {"schema_version", "already_linear_for_p35", "equation_order", "variable_order", "partitions"}:
        if type(value[key]) is not str or not value[key]:
            raise ModelError("MALFORMED_MANIFEST", f"Invalid text field {key}")
        if (key.endswith("hash") or key.endswith("sha256")) and not re.fullmatch("[0-9a-f]{64}", value[key]):
            raise ModelError("MALFORMED_MANIFEST", f"Invalid SHA256 {key}")
    for key in ("equation_order", "variable_order"):
        if type(value[key]) is not list or not value[key] or any(type(x) is not str or not x for x in value[key]):
            raise ModelError("MALFORMED_MANIFEST", f"Invalid order {key}")
    p = value["partitions"]
    if type(p) is not dict or set(p) != {"k", "z", "j"} or any(
            type(v) is not list or any(type(x) is not str or not x for x in v) for v in p.values()):
        raise ModelError("MALFORMED_MANIFEST", "Require explicit k/z/j arrays")
    expected = digest({k: v for k, v in value.items() if k != "manifest_sha256"})
    if value["manifest_sha256"] != expected:
        raise ModelError("TAMPERED_MANIFEST", "Manifest checksum mismatch; no silent refresh")
    if not value["already_linear_for_p35"]:
        raise ModelError("ASSERTION_FALSE", "Caller has not admitted this input state")
    if value["scope"] != SCOPE:
        raise ModelError("UNSUPPORTED_ADMISSION_SCOPE", "Admission is assembly only")
    return AdmissionManifest(canonical_json(value))


def validate_admission(model, evidence, manifest, *, math_spec_bytes):
    """Validate bindings, raising a specific ModelError; returns typed assertion.

    This is admission only, not P3.4 numerical verification or assembly. The
    assembly consumer must additionally verify structure and P3.4 consistency.
    No helper in production generates an affirmative assertion.
    """
    admitted = parse_manifest(manifest)
    m = admitted.to_dict()
    require_valid_model(model)
    if type(math_spec_bytes) is not bytes:
        raise ModelError("MATH_SPEC_MISMATCH", "Supply the actual frozen specification bytes")
    actual_math = hashlib.sha256(math_spec_bytes).hexdigest()
    if actual_math != MATH_SHA256 or m["math_sha256"] != actual_math or m["math_revision"] != SPEC_REVISION:
        raise ModelError("MATH_SPEC_MISMATCH", "Frozen revision/content differs")
    if m["model_id"] != model.model_id or m["model_hash"] != model.digest or m["canonical_model_hash"] != digest(json.loads(model.source_json)):
        raise ModelError("MODEL_HASH_MISMATCH", "Model identity/source/canonical digest differs")
    if type(evidence) is not dict or type(evidence.get("metadata")) is not dict:
        raise ModelError("MALFORMED_P34_RESULT", "Require P3.4 JSON result with metadata")
    meta = evidence["metadata"]
    if meta.get("model_hash") != model.digest or meta.get("model_id") != model.model_id:
        raise ModelError("P34_MODEL_MISMATCH", "P3.4 evidence belongs to another model")
    ref = evidence.get("reference")
    if type(ref) is not list or len(ref) != len(model.variable_order) or any(type(x) is not float or not math.isfinite(x) for x in ref):
        raise ModelError("MALFORMED_P34_REFERENCE", "Require exact finite binary64 reference array")
    # Reuse the accepted residual evaluation hash convention verbatim.
    reference_hash = hashlib.sha256((model.digest + json.dumps(ref, allow_nan=False)).encode()).hexdigest()
    if m["reference_hash"] != reference_hash or meta.get("evaluation_hash") != reference_hash:
        raise ModelError("REFERENCE_HASH_MISMATCH", "Exact deterministic reference differs")
    if m["equation_order"] != list(model.equation_order) or evidence.get("equation_order") != list(model.equation_order):
        raise ModelError("EQUATION_ORDER_MISMATCH", "Require exact canonical row identities")
    if m["variable_order"] != list(model.variable_order) or evidence.get("variable_order") != list(model.variable_order):
        raise ModelError("VARIABLE_ORDER_MISMATCH", "Require exact canonical column identities")
    if m["partitions"] != partitions(model):
        raise ModelError("PARTITION_MISMATCH", "Declared k/z/j partitions differ (no inference)")
    if any(key not in meta for key in COORDINATE_KEYS):
        raise ModelError("COORDINATE_MISMATCH", "Missing explicit coordinate/scaling metadata")
    if m["coordinate_sha256"] != digest({k: meta[k] for k in COORDINATE_KEYS}):
        raise ModelError("COORDINATE_MISMATCH", "Coordinate convention digest differs")
    if m["p34_capability"] != CAPABILITY or type(evidence.get("schema_version")) is not int or evidence["schema_version"] != 2 or evidence.get("claim") != CLAIM:
        raise ModelError("P34_CAPABILITY_MISMATCH", "Unsupported P3.4 result capability/version")
    try:
        actual_evidence = digest(evidence)
    except ModelError as exc:
        raise ModelError("MALFORMED_P34_RESULT", "P3.4 evidence is not finite plain JSON") from exc
    if m["p34_result_sha256"] != actual_evidence:
        raise ModelError("STALE_P34_EVIDENCE", "Exact P3.4 result hash differs; stale/tampered admission")
    return admitted
