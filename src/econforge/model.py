"""Data-only model admission for supplied-point evaluation, not dynamic solving."""

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
from typing import Any, Callable

from econforge.model_ast import (
    ExpressionParser, IDENTIFIER, Node, Symbol, Unit, evaluate_ast, references,
    validate_ast,
)
from econforge.model_errors import ModelError, real
from econforge.residual_policy import (
    CANONICAL_ROWS_MAX, EXPRESSION_DEPTH_MAX, EXPRESSION_NODES_MAX,
    MODEL_BYTES_MAX, MODEL_SYMBOLS_MAX, NUMERIC_LITERAL_CHARS_MAX,
    residual_thresholds,
)


def fields(value: object, required: set[str], optional: set[str], source: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ModelError("INVALID_METADATA", "Expected object", source)
    missing, extra = required - value.keys(), value.keys() - required - optional
    if missing or extra:
        raise ModelError("INVALID_METADATA", f"Missing keys {sorted(missing)}; unknown keys {sorted(extra)}", source)
    return value


def identifier(value: object, source: str) -> str:
    if type(value) is not str or not IDENTIFIER.fullmatch(value) or value in ("exp", "log"):
        raise ModelError("INVALID_IDENTIFIER", "Expected nonreserved ASCII identifier", source)
    return value


def text(value: object, source: str) -> str:
    if type(value) is not str or not value.strip():
        raise ModelError("INVALID_METADATA", "Expected nonempty text", source)
    return value


def unit_metadata(value: object, source: str) -> Unit:
    item = fields(value, {"label", "dimensions"}, set(), source)
    text(item["label"], source)
    if type(item["dimensions"]) is not dict:
        raise ModelError("INVALID_METADATA", "Unit dimensions must be an object", source)
    result = []
    for key, exponent in sorted(item["dimensions"].items()):
        identifier(key, source)
        if type(exponent) is not int:
            raise ModelError("INVALID_METADATA", "Dimension powers must be integers", source)
        if exponent:
            result.append((key, Fraction(exponent)))
    return tuple(result)


def domain_metadata(value: object, source: str) -> tuple[float | None, float | None]:
    item = fields(value, {"lower", "upper"}, set(), source)
    lower = None if item["lower"] is None else real(item["lower"], source)
    upper = None if item["upper"] is None else real(item["upper"], source)
    if lower is not None and upper is not None and lower >= upper:
        raise ModelError("INVALID_DOMAIN", "Open domain must have lower < upper", source)
    return lower, upper


def check_domain(value: float, bounds: tuple[float | None, float | None], source: str) -> None:
    lower, upper = bounds
    if (lower is not None and value <= lower) or (upper is not None and value >= upper):
        raise ModelError("INVALID_DOMAIN", "Value is outside the declared open domain", source, symbol=source)


@dataclass(frozen=True)
class Variable:
    symbol: Symbol
    representation: str
    coordinate: str
    display_unit: str
    underlying: str
    domain: tuple[float | None, float | None]
    scale: float
    baseline: float


@dataclass(frozen=True)
class Equation:
    equation_id: str
    kind: str
    target: str | None
    expression: str
    tree: Node
    scale: float
    magnitude: float
    source_index: int


@dataclass(frozen=True)
class ValidatedModel:
    model_id: str
    frequency: str
    branch_id: str
    variables: tuple[Variable, ...]
    symbols: tuple[Symbol, ...]
    parameters: tuple[tuple[str, float], ...]
    aliases: tuple[tuple[str, Node], ...]
    observables: tuple[tuple[str, Node], ...]
    equations: tuple[Equation, ...]
    source_variable_order: tuple[str, ...]
    variable_order: tuple[str, ...]
    equation_order: tuple[str, ...]
    source_to_canonical_variables: tuple[int, ...]
    canonical_to_source_variables: tuple[int, ...]
    source_to_canonical_equations: tuple[int, ...]
    canonical_to_source_equations: tuple[int, ...]
    reference: tuple[float, ...]
    process_means: tuple[tuple[str, float], ...]
    source_json: str
    digest: str


def expand_aliases(tree: Node, aliases: dict[str, Node], symbols: dict[str, Symbol]) -> Node:
    """Acyclic substitutions retain defining source spans and add explicit dates."""
    count = 0

    def visit(node: Node, shift: int, active: tuple[str, ...], depth: int) -> Node:
        nonlocal count
        count += 1
        if depth > EXPRESSION_DEPTH_MAX or count > EXPRESSION_NODES_MAX:
            raise ModelError("RESOURCE_LIMIT", "Expanded AST exceeds bounds", node.source)
        if node.kind == "symbol":
            name = str(node.value)
            offset = node.offset + shift if symbols[name].role != "parameter" else 0
            if offset not in (-1, 0, 1):
                raise ModelError("INVALID_TIMING", "Alias expansion leaves available slots", node.source, symbol=name)
            if name in aliases:
                if name in active:
                    raise ModelError("CYCLIC_DEFINITION", "Cyclic alias definition", node.source, symbol=name)
                return visit(aliases[name], offset, (*active, name), depth+1)
            return replace(node, offset=offset)
        return replace(node, children=tuple(visit(child, shift, active, depth+1) for child in node.children))

    return validate_ast(visit(tree, 0, (), 1), symbols)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelError("DUPLICATE_KEY", f"Duplicate JSON key {key}")
        result[key] = value
    return result


def _number(token: str) -> float:
    if len(token) > NUMERIC_LITERAL_CHARS_MAX:
        raise ModelError("RESOURCE_LIMIT", "JSON numeric literal too long")
    return real(float(token), "JSON literal")


def _constant(token: str) -> None:
    raise ModelError("NAN_VALUE" if token == "NaN" else "INF_VALUE", "Nonfinite JSON literal")


def _integer(token: str) -> int:
    if len(token) > NUMERIC_LITERAL_CHARS_MAX:
        raise ModelError("RESOURCE_LIMIT", "JSON integer literal too long")
    return int(token)


def _matching(incidences: tuple[tuple[str, ...], ...]) -> bool:
    """Combinatorial incidence admission; makes no numerical rank claim."""
    owners: dict[str, int] = {}

    def assign(row: int, seen: set[str]) -> bool:
        for symbol in incidences[row]:
            if symbol in seen:
                continue
            seen.add(symbol)
            if symbol not in owners or assign(owners[symbol], seen):
                owners[symbol] = row
                return True
        return False

    return all(assign(row, set()) for row in range(len(incidences)))


def _variables(items: list[dict[str, Any]], declare: Callable[[dict[str, Any], str], Symbol]) -> list[Variable]:
    """Admit stored representations, units and strict variable domains."""
    variables = []
    for item in items:
        fields(item, {"id", "role", "unit", "representation", "coordinate", "display_unit", "underlying", "domain", "scale", "baseline"}, set(), "variable")
        if item["role"] not in ("k", "z", "j"):
            raise ModelError("UNSUPPORTED_MODEL_CONSTRUCT", "Variable role must be k, z or j", unsupported=True)
        symbol = declare(item, item["role"])
        rep, coordinate = item["representation"], item["coordinate"]
        allowed = {"level": ("scaled_level", "log_deviation"), "log_level": ("unscaled_deviation",), "declared_deviation": ("unscaled_deviation", "scaled_level")}
        if rep not in allowed or coordinate not in allowed[rep]:
            raise ModelError("INVALID_REPRESENTATION", "Conflicting stored representation/coordinate", symbol.name)
        scale, baseline = real(item["scale"], symbol.name), real(item["baseline"], symbol.name)
        if scale <= 0 or (coordinate == "unscaled_deviation" and scale != 1):
            raise ModelError("INVALID_SCALE", "Positive explicit scale, one for unscaled coordinates, required", symbol.name)
        if rep == "declared_deviation" and baseline != 0:
            raise ModelError("INVALID_REPRESENTATION", "Declared deviation baseline must be zero", symbol.name)
        if rep == "log_level" and symbol.unit:
            raise ModelError("UNIT_MISMATCH", "Stored log must be dimensionless", symbol.name)
        if coordinate == "log_deviation" and baseline <= 0:
            raise ModelError("INVALID_DOMAIN", "Positive log-deviation baseline required", symbol.name)
        bounds = domain_metadata(item["domain"], symbol.name)
        check_domain(baseline, bounds, symbol.name)
        variables.append(Variable(symbol, rep, coordinate, text(item["display_unit"], symbol.name), text(item["underlying"], symbol.name), bounds, scale, baseline))
    return variables


def _parameters(parameter_items: list[tuple[Symbol, dict[str, Any]]], registry: dict[str, Symbol]) -> dict[str, float]:
    """Resolve only the explicitly declared acyclic calibration graph."""
    parameters: dict[str, float] = {}
    pending = dict((s.name, (s, item)) for s, item in parameter_items)

    def parameter(name: str, active: tuple[str, ...]) -> float:
        if name in parameters:
            return parameters[name]
        if name in active:
            raise ModelError("CYCLIC_DEFINITION", "Dependent parameter cycle", name)
        symbol, item = pending[name]
        if "value" in item:
            value = real(item["value"], name)
        else:
            tree = ExpressionParser(item["expression"], registry, name).parse()
            if tree.unit != symbol.unit:
                raise ModelError("UNIT_MISMATCH", "Parameter definition unit", name)
            refs = references(tree)
            if any(registry[ref].role != "parameter" for ref, _ in refs):
                raise ModelError("INVALID_PARAMETER", "Dependent parameter may reference only parameters", name)
            slots = {(ref, 0): parameter(ref, (*active, name)) for ref, _ in refs}
            value = evaluate_ast(tree, registry, slots)
        try:
            check_domain(value, domain_metadata(item["domain"], name), name)
        except ModelError as exc:
            raise ModelError("INVALID_PARAMETER", exc.diagnostic.message, name, symbol=name) from exc
        parameters[name] = value
        return value

    for symbol, _ in parameter_items:
        parameter(symbol.name, ())
    return parameters


def _definitions(definitions: list[tuple[Symbol, dict[str, Any]]], registry: dict[str, Symbol]) -> tuple[dict[str, Node], dict[str, Node]]:
    """Validate aliases and separate reporting definitions."""
    aliases, observables = {}, {}
    for symbol, item in definitions:
        tree = ExpressionParser(item["expression"], registry, symbol.name).parse()
        if tree.unit != symbol.unit:
            raise ModelError("UNIT_MISMATCH", "Definition units differ", symbol.name)
        if any(registry[name].role == "observable" for name, _ in references(tree)):
            raise ModelError("UNSUPPORTED_MODEL_CONSTRUCT", "Reporting observables cannot feed definitions/equations", symbol.name, unsupported=True)
        (aliases if symbol.role == "alias" else observables)[symbol.name] = tree
    for name, tree in (*aliases.items(), *observables.items()):
        expanded = expand_aliases(tree, aliases, registry)
        if name in observables and any(offset != 0 or registry[ref].role == "innovation" for ref, offset in references(expanded)):
            raise ModelError("UNSUPPORTED_OBSERVABLE_TIMING", "Reporting maps use current variables and parameters only", name, unsupported=True)
    return aliases, observables


def _equations(items: list[dict[str, Any]], registry: dict[str, Symbol], aliases: dict[str, Node]) -> list[Equation]:
    """Validate signed residual rows without guessing equation meanings."""
    equations = []
    ids = set()
    for index, item in enumerate(items):
        fields(item, {"id", "kind", "target", "expression", "unit", "scale", "magnitude"}, set(), "equation")
        name = identifier(item["id"], "equation")
        if name in ids:
            raise ModelError("DUPLICATE_EQUATION", name)
        ids.add(name)
        kind = item["kind"]
        if kind not in ("k_transition", "z_process", "conditional", "identity"):
            raise ModelError("EXPECTATION_UNSUPPORTED", "Only whole-row current-information conditional tags or pathwise rows", name, unsupported=True)
        tree = ExpressionParser(item["expression"], registry, name).parse()
        if tree.unit != unit_metadata(item["unit"], name):
            raise ModelError("UNIT_MISMATCH", "Equation residual unit differs", name)
        expanded = expand_aliases(tree, aliases, registry)
        refs = references(expanded)
        if any(registry[ref].role == "observable" for ref, _ in refs):
            raise ModelError("UNSUPPORTED_MODEL_CONSTRUCT", "Observable used in equilibrium", name, unsupported=True)
        if any(registry[ref].role == "innovation" and (kind != "z_process" or offset not in (0, 1)) for ref, offset in refs):
            raise ModelError("INVALID_TIMING", "Innovation slots belong to explicit process rows", name)
        if kind in ("k_transition", "z_process"):
            role = "k" if kind == "k_transition" else "z"
            target = item["target"]
            if type(target) is not str or target not in registry or registry[target].role != role or not any(ref == target for ref, _ in refs):
                raise ModelError("INVALID_TIMING", "Transition target/role/incidence mismatch", name)
            if kind == "k_transition" and (target, 1) not in refs:
                raise ModelError("INVALID_TIMING", "Beginning-stock transition must declare its next stock", name)
            if kind == "z_process" and any(registry[ref].role in ("k", "j") for ref, _ in refs):
                raise ModelError("INVALID_TIMING", "Exogenous process depends on endogenous choice", name)
        elif item["target"] is not None:
            raise ModelError("INVALID_ORDERING", "Equilibrium/reporting row has no transition target", name)
        scale, magnitude = real(item["scale"], name), real(item["magnitude"], name)
        if scale <= 0 or magnitude < 0:
            raise ModelError("INVALID_SCALE", "Require fixed S>0 and L>=0", name)
        equations.append(Equation(name, kind, item["target"], item["expression"], tree, scale, magnitude, index))
    return equations


def _parse_model(source: str) -> ValidatedModel:
    """Parse model JSON and enforce bounded expression/metadata consistency."""
    if type(source) is not str:
        raise ModelError("MALFORMED_MODEL", "Model input must be JSON text")
    if len(source.encode("utf-8")) > MODEL_BYTES_MAX:
        raise ModelError("RESOURCE_LIMIT", "Model byte limit exceeded")
    try:
        data = json.loads(source, object_pairs_hook=_pairs, parse_float=_number, parse_int=_integer, parse_constant=_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ModelError("MALFORMED_MODEL", str(exc)) from exc
    required = {"schema_version", "model_id", "frequency", "branch_id", "variables", "parameters", "aliases", "observables", "innovations", "equations", "process_means", "covariance", "reference"}
    data = fields(data, required, set(), "model")
    if data["schema_version"] != 1 or type(data["schema_version"]) is bool:
        raise ModelError("UNSUPPORTED_MODEL_VERSION", "Only model version 1", unsupported=True)
    for key in ("variables", "parameters", "aliases", "observables", "innovations", "equations"):
        if type(data[key]) is not list:
            raise ModelError("INVALID_METADATA", f"{key} must be an ordered list")
    declarations = sum(len(data[key]) for key in ("variables", "parameters", "aliases", "observables", "innovations"))
    if declarations > MODEL_SYMBOLS_MAX or not 0 < len(data["equations"]) <= CANONICAL_ROWS_MAX:
        raise ModelError("RESOURCE_LIMIT", "Symbol/row count outside resource limits")
    registry: dict[str, Symbol] = {}

    def declare(item: dict[str, Any], role: str) -> Symbol:
        name = identifier(item["id"], role)
        if name in registry:
            raise ModelError("DUPLICATE_SYMBOL", name)
        symbol = Symbol(name, role, unit_metadata(item["unit"], name))
        registry[name] = symbol
        return symbol

    variables = _variables(data["variables"], declare)
    parameter_items = []
    for item in data["parameters"]:
        fields(item, {"id", "unit", "domain"}, {"value", "expression"}, "parameter")
        if ("value" in item) == ("expression" in item):
            raise ModelError("MISSING_PARAMETER", "Supply exactly one parameter value or definition")
        parameter_items.append((declare(item, "parameter"), item))
    definitions = []
    for category, role in (("aliases", "alias"), ("observables", "observable")):
        for item in data[category]:
            fields(item, {"id", "unit", "expression"}, set(), category)
            definitions.append((declare(item, role), item))
    for item in data["innovations"]:
        fields(item, {"id", "unit", "scale"}, set(), "innovation")
        symbol = declare(item, "innovation")
        if real(item["scale"], symbol.name) <= 0:
            raise ModelError("INVALID_SCALE", "Innovation scale must be positive", symbol.name)
    parameters = _parameters(parameter_items, registry)
    aliases, observables = _definitions(definitions, registry)
    equations = _equations(data["equations"], registry, aliases)
    ordered_variables = tuple(v for role in ("k", "z", "j") for v in variables if v.symbol.role == role)
    variable_order = tuple(v.symbol.name for v in ordered_variables)
    ordered_equations = []
    for variable in ordered_variables:
        if variable.symbol.role == "j":
            continue
        rows = [e for e in equations if e.target == variable.symbol.name]
        if len(rows) != 1:
            raise ModelError("MODEL_DIMENSION", "Each state requires exactly one tagged transition", variable.symbol.name)
        ordered_equations.extend(rows)
    ordered_equations.extend(e for e in equations if e.kind == "conditional")
    if len(ordered_equations) != len(ordered_variables) or not ordered_variables:
        raise ModelError("MODEL_DIMENSION", "Require n canonical residuals for n variables")
    incidence = tuple(tuple(v for v in variable_order if any(name == v for name, _ in references(expand_aliases(e.tree, aliases, registry)))) for e in ordered_equations)
    if not _matching(incidence):
        raise ModelError("STRUCTURAL_INCIDENCE", "No complete equation-variable incidence matching")
    ordered_equations.extend(e for e in equations if e.kind == "identity")
    means = fields(data["process_means"], {v.symbol.name for v in variables if v.symbol.role == "z"}, set(), "process_means")
    process_means = tuple((v.symbol.name, real(means[v.symbol.name], "process_means")) for v in ordered_variables if v.symbol.role == "z")
    if len(data["innovations"]) != len(process_means):
        raise ModelError("MODEL_DIMENSION", "One physical innovation per exogenous state is required")
    for name, value in process_means:
        variable = next(v for v in variables if v.symbol.name == name)
        if value != variable.baseline:
            raise ModelError("INCONSISTENT_MODEL", "Declared process mean must equal declared baseline", name)
    mean_slots = {(name, offset): value for name, value in process_means for offset in (-1, 0, 1)}
    mean_slots.update({(name, 0): value for name, value in parameters.items()})
    mean_slots.update({(s.name, offset): 0.0 for s in registry.values() if s.role == "innovation" for offset in (-1, 0, 1)})
    for row in ordered_equations:
        if row.kind == "z_process":
            residual = evaluate_ast(expand_aliases(row.tree, aliases, registry), registry, mean_slots)
            raw_limit, scaled_limit = residual_thresholds(row.scale, row.magnitude, row.equation_id)
            if abs(residual) > raw_limit or abs(real(residual/row.scale, row.equation_id)) > scaled_limit:
                raise ModelError("INCONSISTENT_PROCESS_MEAN", "Declared process mean does not satisfy its process residual", row.equation_id)
    covariance = fields(data["covariance"], {"kind", "values"}, set(), "covariance")
    if covariance["kind"] != "diagonal":
        raise ModelError("UNSUPPORTED_MODEL_CONSTRUCT", "This residual-only slice admits diagonal covariance metadata; full/factor admission is deferred", "covariance", unsupported=True)
    if type(covariance["values"]) is not list or len(covariance["values"]) != len(process_means):
        raise ModelError("MODEL_DIMENSION", "Covariance dimension differs from innovation order")
    if any(real(value, "covariance") < 0 for value in covariance["values"]):
        raise ModelError("INVALID_COVARIANCE", "Negative variance")
    reference_data = fields(data["reference"], set(variable_order), set(), "reference")
    reference = tuple(real(reference_data[name], "reference", symbol=name) for name in variable_order)
    for v, value in zip(ordered_variables, reference):
        check_domain(value, v.domain, v.symbol.name)
        if v.coordinate == "log_deviation" and value <= 0:
            raise ModelError("INVALID_DOMAIN", "Positive supplied log-coordinate reference required", v.symbol.name)
    source_order = tuple(v.symbol.name for v in variables)
    equation_order = tuple(e.equation_id for e in ordered_equations)
    return ValidatedModel(
        identifier(data["model_id"], "model_id"), text(data["frequency"], "frequency"), text(data["branch_id"], "branch_id"),
        ordered_variables, tuple(registry.values()), tuple((s.name, parameters[s.name]) for s, _ in parameter_items),
        tuple(aliases.items()), tuple(observables.items()), tuple(ordered_equations), source_order, variable_order, tuple(e.equation_id for e in ordered_equations if e.kind != "identity"),
        tuple(variable_order.index(name) for name in source_order), tuple(source_order.index(name) for name in variable_order),
        tuple(equation_order.index(e.equation_id) for e in equations), tuple(e.source_index for e in ordered_equations),
        reference, process_means, source, hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def parse_model(source: str) -> ValidatedModel:
    """Public admission boundary with structured malformed-metadata failures."""
    try:
        return _parse_model(source)
    except ModelError:
        raise
    except (TypeError, KeyError, IndexError, RecursionError, UnicodeError) as exc:
        raise ModelError("INVALID_METADATA", f"Malformed model data: {type(exc).__name__}") from exc


def require_valid_model(model: ValidatedModel) -> ValidatedModel:
    """Reject tampered/hand-constructed records at the public evaluation boundary."""
    if type(model) is not ValidatedModel:
        raise ModelError("INVALID_MODEL", "A parsed ValidatedModel is required")
    original = parse_model(model.source_json)
    if original != model:
        raise ModelError("INCONSISTENT_MODEL", "Model metadata/AST differs from validated source")
    return model

