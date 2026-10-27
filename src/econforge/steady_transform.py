"""Deterministic substitution only; no equilibrium search or expectation integration."""

from dataclasses import dataclass, replace

from econforge.model import Equation, ValidatedModel, expand_aliases, require_valid_model
from econforge.model_ast import Node, references, validate_ast


@dataclass(frozen=True)
class SteadyEquation:
    original: Equation
    tree: Node
    original_slots: tuple[tuple[str, int], ...]
    interpretation: str = "deterministic_steady_substitution"


def steady_transform(model: ValidatedModel) -> tuple[SteadyEquation, ...]:
    """Collapse every dated variable to its stored candidate; innovations to zero."""
    require_valid_model(model)
    registry = {s.name: s for s in model.symbols}
    aliases = dict(model.aliases)

    def collapse(node: Node) -> Node:
        if node.kind == "symbol":
            if registry[str(node.value)].role == "innovation":
                return replace(node, kind="innovation_zero", offset=0)
            return replace(node, offset=0)
        return replace(node, children=tuple(collapse(child) for child in node.children))

    result = []
    for equation in model.equations:
        expanded = expand_aliases(equation.tree, aliases, registry)
        # Internal typed zero retains the physical innovation's unit and source.
        transformed = validate_ast(collapse(expanded), registry)
        result.append(SteadyEquation(equation, transformed, references(expanded)))
    return tuple(result)
