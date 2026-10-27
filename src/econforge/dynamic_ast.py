"""Original integrand JAX lowering with independent -1/0/+1 stored slots."""

from dataclasses import dataclass
from typing import Any, Callable

from econforge.ad_backend import ADBackend
from econforge.model import ValidatedModel, expand_aliases, require_valid_model
from econforge.model_ast import Node, integer_literal, references, validate_ast
from econforge.model_errors import ModelError


DATE_OFFSETS = (-1, 0, 1)


@dataclass(frozen=True)
class LoweredDynamicResidual:
    function: Callable[[Any], tuple[Any, tuple[Any, Any]]]
    nodes: tuple[Node, ...]
    equation_order: tuple[str, ...]
    variable_order: tuple[str, ...]
    slot_order: tuple[tuple[int, str], ...]
    row_kinds: tuple[str, ...]
    original_slots: tuple[tuple[tuple[str, int], ...], ...]
    model_hash: str


def lower_dynamic_ast(model: ValidatedModel, backend: ADBackend) -> LoweredDynamicResidual:
    """No steady_transform, date shift, expectation integration or new unknowns."""
    require_valid_model(model)
    registry = {s.name: s for s in model.symbols}
    rows = tuple(e for e in model.equations if e.kind != "identity")
    trees = tuple(validate_ast(expand_aliases(e.tree, dict(model.aliases), registry), registry) for e in rows)
    slots = tuple((offset, name) for offset in DATE_OFFSETS for name in model.variable_order)
    columns = {key: i for i, key in enumerate(slots)}
    parameters = dict(model.parameters)
    jnp = backend.numpy
    nodes: list[Node] = []

    def collect(node: Node) -> None:
        for child in node.children:
            collect(child)
        nodes.append(node)

    for tree in trees:
        collect(tree)

    def evaluate_node(node: Node, x: Any, trace: list[Any]) -> Any:
        # Branch only on validated immutable syntax; never inspect tracer values.
        children = tuple(evaluate_node(child, x, trace) for child in node.children)
        if node.kind == "number":
            value = jnp.asarray(node.value, dtype=jnp.float64)
        elif node.kind == "symbol":
            name = str(node.value)
            role = registry[name].role
            if role == "parameter" and node.offset == 0:
                value = jnp.asarray(parameters[name], dtype=jnp.float64)
            elif role == "innovation":
                # Fixed zero innovations; their original slots remain in metadata.
                value = jnp.asarray(0.0, dtype=jnp.float64)
            elif (node.offset, name) in columns:
                value = x[columns[(node.offset, name)]]
            else:
                raise ModelError("DYNAMIC_SLOT_UNSUPPORTED", "Unsupported differentiated symbol slot", node.source,
                                 start=node.start, end=node.end, unsupported=True)
        elif node.kind == "unary":
            value = children[0] if node.value == "+" else -children[0]
        elif node.kind == "call" and node.value == "exp":
            value = jnp.exp(children[0])
        elif node.kind == "call" and node.value == "log":
            value = jnp.log(children[0])
        elif node.kind == "binary" and node.value == "+":
            value = children[0] + children[1]
        elif node.kind == "binary" and node.value == "-":
            value = children[0] - children[1]
        elif node.kind == "binary" and node.value == "*":
            value = children[0] * children[1]
        elif node.kind == "binary" and node.value == "/":
            value = children[0] / children[1]
        elif node.kind == "binary" and node.value == "^":
            exponent = integer_literal(node.children[1])
            value = children[0] ** (exponent if exponent is not None else children[1])
        else:
            raise ModelError("AD_OPERATION_UNSUPPORTED", "Unsupported differentiated AST operation", node.source,
                             start=node.start, end=node.end, unsupported=True)
        trace.append(value)
        return value

    def evaluate_vector(x: Any) -> tuple[Any, tuple[Any, Any]]:
        trace: list[Any] = []
        vector = jnp.stack(tuple(evaluate_node(tree, x, trace) for tree in trees))
        return vector, (vector, jnp.stack(trace))

    return LoweredDynamicResidual(evaluate_vector, tuple(nodes), tuple(e.equation_id for e in rows),
                                  model.variable_order, slots, tuple(e.kind for e in rows),
                                  tuple(references(tree) for tree in trees), model.digest)
