"""JAX lowering of the SAME P3.1A substituted AST (MATH_SPEC section 15)."""

from dataclasses import dataclass
from typing import Any, Callable

from econforge.ad_backend import ADBackend
from econforge.model import ValidatedModel
from econforge.model_ast import Node, integer_literal
from econforge.model_errors import ModelError
from econforge.steady_transform import steady_transform


@dataclass(frozen=True)
class LoweredSteadyResidual:
    function: Callable[[Any], tuple[Any, tuple[Any, Any]]]
    nodes: tuple[Node, ...]
    equation_order: tuple[str, ...]
    variable_order: tuple[str, ...]


def lower_steady_ast(model: ValidatedModel, backend: ADBackend) -> LoweredSteadyResidual:
    """Lower canonical rows only; typed zero shocks and parameters are constants."""
    rows = tuple(row for row in steady_transform(model) if row.original.kind != "identity")
    columns = {name: i for i, name in enumerate(model.variable_order)}
    parameters = dict(model.parameters)
    jnp = backend.numpy
    nodes: list[Node] = []

    def collect(node: Node) -> None:
        for child in node.children:
            collect(child)
        nodes.append(node)

    for row in rows:
        collect(row.tree)
    # All branching is on immutable syntax, never on an AD tracer.
    def evaluate_node(node: Node, x: Any, trace: list[Any]) -> Any:
        children = tuple(evaluate_node(child, x, trace) for child in node.children)
        if node.kind == "number":
            value = jnp.asarray(node.value, dtype=jnp.float64)
        elif node.kind == "innovation_zero":
            value = jnp.asarray(0.0, dtype=jnp.float64)
        elif node.kind == "symbol" and node.offset == 0:
            name = str(node.value)
            value = x[columns[name]] if name in columns else jnp.asarray(parameters[name], dtype=jnp.float64)
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
            # An integer syntax constant must stay an integer JAX power:
            # floating power at a negative/zero base can corrupt its tangent.
            value = children[0] ** (exponent if exponent is not None else children[1])
        else:
            raise ModelError("AD_OPERATION_UNSUPPORTED", "Unsupported differentiated AST operation", node.source,
                             start=node.start, end=node.end, unsupported=True)
        trace.append(value)
        return value

    def evaluate_vector(x: Any) -> tuple[Any, tuple[Any, Any]]:
        trace: list[Any] = []
        vector = jnp.stack(tuple(evaluate_node(row.tree, x, trace) for row in rows))
        return vector, (vector, jnp.stack(trace))

    return LoweredSteadyResidual(evaluate_vector, tuple(nodes),
                                 tuple(row.original.equation_id for row in rows), model.variable_order)
