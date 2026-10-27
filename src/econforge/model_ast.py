"""Bounded expression parser and scalar interpreter, never Python execution."""

from dataclasses import dataclass, replace
from fractions import Fraction
import math
import re
from typing import Mapping

from econforge.model_errors import ModelError, real
from econforge.residual_policy import (
    EXPRESSION_BYTES_MAX, EXPRESSION_DEPTH_MAX, EXPRESSION_NODES_MAX,
    NUMERIC_LITERAL_CHARS_MAX,
)

Unit = tuple[tuple[str, Fraction], ...]
TOKEN = re.compile(r"\s*(?:(?P<num>(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)|(?P<id>[A-Za-z_][A-Za-z_0-9]*)|(?P<op>[()+*/^,-]))", re.ASCII)
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z", re.ASCII)


@dataclass(frozen=True)
class Node:
    kind: str
    value: str | float
    children: tuple["Node", ...]
    offset: int
    source: str
    start: int
    end: int
    unit: Unit = ()


@dataclass(frozen=True)
class Symbol:
    name: str
    role: str
    unit: Unit


def unit_product(a: Unit, b: Unit, sign: int = 1) -> Unit:
    result = dict(a)
    for name, power in b:
        result[name] = result.get(name, Fraction(0)) + sign * power
    return tuple(sorted((name, power) for name, power in result.items() if power))


def integer_literal(node: Node) -> int | None:
    """Recognize integer constant expressions, never calibrated/dynamic powers."""
    if node.kind == "number" and float(node.value).is_integer():
        return int(node.value)
    if node.kind == "unary":
        child = integer_literal(node.children[0])
        if child is not None:
            return child if node.value == "+" else -child
    def is_constant(item: Node) -> bool:
        return item.kind in ("number", "unary", "binary", "call") and all(is_constant(child) for child in item.children)

    if node.kind in ("binary", "call") and is_constant(node):
        value = evaluate_ast(node, {}, {})
        if value.is_integer():
            return int(value)
    return None


class ExpressionParser:
    """Recursive descent with independent syntax and AST depth limits."""

    def __init__(self, text: str, symbols: Mapping[str, Symbol], source: str):
        if type(text) is not str:
            raise ModelError("MALFORMED_EXPRESSION", "Expression must be a string", source)
        if len(text.encode("utf-8")) > EXPRESSION_BYTES_MAX:
            raise ModelError("RESOURCE_LIMIT", "Expression byte limit exceeded", source)
        self.text, self.symbols, self.source = text, symbols, source
        self.tokens: list[tuple[str, str, int, int]] = []
        pos = 0
        while pos < len(text):
            if text[pos:].isspace():
                break
            token = TOKEN.match(text, pos)
            if token is None:
                raise ModelError("PARSE_UNSUPPORTED", "Unsupported expression character", source, start=pos, end=pos+1, unsupported=True)
            kind = token.lastgroup
            assert kind is not None
            self.tokens.append((kind, token.group(kind), token.start(kind), token.end()))
            pos = token.end()
            if len(self.tokens) > EXPRESSION_NODES_MAX * 4:
                raise ModelError("RESOURCE_LIMIT", "Expression token limit exceeded", source)
        self.tokens.append(("end", "", len(text), len(text)))
        self.index, self.count, self.depth = 0, 0, 0

    def peek(self) -> str:
        return self.tokens[self.index][1]

    def take(self, expected: str | None = None) -> tuple[str, str, int, int]:
        token = self.tokens[self.index]
        if expected is not None and token[1] != expected:
            raise ModelError("MALFORMED_EXPRESSION", f"Expected {expected!r}", self.source, start=token[2], end=token[3])
        if token[0] == "end":
            raise ModelError("MALFORMED_EXPRESSION", "Unexpected end", self.source, start=token[2], end=token[3])
        self.index += 1
        return token

    def node(self, kind: str, value: str | float, children: tuple[Node, ...], start: int, end: int, offset: int = 0) -> Node:
        self.count += 1
        if self.count > EXPRESSION_NODES_MAX:
            raise ModelError("RESOURCE_LIMIT", "AST node limit exceeded", self.source)
        return Node(kind, value, children, offset, self.source, start, end)

    def expression(self, minimum: int = 0) -> Node:
        self.depth += 1
        if self.depth > EXPRESSION_DEPTH_MAX:
            raise ModelError("RESOURCE_LIMIT", "Expression depth exceeded", self.source)
        kind, value, start, end = self.take()
        if value in ("+", "-"):
            child = self.expression(30)
            left = self.node("unary", value, (child,), start, child.end)
        elif value == "(":
            left = self.expression()
            self.take(")")
        elif kind == "num":
            if len(value) > NUMERIC_LITERAL_CHARS_MAX:
                raise ModelError("RESOURCE_LIMIT", "Numeric literal too long", self.source, start=start, end=end)
            left = self.node("number", real(float(value), self.source), (), start, end)
        elif kind == "id":
            if value in ("exp", "log"):
                self.take("(")
                child = self.expression()
                close = self.take(")")
                left = self.node("call", value, (child,), start, close[3])
            else:
                if value not in self.symbols:
                    raise ModelError("UNKNOWN_SYMBOL", f"Undeclared symbol {value}", self.source, start=start, end=end, symbol=value)
                offset = 0
                if self.peek() == "(":
                    self.take("(")
                    sign = self.take()[1]
                    number = self.take()[1]
                    close = self.take(")")
                    if sign not in ("+", "-") or number != "1" or self.symbols[value].role == "parameter":
                        raise ModelError("INVALID_TIMING", "Only variable (-1)/(+1) slots are supported", self.source, start=start, end=close[3])
                    offset = 1 if sign == "+" else -1
                    end = close[3]
                left = self.node("symbol", value, (), start, end, offset)
        else:
            raise ModelError("PARSE_UNSUPPORTED", "Unsupported expression syntax", self.source, start=start, end=end, unsupported=True)
        precedence = {"+": 10, "-": 10, "*": 20, "/": 20, "^": 40}
        while self.peek() in precedence and precedence[self.peek()] >= minimum:
            operator = self.take()[1]
            priority = precedence[operator]
            right = self.expression(priority if operator == "^" else priority + 1)
            left = self.node("binary", operator, (left, right), left.start, right.end)
        self.depth -= 1
        return left

    def parse(self) -> Node:
        tree = self.expression()
        if self.tokens[self.index][0] != "end":
            token = self.tokens[self.index]
            raise ModelError("PARSE_UNSUPPORTED", "Trailing expression tokens", self.source, start=token[2], end=token[3], unsupported=True)
        return validate_ast(tree, self.symbols)


def validate_ast(tree: Node, symbols: Mapping[str, Symbol]) -> Node:
    """Revalidate structure/units even when callers supply a constructed AST."""
    count = 0

    def visit(node: Node, depth: int) -> Node:
        nonlocal count
        count += 1
        if depth > EXPRESSION_DEPTH_MAX or count > EXPRESSION_NODES_MAX:
            raise ModelError("RESOURCE_LIMIT", "AST resource bound exceeded")
        if type(node) is not Node or type(node.children) is not tuple:
            raise ModelError("UNSUPPORTED_AST_NODE", "Expected immutable Node", unsupported=True)
        if type(node.kind) is not str or type(node.value) not in (str, int, float) or type(node.source) is not str:
            raise ModelError("UNSUPPORTED_AST_NODE", "Invalid AST field types", unsupported=True)
        if type(node.start) is not int or type(node.end) is not int or not 0 <= node.start <= node.end:
            raise ModelError("UNSUPPORTED_AST_NODE", "Invalid source span", unsupported=True)
        if type(node.offset) is not int or node.offset not in (-1, 0, 1):
            raise ModelError("INVALID_TIMING", "Unavailable slot", node.source)
        if node.kind not in ("symbol", "innovation_zero") and node.offset != 0:
            raise ModelError("INVALID_TIMING", "Only symbols have time offsets", node.source)
        children = tuple(visit(child, depth+1) for child in node.children)
        unit: Unit = ()
        if node.kind == "number" and not children:
            real(node.value, node.source)
        elif node.kind == "innovation_zero" and not children and node.value in symbols and symbols[str(node.value)].role == "innovation" and node.offset == 0:
            unit = symbols[str(node.value)].unit
        elif node.kind == "symbol" and not children and node.value in symbols:
            symbol = symbols[str(node.value)]
            if symbol.role == "parameter" and node.offset:
                raise ModelError("INVALID_TIMING", "Dated parameter", node.source)
            unit = symbol.unit
        elif node.kind == "unary" and node.value in ("+", "-") and len(children) == 1:
            unit = children[0].unit
        elif node.kind == "call" and node.value in ("exp", "log") and len(children) == 1:
            if children[0].unit:
                raise ModelError("UNIT_MISMATCH", "exp/log requires dimensionless argument", node.source)
        elif node.kind == "binary" and node.value in ("+", "-", "*", "/", "^") and len(children) == 2:
            a, b = children
            if node.value in ("+", "-"):
                if a.unit != b.unit:
                    raise ModelError("UNIT_MISMATCH", "Addition/subtraction units differ", node.source)
                unit = a.unit
            elif node.value in ("*", "/"):
                unit = unit_product(a.unit, b.unit, 1 if node.value == "*" else -1)
            else:
                if b.unit:
                    raise ModelError("UNIT_MISMATCH", "Exponent must be dimensionless", node.source)
                exponent = integer_literal(b)
                if a.unit and exponent is None:
                    raise ModelError("UNIT_MISMATCH", "Normalize a dimensionful base before a variable/noninteger power", node.source)
                unit = tuple((name, power*exponent) for name, power in a.unit if power*exponent) if exponent is not None else ()
        else:
            raise ModelError("UNSUPPORTED_AST_NODE", "Unrecognized AST structure", node.source, unsupported=True)
        return replace(node, children=children, unit=unit)

    return visit(tree, 1)


def references(tree: Node) -> tuple[tuple[str, int], ...]:
    """Stable source-order slot incidence, without inferred timing roles."""
    if tree.kind == "symbol":
        return ((str(tree.value), tree.offset),)
    return tuple(ref for child in tree.children for ref in references(child))


def evaluate_ast(tree: Node, symbols: Mapping[str, Symbol], slots: Mapping[tuple[str, int], float]) -> float:
    """Evaluate a validated integrand at explicit slots, not an expectation."""
    checked = validate_ast(tree, symbols)

    def visit(node: Node) -> float:
        try:
            if node.kind == "number":
                value = node.value
            elif node.kind == "innovation_zero":
                value = 0.0
            elif node.kind == "symbol":
                key = (str(node.value), node.offset)
                if key not in slots:
                    raise ModelError("MISSING_SLOT", f"Missing slot {key}", node.source, symbol=str(node.value))
                value = slots[key]
            elif node.kind == "unary":
                argument = visit(node.children[0])
                value = argument if node.value == "+" else -argument
            elif node.kind == "call":
                argument = visit(node.children[0])
                value = math.exp(argument) if node.value == "exp" else math.log(argument)
            else:
                a, b = (visit(child) for child in node.children)
                if node.value == "+":
                    value = a+b
                elif node.value == "-":
                    value = a-b
                elif node.value == "*":
                    value = a*b
                elif node.value == "/":
                    value = a/b
                else:
                    exponent = integer_literal(node.children[1])
                    if (exponent is None and a <= 0) or (a == 0 and b <= 0):
                        raise ValueError("Power is outside its open real domain")
                    value = a**b
            return real(value, node.source, symbol=str(node.value) if node.kind == "symbol" else None)
        except ModelError as exc:
            d = exc.diagnostic
            if d.start is not None:
                raise
            raise ModelError(d.code, d.message, d.source or node.source, start=node.start, end=node.end, symbol=d.symbol, unsupported=d.unsupported) from exc
        except (ValueError, ZeroDivisionError) as exc:
            raise ModelError("EXPRESSION_DOMAIN", str(exc), node.source, start=node.start, end=node.end) from exc
        except OverflowError as exc:
            raise ModelError("INF_VALUE", "Expression overflow", node.source, start=node.start, end=node.end) from exc
        except RuntimeError as exc:
            raise ModelError("EQUATION_EVALUATION_EXCEPTION", str(exc), node.source, start=node.start, end=node.end) from exc

    return visit(checked)

