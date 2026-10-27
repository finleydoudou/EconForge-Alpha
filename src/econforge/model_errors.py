"""Source-local input and numerical failures for the bounded model language."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    source: str = ""
    start: int | None = None
    end: int | None = None
    symbol: str | None = None
    unsupported: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ModelError(ValueError):
    """An explicit diagnostic; never a substituted residual."""

    def __init__(self, code: str, message: str, source: str = "", **details: object):
        self.diagnostic = Diagnostic(code, message, source, **details)
        super().__init__(f"{code}: {message}")


def real(value: object, source: str, *, symbol: str | None = None) -> float:
    """Accept plain finite real data only, without executing coercion hooks."""
    if type(value) is complex:
        raise ModelError("COMPLEX_VALUE", "Real value required", source, symbol=symbol)
    if type(value) not in (int, float):
        raise ModelError("INVALID_VALUE_TYPE", "Expected a plain int or float", source, symbol=symbol)
    try:
        number = float(value)
    except OverflowError as exc:
        raise ModelError("INF_VALUE", "Real conversion overflow", source, symbol=symbol) from exc
    if math.isnan(number):
        raise ModelError("NAN_VALUE", "NaN is not admissible", source, symbol=symbol)
    if not math.isfinite(number):
        raise ModelError("INF_VALUE", "Infinity is not admissible", source, symbol=symbol)
    return number

