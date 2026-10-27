from pathlib import Path
from econforge.model import parse_model
from econforge.residuals import evaluate_residuals
m = parse_model(Path(__file__).with_name("residual_model.json").read_text())
r = evaluate_residuals(m, {"x": 2.0})
print(r.to_dict())
assert r.status.value == "PASS" and r.residual_vector == (0.0,)
