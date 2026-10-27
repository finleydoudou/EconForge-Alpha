"""Run the frozen RBC-A_primary example with an explicit supplied start."""
import json
from pathlib import Path
from econforge.model import parse_model
from econforge.steady_state import solve_steady_state
root = Path(__file__).resolve().parent
case = next(c for c in json.loads((root / "cases.json").read_text()) if c["case_id"] == 'RBC-A_primary')
model = parse_model((root / case["model_file"]).read_text())
result = solve_steady_state(model, initial_guess=case["initial_guess"])
print(json.dumps({"case":case["case_id"],"initial_guess":case["initial_guess"],"result":result.to_dict()},indent=2))
if result.status.value != "PASS" or result.solution_candidate is None:
    raise RuntimeError("Steady-state check did not pass; inspect the structured result")
