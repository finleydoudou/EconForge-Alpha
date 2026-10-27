"""Installed-package check against frozen analytic roots. No test-module imports."""
import json
from pathlib import Path
from econforge.model import parse_model
from econforge.steady_state import solve_steady_state
root = Path(__file__).resolve().parent
results = []
for case in json.loads((root / "cases.json").read_text()):
    m = parse_model((root / case["model_file"]).read_text())
    r = solve_steady_state(m, initial_guess=case["initial_guess"])
    target, scales = case["target"], case["scales"]
    errors = None if r.solution_candidate is None else [abs(a-b)/d for a,b,d in zip(r.solution_candidate,target,scales)]
    bounds = [case["reference_atol"]+case["reference_rtol"]*abs(b/d) for b,d in zip(target,scales)]
    passed = r.status.value == "PASS" and errors is not None and all(e<=b for e,b in zip(errors,bounds))
    results.append({"case_id":case["case_id"],"status":"PASS" if passed else "FAIL","normalized_errors":errors,"bounds":bounds,"actual_result":r.to_dict(),"oracle_origin":case["oracle_origin"]})
print(json.dumps(results,indent=2,allow_nan=False))
assert all(r["status"] == "PASS" for r in results)
