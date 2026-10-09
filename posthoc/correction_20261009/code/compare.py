"""Compare stored results with the re-runs: stored vs orig (must match), orig vs new (only pilot entries may differ)."""
import json
import math

J = "/N/project/AiLab/jev"
A = f"{J}/review_pipeline/rounds/round_0001/revision/analysis"
R = f"{J}/relabel_20261009/rerun"
SKIP = {"runtime_seconds", "elapsed_seconds", "finished", "started", "runtime", "secs"}


def flat(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            if k in SKIP:
                continue
            out.update(flat(v, f"{p}/{k}"))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(flat(v, f"{p}[{i}]"))
    else:
        out[p] = o
    return out


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-12 * max(1, abs(a))
    return a == b


report = {}
for an in ["AN-0001-01", "AN-0001-02", "AN-0001-06", "AN-0001-10"]:
    st = flat(json.load(open(f"{A}/{an}/results.json")))
    og = flat(json.load(open(f"{R}/orig_{an}/results.json")))
    nw = flat(json.load(open(f"{R}/new_{an}/results.json")))
    d1 = [k for k in set(st) | set(og) if not same(st.get(k), og.get(k))]
    d2 = sorted(k for k in set(og) | set(nw) if not same(og.get(k), nw.get(k)))
    nonpilot = [k for k in d2 if "pilot" not in k.lower()]
    report[an] = {"stored_vs_orig_differences": sorted(d1)[:20], "n_stored_vs_orig": len(d1),
                  "n_orig_vs_new": len(d2), "orig_vs_new_outside_pilot": nonpilot[:20]}
    print(an, "| stored vs orig differences:", len(d1), sorted(d1)[:8], "| orig vs new:", len(d2), "| outside pilot keys:", nonpilot[:8])
json.dump(report, open(f"{R}/compare_report.json", "w"), indent=1)
