"""Measure the pre-filter on labelled conversations (offline, no API calls).

    python eval_prefilter.py            # summary
    python eval_prefilter.py -v         # every turn with reasons

Reports:
  recall          - risky turns sent to Chat Safety (most important; aim for 100%)
  false positives - safe turns sent to Chat Safety (cost only, not a safety failure)
  skipped         - share of all turns that never needed the expensive model
"""

import json
import sys
from pathlib import Path

from classifier import PreFilter, SessionState

verbose = "-v" in sys.argv
convs = json.loads((Path(__file__).parent / "data" / "eval_set.json").read_text())
pf = PreFilter()
if not pf.semantic_enabled:
    print(f"! embedding model not loaded ({pf.semantic_error}) - rules + drift only\n")

tp = fn = fp = tn = 0
latencies, misses = [], []
for conv in convs:
    state = SessionState()
    if verbose:
        print(f"\n== {conv['name']}")
    for text, label in conv["turns"]:
        d = pf.assess(text, state)
        latencies.append(d.latency_ms)
        got = d.route
        if label == "escalate":
            tp += got == "escalate"
            fn += got == "safe"
        else:
            fp += got == "escalate"
            tn += got == "safe"
        if got != label:
            misses.append((conv["name"], text, label, got, d.risk))
        if verbose:
            mark = "ok " if got == label else "XX "
            drift = " [drift]" if d.drift else ""
            print(f"  {mark}{got:8} {d.risk:.2f} {d.top_family or '-':24}{drift} | {text}")
            if d.reasons and got == "escalate":
                print("       " + "; ".join(d.reasons))
        # the agent also scores its own replies; here we have none, so nothing to add

total = tp + fn + fp + tn
print(f"\nTurns: {total}   (risky {tp + fn}, safe {fp + tn})")
print(f"Recall (risky caught):        {tp}/{tp + fn} = {tp / max(1, tp + fn):.0%}")
print(f"False positives (safe flagged): {fp}/{fp + tn} = {fp / max(1, fp + tn):.0%}")
print(f"Skipped Chat Safety:          {tn + fn}/{total} = {(tn + fn) / total:.0%}")
print(f"Pre-filter latency:           avg {sum(latencies) / len(latencies):.1f} ms, max {max(latencies):.1f} ms")
if misses:
    print("\nMismatches:")
    for name, text, label, got, risk in misses:
        print(f"  [{name}] expected {label}, got {got} ({risk:.2f}): {text}")
