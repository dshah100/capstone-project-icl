"""
Add Round 7 results to query_history.json.
Run this AFTER you receive Round 7 results from the capstone portal.
Update the ROUND7 dict below with your actual portal results.
"""
import json
from pathlib import Path

# UPDATE THESE with your actual Round 7 results from the portal
ROUND7 = {
    1: {"query": [0.550123, 0.499876], "result": None},  # Replace None with portal result
    2: {"query": [0.918234, 0.452341, 0.391567], "result": None},
    3: {"query": [0.478234, 0.579123, 0.547890, 0.586543], "result": None},
    4: {"query": [0.451234, 0.498765, 0.448901, 0.532109, 0.515678], "result": None},
    5: {"query": [0.524567, 0.561234, 0.491234, 0.552345, 0.508901, 0.442345], "result": None},
    6: {"query": [0.489012, 0.401234, 0.448901, 0.538901, 0.464567, 0.572345, 0.519876], "result": None},
    7: {"query": [0.521234, 0.158901, 0.018765, 0.331234, 0.984567, 0.512345, 0.861234, 0.072345], "result": None},
    8: {"query": [0.291234, 0.472345, 0.751234, 0.908901, 0.401234, 0.918901, 0.501234, 0.791234, 0.891234], "result": None},
}

path = Path(__file__).parent.parent / "module17" / "query_history.json"
with open(path) as f:
    h = json.load(f)

# Skip _metadata when iterating
for i in range(1, 9):
    k = f"function_{i}"
    if k not in h or not isinstance(h[k], list):
        continue
    if 7 not in [o["round"] for o in h[k]]:
        if ROUND7[i]["result"] is not None:
            h[k].append({"round": 7, "query": ROUND7[i]["query"], "result": ROUND7[i]["result"]})
            print(f"Added Round 7 for Function {i}")
        else:
            print(f"SKIP Function {i}: Update ROUND7[{i}]['result'] with portal result first")

with open(path, "w") as f:
    json.dump(h, f, indent=2)

print("\nDone. Run round8_generator.py to generate Round 8 queries.")
