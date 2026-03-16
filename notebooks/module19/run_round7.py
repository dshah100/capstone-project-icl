"""One-shot script to add Round 6 and generate Round 7."""
import sys
import json
from pathlib import Path

# Add parent for module17 import
sys.path.insert(0, str(Path(__file__).parent.parent))

# Round 6 data
ROUND6 = {
    1: {"query": [0.55, 0.5], "result": 2.133315e-16},
    2: {"query": [0.895432, 0.465123, 0.405678], "result": 0.354561},
    3: {"query": [0.465890, 0.568234, 0.545678, 0.587654], "result": 0.025950},
    4: {"query": [0.456789, 0.492345, 0.445678, 0.534567, 0.518234], "result": 3.061625},
    5: {"query": [0.518345, 0.556789, 0.492345, 0.545678, 0.512345, 0.441234], "result": 15.432354},
    6: {"query": [0.492345, 0.408765, 0.452345, 0.536789, 0.468765, 0.567890, 0.518234], "result": 0.904261},
    7: {"query": [0.534567, 0.176543, 0.028765, 0.342345, 0.965432, 0.523456, 0.854321, 0.089876], "result": 0.002902},
    8: {"query": [0.298765, 0.487654, 0.745678, 0.905432, 0.408765, 0.912345, 0.512345, 0.789012, 0.883456], "result": 7.653867},
}

path = Path(__file__).parent.parent / "module17" / "query_history.json"
with open(path) as f:
    h = json.load(f)

for i in range(1, 9):
    k = f"function_{i}"
    if 6 not in [o["round"] for o in h[k]]:
        h[k].append({"round": 6, "query": ROUND6[i]["query"], "result": ROUND6[i]["result"]})

with open(path, "w") as f:
    json.dump(h, f, indent=2)

print("Added Round 6 to query_history.json")

# Generate Round 7
from module17.bbo_capstone_framework import BBOCapstoneManager

m = BBOCapstoneManager()
m.load_query_history_from_file(str(path))
q = m.generate_round_queries(round_num=7, acquisition="ei", xi=0.008, kappa=2.0)

out = Path(__file__).parent / "round_7_portal_submission.txt"
with open(out, "w") as f:
    f.write("ROUND 7 QUERIES - 16 Data Points\n")
    f.write("="*60 + "\n\n")
    for fid, qstr in q.items():
        f.write(f"Function {fid}:\n{qstr}\n\n")

print(f"Saved Round 7 queries to {out}")
for fid, qstr in sorted(q.items()):
    print(f"  Function {fid}: {qstr}")
