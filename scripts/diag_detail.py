"""Detailed per-qari matched segment timing check."""
import json
from pathlib import Path

for js in sorted(Path("output").glob("*/*reference_segments.json")):
    d = json.load(open(js, encoding="utf-8"))
    qari = d["qari"]["name"]
    dur = d["summary"]["total_duration_sec"]
    parts = []
    for s in d["segments"]:
        if s.get("matched"):
            parts.append(f"{s['index']}({s['start_sec']:.1f}-{s['end_sec']:.1f})")
    print(f"{qari:<34} dur={dur:6.1f}s  {' '.join(parts)}")