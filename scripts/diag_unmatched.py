"""Show which Surah 113 segments stayed unmatched across qaris."""
import json
import sys
from pathlib import Path

surah = sys.argv[1] if len(sys.argv) > 1 else "113"

from collections import Counter

unmatched = Counter()
for js in sorted(Path("output").glob(f"*/surah_{surah:0>3}*reference_segments.json")):
    d = json.load(open(js, encoding="utf-8"))
    qari = d["qari"]["name"]
    bad = [s for s in d["segments"] if not s.get("matched")]
    if bad:
        print(f"{qari}: {len(d['segments'])}/{len(d['segments'])} segments, unmatched ids:", [s["index"] for s in bad])
        for s in bad:
            unmatched[(s["index"], s["transcription"])] += 1

print("\n== Unmatched phrase frequency ==")
for (i, txt), n in sorted(unmatched.items()):
    print(f"  id={i:>2}  {txt}  -> unmatched in {n} qaris")