"""Live status of the word-clips build (safe to run while the job is active)."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
BASE = Path(__file__).resolve().parent.parent
WC = BASE / "dataset" / "word_clips"

TOTAL_VERSES = 6236
TOTAL_WORDS = 77433

rows = []
if (WC / "manifest_tmp.jsonl").exists():
    rows = [json.loads(l) for l in (WC / "manifest_tmp.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

done = len(json.load(open(WC / "progress.json", encoding="utf-8"))) if (WC / "progress.json").exists() else 0
matched = sum(1 for r in rows if r.get("matched"))
clipped = sum(1 for r in rows if r.get("clip_file"))
dur = sum((r.get("duration_sec") or 0) for r in rows)

print(f"verses done       : {done}/{TOTAL_VERSES} ({100*done/TOTAL_VERSES:.1f}%)")
print(f"word rows cached  : {len(rows)}/{TOTAL_WORDS}")
print(f"matched words     : {matched} ({100*matched/max(1,len(rows)):.1f}% of rows)")
print(f"clips written     : {clipped}")
print(f"audio cut so far  : {dur/3600:.2f} h")
wavs = sum(1 for _ in (WC).rglob("*.wav")) if WC.exists() else 0
print(f"wav files on disk : {wavs}")

if (WC / "run.log").exists() and (WC / "run.log").stat().st_size:
    tail = "\n".join((WC / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()[-3:])
    print("--- run.log tail ---")
    print(tail)

err = WC / "run_err.log"
if err.exists() and err.stat().st_size:
    print("--- run_err.log tail ---")
    print("\n".join(err.read_text(encoding="utf-8", errors="replace").splitlines()[-3:]))
else:
    print("errors: none")