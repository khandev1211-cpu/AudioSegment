"""Check duration + matched phrase id list for every qari."""
import json
from pathlib import Path

import soundfile as sf

base = Path("output")
for js in sorted(base.glob("*/*reference_segments.json")):
    d = json.load(open(js, encoding="utf-8"))
    qari = d["qari"]["name"]
    dur = d["audio_file"]["duration_sec"]
    matched = [s["index"] for s in d["segments"] if s.get("matched")]
    txt = " ".join(s["transcription"][:12] for s in d["segments"] if s.get("matched"))
    print(f"{qari:<34} dur={dur:6.1f}s  matched={matched}")