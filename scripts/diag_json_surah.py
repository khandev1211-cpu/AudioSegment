"""Inspect a generated reference-segments JSON (any surah/qari)."""
import json
import sys

p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
print("qari  :", d["qari"])
print("surah :", d["surah"]["number"], d["surah"]["name_roman"], "|", d["surah"]["name_ar"])
print("model :", d["model"]["engine"], d["model"]["device"], d["model"]["compute_type"])
print("summary:", d["summary"])
print()
print(f"{'idx':<5}{'start':<9}{'end':<9}{'dur':<9}{'ayah':<12}{'f0_med':<8}matched  text")
for s in d["segments"]:
    pm = s["pitch"]["statistics_hz"]["median"] if s.get("pitch") and s["pitch"].get("statistics_hz") else None
    if s.get("start_sec") is None:
        print(f"{s['index']:<5}{'--':<9}{'--':<9}{'--':<9}{str(s['surah_ayah']):<12}{'--':<8}{str(s['matched']):<8} {s['transcription']}")
    else:
        print(f"{s['index']:<5}{s['start_sec']:<9.3f}{s['end_sec']:<9.3f}{s['duration_sec']:<9.3f}{str(s['surah_ayah']):<12}{str(pm):<8}{str(s['matched']):<8} {s['transcription']}")
print()
print("global word count:", len(d["global_words"]))