"""Inspect the reference-segments JSON for minshawi."""
import json

d = json.load(open(r"output/muhammad-siddiq-al-minshawi/surah_114_an-nas__reference_segments.json", encoding="utf-8"))
print("qari  :", d["qari"])
print("surah :", d["surah"]["number"], d["surah"]["name_roman"])
print("summary:", d["summary"])
print()
print(f"{'idx':<5}{'start':<9}{'end':<9}{'dur':<9}{'text':<24}{'pitch_med'}")
for s in d["segments"]:
    pm = s["pitch"]["statistics_hz"]["median"] if s["pitch"] and s["pitch"]["statistics_hz"] else None
    print(f"{s['index']:<5}{s['start_sec']:<9.3f}{s['end_sec']:<9.3f}{s['duration_sec']:<9.3f}{s['transcription']:<24}{pm if pm else '-'}")
print()
print("global word count:", len(d["global_words"]))
for w in d["global_words"]:
    print(f"   {w['start_sec']:6.2f}-{w['end_sec']:6.2f}  {w['text']}")
print()
print("model:", d["model"]["engine"], d["model"]["device"], d["model"]["compute_type"])