"""Verify structure of one reference-segments JSON in detail."""
import json

d = json.load(open(r"output/muhammad-siddiq-al-minshawi/surah_114_an-nas__reference_segments.json", encoding="utf-8"))

print("TOP KEYS:", list(d.keys()))
print("\nqari:", d["qari"])
print("audio_file:", {k: v for k, v in d["audio_file"].items() if k != "path"})
print("model:", d["model"])
print("segmentation:", d["segmentation"])
print("summary:", d["summary"])

s = d["segments"][0]
print("\nSEG 1 keys:", list(s.keys()))
print("  index/ayah/matched:", s["index"], s["surah_ayah"], s["matched"])
print("  audio_segment_file:", s["audio_segment_file"])
print("  transcription:", s["transcription"])
print("  words:", s["words"])
print("  pitch keys:", list(s["pitch"].keys()))
print("  pitch.stats_hz:", s["pitch"]["statistics_hz"])
print("  pitch.voiced_frames:", len(s["pitch"]["contour_full"]["f0_hz"]),
      "| voiced:", sum(1 for x in s["pitch"]["contour_full"]["f0_hz"] if x is not None))
print("  audio_info rms:", s["audio_info"]["rms"])
print("  audio_info spectral:", s["audio_info"]["spectral"])

print("\nglobal_words count:", len(d["global_words"]))
print("first 3:", d["global_words"][:3])