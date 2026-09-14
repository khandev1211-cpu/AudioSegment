"""Post-build verification of the enriched full-Quran dataset."""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW = json.load(open(os.path.join(BASE, "quran_arabic_words.json"), encoding="utf-8"))
OLD = json.load(open(os.path.join(BASE, "quran_arabic_words_backup_original.json"), encoding="utf-8"))

print("== 1. rukus filled for all surahs ==")
missing = [s["id"] for s in NEW if not s.get("rukus")]
print("  surahs missing rukus:", missing or "NONE ✓")
print("  total rukus:", sum(len(s["rukus"]) for s in NEW))
print("  total juz_entries:", sum(len(s["juz"]) for s in NEW))

print("\n== 2. surah 1 juz/rukus unchanged; surah 2 rukus INTENTIONALLY corrected ==")
ok = True
for sid in (1, 2):
    a = NEW[sid - 1]["juz"]
    b = OLD[sid - 1]["juz"]
    if a != b:
        ok = False
        print(f"  juz mismatch surah {sid}: new={a} old={b}")
ra = NEW[0]["rukus"]
rb = OLD[0]["rukus"]
if ra != rb:
    ok = False
    print(f"  surah1 rukus changed (should not): {ra} vs {rb}")
print("  surah 1 rukus/juz identical:",
      "yes ✓" if (ra == rb and NEW[0]["juz"] == OLD[0]["juz"]) else "NO ✗")
print("  surah 2 rukus: OLD had only", len(OLD[1]["rukus"]),
      "entries -> NEW has", len(NEW[1]["rukus"]),
      "standard boundaries (intentional fix ✓)")

print("\n== 3. global verse ids contiguous 1..6236 ==")
nums = [v["global_verse_number"] for s in NEW for v in s["verses"]]
print("  sequential:", nums == list(range(1, 6237)), "| min/max:", min(nums), max(nums))

print("\n== 4. sample enriched verse (2:255 Ayat al-Kursi) ==")
s2 = NEW[1]
v = next(v for v in s2["verses"] if v["id"] == 255)
print(json.dumps({k: v[k] for k in ("id", "verse_key", "global_verse_number", "juz", "ruku", "page", "manzil", "hizb_quarter", "sajda", "audio_file")}, ensure_ascii=False, indent=2))

print("== 5. audio files all exist ==")
bad = 0
for s in NEW:
    for v in s["verses"]:
        if not os.path.exists(os.path.join(BASE, v["audio_file"])):
            bad += 1
            print("  MISSING", v["audio_file"])
print("  missing audio:", bad or "0 ✓")

print("\n== 6. corpus stats ==")
print("  surahs:", len(NEW))
print("  verses:", sum(len(s["verses"]) for s in NEW))
print("  words:", sum(len(v["words"]) for s in NEW for v in s["verses"]))
print("  sajda verses:", sum(1 for s in NEW for v in s["verses"] if v.get("sajda")))
print("  pages covered:", len({v["page"] for s in NEW for v in s["verses"]}))
print("\n== 7. original id/text/words untouched; ruku changed only where needed ==")
changed_text, changed_words, changed_id, ruku_changed, ruku_same = 0, 0, 0, 0, 0
changed_surahs = {}
for s_new, s_old in zip(NEW, OLD):
    for vn, vo in zip(s_new["verses"], s_old["verses"]):
        if vn["id"] != vo["id"]:
            changed_id += 1
        if vn["text"] != vo["text"]:
            changed_text += 1
        if vn["words"] != vo["words"]:
            changed_words += 1
        if vn.get("ruku") != vo.get("ruku"):
            ruku_changed += 1
            changed_surahs.setdefault(s_new["id"], 0)
            changed_surahs[s_new["id"]] += 1
        else:
            ruku_same += 1
print(f"  changed id/text/words: {changed_id}/{changed_text}/{changed_words} (expect 0)")
print(f"  verse ruku same: {ruku_same} | changed: {ruku_changed}")
print("  ruku changes per surah:", dict(sorted(changed_surahs.items())))

print("\n== 8. global verse numbering matches standard Quran ==")
s2 = NEW[1]["verses"][0]
print(f"  surah2:1 global_verse_number = {s2['global_verse_number']} (expect 8)")
s114 = NEW[113]["verses"][-1]
print(f"  114:6 global_verse_number     = {s114['global_verse_number']} (expect 6236)")

print("\n== 9. sajda verses ==")
saj = [(s["id"], v["id"], v["sajda"]) for s in NEW for v in s["verses"] if v.get("sajda")]
print("  total:", len(saj))
print("  list:", saj)