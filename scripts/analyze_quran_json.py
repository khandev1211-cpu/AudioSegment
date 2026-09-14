"""Deep analysis of quran_arabic_words.json — checks data quality / completeness."""
import json
import collections
import re
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE, "quran_arabic_words.json")

data = json.load(open(JSON_PATH, encoding="utf-8"))

print(f"total surahs: {len(data)}")

# 1. Duplicate / sequential surah ids
ids = [s["id"] for s in data]
print("duplicate surah ids:", [i for i, c in collections.Counter(ids).items() if c > 1])
print("surah ids sequential:", ids == list(range(1, 115)))

# 2. Verse count mismatches
total_verses = 0
mism = []
for s in data:
    if s["total_verses"] != len(s["verses"]):
        mism.append((s["id"], s["total_verses"], len(s["verses"])))
    total_verses += len(s["verses"])
print("surah verse-count mismatches:", mism)
print("total verses:", total_verses)

# 3. Verse id sequence / duplicates within each surah
bad_seq = []
for s in data:
    vids = [v["id"] for v in s["verses"]]
    if vids != list(range(1, len(vids) + 1)):
        bad_seq.append((s["id"], vids[:10]))
print("verse-id sequence issues:", bad_seq[:5] if bad_seq else "None")

# 4. words vs text split check (approx: split on spaces)
word_split_mismatch = []
for s in data:
    for v in s["verses"]:
        n = len(v["words"])
        # some verses keep bismillah + text merged; count split by spaces but strip empty tokens
        split_n = len([x for x in re.split(r"\s+", v["text"]) if x.strip()])
        if n != split_n:
            word_split_mismatch.append((s["id"], v["id"], n, split_n, v["text"][:40]))
print("words-vs-textspace mismatches:", len(word_split_mismatch))
for m in word_split_mismatch[:8]:
    print("   ", m)

# 5. Empty words lists / empty texts
empty = []
for s in data:
    for v in s["verses"]:
        if not v.get("words") or not v.get("text", "").strip():
            empty.append((s["id"], v["id"]))
print("verses with empty text/words:", empty[:10] if empty else "None")

# 6. juz / rukus coverage summary
juzless = [s["id"] for s in data if not s.get("juz")]
rukless = [s["id"] for s in data if not s.get("rukus")]
print("surahs without juz:", juzless or "None")
print("surahs without rukus:", rukless or "None")

juz_indexes = collections.Counter()
for s in data:
    for j in s.get("juz", []):
        juz_indexes[j.get("index")] += 1
print("juz indexes covered:", sorted(juz_indexes.keys(), key=lambda x: int(x)))

# 7. ruku indexes coherence
n_ruku_total = 0
ruku_errs = []
for s in data:
    rukus = s.get("rukus", [])
    n_ruku_total += len(rukus)
    idxs = []
    for r in rukus:
        try:
            idxs.append(int(r["index"]))
        except Exception:
            ruku_errs.append((s["id"], r))
    if idxs and idxs != list(range(1, len(idxs) + 1)):
        ruku_errs.append((s["id"], "non-sequential", idxs[:5]))
print("total rukus:", n_ruku_total, "| ruku errors:", ruku_errs[:5] or "None")

# 8. type field check
print("surah types:", collections.Counter(s.get("type") for s in data))

# 9. basmalah in verse 1?
import unicodedata
def strip_tashkeel(t):
    return "".join(c for c in t if not unicodedata.combining(c) and c not in "۞۩")
for sid in (1, 2, 113, 114):
    s = data[sid - 1]
    v1 = strip_tashkeel(s["verses"][0]["text"]).replace(" ", "")
    proto = strip_tashkeel("بسم الله الرحمن الرحيم").replace(" ", "")
    print(f"surah {sid} verse1 contains basmalah-ish: {proto in v1}", v1[:30])

# 10. audio files available for alafasy full quran
AQ = os.path.join(BASE, "fullQuran", "alafasy_quran_complete")
have, missing = 0, []
if os.path.isdir(AQ):
    for s in data:
        folder = os.path.join(AQ, f"{s['id']:03d}")
        for v in s["verses"]:
            fname = os.path.join(folder, f"{s['id']:03d}{v['id']:03d}.mp3")
            if os.path.exists(fname):
                have += 1
            else:
                missing.append(fname)
else:
    print("audio dir not found")
print(f"alafasy verse-audio available: {have}/{total_verses} | missing files: {len(missing)}")
if missing[:5]:
    print("   missing sample:", missing[:5])