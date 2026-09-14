"""
Build a complete/enriched Quran dataset from quran_arabic_words.json.

What it does
------------
1. Loads the existing quran_arabic_words.json (114 surahs, 6236 verses, 77k words).
2. Fetches authoritative per-ayah metadata (juz, ruku, manzil, page, hizbQuarter, sajda)
   from the alquran.cloud API for ALL 114 surahs (cached under scripts/_cache/).
3. Enriches every ayah with:
     - "verse_key"           e.g. "2:255"
     - "global_verse_number" absolute count across the Quran
     - "juz", "manzil", "page", "hizb_quarter", "sajda"  (authoritative numbers)
     - "audio_file"          relative path to the Alafasy verse mp3
4. Fills the missing surah-level "rukus" arrays (only surah 1 & 2 had them)
   and rebuilds surah-level "juz" arrays from the verse-level data.
5. Adds surah-level "english_name", "word_count", "juz_count", "ruku_count".

Output
------
Overwrites  quran_arabic_words.json  with an enriched, self-consistent dataset.
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE, "quran_arabic_words.json")
CACHE_DIR = os.path.join(BASE, "scripts", "_cache")
CACHE_FILE = os.path.join(CACHE_DIR, "alquran_surah_meta.json")

USER_AGENT = "QuranDatasetBuilder/1.0"
API = "https://api.alquran.cloud/v1/surah/{n}"


# --------------------------------------------------------------------------- #
# 1. meta fetch (cached, offline-safe on re-runs)
# --------------------------------------------------------------------------- #
def fetch_meta(force=False):
    """Return {surah_number: {ayah_numberInSurah: meta}} for all 114 surahs."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = {}
    if os.path.exists(CACHE_FILE) and not force:
        with open(CACHE_FILE, encoding="utf-8") as fh:
            cache = json.load(fh)
    missing = [n for n in range(1, 115) if str(n) not in cache]
    if missing:
        print(f"fetching meta for {len(missing)} surahs ...")
        for n in missing:
            last_err = None
            for attempt in range(4):
                try:
                    req = urllib.request.Request(API.format(n=n),
                                                 headers={"User-Agent": USER_AGENT})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        payload = json.loads(r.read().decode("utf-8"))
                    if payload.get("code") != 200:
                        raise RuntimeError(f"API code {payload.get('code')}")
                    meta = {}
                    for a in payload["data"]["ayahs"]:
                        meta[str(a["numberInSurah"])] = {
                            "global": int(a["number"]),
                            "juz": int(a["juz"]),
                            "manzil": int(a["manzil"]),
                            "page": int(a["page"]),
                            "ruku": int(a["ruku"]),
                            "hizb_quarter": int(a["hizbQuarter"]),
                            "sajda": a.get("sajda") or None,
                        }
                    meta["_meta"] = {
                        "english_name": payload["data"].get("englishName"),
                        "english_name_translation": payload["data"].get("englishNameTranslation"),
                        "revelation_type": payload["data"].get("revelationType"),
                    }
                    cache[str(n)] = meta
                    print(f"  surah {n:3d}: {len(meta)-1} ayahs  "
                          f"juz {min(m['juz'] for m in meta.values() if 'juz' in m)}-"
                          f"{max(m['juz'] for m in meta.values() if 'juz' in m)}  "
                          f"ruku {min(m['ruku'] for m in meta.values() if 'ruku' in m)}-"
                          f"{max(m['ruku'] for m in meta.values() if 'ruku' in m)}")
                    with open(CACHE_FILE, "w", encoding="utf-8") as fh:
                        json.dump(cache, fh, ensure_ascii=False, indent=1)
                    time.sleep(0.2)
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    time.sleep(2.0 * (attempt + 1))
            else:
                sys.exit(f"FATAL: failed to fetch surah {n}: {last_err}")
    else:
        print(f"meta cache complete ({len(cache)} surahs found).")
    return cache


# --------------------------------------------------------------------------- #
# 2. enrich the dataset
# --------------------------------------------------------------------------- #
def build():
    data = json.load(open(JSON_PATH, encoding="utf-8"))
    if len(data) != 114:
        sys.exit(f"unexpected surah count in dataset: {len(data)}")
    meta = fetch_meta()

    global_verse = 0
    words_total = 0
    total_rukus = 0
    for s in data:
        n = s["id"]
        s_meta = meta[str(n)]
        if len(s_meta) - 1 != len(s["verses"]):
            sys.exit(f"surah {n}: api {len(s_meta)-1} ayahs vs json {len(s['verses'])}")

        verses = s["verses"]
        ruku_min = min(m["ruku"] for m in s_meta.values() if "ruku" in m)
        s["english_name"] = s_meta["_meta"]["english_name"]
        s["revelation_type"] = s_meta["_meta"]["revelation_type"]

        # --- enrich each verse ---
        for v in verses:
            m = s_meta[str(v["id"])]
            global_verse += 1
            v["verse_key"] = f"{n}:{v['id']}"
            v["global_verse_number"] = global_verse
            v["juz"] = m["juz"]
            v["manzil"] = m["manzil"]
            v["page"] = m["page"]
            v["hizb_quarter"] = m["hizb_quarter"]
            v["sajda"] = m["sajda"]
            # per-surah ruku index (consistent with existing field semantics)
            v["ruku"] = m["ruku"] - ruku_min + 1
            v["audio_file"] = os.path.join(
                "fullQuran", "alafasy_quran_complete",
                f"{n:03d}", f"{n:03d}{v['id']:03d}.mp3",
            ).replace("\\", "/")
            full = os.path.join(BASE, v["audio_file"])
            if not os.path.exists(full):
                sys.exit(f"missing audio: {full}")

        # --- rebuild surah-level juz ---
        juz_entries = []
        for jn in sorted({m["juz"] for m in s_meta.values() if "juz" in m}):
            ids = [v["id"] for v in verses if v["juz"] == jn]
            juz_entries.append({
                "index": f"{jn:02d}",
                "verse": {"start": f"verse_{min(ids)}", "end": f"verse_{max(ids)}"},
            })
        s["juz"] = juz_entries

        # --- rebuild surah-level rukus (fills the missing ones) ---
        ruku_entries = []
        for rn in sorted({v["ruku"] for v in verses}):
            ids = [v["id"] for v in verses if v["ruku"] == rn]
            ruku_entries.append({
                "index": str(rn),
                "verse": {"start": str(min(ids)), "end": str(max(ids))},
            })
        s["rukus"] = ruku_entries

        s["juz_count"] = len(juz_entries)
        s["ruku_count"] = len(ruku_entries)
        s["word_count"] = sum(len(v["words"]) for v in verses)
        words_total += s["word_count"]
        total_rukus += len(ruku_entries)

    with open(JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print("\nDONE.")
    print(f"  surahs          : {len(data)}")
    print(f"  verses          : {global_verse}")
    print(f"  words           : {words_total}")
    print(f"  rukus (total)   : {total_rukus}")


if __name__ == "__main__":
    build()