"""Verify auto (high-confidence) person surfaces against zh.wikipedia + Wikidata.

For each auto canonical_name:
  1. zh.wikipedia pageprops -> wikibase_item (follows redirects, e.g. 卢龙 -> 卢龙县)
  2. Wikidata P31 (instance-of) for that entity
  3. classify: human (Q5) -> keep; disambiguation -> keep (uncertain);
     no article / no qid -> keep (can't disprove); a PLACE or other non-person
     entity -> DROP (false positive like 卢龙=county, 陆梁=region).

Writes wiki_verdicts.json (full) + wiki_nonperson.json (the drop list).
Rerunnable; re-reads people.json each time.
"""
import json, pathlib, time, urllib.parse, urllib.request
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
PDIR = HERE.parents[1] / "web" / "public" / "text" / "persons"
UA = "ZizhiTongjianReader/1.0 (person-NER verification; maintainer contact)"
Q_HUMAN = "Q5"
Q_DISAMBIG = "Q4167410"

PLACE_KW_EN = ("county", "city", "province", "prefecture", "town", "village",
               "region", "district", "mountain", "river", "lake", "commandery",
               "administrative", "geographic", "settlement", "capital",
               "township", "locality", "place in", "ancient city", "state of",
               "fortress", "palace", "temple")
PLACE_KW_ZH = ("县", "市", "省", "地区", "地名", "行政区", "山", "河", "湖", "乡",
               "镇", "古城", "自治", "地方", "流域", "关隘", "宫殿", "要塞", "城池")


def _get(url):
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", 0) or 0) or (5 * (attempt + 1))
                print("    429 backoff", wait, "s")
                time.sleep(wait)
                continue
            if attempt == 5:
                raise
            time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def wiki_qids(surfaces):
    out = {}
    base = ("https://zh.wikipedia.org/w/api.php?action=query&format=json"
            "&redirects=1&prop=pageprops&ppprop=wikibase_item&titles=")
    done = 0
    for batch in chunks(surfaces, 50):
        data = _get(base + urllib.parse.quote("|".join(batch)))
        q = data.get("query", {})
        norm = {x["from"]: x["to"] for x in q.get("normalized", [])}
        redir = {x["from"]: x["to"] for x in q.get("redirects", [])}
        title2qid = {}
        for pg in q.get("pages", {}).values():
            qid = pg.get("pageprops", {}).get("wikibase_item")
            if qid:
                title2qid[pg["title"]] = qid
        for s in batch:
            t = norm.get(s, s)
            t = redir.get(t, t)
            out[s] = title2qid.get(t)
        done += len(batch)
        if done % 500 == 0 or done >= len(surfaces):
            print("  wiki", done, "/", len(surfaces))
        time.sleep(0.5)
    return out


def wd_p31(qids):
    out = {}
    base = ("https://www.wikidata.org/w/api.php?action=wbgetentities"
            "&format=json&props=claims&ids=")
    for batch in chunks(list(qids), 50):
        data = _get(base + "|".join(batch))
        for qid, ent in data.get("entities", {}).items():
            tgts = []
            for cl in ent.get("claims", {}).get("P31", []):
                try:
                    tgts.append(cl["mainsnak"]["datavalue"]["value"]["id"])
                except Exception:
                    pass
            out[qid] = tgts
        time.sleep(0.4)
    return out


def wd_classify_targets(target_qids):
    out = {}
    base = ("https://www.wikidata.org/w/api.php?action=wbgetentities"
            "&format=json&props=descriptions|labels&languages=en|zh&ids=")
    for batch in chunks(list(target_qids), 50):
        data = _get(base + "|".join(batch))
        for qid, ent in data.get("entities", {}).items():
            if qid == Q_HUMAN:
                out[qid] = "person"
                continue
            desc_en = ent.get("descriptions", {}).get("en", {}).get("value", "").lower()
            label_en = ent.get("labels", {}).get("en", {}).get("value", "").lower()
            desc_zh = ent.get("descriptions", {}).get("zh", {}).get("value", "")
            label_zh = ent.get("labels", {}).get("zh", {}).get("value", "")
            blob_en = desc_en + " " + label_en
            blob_zh = desc_zh + label_zh
            if any(k in blob_en for k in PLACE_KW_EN) or any(k in blob_zh for k in PLACE_KW_ZH):
                out[qid] = "place"
            else:
                out[qid] = "other"
        time.sleep(0.4)
    return out


def main():
    people = json.load(open(PDIR / "people.json", encoding="utf-8"))["people"]
    autos = sorted({p["canonical_name"] for p in people if p.get("confidence") == "high"})
    print("auto surfaces to verify:", len(autos))

    s2qid = wiki_qids(autos)
    qids = sorted({q for q in s2qid.values() if q})
    print("distinct wikidata entities:", len(qids))

    qid2p31 = wd_p31(qids)
    targets = sorted({t for lst in qid2p31.values() for t in lst})
    print("distinct P31 targets:", len(targets))
    tclass = wd_classify_targets(targets)

    verdicts = {}
    for s in autos:
        qid = s2qid.get(s)
        if not qid:
            verdicts[s] = {"qid": None, "verdict": "none", "p31": []}
            continue
        p31 = qid2p31.get(qid, [])
        if Q_HUMAN in p31:
            v = "person"
        elif Q_DISAMBIG in p31:
            v = "disambig"
        elif any(tclass.get(t) == "place" for t in p31):
            v = "place"
        elif p31:
            v = "other"
        else:
            v = "none"
        verdicts[s] = {"qid": qid, "verdict": v,
                       "p31": [[t, tclass.get(t)] for t in p31]}

    drop = sorted(s for s, v in verdicts.items() if v["verdict"] in ("place", "other"))
    json.dump(verdicts, open(HERE / "wiki_verdicts.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(drop, open(HERE / "wiki_nonperson.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("verdicts:", dict(Counter(v["verdict"] for v in verdicts.values())))
    print("DROP (place+other):", len(drop))


if __name__ == "__main__":
    main()
