"""Curate the final wiki non-person drop list from wiki_verdicts.json + descriptions.

Conservative: only drop surfaces whose Wikidata entity is UNAMBIGUOUSLY a
non-person concept. KEEP disambiguation pages (they contain real people), any
entity described as a ruler/official/etc., and empty-description entities
(can't disprove a person). PLACE-verdict surfaces are dropped wholesale except a
tiny allowlist of real historical persons whose name homographs a foreign place.
"""
import json, pathlib, time
import urllib.request, urllib.error

HERE = pathlib.Path(__file__).resolve().parent
v = json.load(open(HERE / "wiki_verdicts.json", encoding="utf-8"))
cand = {s: d for s, d in v.items() if d["verdict"] in ("place", "other")}

UA = "ZizhiTongjianReader/1.0 (verification)"


def get(url):
    for a in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(int(e.headers.get("Retry-After", 0) or 0) or 6 * (a + 1)); continue
            if a == 5: raise
            time.sleep(2 * (a + 1))
        except Exception:
            if a == 5: raise
            time.sleep(2 * (a + 1))


# real persons whose zh.wikipedia title resolves to a homographic place/dynasty
PLACE_KEEP = {"韦伦"}
OTHER_KEEP = {"桓楚", "梁龙", "酋龙", "胡才", "马元义", "耿国", "秦兴", "吴瑶",
              "马腹", "王吴", "李承裕", "刘展", "袁公"}

PERSON_KW = ("disambiguation", "ruler", "emperor", "queen", "leader", "official",
             "general", "poet", "king", "prince", "politician", "empress",
             "chancellor", "statesman", "military leaders",
             "人物", "君主", "皇帝", "将领", "丞相", "宰相", "诗人")

NONPERSON_KW = (
    "people", "nomadic", "tribe", "ethnic", "month of the chinese", "festival",
    "suicide", "put to death", "rank in", "poem", "state office", "chinese text",
    "history book", "book by", "book about", "work on", "classic", "varna",
    "peafowl", "bird", "child's child", "grandparent", "feudal state",
    "chinese state", "states in ancient", "constellation", "era name",
    "mythology", "symbols of chinese", "tree", "fruit", "moth", "dinosaur",
    "genus", "reptiles", "sauropod", "hairstyle", "domesticated", "herbivore",
    "annelid", "filament", "incense", "thunder", "weather", "storm", "language",
    "dialect", "family name", "nobility", "state office", "rebellion",
    "former country", "country after", "combat", "system of rules", "printing",
    "arthropod", "daughter of an emperor", "period of chinese history",
    "circuit in chinese", "varna",
    "民族", "游牧", "部落", "農曆月", "节日", "節日", "史書", "史书", "经典", "經典",
    "諸侯國", "年號", "神獸", "神兽", "神祇", "植物", "果实", "果實", "髮型", "哺乳",
    "信仰", "姓氏", "爵", "頭銜", "头衔", "起義", "起义", "历史国家", "歷史國家",
    "方言", "刑罰", "刑罚", "關係", "关系", "政權", "政权", "概念", "技術", "技术",
    "香料", "天气", "星", "诸侯国",
)

qids = sorted({d["qid"] for d in cand.values() if d["qid"]})
desc = {}
base = ("https://www.wikidata.org/w/api.php?action=wbgetentities&format=json"
        "&props=descriptions&languages=en|zh&ids=")
for i in range(0, len(qids), 50):
    data = get(base + "|".join(qids[i:i + 50]))
    for qid, ent in data.get("entities", {}).items():
        dz = ent.get("descriptions", {}).get("zh", {}).get("value", "")
        de = ent.get("descriptions", {}).get("en", {}).get("value", "")
        desc[qid] = (dz + " " + de).lower()
    time.sleep(0.5)

drop, keep = [], []
for s, d in cand.items():
    if d["verdict"] == "place":
        (keep if s in PLACE_KEEP else drop).append((s, "place", desc.get(d["qid"], "")))
        continue
    if s in OTHER_KEEP:
        keep.append((s, "other-allow", desc.get(d["qid"], ""))); continue
    blob = desc.get(d["qid"], "")
    if not blob.strip():
        keep.append((s, "other-empty", "")); continue
    if any(k in blob for k in PERSON_KW):
        keep.append((s, "other-person", blob)); continue
    if any(k in blob for k in NONPERSON_KW):
        drop.append((s, "other", blob)); continue
    keep.append((s, "other-unknown", blob))

drop_surfaces = sorted(s for s, _, _ in drop)
json.dump(drop_surfaces, open(HERE / "wiki_nonperson.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

lines = ["=== DROP (%d) ===" % len(drop_surfaces)]
for s, tag, b in sorted(drop):
    lines.append("%-5s %-6s %s" % (s, tag, b[:70]))
lines.append("\n=== KEEP other (%d shown) ===" % len([1 for s, t, _ in keep if t.startswith('other')]))
for s, tag, b in sorted(keep):
    if tag.startswith("other"):
        lines.append("%-5s %-14s %s" % (s, tag, b[:60]))
open(HERE / "_curate.txt", "w", encoding="utf-8").write("\n".join(lines))
print("DROP", len(drop_surfaces), "KEEP", len(keep))
