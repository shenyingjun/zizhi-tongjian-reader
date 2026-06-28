"""Auto-seed cast layer — scales person identity from 卷001–010 to all 294 卷.

The hand-curated cast (cast.py, confidence 'reviewed') stays authoritative for
editorially-important figures. This module *extends* coverage to the whole work
deterministically, seeded from each 卷's 白话导读 `key_people`:

  1. Hand entries keep their content; their `juans` are GROWN (contiguity-gated)
     to every 卷 where the guide names that person — this is how a recurring
     figure (刘邦, 项羽, …) accumulates appearances ACROSS batches under one
     stable id (the cross-batch reference mechanism).
  2. Names not owned by a hand entry become 'auto' people. A name appearing in
     non-contiguous 卷 (gap > GAP) is SPLIT into separate person instances, so a
     generic 2-char name reused centuries apart is never merged into one person.
  3. Generic titles / reused 庙号·谥号 (太后, 高祖, 文帝, …) and single chars are
     never used as match surfaces — they denote a role, not a unique referent.
  4. Per-卷 collision resolution: if a surface would resolve to >1 person in a
     卷, it is dropped there, so validate_persons' collision invariant holds.

Output consumed by build_persons.py:
  build_seed(hand_people, juans_allowed) -> (people_list, rules_by_juan)
    people_list   merged hand+auto cast (hand juans grown, auto split)
    rules_by_juan {juan: {surface: person_id}} collision-free match table
"""
from __future__ import annotations
import json, glob, collections, hashlib, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEXT = REPO / "web" / "public" / "text"
GUIDE = TEXT / "guide"

# Contiguity window: a name recurring across 卷 no more than GAP apart is the
# same person; a larger jump starts a new person instance (and, for hand
# entries, is simply not absorbed).
GAP = 8

# Surfaces that denote a ROLE or a reused honorific, never a unique person.
# Auto entries never emit these as match surfaces. (Hand entries are vetted and
# may use context-unique appellations; they are still collision-resolved.)
BANNED_SURF: set[str] = {
    # bare ranks / appellations
    "王", "公", "侯", "君", "上", "帝", "大王", "太上皇", "太子", "公子",
    "太后", "皇后", "皇太后", "太皇太后", "皇帝", "天子", "皇子", "皇孙",
    "世子", "王后", "国君", "大夫", "夫人", "公主", "太妃", "贵妃", "婕妤",
    "昭仪", "群臣", "百姓", "诸侯", "左右", "孝公", "惠王", "武王", "文王",
    "威王", "成侯",
    # offices
    "丞相", "相国", "大将军", "将军", "太尉", "司徒", "司空", "御史大夫",
    "刺史", "都督", "太守", "长史", "司马", "中郎将", "尚书", "侍中", "仆射",
    "单于", "可汗", "节度使", "观察使",
    # reused 庙号 / 谥号 (resolve only with dynasty context)
    "高祖", "太祖", "世祖", "高宗", "太宗", "中宗", "睿宗", "玄宗", "肃宗",
    "代宗", "德宗", "顺宗", "宪宗", "穆宗", "敬宗", "文宗", "武宗", "宣宗",
    "懿宗", "僖宗", "昭宗", "世宗", "仁宗", "神宗", "高帝", "文帝", "武帝",
    "明帝", "章帝", "和帝", "安帝", "顺帝", "桓帝", "灵帝", "献帝", "宣帝",
    "元帝", "成帝", "哀帝", "平帝", "景帝", "惠帝", "少帝", "废帝", "后主",
    # peoples / polities sometimes mislabelled as a person name
    "匈奴", "突厥", "鲜卑", "吐蕃", "契丹", "柔然", "羌", "氐", "胡", "羯",
    "天下", "中国", "朝廷", "京师",
    # region / object / idiom fragments jieba mislabels nr (clean-surname start)
    "陆梁", "侯印", "龙颜",
    # the historian's own editorial voice (臣光曰 / 史臣曰) — not a character
    "臣光", "史臣", "臣", "光",
}


# RC-1/RC-3a — common 文言 words, idioms, objects and 爵号 fragments whose first
# char is ALSO a 姓 (黄/雷/徐/符/顾/陈/神…), so the surname-prefix gate wrongly
# admits them as a 2-char 姓+名. These are never a person card. Seeded from the
# 卷174 historian audit; extend as new false positives surface.
COMMON_WORD_NONPERSON: set[str] = {
    "雷霆",   # 声如雷霆 — thunder
    "符玺",   # 天子符玺 — the imperial seal
    "黄钺",   # 假黄钺 — a ceremonial axe / mark of authority
    "徐行",   # 称疾徐行 — to walk slowly
    "顾托",   # 猥蒙顾托 — to entrust
    "陈谢",   # 陈情谢罪 — to apologise
    "黄龙",   # 黄龙兵 / 年号 / 地名 — never a person here
    "阿衡",   # 伊尹's office, used allusively (阿衡之任)
    "神武",   # 爵号 (神武公) / 谥 / adjective — too ambiguous as a standalone card
    "魏安",   # 魏安公 place-title fragment (→ 尉迟惇)
    "申公",   # 爵号 fragment (申公·李穆)
}

# RC-6/RC-3c — 3-char 鲜卑/胡 复姓 (clan names). On their own they are a CLAN, not
# a person (阿史那 = the Türk royal house), so a bare occurrence is dropped; but as
# the head of a longer name (阿史那社尔) the model SHOULD keep them — handled in
# load_model_ner_people (bare-clan drop only).
COMPOUND3: set[str] = {
    "阿史那", "阿史德", "破六韩", "纥豆陵", "费也利", "吐谷浑", "是云", "没鹿回",
}


def bad_auto_surface(s: str) -> bool:
    """A surface unsafe to auto-match: too short, an explicit banned form, or a
    generic title / honorific / clan pattern (X王, X后, X帝, X太子, X氏, …) that
    denotes a rotating role rather than one fixed referent."""
    if len(s) < 2 or s in BANNED_SURF:
        return True
    if s in COMMON_WORD_NONPERSON or s in COMPOUND3:   # RC-1/RC-3: common words & bare 胡 clans
        return True
    if s[-1] in "王后妃帝":            # 赵王 / 窦太后 / 许皇后 / 魏文帝 …
        return True
    if s.endswith("太子") or s.endswith("世子"):
        return True
    if len(s) <= 3 and s.endswith("氏"):  # 王氏 / 窦氏 clan reference
        return True
    return False


# ── NER-proposal guards (shared by ner_extract.py harvesting AND seed loading,
#    so tuning these needs only a rebuild, never a re-run of the slow jieba pass).

# 百家姓 — a 2-char auto name must begin with a known 姓; this single gate removes
# most place names (秦/楚/阿房宫) and titles (大将军/二世) that jieba mislabels nr.
SURNAMES = set(
    "王李张刘陈杨赵黄周吴徐孙朱马胡郭林何高梁郑罗宋谢唐韩冯邓曹彭曾萧田董袁潘"
    "于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱"
    "江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤聂耿牛桓乐祖虞嬴芈"
    "苻寇桑屈雍简栾荀郦郗桥伍卞邴臧鲍栗蒙樊荆关召展屠晁鼌审英布彭越黥郅郤庞蒯"
    "华都国佘仲赫尔独窦伏")

# Curated expansion of the 姓 whitelist (reference: surnames_baijiaxing.json, the
# canonical 百家姓). NOT a wholesale union — a wholesale 百家姓 swap admits ~9k new
# jieba-nr surfaces, ~80% garbage, because the surname-gate is the ONLY filter and
# many 姓 are homographs of states (晋/齐/楚/燕), offices (诸/左/公/司) or function
# words (有/言/益). These chars were scored against the corpus (fraction of the
# surfaces they admit that are real names vs. titles/idioms); only 姓 whose
# admissions are dominantly real names are added. Excluded: state/office/function
# homographs and given-name fragments (全忠/弘俶). This recovers genuinely-missing
# 姓 like 柴 (柴绍), 裴, 项, 庾, 霍, 房, 辛, 班, 葛, 颜, 廉, 狄 … without the blowup.
SURNAMES_EXTRA = set(
    "裴项翟奚褚岑应隗郁娄柴路闵甄滕赖仇靳舒毕禹冉焦祝梅湛贡乔竺谯茹习裘席祁嵇"
    "蔺扈倪缪瞿阚茅夔巩支俞巢包查逄姬缑訾池乜鞠贲党喻柯滑钦逯佟璩芮郜酆浦钭傅"
    "景房阎时羊章索庄邢辛盖牧班鄂申柏葛甘季法皮解边刁庾蒲霍费颜廉狄卓阮骆殷温"
    "符管苗麻单纪吉")
SURNAMES |= SURNAMES_EXTRA

# Compound (2-char) surnames → a 3-char auto name must begin with one of these.
COMPOUND = {
    "司马", "诸葛", "欧阳", "上官", "夏侯", "公孙", "令狐", "慕容", "拓跋",
    "宇文", "长孙", "赫连", "尔朱", "独孤", "侯莫", "皇甫", "钟离", "东方",
    "西门", "公子", "公叔", "申屠", "万俟", "尉迟", "鲜于", "闾丘", "南宫",
    # 百家姓 复姓 present in 资治通鉴; offices (司徒/司空/司寇), titles (单于) and
    # 公-prefixed / place-ambiguous forms are deliberately excluded.
    "呼延", "淳于", "太叔", "仲孙", "轩辕", "段干", "百里", "羊舌", "微生",
    "梁丘", "澹台", "宗政",
    # RC-6 — 鲜卑 / 十六国 / 胡 复姓 attested in 资治通鉴 (达奚儒, 乙弗虔, 豆卢勣,
    # 贺娄子干, 叱列长义, 斛律光 …). Enables the 复姓+单名 3-char path for jieba.
    "达奚", "乙弗", "豆卢", "贺娄", "叱列", "乌丸", "贺兰", "斛律", "屈突",
    "库狄", "乞伏", "秃发", "沮渠", "出连", "可朱", "叱罗", "叱干", "慕舆",
    "费连", "莫多", "厍狄",
}

# A name char-2/3 that is a particle / common verb / admin·geographic unit marks
# a glued fragment (王闻 / 李信奔 / 赵地), not a name. Also a few common nouns that
# glue to a surname into an ordinary word (胡客 = "a 胡 visitor", not a name).
STOP_CHARS = set(
    "之乎者也矣焉耳耶邪兮哉乃遂复皆其与及亦即既因故则且若所是此彼于为以而"
    "曰谓闻令使遣将击攻伐入出收徇略守拔围降破斩杀走奔死亡立得"
    "军师兵卒众人民吏臣使者党"
    "郡县城邑乡里亭聚关塞陵庙宫殿台门阙境地山川池阳中"
    "官家问甲马财物粟谷货怀心意吟儿州置帛服朝族母病鼎父世灭客主谏牛羊")

# A 3-char auto name with one of these as its MIDDLE char is a polity/title +
# name glue (魏王操 = 魏王·曹操, 魏主赐, 赵主曜), not a single given name. Such a
# surface is never seeded as a STANDALONE auto person; the genuine title+given-name
# aliases among them are instead routed into TITLE_GLUE_ALIASES below.
MID_TITLE = set("王主帝公侯后太")

# Title-glue ALIASES (魏王操 = 魏王·曹操). A 〔polity〕〔title〕〔given-name〕 form is a
# real reference to a person under a titular alias, NOT noise — but it cannot be
# resolved automatically: a referential title (魏主/赵王/秦主) denotes a DIFFERENT
# ruler every few 卷 across the centuries, and matching the bare given-name char
# against the people set collides hard with same-char auto people (a dry run bound
# 魏主命→顾命, 赵公子→孟子, 周公瑾→诸葛瑾). So these are hand-verified.
#
# Attaching the surface to the canonical person and extending that person's
# nearest 卷 window to the alias's own 卷 keeps the bind era-local: 赵主曜 lands on
# 刘曜's 318–329 window, never on an unrelated 曜. Only entries whose canonical
# person already exists take effect; the rest are reported by build_persons as a
# "cast to add" backlog (拓跋嗣=魏主嗣, 慕容俊=燕主俊, 苻登=秦主登, 司马伦=赵王伦 …).
TITLE_GLUE_ALIASES = {
    "魏王操": "曹操", "魏公操": "曹操",
    "赵主曜": "刘曜", "汉主曜": "刘曜",
    "燕主垂": "慕容垂", "燕王垂": "慕容垂",
    "赵王虎": "石虎",
    "赵主勒": "石勒",
    "汉主渊": "刘渊",
    "魏王豹": "魏豹",
    # Present auto-canonicals (本名已自动入库) — bind their title-glue forms.
    "燕王跋": "冯跋",
    "汉主寿": "李寿",
    "晋公护": "宇文护",
    "秦主健": "苻健",
    "秦主泓": "姚泓",
    "宋公裕": "刘裕", "宋王裕": "刘裕",
}

# Surname chars that are ALSO very common function words / nouns in 文言. A name
# starting with one of these is too easily a glued phrase (于今, 何谓, 方略,
# 丁壮…), so such a candidate is accepted ONLY when jieba's name lexicon already
# knows the whole token (the `d` flag). Unambiguous surnames take the cheap path.
AMBIGUOUS_SURNAMES = set("于何方白向都任武史召国金田文成安平广万丁乐华牛高严时后那东")
CLEAN_SURNAMES = SURNAMES - AMBIGUOUS_SURNAMES

_TIANGAN = set("甲乙丙丁戊己庚辛壬癸")
_DIZHI = set("子丑寅卯辰巳午未申酉戌亥")


def ner_surface_ok(tok: str, in_dict: bool = False) -> bool:
    """Accept a jieba `nr` proposal as an auto person surface — conservative, to
    keep the 自动识别 tier clean.

    Pure surname-gate (precision over recall). jieba's own `nr` lexicon is NOT a
    clean gazetteer — it lists offices/idioms/fragments (都尉, 士大夫, 万世, 谢病,
    even mis-segmented 西击秦), so the `in_dict` signal is ignored; the famous
    names it would add (李斯, 荆轲, 东方朔) already pass on the surname paths.

      * 3-char: a compound 姓 (司马/诸葛/东方…) OR a clean single 姓 + 2-char given
        name (张延赏, 李德裕), with char-2/3 not a stop char / 地支. The single-姓
        3-char class is noisier, so load_ner_people gates it on stronger evidence.
      * 2-char: must begin with an UNAMBIGUOUS 姓 (王/蒙/曹…), char-2 not a particle
        / 地支 / admin·geographic unit. Ambiguous function-word 姓 (于/何/方/万/都…)
        are rejected outright — they glue into phrases (于今, 万世, 都尉) far more
        often than they head a real name we'd miss.
    干支 date pairs (丁酉, 庚午…) and stop-char fragments are always rejected.
    """
    L = len(tok)
    if L not in (2, 3):
        return False
    if bad_auto_surface(tok):
        return False
    if tok[0] in _TIANGAN and tok[1] in _DIZHI:   # 干支 sexagenary date
        return False
    if L == 3:
        if tok[1] in STOP_CHARS or tok[2] in STOP_CHARS or tok[1] in _DIZHI:
            return False
        if tok[1] in MID_TITLE:                    # 魏王操 / 魏主赐 polity+name glue
            return False
        if tok[:2] in COMPOUND:                    # 司马懿 / 诸葛亮
            return True
        return tok[0] in CLEAN_SURNAMES            # 张延赏 / 李德裕 (single 姓)
    # L == 2
    if tok[1] in STOP_CHARS or tok[1] in _DIZHI:
        return False
    return tok[0] in CLEAN_SURNAMES


def load_guide_people():
    """name -> list of (juan_no, ce_year, role, query)."""
    recs: dict[str, list] = collections.defaultdict(list)
    for f in sorted(glob.glob(str(GUIDE / "juan_*.json"))):
        g = json.loads(Path(f).read_text(encoding="utf-8"))
        jn = g["juan_no"]
        for s in g.get("summaries", []):
            cy = s.get("ce_year")
            for kp in s.get("key_people", []):
                nm = (kp.get("name") or "").strip()
                if not nm or nm in WIKI_NONPERSON:
                    continue
                recs[nm].append((jn, cy, (kp.get("role") or "").strip(),
                                 (kp.get("query") or "").strip()))
    return recs


def load_ner_people(min_occ=2):
    """NER proposals harvested from the 原文 (ner_extract.py) → same shape as
    load_guide_people but with no role/query/year. Surface-guarded here so the
    guard can be tuned and re-applied without re-running the jieba pass. A surface
    seen fewer than `min_occ` times corpus-wide is dropped: a lone hapax `nr` tag
    is the least reliable signal and the least useful card (a one-off mention).
    Returns {} when ner_candidates.json is absent (NER layer simply off)."""
    f = Path(__file__).resolve().parent / "ner_candidates.json"
    if not f.exists():
        return collections.defaultdict(list)
    raw = json.loads(f.read_text(encoding="utf-8"))
    recs: dict[str, list] = collections.defaultdict(list)
    for surf, info in raw.items():
        if surf in WIKI_NONPERSON:
            continue
        n = info.get("n", 1)
        # The single-姓 3-char class (张延赏, 李德裕) is noisier than the 2-char /
        # compound-姓 classes — a glued 姓+2-char verb phrase can slip the
        # structural gate. Require stronger corpus evidence for it (≥3 vs ≥2).
        is_single3 = len(surf) == 3 and surf[:2] not in COMPOUND
        need = max(min_occ, 3) if is_single3 else min_occ
        if n < need:
            continue
        if not ner_surface_ok(surf, bool(info.get("d"))):
            continue
        for jn in info.get("j", []):
            recs[surf].append((jn, None, "", ""))
    return recs


_HAN_ONLY = re.compile(r"^[\u4e00-\u9fff]+$")

# Titles, kinship and reign/posthumous designations the UPOS model tags as PROPN
# person but which are not a personal name. Small + auditable.
_MODEL_TITLE_BLOCK = {
    "光武", "文武", "道宗", "赞普", "谷蠡", "述律", "单于", "可汗", "阏氏", "须卜",
    "居次", "大单于", "骨都侯", "当户", "赤松子", "羲和", "大良造", "左贤", "右贤",
    "左谷蠡", "右谷蠡",
}


def load_model_ner_people(hand_people):
    """Tier-2: a classical-Chinese UPOS NER pass (ner_model.py →
    ner_model_candidates.json) used ONLY to recover *foreign-headed* person names
    the Han-surname gate in `load_ner_people` structurally cannot reach — 突厥/鲜卑/
    匈奴/吐蕃 names like 突利, 阿史那思摩, 颉利, 默啜, 斛律光, 呼韩邪.

    The model's raw output also contains bare given-name fragments (世民 ⊂ 李世民),
    clan/compound surnames (宇文, 慕容), reign/posthumous designations (孝武, 睿武孝文)
    and title/state glues (柔然处罗). We keep ONLY surfaces that:
      - are all-Han, 2–5 chars, not a known non-person / bad surface,
      - are NOT surname-headed (that's `load_ner_people`'s job — avoids duplicates),
      - are NOT a 1–2 char suffix of any known full name (drops given-name frags),
      - are NOT a bare compound surname / title / 孝X posthumous,
      - do NOT begin with a known non-person prefix (state/title glue),
    at frequency n≥3, OR len≥4 & n≥2 (the long foreign compounds are reliable even
    at n=2, e.g. 阿史那思摩). Returns {} when the model feed is absent (tier simply
    off). Records carry no role/query/year, like `load_ner_people`."""
    f = Path(__file__).resolve().parent / "ner_model_candidates.json"
    if not f.exists():
        return collections.defaultdict(list)
    raw = json.loads(f.read_text(encoding="utf-8"))

    def surname_headed(s: str) -> bool:
        return (bool(s) and s[0] in CLEAN_SURNAMES) or (len(s) >= 3 and s[:2] in COMPOUND)

    # Known full names → their 1–2 char suffixes (given-name fragments to drop).
    full_names: set[str] = set()
    for p in hand_people:
        for s in [p.get("canonical_name", ""), *p.get("names", []), *p.get("match", [])]:
            if s and 2 <= len(s) <= 6:
                full_names.add(s)
    for nm in load_guide_people().keys():
        if 2 <= len(nm) <= 6:
            full_names.add(nm)
    full_names |= set(load_ner_people().keys())
    for s in raw:
        if surname_headed(s) and 2 <= len(s) <= 6:
            full_names.add(s)
    tails: set[str] = set()
    for fn in full_names:
        for k in (1, 2):
            if len(fn) - k >= 2:
                tails.add(fn[k:])

    def prefix_blocked(s: str) -> bool:
        return any(len(s) > k and s[:k] in WIKI_NONPERSON for k in (2, 3))

    recs: dict[str, list] = collections.defaultdict(list)
    for surf, info in raw.items():
        n = info.get("n", 1)
        if not _HAN_ONLY.match(surf):
            continue
        if not (2 <= len(surf) <= 5):
            continue
        if surf in WIKI_NONPERSON or bad_auto_surface(surf):
            continue
        if surf in COMPOUND or surf in _MODEL_TITLE_BLOCK:
            continue
        if surf in COMPOUND3:                            # bare 胡 clan (阿史那)
            continue
        if len(surf) == 2 and surf[0] == "孝":          # 孝武/孝文/孝宣… posthumous
            continue
        if len(surf) >= 4 and ("孝" in surf or "睿" in surf or "愍" in surf):
            continue                                     # long posthumous strings
        if surname_headed(surf):
            continue
        if surf in tails:
            continue
        if prefix_blocked(surf):
            continue
        if not (n >= 3 or (len(surf) >= 4 and n >= 2)):
            continue
        for jn in info.get("j", []):
            recs[surf].append((jn, None, "", ""))
    return recs


def _surname_of(full: str):
    """Leading 姓 of a full 姓名: a 2-char COMPOUND prefix, else a single clean 姓."""
    if len(full) >= 3 and full[:2] in COMPOUND:
        return full[:2]
    if full and full[0] in CLEAN_SURNAMES:
        return full[0]
    return None


# RC-2 — 省姓回指 (surname-elided anaphora). On second mention 资治通鉴 routinely drops
# the surname — 「于仲文…仲文」, 「庾季才…季才」, 「崔仲方…仲方」 — so the bare given name
# splits off as its own spurious 2-char card. Fix: within a single 卷, if a 2-char auto
# surface S equals a full 姓名 F (also present in that 卷) with its leading 姓 removed —
# i.e. F = 〔姓〕+ S where 姓 is a real surname (1-char clean/ambiguous or 2-char compound)
# — bind S to F as a 卷-local alias and drop its standalone presence in that 卷.
#
# Deliberately SUFFIX-only (省姓回指), never prefix: a prefix match like 李广 ⊂ 李广利 or
# 张昌 ⊂ 张昌宗 would silently fold one famous person into another. jieba head-truncations
# (辛彦之→辛彦, 李圆通→李圆) are therefore left to recall fixes, not merged here.
def _elided_surname(full: str, given: str):
    """If `full` == 〔姓〕+ `given` and the removed head is a real surname, return that
    surname; else None. Models 省姓回指 surname elision precisely."""
    if not full.endswith(given) or len(full) <= len(given):
        return None
    head = full[:len(full) - len(given)]
    if len(head) == 2 and head in COMPOUND:
        return head
    if len(head) == 1 and head in (CLEAN_SURNAMES | AMBIGUOUS_SURNAMES):
        return head
    return None


def merge_anaphora(people):
    """Fold 省姓回指 surname-elided 2-char fragments into the full 姓名 they belong to,
    within each 卷. Mutates `people` in place; returns (merged, dropped) counts."""
    by_juan_full: dict[int, list] = collections.defaultdict(list)
    for p in people:
        cn = p["canonical_name"]
        if 3 <= len(cn) <= 5 and _HAN_ONLY.match(cn):
            for j in p["juans"]:
                by_juan_full[j].append(p)

    merged = 0
    dropped_juans = collections.defaultdict(set)  # frag_id -> {juan,...}
    for p in people:
        cn = p["canonical_name"]
        if len(cn) != 2 or p["confidence"] != "high" or not _HAN_ONLY.match(cn):
            continue
        for j in list(p["juans"]):
            host = None
            for F in by_juan_full.get(j, ()):
                fn = F["canonical_name"]
                if fn == cn or not _elided_surname(fn, cn):
                    continue
                if host is not None and host["canonical_name"] != fn:
                    host = "AMBIG"
                    break
                host = F
            if host and host != "AMBIG":
                if cn not in host["match"]:
                    host["match"].append(cn)
                if cn not in host.get("names", []):
                    host.setdefault("names", []).append(cn)
                dropped_juans[p["id"]].add(j)
                merged += 1

    dropped = 0
    survivors = []
    for p in people:
        rm = dropped_juans.get(p["id"])
        if rm:
            p["juans"] = sorted(set(p["juans"]) - rm)
            if not p["juans"]:
                dropped += 1
                continue
        survivors.append(p)
    people[:] = survivors
    return merged, dropped


def merge_people_sources(*sources):
    """Union several {name: [recs]} maps into one."""
    out: dict[str, list] = collections.defaultdict(list)
    for src in sources:
        for nm, recs in src.items():
            out[nm].extend(recs)
    return out


def _load_wiki_nonperson():
    """Surfaces confirmed NON-person via zh.wikipedia + Wikidata P31 (wiki_verify.py
    → wiki_curate.py): places (卢龙=county), peoples (乌桓), books (周礼), states
    (赵国), months (仲冬), titles/kinship (侯爵, 曾孙), animals/plants (孔雀, 杜仲),
    eras (正德)… Dropped from the auto tier. Disambiguation pages and legendary
    rulers (唐尧, 虞舜) are deliberately NOT in this list. Empty if file absent."""
    f = Path(__file__).resolve().parent / "wiki_nonperson.json"
    if not f.exists():
        return set()
    try:
        return set(json.loads(f.read_text(encoding="utf-8")))
    except Exception:
        return set()


WIKI_NONPERSON = _load_wiki_nonperson()


# ── Wikipedia/Wikidata biographical enrichment (wiki_enrich.py) ──────────────
# wiki_person_info.json maps a verified-person surface to {qid, title, desc,
# extract}. We turn that into an informative `brief` (one-line identity) and a
# fuller `identity` (lead-paragraph bio behind the spoiler toggle), replacing
# the uninformative "见于卷216（748年）" template for ~2,500 auto people. A period
# gate drops modern / post-959 namesakes (资治通鉴 ends 959 CE), so a name that
# happens to match a 1980s actor's article is left with the generic brief.
def _load_wiki_info():
    f = Path(__file__).resolve().parent / "wiki_person_info.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


WIKI_INFO = _load_wiki_info()

_DYN_PREFIXES = (
    "汉", "漢", "東漢", "东汉", "西漢", "西汉", "蜀漢", "蜀汉", "唐", "隋", "宋", "梁",
    "陈", "陳", "齐", "齊", "周", "秦", "晋", "晉", "魏", "三國", "三国", "南朝", "北朝",
    "五代", "新", "後", "后", "東晉", "东晋", "西晉", "西晋", "曹魏", "孫吳", "孙吴",
    "東吳", "东吴", "北魏", "東魏", "东魏", "西魏", "北齊", "北齐", "北周", "南齊",
    "南齐", "劉宋", "刘宋", "隋朝", "唐朝", "唐代", "明", "清", "元", "金", "遼", "辽",
    "五胡", "十六國", "十六国",
)
_ID_KW = (
    "政治家", "軍事家", "军事家", "思想家", "文学家", "文學家", "詩人", "诗人", "宰相",
    "丞相", "皇帝", "皇后", "太后", "公主", "将领", "將領", "名将", "名將", "大臣",
    "官员", "官員", "人物", "节度使", "節度使", "刺史", "太守", "将军", "將軍", "君主",
    "国君", "國君", "权臣", "權臣", "外戚", "宦官", "領袖", "领袖", "开国", "開國",
    "史学家", "史學家", "经学家", "經學家", "僧人", "道士", "画家", "畫家", "书法家",
    "書法家", "起义", "起義", "皇子", "王", "帝", "侯", "公",
)
_LATE_TOKENS = (
    "明朝", "明代", "清朝", "清代", "元朝", "元代", "宋朝", "北宋", "南宋", "宋代",
    "金朝", "民國", "民国", "中华民国", "中華民國", "中华人民共和国", "中国大陆",
    "中國大陸", "現代", "现代", "當代", "当代",
)


def _cjk_len(s):
    return sum(1 for ch in s if '\u3400' <= ch <= '\u9fff')


def _strip_lifespan(s):
    return re.sub(r'（[^）]*?[\d?？][^）]*?）', '', s).strip()


def _cap_cjk(s, limit):
    if _cjk_len(s) <= limit:
        return s
    out, c, brk = '', 0, -1
    for ch in s:
        if '\u3400' <= ch <= '\u9fff':
            c += 1
        out += ch
        if ch in '、，':
            brk = len(out) - 1
        if c >= limit:
            break
    if brk > 0:
        out = out[:brk]
    return out.rstrip('、，·')


def _head_strip(name, extract):
    e = (extract or '').strip()
    if name and e.startswith(name):
        e = e[len(name):]
        if e[:1] == '，':
            e = e[1:]
    return e.strip()


def _wiki_year(extract):
    """Earliest birth/death year (CE; BCE negative) from the lead date paren,
    parsing both 年 and 世紀. None if no date is present."""
    if not extract:
        return None
    m = re.search(r'（([^）]*)）', extract[:48])
    if not m:
        return None
    span, yrs = m.group(1), []
    for ym in re.finditer(r'(前)?(\d{1,4})\s*年', span):
        y = int(ym.group(2))
        yrs.append(-y if ym.group(1) else y)
    for cm in re.finditer(r'(\d{1,2})\s*世[紀纪]', span):
        yrs.append(int(cm.group(1)) * 100 - 50)
    return min(yrs) if yrs else None


def _wiki_out_of_period(info):
    """True if this Wikipedia match is a post-959 / modern namesake that cannot
    be the figure 资治通鉴 refers to. Year-first, then late-dynasty tokens."""
    y = _wiki_year(info.get('extract'))
    if y is not None:
        return y > 1000
    text = (info.get('desc') or '') + (info.get('extract') or '')[:80]
    return any(t in text for t in _LATE_TOKENS)


def _wiki_clean_desc(desc):
    if not desc:
        return None
    d = _strip_lifespan(desc.strip())
    d = re.sub(r'\s+', '', d).rstrip('，、。；')
    return d or None


def _wiki_brief_from_extract(name, extract):
    e = _head_strip(name, extract)
    if not e:
        return None
    para = e.split('\n')[0]
    m = re.search(r'人[，。、]', para)
    cand = para[m.end():] if m else para
    cand = _strip_lifespan(cand.split('。')[0]).strip('，、 ')
    if cand and not any(k in cand for k in _ID_KW):
        for sent in para.split('。'):
            if any(k in sent for k in _ID_KW):
                cand = _strip_lifespan(sent).strip('，、 ')
                break
    if not cand:
        return None
    return _cap_cjk(cand, 40)


def wiki_brief(name, dyn, info):
    """One-line identity brief from wiki info; dynasty-prefixed when the source
    doesn't already lead with an era. None when nothing usable can be derived."""
    src = _wiki_clean_desc(info.get('desc')) or _wiki_brief_from_extract(name, info.get('extract'))
    if not src:
        return None
    src = re.sub(r'^(中國|中国)', '', src)
    if not src.startswith(_DYN_PREFIXES) and dyn and dyn != '—':
        src = f"{dyn}·{src}"
    return src


def wiki_identity(name, info):
    """Fuller lead-paragraph bio (≤120 漢字 / ≤2 sentences, whichever yields more
    complete sentences; name head stripped, dates kept). For the spoiler toggle."""
    e = _head_strip(name, info.get('extract')).replace('\n', '')
    if not e:
        return None
    sents = [s for s in re.findall(r'[^。]*。|[^。]+$', e) if s.strip()]
    if not sents:
        return None
    a, tot = [], 0
    for s in sents:
        l = _cjk_len(s)
        if a and tot + l > 120:
            break
        a.append(s); tot += l
    if not a:
        a = sents[:1]
    b = sents[:2]
    chosen = a if len(a) >= len(b) else b
    out = ''.join(chosen).strip()
    if len(chosen) < len(sents):
        out = out.rstrip() + '…'
    return out or None


def wiki_enrich(name, dyn):
    """(brief, identity) from Wikipedia for an auto person, or (None, None) when
    there's no usable / period-consistent match."""
    info = WIKI_INFO.get(name)
    if not info or _wiki_out_of_period(info):
        return None, None
    return wiki_brief(name, dyn, info), wiki_identity(name, info)


def juan_meta():
    m = json.loads((TEXT / "manifest.json").read_text(encoding="utf-8"))
    meta = {}
    for j in m["juans"]:
        meta[j["juan_no"]] = {
            "dynasty": (j.get("dynasty") or "").replace("纪", ""),
            "ce_start": j.get("ce_start"), "ce_end": j.get("ce_end"),
        }
    return meta


def split_windows(juans, gap=GAP):
    js = sorted(set(juans))
    if not js:
        return []
    out, cur = [], [js[0]]
    for j in js[1:]:
        if j - cur[-1] <= gap:
            cur.append(j)
        else:
            out.append(cur)
            cur = [j]
    out.append(cur)
    return out


def grow(seed_juans, candidate_juans, gap=GAP):
    """Absorb candidate 卷 into the seed set only while they stay within `gap`
    of the (growing) set — keeps a hand figure contiguous, never teleporting a
    posthumous-honorific match into an unrelated dynasty."""
    s = set(seed_juans)
    cand = sorted(set(candidate_juans))
    changed = True
    while changed:
        changed = False
        for j in cand:
            if j in s:
                continue
            if any(abs(j - x) <= gap for x in s):
                s.add(j)
                changed = True
    return sorted(s)


def _auto_id(name, widx):
    h = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"a:{h}-{widx}"


def build_seed(hand_people, juans_allowed):
    allowed = set(juans_allowed)
    # Two proposal sources, unified: the 白话导读 key_people (editorially chosen)
    # and NER over the 原文 itself (closes the gap where a body-text figure was
    # never named in any summary). Both flow through the same grow/split/collision
    # machinery, so a name seen in both just accumulates 卷.
    people_src = merge_people_sources(load_guide_people(), load_ner_people(),
                                      load_model_ner_people(hand_people))
    meta = juan_meta()

    # Recognisable surface -> hand person id (canonical + aliases + match).
    name_to_hand: dict[str, str] = {}
    for p in hand_people:
        for s in [p["canonical_name"], *p.get("names", []), *p.get("match", [])]:
            name_to_hand.setdefault(s, p["id"])

    # 1. Grow hand juans from mentions (contiguity-gated), and record which names
    #    are thereby "consumed" by a hand entry.
    hand_grow: dict[str, set] = collections.defaultdict(set)
    consumed: set[str] = set()
    for nm, recs in people_src.items():
        pid = name_to_hand.get(nm)
        if pid is None:
            continue
        consumed.add(nm)
        for (jn, _cy, _role, _q) in recs:
            if jn in allowed:
                hand_grow[pid].add(jn)

    people: list[dict] = []
    for p in hand_people:
        cand = hand_grow.get(p["id"], set())
        grown = grow([j for j in p["juans"] if j in allowed], cand)
        grown = [j for j in grown if j in allowed]
        q = dict(p)
        q["juans"] = grown
        q["confidence"] = "reviewed"
        people.append(q)

    # 2. Auto people for the remaining (unconsumed) names.
    for nm, recs in people_src.items():
        if nm in consumed or bad_auto_surface(nm):
            continue
        juans_here = [jn for (jn, _, _, _) in recs if jn in allowed]
        if not juans_here:
            continue
        for widx, win in enumerate(split_windows(juans_here)):
            wrecs = [r for r in recs if r[0] in win]
            roles, seen_r = [], set()
            for (_, _, role, _) in wrecs:
                if role and role not in seen_r:
                    roles.append(role)
                    seen_r.add(role)
            years = [cy for (_, cy, _, _) in wrecs if cy is not None]
            j0 = min(win)
            dyn = meta.get(j0, {}).get("dynasty", "")
            ystr = ""
            if years:
                y = min(years)
                ystr = f"前{-y}年" if y < 0 else f"{y}年"
            # Card content is BOOK-DERIVED ONLY — never Wikipedia bio text. A
            # bare surname+name string collides with same-name figures from other
            # eras on zh.wikipedia (e.g. the Tang general 李晟 vs. a 1985 actress),
            # so pulling a wiki lead paragraph risks captioning a 资治通鉴 figure
            # with a modern namesake's biography. Wikipedia is used ONLY to
            # verify/disambiguate a surface (person vs. place, era consistency in
            # wiki_verify.py), not as a content source. The brief is the book
            # locator; identity is the 白话导读 editorial roles when present.
            brief = (f"{dyn + '·' if dyn else ''}见于卷{j0:03d}"
                     + (f"（{ystr}）" if ystr else "") + "。")
            identity = "；".join(roles) if roles else brief
            aliases = sorted({q for (_, _, _, q) in wrecs
                              if q and q != nm and len(q) >= 2 and not bad_auto_surface(q)})
            people.append({
                "id": _auto_id(nm, widx),
                "canonical_name": nm,
                "names": aliases,
                "dynasty": dyn or "—",
                "era_hint": (roles[0][:16] if roles else (dyn or "人物")),
                "floruit": [min(years), max(years)] if years else [None, None],
                "brief": brief,
                "identity": identity,
                "match": [nm],
                "juans": sorted(win),
                "confidence": "high",  # auto-seeded: program-identified, 非原文
            })

    # 2b. Curated title-glue aliases (魏王操 → 曹操). For each alias surface, gather
    #     the 卷 where it actually occurs (ner_candidates + any 白话导读 duplicate the
    #     guide seeded under the glue form), pick the canonical person's window
    #     NEAREST those 卷, extend it to cover them, attach the surface, and drop the
    #     duplicate so attribution is single. Extending the nearest window keeps the
    #     bind era-local (no century teleport) even when a ruler has several windows.
    _gnf = Path(__file__).resolve().parent / "ner_candidates.json"
    _graw = json.loads(_gnf.read_text(encoding="utf-8")) if _gnf.exists() else {}
    by_canon: dict[str, list] = collections.defaultdict(list)
    for p in people:
        by_canon[p["canonical_name"]].append(p)

    def _juan_dist(wjuans, sjuans):
        if not wjuans or not sjuans:
            return 10**6
        return min(abs(a - b) for a in wjuans for b in sjuans)

    glue_bound, glue_missing, remove_ids = 0, [], set()
    for surf, canon in TITLE_GLUE_ALIASES.items():
        targets = by_canon.get(canon)
        if not targets:
            glue_missing.append(f"{surf}→{canon}")
            continue
        surf_juans = {j for j in _graw.get(surf, {}).get("j", []) if j in allowed}
        for dup in by_canon.get(surf, []):
            surf_juans |= {j for j in dup["juans"] if j in allowed}
            remove_ids.add(dup["id"])
        target = min(targets, key=lambda t: _juan_dist(t["juans"], surf_juans))
        if surf_juans:
            target["juans"] = sorted(set(target["juans"]) | surf_juans)
        if surf not in target["match"]:
            target["match"].append(surf)
        if surf not in target.get("names", []):
            target.setdefault("names", []).append(surf)
        glue_bound += 1
    if remove_ids:
        people = [p for p in people if p["id"] not in remove_ids]

    # 2c. RC-2 anaphora / truncation merge — fold 省姓回指 + jieba-truncated 2-char
    #     fragments into the full 姓名 they belong to, per 卷 (see merge_anaphora).
    anaphora_merged, anaphora_dropped = merge_anaphora(people)

    # 3. Per-卷 collision resolution. A surface owned by >1 person in a 卷 is
    #    ambiguous there and dropped from the rule table for that 卷.
    owners: dict[int, dict[str, set]] = collections.defaultdict(lambda: collections.defaultdict(set))
    for p in people:
        for s in p["match"]:
            if len(s) < 2:
                continue
            for j in p["juans"]:
                owners[j][s].add(p["id"])

    rules: dict[int, dict[str, str]] = collections.defaultdict(dict)
    dropped = 0
    for j, sm in owners.items():
        for s, pids in sm.items():
            if len(pids) == 1:
                rules[j][s] = next(iter(pids))
            else:
                dropped += 1

    return people, dict(rules), {"ambiguous_dropped": dropped,
                                 "glue_bound": glue_bound,
                                 "glue_missing": glue_missing,
                                 "anaphora_merged": anaphora_merged,
                                 "anaphora_dropped": anaphora_dropped}
