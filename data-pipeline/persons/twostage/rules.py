"""AGENT 1 — TAGGER as a RULES ENGINE.

Design principle (per user): Agent 1 decides whether a span denotes a person without
consulting the global person/identity KB. It uses POS/BIO morphology, local syntax,
same-jie earlier anchors, and paragraph-local translation evidence. Agent 2 alone may
consult people.json to bind an occurrence to an identity. Juan number is passed only as
a time proxy for CE-aware geographic vetoes, never for person-surface admission.

Each rule is a named function that yields candidate spans. The driver runs them in
priority order over a shared `consumed[]`, so a higher-priority rule reserves a span
before a lower one can. Every emitted OccurrenceCard carries its `rule` + `scope`
provenance so the evaluator can score each rule independently.

Rules (scope):
  pos_person_name (jie)     complete high-confidence POS/BIO personal-name span.
  title_appellation(jie)    morphosyntactically licensed personal title/appellation.
  foreign_title_name(jie)   POS-backed two-char name + 可汗/单于.
  foreign_suffix_name(jie)  compound surname + BIO given + foreign suffix (拓跋沙漠汗).
  royal_title_name(jie)     太子/世子/皇子/王子 + known two-char name.
  polity_appos    (jie)     polity/place marker + POS-backed full name (齐田和).
  block_appos     (jie)     BLOCK title/role + POS-backed known full name.
  corpus_given2   (corpus)  literal 2-char bare-given 省称 KB alias (道济, 子胥) — risky.
  role             (corpus)  controlled polity+ruler appellation (吴主, 契丹主).
  polity_king      (corpus)  controlled polity+王 person title (汉王, 吴王).
  jue_name         (jie)     polity/title+given full-name-equivalent anchor (赵王虎).
  office_name      (jie)     office+given full-name-equivalent anchor (大将军光).
  office_fullname  (jie)     office+POS surname/given anchor (司马班超).
  office_alias2    (jie)     office ending 史 + POS-backed name (内史汲黯).
  appointment      (jie)     以 + POS-backed person + 为 + office/title.
  empress_title    (jie)     inherently personal titles 太后 / 皇后.
  surname_empress  (jie)     surname morphology + 后 title (贾后, 独孤后).
  genealogy_given (jie)     kinship + POS-backed name (弟亮 / 其子仁果; tags name only).
  struct_fuxing   (jie)     复姓 + POS.Giv given.
  struct_xingming (jie)     clean 姓 + POS.Giv given.
  semantic_given2 (jie)     complete POS.Giv + person-predicate frame.
  gloss_geneal     (jie)     genealogy / 谥曰 / enumerated-name prepass.
  jie_anaphora     (jie)     same-jie, earlier-anchor-only given-name anaphora.
Guards (identity-independent): POS function/geo vetoes, BIO boundaries, 复姓
  left-guard, BLOCK1/BLOCK2, 爵-head gate, 仆射, and 谥号.
"""
from __future__ import annotations
import sys, os, json, collections, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import seed as S  # noqa: E402
import pos_giv  # noqa: E402
import evidence as E  # noqa: E402
from pathlib import Path

# Process every rule at 节 (jie / numbered-block) scope: a block's paragraphs are
# joined into one text so anchors, gloss prepass and anaphora all see the whole 节
# (a bare given resolves to a full name in a sibling paragraph). ON by default;
# set to "0" for the old per-paragraph scope.
ANAPHORA_BLOCK = os.environ.get("ZTJ_ANAPHORA_BLOCK", "1") != "0"


def rules_bundle_sha256():
    """Hash every source file that can change Agent-1 admission behavior."""
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), Path(E.__file__).resolve()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

# ── surname / boundary tables ────────────────────────────────────────────────
CLEAN = S.CLEAN_SURNAMES
COMPOUND = S.COMPOUND
COMPOUND3 = S.COMPOUND3
SUR_EXTRA = set("高严任武史田文安万华成牛丁乐金时后元凌楼洪云柳鲁穆归楚许")
SUR_ALL = CLEAN | SUR_EXTRA               # for classifying 2-char names 姓-headed?
BLOCK1 = set("齐韩汉魏赵楚燕秦吴越梁陈宋鲁卫郑蔡曹许滕薛邾莒巴蜀晋隋周唐虞夏商"
             "大王侯公伯子男帝后妃太世储君贰")
BLOCK2 = {"中山", "常山", "长沙", "河间", "东平", "太子", "世子", "公子", "王子", "嗣王"}
KING_POLITY = BLOCK1 - set("大王侯公伯子男帝后妃太世储君贰")
NAMESTART = set("、，。；：︰「」『』（）〔〕·！？　 \n\u0001,.;:!?\"'")
APPOS_TAIL = set("使军尉守丞卿郎牧将相者长监令傅师保都统帅侯公王后妃主民酋蛮曹夫")
FANGWEI = set("东西南北")
JUE_HEAD = set("\u738b\u516c\u4faf")
# A two-char KB alias still needs local syntactic evidence when it is embedded in
# prose rather than introduced at a name boundary. These are person-selecting verbs
# on the left and predicates that conventionally take a named person as subject.
PERSON_LEFT_VERBS = set("召使遣命拜封杀执斩诛攻伐击问谓告从见留逐废立迎送任赐讨擒袭说劝诣遗谗谮劾")
FUNCTION_HANDLE_OBJECT_VERBS = set("劝从遣")
FUNCTION_HANDLE_SUBJECT_PREDICATES = set("破出知欲为应还归走奔死降入如")
FUNCTION_HANDLE_SUBJECT_VETO = {"当"}
PERSON_RIGHT_PRED = set("曰云言白为欲乃遂亦因与及从见请谏谓告使率帅攻击杀死亡降走奔至来去还归领兼屯卒议困")
PERSON_SUBJECT_PRED = set("曰云言白请谏率帅攻击杀死亡降走奔至来去还归遣命引闻问怒使从纳听拒救议困")
PERSON_REPORTING_VERBS = set("白")
PERSON_FOLLOW_OBJECT = set("上帝王主驾军之")
PERSON_OBJECT_TAIL = NAMESTART | set("为曰云言将帅相王公侯卿")
PERSON_ZHI_PRED = set("为言罪死谋策党徒子孙弟兄父母功时")
COORD_HEAD = set("与及")
COORD_TAIL = set("、等谋战俱共同合相")
COORD_PERSON_VERBS = set("用如若举采征荐依慕历诏以钉生")
AMBIGUOUS_PERSONAL_TITLES = {"长君", "嗣君", "郎君", "王君"}
APPELLATION_MARKERS = ("字", "号曰", "名曰", "谓之", "称之", "称为", "号为", "名为")
APP_DEFINITION_PREFIXES = ("号曰", "字")
TITLE_SUBJECT_LEFT = NAMESTART | set("而则乃故子弟兄")
OFFICE_TITLES = tuple(sorted((
    "大将军", "骠骑将军", "车骑将军", "卫将军", "衞将军",
    "前将军", "后将军", "左将军", "右将军", "将军",
    "大司马", "司徒", "司空", "丞相", "相国", "太尉", "太傅", "太师",
    "散骑常侍", "常侍",
), key=len, reverse=True))
OFFICE_SHI_TITLES = ("侍御史", "御史", "刺史", "长史", "内史", "令史", "黄门史")
PERSON_APPOS_TAIL = APPOS_TAIL | set("徒督史人贼书")
PERSON_ROLE_PREFIXES = tuple(sorted((
    "比丘尼", "胡僧", "梵僧", "沙门", "释子", "道人", "道士", "女冠",
    "处士", "隐士", "方士", "术士", "相士", "医者", "优人", "伶人",
    "贼臣", "奸臣", "权臣", "近臣", "嬖臣",
    "僧", "尼", "医", "巫", "优", "伶",
), key=len, reverse=True))
NAMING_PERSON_RELATIONS = (
    "兄子", "弟子", "长子", "次子", "少子", "幼子", "庶子", "养子",
    "其子", "之子", "生子", "生男", "太子", "皇子",
)
NAMING_IMPLICIT_MARKERS = (
    "更其姓名", "更姓名", "复其姓名", "复姓名", "赐姓名",
    "更其名", "更名", "改名", "易名", "自名", "赐名",
)
ADMIN_INTRO_TAIL = set("郡州县國国邑")
APPOINT_TITLES = (
    "皇太子", "大将军", "中郎将", "左庶长", "丞相", "太傅", "太守", "刺史",
    "内史", "御史", "校尉", "都尉", "尚书", "侍中", "司马", "司徒", "司空",
    "将军", "皇后", "太子", "大夫", "国尉", "左贤王", "右贤王",
    "王", "公", "侯", "相", "郎", "令", "长", "尉", "牧",
)
HUMAN_APPOINT_TITLES = tuple(sorted(
    set(APPOINT_TITLES) | {"女官"},
    key=len,
    reverse=True,
))
FULLNAME_TITLE_CONTINUATIONS = ("马步都指挥使", "都指挥使")
PU, SHE = "\u4ec6", "\u5c04"
SHI_YUE = ("\u8c25\u66f0", "\u8c25\u4e3a")
SHI = "\u8c25"
FUNCTION_POS = {"AUX", "ADV", "PART", "ADP", "CCONJ", "SCONJ", "PRON", "NUM", "SYM"}
POS_FUNCTION_VETO_SCORE = 0.9
KNOWN_FULLNAME_POS_SCORE = 0.7
TRANSLATION_OVEREXTENSION_TAIL = set(
    "不与举于军分又召启命复多奏尊常托新有望权来格治深知等草见诈诳辞退遂闻妃"
)
TRANSLATION_BARE_PERSON_TITLES = {"赞普"}
PERSON_TITLE_SUFFIXES = (
    "长公主", "神皇后", "皇太后", "皇后", "太后", "公主", "可汗", "单于",
    "王", "后", "公", "侯",
)
INHERENT_PERSON_TITLE_SUFFIXES = frozenset(PERSON_TITLE_SUFFIXES[:-4])
TITLE_EPITHET_END = set("穆哀简武昭襄灵元成康靖烈孝文宣景惠悼献懿恭肃")
FIEF_POSTHUMOUS_EPITHETS = set("孝武文昭宣景惠哀悼简成平康靖烈灵")
LEXICAL_RULER_TITLES = {"始皇", "主父"}
TITLE_NONPERSON_COMPONENTS = {
    "先帝", "匈奴", "乌桓", "右贤", "左贤", "前部", "南越", "月氏",
    "莎车", "鲜卑", "柔然", "突厥", "蠕蠕", "扶余", "波斯", "谷浑",
    "回纥", "回鹘", "黑姓", "乌长", "诸侯", "诸国", "臣国", "郡县",
}
TITLE_COMPONENT_BAD_HEAD = set("曰葬杀称谓召是今知等")
TITLE_NOMINAL_COMPONENTS = {"天亲"}
TITLE_EMBEDDED_COMPONENTS = {"天亲", "崇德"}
TITLE_PREDICATE_MODIFIERS = set("每尝常辄遂乃亦皆复又因即东西南北")
TITLE_SUBJECT_PREDICATES = PERSON_RIGHT_PRED | set("游叹巡")

REPO = Path(__file__).resolve().parents[3]
_TEXT = REPO / "web" / "public" / "text"
_PERS = Path(__file__).resolve().parents[1]
_ADMIN_PLACES = Path(__file__).resolve().parent / "admin-places.json"


# ── identity-free shared evidence ────────────────────────────────────────────
class Corpus:
    def __init__(self, ner, admin_places, geo_names):
        self.ner = ner            # set of NER candidate surfaces (corroboration)
        self.ner_maxL = max(2, max(map(len, ner), default=0))
        self.admin_places = admin_places  # surface -> exactly attested CE years
        self.geo_names = geo_names


def _is_xing_headed(nm):
    return nm[:1] in SUR_ALL or nm[:2] in COMPOUND or nm[:3] in COMPOUND3


def load_corpus():
    ner = set()
    for f in ("ner_candidates.json", "ner_model_candidates.json"):
        fp = _PERS / f
        if fp.exists():
            ner |= set(json.loads(fp.read_text(encoding="utf-8")).keys())
    if not _ADMIN_PLACES.is_file():
        raise FileNotFoundError(
            f"missing temporal administrative-place lexicon: {_ADMIN_PLACES}; "
            "run twostage/build_admin_places.py"
        )
    admin_blob = json.loads(_ADMIN_PLACES.read_text(encoding="utf-8"))
    if admin_blob.get("version") != 1:
        raise ValueError(f"unsupported admin-place lexicon version: {_ADMIN_PLACES}")
    admin_places = collections.defaultdict(set)
    geo_names = {
        row["surface"]
        for row in admin_blob.get("high_confidence_geo_entities", [])
        if isinstance(row.get("surface"), str)
    }
    for row in admin_blob.get("fallback_admin_origins", []):
        surface = row.get("surface")
        years = row.get("attested_years")
        if not isinstance(surface, str) or not surface or not isinstance(years, list):
            raise ValueError(f"invalid admin-place row in {_ADMIN_PLACES}")
        admin_places[surface].update(int(year) for year in years)
    return Corpus(
        ner,
        {surface: frozenset(years) for surface, years in admin_places.items()},
        geo_names,
    )


# ── shared guards ────────────────────────────────────────────────────────────
def _shi_guard(t, i):
    return t[max(0, i - 2):i] in SHI_YUE or t[max(0, i - 1):i] == SHI


def _fuxing_left(t, i):
    return t[i - 1:i + 1] in COMPOUND or t[i - 2:i + 1] in COMPOUND3


def _jue_ok(t, i, gset):
    prev = t[i - 1] if i > 0 else "\u0001"
    if prev in FANGWEI:
        return False
    return prev in NAMESTART or prev in APPOS_TAIL or (i + 1) in gset


def _all_high_confidence_function_pos(ctx, start, end):
    tokens = ctx.tokens_for(start, end)
    if not tokens or tokens[0].start != start or tokens[-1].end != end:
        return False
    if any(left.end != right.start for left, right in zip(tokens, tokens[1:])):
        return False
    all_function = all(
        token.pos in FUNCTION_POS
        and token.score is not None
        and token.score >= POS_FUNCTION_VETO_SCORE
        for token in tokens
    )
    prev = ctx.t[start - 1:start] or "\u0001"
    nxt = ctx.t[end:end + 1]
    person_subject = prev in NAMESTART and nxt in PERSON_SUBJECT_PRED
    return all_function and not person_subject


def _translation_fullname_pos_ok(ctx, start, end):
    """Require the original POS model to describe the whole mapped span as a name."""
    tokens = ctx.tokens_for(start, end)
    if not tokens:
        return False
    if any(
        token.pos != "PROPN"
        or not any(
            name_type in token.tag
            for name_type in ("NameType=Sur", "NameType=Giv", "NameType=Prs")
        )
        for token in tokens
    ):
        return False
    previous = ctx.token_at(start - 1) if start > 0 else None
    if (
        ctx.t[start:start + 1] == "王"
        and (
            ctx.t[start - 1:start] in "前后左右大小"
            or (
                previous is not None
                and previous.end == start
                and (
                    "NameType=Geo" in previous.tag
                    or "NameType=Nat" in previous.tag
                )
            )
        )
    ):
        # 鄯善王广: modern NER may call 王广 a full name, but 王 is the title.
        return False
    return True


def _surface_has_local_person_token(ctx, surface):
    """Whether another exact local occurrence carries explicit person-name morphology."""
    cursor = ctx.t.find(surface)
    while cursor >= 0:
        end = cursor + len(surface)
        tokens = ctx.tokens_for(cursor, end)
        if tokens and any(
            token.pos == "PROPN"
            and any(
                name_type in token.tag
                for name_type in ("NameType=Sur", "NameType=Giv", "NameType=Prs")
            )
            and "NameType=Geo" not in token.tag
            and "NameType=Nat" not in token.tag
            for token in tokens
        ):
            return True
        cursor = ctx.t.find(surface, cursor + 1)
    return False


def _is_person_name_token(token):
    return (
        token is not None
        and token.pos == "PROPN"
        and any(
            name_type in token.tag
            for name_type in ("NameType=Sur", "NameType=Giv", "NameType=Prs")
        )
        and "NameType=Geo" not in token.tag
        and "NameType=Nat" not in token.tag
    )


def _is_named_entity_token(token):
    return (
        token is not None
        and token.pos == "PROPN"
        and "NameType=" in token.tag
    )


def _occurrence_has_polity_frame(ctx, start, end):
    """Whether this occurrence, rather than another same-surface use, is a polity."""
    suffix = ctx.t[end:end + 4]
    surface = ctx.t[start:end]
    if suffix.startswith(
        ("小国", "部众", "之众", "举部", "军士")
    ) or any(
        surface + marker in ctx.t
        for marker in ("小国", "举部", "入贡")
    ) or "安抚" + surface + "使" in ctx.t:
        return True
    if ctx.t[start - 1:start] in set("五七十") and (
        suffix.startswith(("大啜", "大俟斤"))
        or ctx.t[max(0, start - 2):start] == "号五"
    ):
        return True
    if ctx.t[start - 1:start] == "姓":
        return True
    if (
        suffix.startswith("强盛")
        and "国患" in ctx.t
    ) or (
        suffix.startswith(("酷逆", "无道"))
        and any(marker in ctx.t for marker in ("人神", "弑君", "虐民"))
    ):
        return True
    tokens = ctx.tokens_for(start, end)
    if not tokens or any(_is_person_name_token(token) for token in tokens):
        return False
    if suffix.startswith(("部", "国", "族")):
        return True
    polity_morphology = any(
        "NameType=Geo" in token.tag or "NameType=Nat" in token.tag
        for token in tokens
    )
    nominal_morphology = all(token.pos in {"NOUN", "PROPN"} for token in tokens)
    return (
        nominal_morphology
        and (
            (
                (
                    ctx.t[start - 1:start] in set("伐击袭侵拔灭寇")
                    or ctx.t[start - 2:start] in {"攻伐", "侵袭"}
                )
                and (
                    ("其王" in ctx.t and "部落" in ctx.t)
                    or "诸夷" in ctx.t
                )
            )
            or ctx.t[end:end + 1] in {"寇", "附"}
            or suffix.startswith(("遣使", "入贡"))
            or (
                polity_morphology
                and len(surface) >= 3
                and surface + "王" in ctx.t
            )
        )
    )


def _has_person_bio_left_continuation(ctx, start):
    current = ctx.token_at(start)
    previous = ctx.token_at(start - 1)
    if (
        current is not None
        and current.start < start
        and _is_named_entity_token(current)
    ):
        return True
    return (
        current is not None
        and current.start == start
        and (
            current.bio == "I"
            or "NameType=Giv" in current.tag
            or "NameType=Prs" in current.tag
        )
        and _is_named_entity_token(current)
        and previous is not None
        and previous.end == start
        and ctx.t[start - 1:start] not in (PERSON_LEFT_VERBS | {"将"})
        and (
            previous.bio in {"B", "I"}
            or "NameType=Sur" in previous.tag
            or "NameType=Prs" in previous.tag
        )
        and _is_named_entity_token(previous)
    )


def _has_person_bio_right_continuation(ctx, start, end):
    tokens = ctx.tokens_for(start, end)
    following = ctx.token_at(end)
    containing = ctx.token_at(end - 1)
    if (
        containing is not None
        and containing.end > end
        and _is_named_entity_token(containing)
    ):
        return True
    return (
        bool(tokens)
        and tokens[-1].end == end
        and (
            tokens[-1].bio in {"B", "I"}
            or "NameType=Sur" in tokens[-1].tag
            or "NameType=Prs" in tokens[-1].tag
        )
        and (
            _is_named_entity_token(tokens[-1])
            or tokens[-1].bio in {"B", "I"}
        )
        and following is not None
        and following.start == end
        and ctx.t[end:end + 1] not in NAMESTART
        and ctx.t[end:end + 1]
        not in {"王", "公", "侯", "君", "卿", "子", "妃", "后"}
        and (
            following.bio == "I"
            or "NameType=Giv" in following.tag
            or "NameType=Prs" in following.tag
        )
        and (
            _is_named_entity_token(following)
            or following.bio == "I"
        )
    )


def _has_repeated_model_extension(ctx, start, end):
    for left in range(0, 3):
        for right in range(0, 3):
            if not (left or right) or start < left or end + right > len(ctx.t):
                continue
            extended = ctx.t[start - left:end + right]
            if (
                extended in ctx.corpus.ner
                and not (set(extended) & NAMESTART)
                and ctx.t.count(extended) >= 2
            ):
                occurrence = ctx.t.find(extended)
                while occurrence >= 0:
                    title_window = ctx.t[
                        occurrence + len(extended):
                        occurrence + len(extended) + 12
                    ]
                    if (
                        title_window[:1] in {"、", "，"}
                        and any(
                            title in title_window
                            for title in ("可汗", "单于")
                        )
                    ):
                        return True
                    occurrence = ctx.t.find(extended, occurrence + 1)
    return False


def _has_polity_title_left_continuation(ctx, start, end):
    surface = ctx.t[start:end]
    previous = ctx.token_at(start - 1)
    return (
        len(surface) >= 2
        and surface[0] in {"王", "公", "侯", "君"}
        and previous is not None
        and previous.end == start
        and (
            "NameType=Geo" in previous.tag
            or "NameType=Nat" in previous.tag
        )
    )


def _location_sequence_end(ctx, start):
    cursor = start
    first = True
    while cursor < len(ctx.t):
        token = ctx.token_at(cursor)
        if (
            token is None
            or token.start != cursor
            or not (
                "NameType=Geo" in token.tag
                or "NameType=Nat" in token.tag
                or (not first and "Case=Loc" in token.tag)
            )
        ):
            break
        cursor = token.end
        first = False
    return cursor


def _has_geo_title_right_continuation(ctx, end):
    location_end = _location_sequence_end(ctx, end)
    return location_end > end and any(
        ctx.t.startswith(title, location_end)
        for title in ("公主", "王", "公", "侯", "君")
    )


def _has_location_office_right_continuation(ctx, start, end):
    tokens = ctx.tokens_for(start, end)
    if not tokens or not all("NameType=Sur" in token.tag for token in tokens):
        return False
    location_end = _location_sequence_end(ctx, end)
    return (
        location_end > end
        and ctx.t.startswith("以来", location_end)
    )


def _has_office_continuation(ctx, end):
    if ctx.t.startswith(("长子", "长女"), end):
        return False
    title = next(
        (
            title
            for title in (
                HUMAN_APPOINT_TITLES
                + OFFICE_TITLES
                + OFFICE_SHI_TITLES
                + ("节度使", "都督", "学士", "诸军", "镇军", "太师")
            )
            if ctx.t.startswith(title, end)
        ),
        None,
    )
    if title is None:
        return False
    if len(title) > 1:
        return True
    tokens = ctx.tokens_for(end, end + len(title))
    if tokens:
        return all(token.pos in {"NOUN", "PROPN"} for token in tokens)
    containing = ctx.token_at(end)
    return (
        containing is not None
        and containing.start <= end
        and containing.end >= end + len(title)
        and containing.pos in {"NOUN", "PROPN"}
    )


def _has_person_designation_right_continuation(ctx, start, end):
    surface = ctx.t[start:end]
    suffixes = (
        "良娣", "妃", "鄕公", "乡公", "将军", "镇军", "太师", "学士",
    )
    for suffix in suffixes:
        if ctx.t.startswith(suffix, end):
            return True
        for overlap in range(1, len(suffix)):
            if (
                surface.endswith(suffix[:overlap])
                and ctx.t.startswith(suffix[overlap:], end)
            ):
                return True
    following = ctx.t[end + 1:end + 2]
    return (
        ctx.t.startswith("君", end)
        and bool(following)
        and following not in NAMESTART
    )


def _has_person_name_before_rank_title(ctx, start, end):
    return (
        ctx.t[end:end + 1] in {"王", "公", "侯"}
        and ctx.t[start - 1:start] in PERSON_LEFT_VERBS
    )


def _foreign_title_followed_by_name(ctx, end):
    title = next(
        (
            candidate
            for candidate in ("可汗", "单于")
            if ctx.t.startswith(candidate, end)
        ),
        None,
    )
    if title is None:
        return False
    name_start = end + len(title)
    following = ctx.token_at(name_start)
    return (
        following is not None
        and following.start == name_start
        and following.pos == "PROPN"
        and (
            "NameType=Giv" in following.tag
            or "NameType=Prs" in following.tag
        )
    )


def _surface_has_jie_collective_frame(ctx, surface):
    """Whether usage in this numbered section explicitly defines a group."""
    text = ctx.t
    if any(
        surface + suffix in text
        for suffix in ("数千骑", "数万骑")
    ):
        return True
    if (
        surface + "将军" in text
        and surface + "军" in text
    ):
        return True
    if "谓之" + surface in text and any(
        marker in text
        for marker in (
            "募" + surface,
            "集" + surface,
            surface + "数千",
            surface + "数万",
        )
    ):
        return True
    return False


def _surface_has_supernatural_frame(ctx, surface):
    """Whether the local passage explicitly treats a surface as a deity."""
    return (
        any(surface + predicate in ctx.t for predicate in ("授", "以"))
        and any(
            marker in ctx.t
            for marker in ("真官", "鸾鹤", "道院", "焚修")
        )
    )


def _translation_model_surface_ok(ctx, start, end):
    """Allow exact translated identities when local model evidence is structurally safe."""
    surface = ctx.t[start:end]
    tokens = ctx.tokens_for(start, end)
    if (
        surface not in ctx.corpus.ner
        or not tokens
        or tokens[0].start != start
        or tokens[-1].end != end
        or any(left.end != right.start for left, right in zip(tokens, tokens[1:]))
        or _shi_guard(ctx.t, start)
        or (
            len(surface) == 2
            and surface[0] in "甲乙丙丁戊己庚辛壬癸"
            and surface[1] in "子丑寅卯辰巳午未申酉戌亥"
        )
        or ctx.t[end:end + 1] in {
            "寺", "军", "城", "州", "郡", "县", "镇", "寨", "关", "谷",
        }
        or ctx.t[end:end + 2].endswith("军")
        or ctx.t[end:end + 1] in {"、", "等", "酋"}
        or _local_nat_or_geo(ctx, start, end)
        or surface in APPOINT_TITLES
        or surface in OFFICE_TITLES
        or _occurrence_has_polity_frame(ctx, start, end)
    ):
        return False
    repeated = ctx.t.find(surface, 0, start) >= 0 or ctx.t.find(surface, end) >= 0
    title = any(
        surface.endswith(suffix) and len(surface) > len(suffix)
        for suffix in MODEL_PERSON_TITLE_SUFFIXES
    )
    return (
        title
        or (repeated and _surface_has_local_person_token(ctx, surface))
    )


def _translation_identity_overextended(ctx, candidate):
    """Reject modern-NER names that swallowed a following predicate or particle."""
    for surface in {candidate["identity_surface"], candidate["surface"]}:
        if len(surface) < 2:
            continue
        stem_end = candidate["end"] - 1
        stem_start = stem_end - len(surface) + 1
        if stem_start < 0 or not _complete_person_pos(ctx, stem_start, stem_end):
            continue
        tail_token = ctx.token_at(candidate["end"] - 1)
        tail_is_name = (
            tail_token is not None
            and tail_token.start == candidate["end"] - 1
            and tail_token.end == candidate["end"]
            and tail_token.pos == "PROPN"
            and any(
                name_type in tail_token.tag
                for name_type in ("NameType=Giv", "NameType=Prs")
            )
        )
        if surface[-1] in TRANSLATION_OVEREXTENSION_TAIL or not tail_is_name:
            return True
    return False


def rule_translation_fullname(ctx, i):
    """Admit a safe exact full-identity mapping as an original-text name anchor."""
    candidate = ctx.translation_fullnames.get(i)
    if candidate is None:
        return None
    end = candidate["end"]
    surface = candidate["surface"]
    start = i
    if surface.startswith("名") and len(surface) >= 2:
        start += 1
        surface = surface[1:]
    previous = ctx.token_at(start - 1) if start > 0 else None
    surface_tokens = ctx.tokens_for(start, end)
    if (
        previous is not None
        and previous.end == start
        and previous.tag == "PROPN|NameType=Sur"
        and (
            previous.text in CLEAN
            or previous.text in COMPOUND
            or previous.text in COMPOUND3
        )
        and surface_tokens
        and all(
            token.pos == "PROPN"
            and (
                "NameType=Giv" in token.tag
                or "NameType=Prs" in token.tag
            )
            for token in surface_tokens
        )
        and end - previous.start <= 4
    ):
        start = previous.start
        surface = ctx.t[start:end]
    next_token = ctx.token_at(end)
    if (
        not 2 <= end - start <= 6
        or ctx.t[start:end] != surface
        or any(ctx.consumed[start:end])
        or set(surface) & (NAMESTART | GLOSS_SEP)
        or ctx.t[end:end + 2] == "之号"
        or ctx.t[end:end + 1] in JUE_HEAD
        or surface in TRANSLATION_BARE_PERSON_TITLES
        or (
            surface.startswith("司马")
            and len(surface) - len("司马") > 2
        )
        or (
            next_token is not None
            and next_token.start == end
            and next_token.pos == "PROPN"
            and any(
                name_type in next_token.tag
                for name_type in ("NameType=Sur", "NameType=Giv", "NameType=Prs")
            )
        )
        or _translation_identity_overextended(ctx, candidate)
        or not (
            _translation_fullname_pos_ok(ctx, start, end)
            or _translation_model_surface_ok(ctx, start, end)
        )
    ):
        return None
    return (start, end, surface, "translation_fullname")


def _translation_given_syntax_ok(ctx, start, end):
    """Original-text admission for an exact mapped given-name occurrence."""
    t = ctx.t
    prev = t[start - 1:start] or "\u0001"
    nxt = t[end:end + 1]
    tokens = ctx.tokens_for(start, end)
    if not tokens:
        return False
    if end - start == 1:
        bio_end = ctx.gspans.get(start)
        if bio_end is not None and bio_end > end:
            return False
    next_token = ctx.token_at(end)
    mapped_object_predicate = (
        prev in PERSON_LEFT_VERBS | {"随"}
        and next_token is not None
        and next_token.start == end
        and next_token.pos in {"VERB", "AUX"}
    )
    embedded_coord_clause = (
        prev in PERSON_LEFT_VERBS | {"闻"}
        and nxt == "与"
        and any(
            predicate in t[end + 1:end + 10]
            for predicate in "争议攻击见从"
        )
    )
    object_frame = (
        (prev in PERSON_LEFT_VERBS | {"畏", "语", "随"} and nxt in PERSON_OBJECT_TAIL)
        or (prev == "语" and nxt == "曰")
        or (prev == "言" and t[end:end + 2] == "罪名")
        or (prev in PERSON_LEFT_VERBS and nxt in "往行")
        or (prev == "从" and end - start == 2 and nxt not in NAMESTART)
        or mapped_object_predicate
        or embedded_coord_clause
    )
    subject_frame = (
        prev in NAMESTART
        and (
            nxt in PERSON_SUBJECT_PRED | {"拔", "拥"}
            or t[end:end + 2] in {"不为", "以为"}
            or t[end:end + 3] == "因上言"
            or (
                nxt == "与"
                and any(
                    predicate in t[end + 1:end + 10]
                    for predicate in "争议攻击见从"
                )
            )
        )
    )
    appointment = prev == "以" and nxt in "为兼"
    return object_frame or subject_frame or appointment


def rule_translation_given(ctx, i):
    """Admit an exact mapped handle only in a controlled original-text frame."""
    candidate = ctx.translation_mentions.get(i)
    if candidate is None:
        return None
    end = candidate["end"]
    surface = candidate["surface"]
    previous_token = ctx.token_at(i - 1) if i > 0 else None
    embedded_fullname = (
        previous_token is not None
        and (
            (
                previous_token.end == i
                and previous_token.tag == "PROPN|NameType=Sur"
                and _complete_person_pos(ctx, previous_token.start, end)
            )
            or (
                previous_token.start < i
                and previous_token.end == end
                and previous_token.pos == "PROPN"
                and any(
                    name_type in previous_token.tag
                    for name_type in ("NameType=Prs", "NameType=Giv")
                )
            )
        )
    ) or any(
        i >= surname_len
        and (
            (surname_len == 1 and ctx.t[i - 1:i] in SUR_ALL)
            or (surname_len == 2 and ctx.t[i - 2:i] in COMPOUND)
            or (surname_len == 3 and ctx.t[i - 3:i] in COMPOUND3)
        )
        and _complete_person_pos(ctx, i - surname_len, end)
        for surname_len in (3, 2, 1)
    )
    prev = ctx.t[i - 1:i] or "\u0001"
    nxt = ctx.t[end:end + 1]
    strict_local_frame = (
        candidate.get("strict_local_owner", False)
        and (
            (prev == "为" and nxt == "所")
            or (prev in NAMESTART and nxt in PERSON_RIGHT_PRED)
        )
    )
    if (
        not 1 <= end - i <= 2
        or ctx.t[i:end] != surface
        or any(ctx.consumed[i:end])
        or set(surface) & (NAMESTART | GLOSS_SEP)
        or surface in BLOCK1
        or surface == "从容"
        or embedded_fullname
        or (
            not strict_local_frame
            and (
                _translation_identity_overextended(ctx, candidate)
                or not _translation_given_syntax_ok(ctx, i, end)
            )
        )
    ):
        return None
    return (i, end, surface, "translation_anaphora")


# ── rules: each yields (start, end, surface, chunk_type) ─────────────────────
def rule_corpus_lit3(ctx, i):
    """Complete POS/BIO personal-name span of at least three characters."""
    t, consumed = ctx.t, ctx.consumed
    for end in range(min(len(t), i + 6), i + 2, -1):
        if any(consumed[i:end]):
            continue
        if _complete_person_pos(ctx, i, end) and not _shi_guard(t, i):
            return (i, end, t[i:end], "pos_person_name")
    return None


def rule_known_title(ctx, i):
    """Compatibility shim: structural title admission lives in title_appellation."""
    return None


def rule_corpus_xing2(ctx, i):
    """Complete two-character POS/BIO surname-plus-given name."""
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    surf = t[i:i + 2]
    if len(surf) != 2 or any(consumed[i:i + 2]):
        return None
    if not _complete_person_pos(ctx, i, i + 2):
        return None
    if _nat_verb_before_coordinated_objects(ctx, i):
        return None
    if _fuxing_left(t, i):
        return None
    if (t[i - 1] if i > 0 else "") in BLOCK1 or t[i - 2:i] in BLOCK2:
        return None
    if surf[0] in JUE_HEAD and not _jue_ok(t, i, gset):
        return None
    if surf[0] == PU and t[i + 2:i + 3] == SHE:
        return None
    if _shi_guard(t, i):
        return None
    return (i, i + 2, surf, "pos_person_name")


def rule_corpus_jue2(ctx, i):
    """POS- and syntax-backed two-character fief/title byname."""
    t, consumed = ctx.t, ctx.consumed
    surf = t[i:i + 2]
    if (
        len(surf) != 2
        or surf[1:2] not in JUE_HEAD
        or surf not in ctx.corpus.ner
        or any(consumed[i:i + 2])
    ):
        return None
    prev = t[i - 1] if i > 0 else "\u0001"
    if _fuxing_left(t, i) or prev in SUR_ALL or prev in FANGWEI:
        return None
    if prev in "一二三四五六七八九十百千万亿":
        return None
    if surf[1] == "公" and t[i + 2:i + 3] == "主":
        return None
    head = ctx.token_at(i)
    if not (
        _complete_person_pos(ctx, i, i + 2)
        or (
            head is not None
            and head.start == i
            and head.end == i + 1
            and head.pos == "PROPN"
            and head.score is not None
            and head.score >= POS_FUNCTION_VETO_SCORE
            and (
                prev in PERSON_LEFT_VERBS | PERSON_APPOS_TAIL
                or t[i + 2:i + 3] in PERSON_RIGHT_PRED
            )
        )
    ):
        return None
    return (i, i + 2, surf, "alias")


ZHU = "\u4e3b"                                     # 主
# ruler-polity allowlist: [北南东西后]?[POLITY1]主  |  契丹主
# single-char polities that mean "ruler of X" when followed by 主 (mined from corpus
# + golden). Common overloaded X主 (公人为之其谋明军戍…) are simply NOT in the set.
POLITY1 = set("\u9b4f\u5434\u5510\u9f50\u8700\u5468\u968b\u95fd\u6c49\u6881\u590f"
              "\u9648\u71d5\u79e6\u8d75\u664b\u6210\u5b8b")   # 魏吴唐齐蜀周隋闽汉梁夏陈燕秦赵晋成宋
POLITY_PREFIX = set("\u5317\u5357\u4e1c\u897f\u540e")          # 北南东西后 (split dynasties)
POLITY_NUM_PREFIX = set("\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d"
                        "\u5341\u767e\u5343\u4e07\u4ebf")
POLITY2 = ("\u5951\u4e39",)                                    # 契丹


def rule_polity_appos(ctx, i):
    """Affiliation marker + known two-char full name, e.g. 齐田和 / 秦张仪.

    BLOCK1 remains the default because state/title characters are highly ambiguous.
    A POS-backed given position turns this specific shape into an apposition: the
    marker identifies polity/place affiliation while the following span is the person.
    """
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    surf = t[i:i + 2]
    if not _complete_person_pos(ctx, i, i + 2) or any(consumed[i:i + 2]):
        return None
    if i == 0 or t[i - 1] not in POLITY1 or i + 1 not in gset:
        return None
    if _fuxing_left(t, i) or _shi_guard(t, i):
        return None
    return (i, i + 2, surf, "xing2_appos")


def rule_block_appos(ctx, i):
    """Title, relationship, or affiliation marker + known two-char full name.

    BLOCK remains the default for structural surname detection. A confirmed KB full
    name whose given position is POS-backed and bounded can safely bypass it in
    expressions such as 平阳侯曹参, 其子宋襄, and 留后韩简.
    """
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    surf = t[i:i + 2]
    if not _complete_person_pos(ctx, i, i + 2) or any(consumed[i:i + 2]):
        return None
    prev = t[i - 1:i]
    office_left = any(
        len(title) >= 2 and t[max(0, i - len(title)):i] == title
        for title in APPOINT_TITLES
    )
    if prev not in BLOCK1 and t[max(0, i - 2):i] not in BLOCK2 and not office_left:
        return None
    if i + 1 not in gset or i + 2 in gset:
        return None
    if _fuxing_left(t, i) or _shi_guard(t, i):
        return None
    return (i, i + 2, surf, "xing2_appos")


def rule_role(ctx, i):
    """Polity ruler title: [北南东西后]?[POLITY1]主 | 契丹主. Corpus-scoped closed
    lexicon; 主 alone is overloaded so only attested ruler polities fire."""
    t, consumed = ctx.t, ctx.consumed
    # 契丹主 (2-char foreign polity)
    for p2 in POLITY2:
        if t[i:i + 2] == p2 and t[i + 2:i + 3] == ZHU and not any(consumed[i:i + 3]):
            return (i, i + 3, t[i:i + 3], "role")
    # [prefix]+POLITY1+主  (3-char, split dynasties: 北汉主 东魏主 后周主)
    if t[i] in POLITY_PREFIX and t[i + 1:i + 2] and t[i + 1] in POLITY1 \
            and t[i + 2:i + 3] == ZHU and not any(consumed[i:i + 3]):
        return (i, i + 3, t[i:i + 3], "role")
    # POLITY1+主  (2-char)
    if t[i] in POLITY1 and t[i + 1:i + 2] == ZHU and not any(consumed[i:i + 2]):
        return (i, i + 2, t[i:i + 2], "role")
    return None


def rule_polity_king(ctx, i):
    """Polity + 王 is personhood even when identity is unresolved."""
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    if t[i:i + 1] in POLITY_PREFIX and t[i + 1:i + 2] in KING_POLITY \
            and t[i + 2:i + 3] == "\u738b":
        end = i + 3
    elif t[i:i + 1] in KING_POLITY and t[i + 1:i + 2] == "\u738b":
        end = i + 2
    else:
        return None
    if end in gset or any(consumed[i:end]):
        return None
    return (i, end, t[i:end], "role")


def rule_jue_name(ctx, i):
    """Title+given-name: [北南东西后]?POLITY1 + 王/公/侯 + given(1-2 char).
    A bare fief-title (秦王/晋王) is ambiguous across many holders so golden does
    NOT tag it, but title+given-name (秦王坚, 秦王世民, 魏王操) is an unambiguous
    person reference -> tag the whole span as alias. The given-name gate (char
    after the title must be a POS given position) is the precision lever; no
    identity is bonded (Agent 2's job)."""
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    n = len(t)
    if t[i] in POLITY_PREFIX and t[i + 1:i + 2] and t[i + 1] in POLITY1 \
            and t[i + 2:i + 3] and t[i + 2] in JUE_HEAD:
        j = i + 3                                    # 后秦王苌, 东魏王…
    elif t[i] in POLITY1 and t[i + 1:i + 2] and t[i + 1] in JUE_HEAD:
        j = i + 2                                    # 秦王坚, 魏公操…
    else:
        return None
    if j >= n or j not in gset:                      # must be title + given name
        return None
    g = 2 if (
        j + 1 < n
        and (j + 1) in gset
        and t[j + 1] not in NAMESTART
    ) else 1
    e = j + g
    if any(consumed[i:e]):
        return None
    return (i, e, t[i:e], "title_name")


TITLE_NAME_STATE_RIGHT = set("幼少长壮老年")


def rule_multifief_jue_name(ctx, i):
    """POS-proven multi-character fief + 王/公/侯 + personal given name."""
    t, cs, consumed = ctx.t, ctx.corpus, ctx.consumed
    title_at = next(
        (j for j in range(i + 2, min(len(t), i + 5)) if t[j] in JUE_HEAD),
        None,
    )
    if title_at is None:
        return None
    if set(t[i:title_at]) & NAMESTART:
        return None
    fief_tokens = ctx.tokens_for(i, title_at)
    if (
        not fief_tokens
        or fief_tokens[0].start != i
        or fief_tokens[-1].end != title_at
        or fief_tokens[0].bio != "B"
        or any(token.bio != "I" for token in fief_tokens[1:])
        or any("Case=Loc" not in token.tag or "NameType=Geo" not in token.tag
               for token in fief_tokens)
    ):
        return None
    name_start = title_at + 1
    if (
        _complete_person_pos(ctx, name_start, name_start + 2)
        or _complete_person_pos(ctx, name_start, name_start + 3)
    ):
        return None
    end = ctx.gspans.get(name_start)
    if end is not None and 1 <= end - name_start <= 2:
        name_tokens = ctx.tokens_for(name_start, end)
        if (
            not name_tokens
            or name_tokens[0].start != name_start
            or name_tokens[-1].end != end
            or any(token.score is None or token.score < 0.7 for token in name_tokens)
        ):
            return None
    elif (
        name_start < len(t)
        and t[name_start] not in NAMESTART
        and t[name_start + 1:name_start + 2] in TITLE_NAME_STATE_RIGHT
        and not _all_high_confidence_function_pos(ctx, name_start, name_start + 1)
    ):
        end = name_start + 1
    else:
        return None
    if any(consumed[i:end]):
        return None
    return (i, end, t[i:end], "title_name")


def rule_office_name(ctx, i):
    """Office-title + given name, e.g. 大将军光 / 左将军桀.

    The full glued expression is a person reference and, like a full name, contributes
    its trailing given name to the same-jie anaphora roster.
    """
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    title = next((x for x in OFFICE_TITLES if t.startswith(x, i)), None)
    if title is None:
        return None
    j = i + len(title)
    if j >= len(t) or j not in gset:
        return None
    g = 2 if (
        j + 1 < len(t)
        and (j + 1) in gset
        and t[j + 1] not in NAMESTART
    ) else 1
    e = j + g
    if any(consumed[i:e]):
        return None
    return (i, e, t[i:e], "title_name")


def rule_office_fullname(ctx, i):
    """Office title followed by a POS-backed surname+given full name."""
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    title = next((
        candidate for candidate in sorted(APPOINT_TITLES, key=len, reverse=True)
        if len(candidate) >= 2 and t.startswith(candidate, i)
    ), None)
    if title is None:
        return None
    start = i + len(title)
    surface = t[start:start + 2]
    if (
        not _complete_person_pos(ctx, start, start + 2)
        or start + 1 not in gset
        or start + 2 in gset
    ):
        return None
    if any(consumed[start:start + 2]) or _shi_guard(t, start):
        return None
    return (start, start + 2, surface, "xing2_appos")


def rule_pos_known_fullname_appos(ctx, i):
    """Complete POS full name after a person role, office, or selecting verb."""
    t, consumed = ctx.t, ctx.consumed
    surname = ctx.token_at(i)
    if (
        surname is None
        or surname.start != i
        or surname.end != i + 1
        or surname.tag != "PROPN|NameType=Sur"
        or surname.score is None
        or surname.score < POS_FUNCTION_VETO_SCORE
    ):
        return None
    ends = []
    bio_end = ctx.gspans.get(surname.end)
    if bio_end is not None:
        ends.append(bio_end)
    cursor = surname.end
    while cursor < min(len(t), i + 3):
        token = ctx.token_at(cursor)
        if (
            token is None
            or token.start != cursor
            or not token.tag.endswith("PROPN|NameType=Giv")
            or token.score is None
            or token.score < POS_FUNCTION_VETO_SCORE
        ):
            break
        cursor = token.end
        ends.append(cursor)
    end = next(
        (
            candidate
            for candidate in sorted(set(ends), reverse=True)
            if _complete_person_pos(ctx, i, candidate)
        ),
        None,
    )
    if end is None or any(consumed[i:end]):
        return None
    prev = t[i - 1:i] or "\u0001"
    office_left = any(
        t[max(0, i - len(title)):i] == title
        for title in APPOINT_TITLES + OFFICE_TITLES + OFFICE_SHI_TITLES
    )
    if (
        prev not in PERSON_APPOS_TAIL
        and prev not in PERSON_LEFT_VERBS
        and not office_left
    ):
        return None
    surface = t[i:end]
    if (
        _shi_guard(t, i)
        or t[end:end + 1] in "郡州县國国"
    ):
        return None
    return (i, end, surface, "appos_fullname")


def rule_empress_title(ctx, i):
    """Inherently personal empress/dowager titles; identity belongs to Agent 2."""
    t, consumed = ctx.t, ctx.consumed
    for title in ("\u592a\u540e", "\u7687\u540e"):  # 太后, 皇后
        if t.startswith(title, i) and not any(consumed[i:i + len(title)]):
            return (i, i + len(title), title, "empress_title")
    return None


def rule_princess_title(ctx, i):
    """Complete local title component + 公主 is inherently person-bearing."""
    component_end = ctx.gspans.get(i)
    first_token = ctx.token_at(i)
    if component_end is None and i + 1 < len(ctx.t) and "\u3400" <= ctx.t[i] <= "\u9fff":
        continued = ctx.token_at(i + 1)
        if (
            continued is not None
            and continued.start == i + 1
            and continued.bio == "I"
        ):
            component_end = ctx.gspans.get(i + 1)
            first_token = continued
    if component_end is None and ctx.t[i + 2:i + 4] == "公主":
        component_tokens = ctx.tokens_for(i, i + 2)
        surface = ctx.t[i:i + 4]
        repeated = ctx.t.find(surface, 0, i) >= 0 or ctx.t.find(surface, i + 4) >= 0
        if (
            component_tokens
            and component_tokens[0].start == i
            and component_tokens[-1].end == i + 2
            and all(left.end == right.start for left, right in zip(
                component_tokens, component_tokens[1:]
            ))
            and repeated
            and ctx.t[i:i + 1] not in PERSON_LEFT_VERBS | {"许"}
            and not any(
                ctx.t[left:i + 4 + right] in ctx.corpus.ner
                for left, right in ((i - 1, 0), (i - 2, 0), (i, 1), (i, 2))
                if left >= 0 and i + 4 + right <= len(ctx.t)
                and (left != i or right)
            )
        ):
            component_end = i + 2
            first_token = component_tokens[0]
    if component_end is None:
        for component_len in range(1, 5):
            candidate_end = i + component_len
            if ctx.t[candidate_end:candidate_end + 2] != "公主":
                continue
            component_tokens = ctx.tokens_for(i, candidate_end)
            if (
                not component_tokens
                or component_tokens[0].start != i
                or component_tokens[-1].end != candidate_end
                or component_tokens[0].bio == "I"
            ):
                continue
            surface = ctx.t[i:candidate_end]
            all_proper = all(token.pos == "PROPN" for token in component_tokens)
            entity_units = sum(token.bio != "I" for token in component_tokens)
            has_person_morphology = any(
                "NameType=Prs" in token.tag
                or "NameType=Giv" in token.tag
                or "NameType=Sur" in token.tag
                for token in component_tokens
            )
            realm_before_complete_entity = (
                len(component_tokens) > 1
                and "NameType=Nat" in component_tokens[0].tag
                and component_tokens[1].bio == "B"
            )
            surname_before_given = (
                len(component_tokens) > 1
                and "NameType=Sur" in component_tokens[0].tag
                and (
                    "NameType=Giv" in component_tokens[1].tag
                    or "NameType=Prs" in component_tokens[1].tag
                )
            )
            reliable_single = (
                len(surface) > 1
                or (
                    component_tokens[0].score is not None
                    and component_tokens[0].score >= KNOWN_FULLNAME_POS_SCORE
                )
            )
            lexical_rank = surface in {"长", "大长"}
            strict_nominal = (
                len(surface) == 2
                and all(token.pos in {"PROPN", "NOUN", "NUM"} for token in component_tokens)
                and (
                    _strict_person_frame(ctx, i, candidate_end + 2)
                    or (
                        ctx.t[i - 1:i] == "为"
                        and ctx.t[candidate_end + 2:candidate_end + 4] == "，妻"
                    )
                )
            )
            if (
                lexical_rank
                or (
                    all_proper
                    and not realm_before_complete_entity
                    and not surname_before_given
                    and reliable_single
                    and ctx.t[i:i + 1] not in PERSON_LEFT_VERBS
                    and (entity_units == 1 or has_person_morphology)
                )
                or strict_nominal
            ):
                component_end = candidate_end
                first_token = component_tokens[0]
            break
    if (
        component_end is None
        or not 1 <= component_end - i <= 4
        or ctx.t[component_end:component_end + 2] != "公主"
        or any(not "\u3400" <= char <= "\u9fff" for char in ctx.t[i:component_end])
        or (
            first_token is not None
            and first_token.start == i
            and first_token.bio == "I"
        )
    ):
        return None
    end = component_end + 2
    if end > len(ctx.t) or any(ctx.consumed[i:end]):
        return None
    suffix_tokens = ctx.tokens_for(component_end, end)
    if (
        len(suffix_tokens) != 2
        or suffix_tokens[0].start != component_end
        or suffix_tokens[0].end != component_end + 1
        or suffix_tokens[1].start != component_end + 1
        or suffix_tokens[1].end != end
        or any(token.pos != "NOUN" for token in suffix_tokens)
    ):
        return None
    return (i, end, ctx.t[i:end], "princess_title")


def rule_surname_empress(ctx, i):
    """Complete surname morphology plus a female court title."""
    t, consumed = ctx.t, ctx.consumed
    for surname_length in (2, 1):
        surname = t[i:i + surname_length]
        if surname not in CLEAN and surname not in COMPOUND:
            continue
        surname_tokens = ctx.tokens_for(i, i + surname_length)
        if not (
            surname_tokens
            and surname_tokens[0].start == i
            and surname_tokens[-1].end == i + surname_length
            and surname_tokens[0].bio != "I"
            and all(
                token.pos == "PROPN" and "NameType=Sur" in token.tag
                for token in surname_tokens
            )
        ):
            continue
        title_start = i + surname_length
        title = t[title_start:title_start + 1]
        if title not in {"后", "姬"}:
            continue
        end = title_start + 1
        title_token = ctx.token_at(title_start)
        following_token = ctx.token_at(end)
        death_predicate_continuation = (
            title == "后"
            and following_token is not None
            and following_token.start == end
            and following_token.end == end + 1
            and end + 1 < len(t)
            and t[end + 1:end + 2] in "卒死亡"
        )
        if (
            any(consumed[i:end])
            or title_token is None
            or title_token.start != title_start
            or title_token.end != end
            or title_token.pos != "NOUN"
            or death_predicate_continuation
        ):
            continue
        confident_surname = all(
            token.score is not None
            and token.score >= KNOWN_FULLNAME_POS_SCORE
            for token in surname_tokens
        )
        kinship_frame = title == "姬" and i > 0 and t[i - 1:i] in "母妻妾女"
        if confident_surname or kinship_frame:
            chunk_type = "empress_title" if title == "后" else "consort_title"
            return (i, end, t[i:end], chunk_type)
    return None


def rule_female_court_title(ctx, i):
    """Occurrence-local named consort forms such as 华阳夫人 and 萧淑妃."""
    t, consumed = ctx.t, ctx.consumed
    for suffix in ("夫人", "妃"):
        for component_length in (2, 1):
            component_end = i + component_length
            if not t.startswith(suffix, component_end):
                continue
            end = component_end + len(suffix)
            if end > len(t) or any(consumed[i:end]):
                continue
            component_tokens = ctx.tokens_for(i, component_end)
            suffix_tokens = ctx.tokens_for(component_end, end)
            if (
                not component_tokens
                or component_tokens[0].start != i
                or component_tokens[-1].end != component_end
                or component_tokens[0].bio == "I"
                or any(not "\u3400" <= char <= "\u9fff" for char in t[i:component_end])
                or not suffix_tokens
                or suffix_tokens[0].start != component_end
                or suffix_tokens[-1].end != end
                or any(token.pos != "NOUN" for token in suffix_tokens)
            ):
                continue
            following = ctx.token_at(end)
            if (
                t[end:end + 1] in {"氏", "妃", "后", "夫"}
                or (
                    following is not None
                    and following.start == end
                    and following.pos == "PROPN"
                    and "NameType=" in following.tag
                )
            ):
                continue
            surname_head = (
                component_tokens[0].pos == "PROPN"
                and "NameType=Sur" in component_tokens[0].tag
                and component_tokens[0].text in CLEAN | COMPOUND
            )
            complete_component = (
                ctx.gspans.get(i) == component_end
                or (
                    len(component_tokens) == 1
                    and component_tokens[0].end == component_end
                    and component_tokens[0].pos == "PROPN"
                    and "NameType=" in component_tokens[0].tag
                )
            )
            explicit_naming = t[max(0, i - 2):i] in {"妃曰", "号曰"} or t[i - 1:i] == "曰"
            person_frame = (
                _strict_person_frame(ctx, i, end)
                or _title_predicate_after(ctx, end)
            )
            if suffix == "夫人":
                if not complete_component or not (explicit_naming or person_frame):
                    continue
            elif not (
                component_length == 2
                and surname_head
                and person_frame
            ):
                continue
            return (i, end, t[i:end], "female_court_title")
    return None


HONORIFIC_RANK_SUFFIXES = {"公", "君", "侯", "卿", "郎"}


def rule_surname_honorific(ctx, i):
    """A complete surname plus an honorific/rank in person syntax."""
    t, consumed = ctx.t, ctx.consumed
    for surname_length in (2, 1):
        surname = t[i:i + surname_length]
        if surname not in CLEAN and surname not in COMPOUND:
            continue
        end = i + surname_length + 1
        suffix = t[i + surname_length:end]
        if suffix not in HONORIFIC_RANK_SUFFIXES or any(consumed[i:end]):
            continue
        surname_tokens = ctx.tokens_for(i, i + surname_length)
        suffix_token = ctx.token_at(i + surname_length)
        if (
            not surname_tokens
            or surname_tokens[0].start != i
            or surname_tokens[-1].end != i + surname_length
            or surname_tokens[0].bio == "I"
            or suffix_token is None
            or suffix_token.start != i + surname_length
            or suffix_token.end != end
            or suffix_token.pos != "NOUN"
        ):
            continue
        surname_morphology = all(
            token.pos == "PROPN"
            and (
                "NameType=Sur" in token.tag
                or (
                    surname_length == 1
                    and token.start == i
                    and token.end == i + 1
                    and "NameType=Prs" in token.tag
                )
            )
            for token in surname_tokens
        )
        if not surname_morphology:
            continue
        following = ctx.token_at(end)
        if (
            t[end:end + 1] in HONORIFIC_RANK_SUFFIXES | {"主", "妃", "后"}
            or (
                following is not None
                and following.start == end
                and following.pos == "PROPN"
                and "NameType=" in following.tag
            )
            or ctx.gspans.get(i) not in {None, i + surname_length}
            or any(
                t[i:end + extension] in ctx.corpus.ner
                for extension in (1, 2)
                if end + extension <= len(t)
            )
        ):
            continue
        vocative = (
            t[i - 1:i] in "「『"
            and any(
                token.start >= end
                and token.start <= end + 2
                and token.pos in {"VERB", "AUX"}
                and token.score is not None
                and token.score >= POS_FUNCTION_VETO_SCORE
                for token in ctx.tokens_for(end, min(len(t), end + 3))
            )
        )
        possessive = t[end:end + 1] == "之" and _title_left_verb(ctx, i)
        if (
            _strict_person_frame(ctx, i, end)
            or _title_predicate_after(ctx, end)
            or vocative
            or possessive
        ):
            return (i, end, t[i:end], "surname_honorific")
    return None


def _local_person_frame(ctx, start, end):
    prev = ctx.t[start - 1:start] or "\u0001"
    nxt = ctx.t[end:end + 1]
    return (
        prev in PERSON_LEFT_VERBS
        or prev in PERSON_APPOS_TAIL
        or prev in COORD_HEAD
        or nxt in PERSON_RIGHT_PRED
        or (prev == "为" and nxt == "所")
        or (prev == "与" and nxt == "有")
    )


def _strict_person_frame(ctx, start, end):
    prev = ctx.t[start - 1:start] or "\u0001"
    nxt = ctx.t[end:end + 1]
    return prev in PERSON_LEFT_VERBS or nxt in PERSON_RIGHT_PRED


def _model_ner_surface(ctx, i, predicate):
    max_length = min(ctx.corpus.ner_maxL, len(ctx.t) - i, 8)
    for length in range(max_length, 1, -1):
        end = i + length
        surface = ctx.t[i:end]
        if (
            surface in ctx.corpus.ner
            and not any(ctx.consumed[i:end])
            and not (set(surface) & (NAMESTART | GLOSS_SEP))
            and not _shi_guard(ctx.t, i)
            and not _all_high_confidence_function_pos(ctx, i, end)
            and not _local_nat_or_geo(ctx, i, end)
            and predicate(surface, end)
        ):
            return surface, end
    return None


def _complete_model_name_tokens(ctx, start, end):
    tokens = ctx.tokens_for(start, end)
    return (
        bool(tokens)
        and tokens[0].start == start
        and tokens[-1].end == end
        and all(left.end == right.start for left, right in zip(tokens, tokens[1:]))
        and all(
            token.pos == "PROPN"
            and "NameType=" in token.tag
            and token.score is not None
            and token.score >= KNOWN_FULLNAME_POS_SCORE
            for token in tokens
        )
    )


def _complete_given_tokens(ctx, start, end):
    tokens = ctx.tokens_for(start, end)
    return (
        bool(tokens)
        and tokens[0].start == start
        and tokens[-1].end == end
        and all(left.end == right.start for left, right in zip(tokens, tokens[1:]))
        and all(
            token.pos == "PROPN"
            and "NameType=Giv" in token.tag
            and token.score is not None
            and token.score >= KNOWN_FULLNAME_POS_SCORE
            for token in tokens
        )
    )


def rule_pos_given_local_frame(ctx, i):
    """Complete POS-given in a strict occurrence-local person frame."""
    end = ctx.gspans.get(i)
    previous = ctx.token_at(i - 1) if i > 0 else None
    following = ctx.token_at(end) if end is not None else None
    if (
        end is None
        or not 1 <= end - i <= 2
        or any(ctx.consumed[i:end])
        or set(ctx.t[i:end]) & (NAMESTART | GLOSS_SEP)
        or not all("\u3400" <= char <= "\u9fff" for char in ctx.t[i:end])
        or _shi_guard(ctx.t, i)
        or not _complete_given_tokens(ctx, i, end)
        or _all_high_confidence_function_pos(ctx, i, end)
        or _local_nat_or_geo(ctx, i, end)
        or ctx.t[i:end] in APPOINT_TITLES
        or (
            previous is not None
            and previous.end == i
            and previous.tag == "PROPN|NameType=Sur"
        )
        or (
            following is not None
            and following.start == end
            and following.pos == "PROPN"
            and (
                "NameType=Giv" in following.tag
                or "NameType=Prs" in following.tag
            )
        )
        or (
            end - i == 1
            and any(
                ctx.t[i:candidate_end] in ctx.corpus.ner
                for candidate_end in range(i + 2, min(len(ctx.t), i + 4) + 1)
            )
        )
        or not _strict_person_frame(ctx, i, end)
    ):
        return None
    return (i, end, ctx.t[i:end], "pos_given_local_frame")


def rule_model_ner_predicate(ctx, i):
    """Model person candidate in an explicit person-selecting/predicate frame."""
    def eligible(surface, end):
        morphology = (
            _is_xing_headed(surface)
            or _complete_model_name_tokens(ctx, i, end)
        )
        return (
            morphology
            and _complete_model_name_tokens(ctx, i, end)
            and surface not in APPOINT_TITLES
            and surface not in OFFICE_TITLES
            and _strict_person_frame(ctx, i, end)
        )

    hit = _model_ner_surface(
        ctx,
        i,
        eligible,
    )
    if hit is None:
        return None
    surface, end = hit
    return (i, end, surface, "model_ner_predicate")


def _title_component_is_proper(ctx, start, end):
    tokens = ctx.tokens_for(start, end)
    return bool(tokens) and tokens[0].start == start and tokens[-1].end == end and all(
        token.pos == "PROPN"
        and token.score is not None
        and token.score >= KNOWN_FULLNAME_POS_SCORE
        for token in tokens
    )


def _title_component_is_relaxed_proper(ctx, start, end):
    tokens = ctx.tokens_for(start, end)
    return bool(tokens) and tokens[0].start == start and tokens[-1].end == end and all(
        token.pos == "PROPN"
        for token in tokens
    )


def rule_model_ner_title(ctx, i):
    """Model person title at a sentence/name boundary or strict person frame."""
    def eligible(surface, end):
        if surface in TITLE_NONPERSON_COMPONENTS or surface in APPOINT_TITLES:
            return False
        suffix = next(
            (
                candidate
                for candidate in MODEL_PERSON_TITLE_SUFFIXES
                if surface.endswith(candidate) and len(surface) > len(candidate)
            ),
            None,
        )
        if suffix is None:
            return False
        component_end = end - len(suffix)
        boundary = ctx.t[i - 1:i] in NAMESTART or ctx.t[end:end + 1] in NAMESTART
        return (
            (boundary or _strict_person_frame(ctx, i, end))
            and _title_component_is_proper(ctx, i, component_end)
        )

    hit = _model_ner_surface(ctx, i, eligible)
    if hit is None:
        return None
    surface, end = hit
    return (i, end, surface, "model_ner_title")


def rule_model_ner_given_boundary(ctx, i):
    """Complete multi-character POS-given model candidate at a hard boundary."""
    hit = _model_ner_surface(
        ctx,
        i,
        lambda surface, end: (
            ctx.gspans.get(i) == end
            and (
                ctx.t[i - 1:i] in NAMESTART
                or ctx.t[end:end + 1] in NAMESTART
            )
            and not _foreign_title_followed_by_name(ctx, end)
        ),
    )
    if hit is None:
        return None
    surface, end = hit
    return (i, end, surface, "model_ner_given")


def rule_model_ner_appos(ctx, i):
    """Model person candidate after an office/role or coordination marker."""
    def eligible(surface, end):
        prev = ctx.t[i - 1:i]
        if prev not in PERSON_APPOS_TAIL | COORD_HEAD:
            return False
        return (
            _complete_model_name_tokens(ctx, i, end)
            and (_is_xing_headed(surface) or len(surface) >= 3)
        )

    hit = _model_ner_surface(ctx, i, eligible)
    if hit is None:
        return None
    surface, end = hit
    return (i, end, surface, "model_ner_appos")


MODEL_PERSON_TITLE_SUFFIXES = (
    "长公主", "皇太后", "皇后", "太后", "夫人", "可汗", "单于",
    "公主", "太子", "世子", "君", "王", "公", "侯", "卿", "妃", "子",
)
EXTRA_MODEL_PERSON_TITLE_SUFFIXES = ("伯", "宗")


def _model_title_parts(ctx, i, suffixes):
    max_length = min(ctx.corpus.ner_maxL, len(ctx.t) - i, 8)
    for length in range(max_length, 1, -1):
        end = i + length
        surface = ctx.t[i:end]
        suffix = next(
            (
                candidate
                for candidate in suffixes
                if surface.endswith(candidate) and len(surface) > len(candidate)
            ),
            None,
        )
        if (
            suffix is not None
            and surface in ctx.corpus.ner
            and not any(ctx.consumed[i:end])
            and not (set(surface) & (NAMESTART | GLOSS_SEP))
            and not _shi_guard(ctx.t, i)
        ):
            return surface, end, suffix
    return None


def rule_model_ner_fief_title(ctx, i):
    """Model title with a Geo fief component and occurrence-local person syntax."""
    hit = _model_title_parts(ctx, i, ("君", "公", "侯"))
    if hit is None:
        return None
    surface, end, suffix = hit
    component_end = end - len(suffix)
    component_tokens = ctx.tokens_for(i, component_end)
    previous_token = ctx.token_at(i - 1)
    left_embedded = (
        previous_token is not None
        and previous_token.start < i
        and previous_token.end > i
    )
    left_noun_continuation = (
        previous_token is not None
        and previous_token.end == i
        and previous_token.pos == "NOUN"
        and not any(ctx.t[:i].endswith(title) for title in OFFICE_TITLES)
    )
    explicit_definition = (
        ctx.t[max(0, i - 3):i] in {"立以为", "封以为"}
        or ctx.t[max(0, i - 2):i] in {"封为", "号为"}
    )
    if (
        not component_tokens
        or left_embedded
        or left_noun_continuation
        or component_tokens[0].start != i
        or component_tokens[-1].end != component_end
        or component_tokens[0].tag.startswith("I-")
        or not all(token.pos == "PROPN" for token in component_tokens)
        or not any("NameType=Geo" in token.tag for token in component_tokens)
        or not (
            explicit_definition
            or _title_left_verb(ctx, i)
            or _title_predicate_after(ctx, end)
        )
        or any(
            ctx.t[candidate_start:end] in ctx.corpus.ner
            for candidate_start in range(max(0, i - 3), i)
        )
        or ctx.t[end:end + 1] in {
            "谷", "营", "城", "山", "水", "门", "亭", "县", "郡",
            "洲", "镇", "堆", "寨", "桥", "口", "驿",
        }
    ):
        return None
    return (i, end, surface, "model_ner_fief_title")


def rule_model_ner_rank_title(ctx, i):
    """Model `X伯` rank title supported by person syntax or a possessive frame."""
    hit = _model_title_parts(ctx, i, ("伯",))
    if hit is None:
        return None
    surface, end, _ = hit
    component_tokens = ctx.tokens_for(i, end - 1)
    right_token = ctx.token_at(end)
    if (
        not component_tokens
        or not all(token.pos == "PROPN" for token in component_tokens)
        or (
            right_token is not None
            and right_token.start == end
            and right_token.pos == "PROPN"
            and "NameType=" in right_token.tag
        )
        or any(
            ctx.t[i:candidate_end] in ctx.corpus.ner
            for candidate_end in range(end + 1, min(len(ctx.t), end + 3) + 1)
        )
        or not (
            _title_left_verb(ctx, i)
            or _title_predicate_after(ctx, end)
            or (
                ctx.t[i - 1:i] in PERSON_LEFT_VERBS | {"于", "与"}
                and ctx.t[end:end + 1] in NAMESTART
            )
            or ctx.t[end:end + 1] in {"之", "军"}
        )
    ):
        return None
    return (i, end, surface, "model_ner_rank_title")


def rule_model_ner_temple_title(ctx, i):
    """Model `X宗` temple title with complete title morphology."""
    hit = _model_title_parts(ctx, i, ("宗",))
    if hit is None:
        return None
    surface, end, _ = hit
    component_tokens = ctx.tokens_for(i, end - 1)
    suffix_token = ctx.token_at(end - 1)
    title_context = (
        ctx.t[max(0, i - 2):i] == "庙号"
        or _title_left_verb(ctx, i)
        or _title_predicate_after(ctx, end)
        or ctx.t[end:end + 1] in {
            "之", "时", "朝", "世", "子", "后", "室", "陵", "丧",
        }
        or i == 0
    )
    motion_place_frame = (
        ctx.t[i - 1:i] in {"奔", "至", "如", "入", "出", "还", "屯", "据", "镇", "保"}
        and ctx.t[end:end + 1] in NAMESTART
    )
    if (
        not component_tokens
        or component_tokens[0].start != i
        or component_tokens[-1].end != end - 1
        or component_tokens[0].tag.startswith("I-")
        or not all(token.pos == "PROPN" for token in component_tokens)
        or any(
            "NameType=Geo" in token.tag or "NameType=Nat" in token.tag
            for token in component_tokens
        )
        or suffix_token is None
        or suffix_token.start != end - 1
        or suffix_token.end != end
        or suffix_token.pos != "NOUN"
        or not title_context
        or motion_place_frame
        or ctx.t[end:end + 1] in {
            "山", "水", "谷", "城", "县", "郡", "王", "侯", "公", "君",
        }
        or any(char in JUE_HEAD for char in ctx.t[end:end + 3])
    ):
        return None
    return (i, end, surface, "model_ner_temple_title")


def rule_model_ner_short_royal_title(ctx, i):
    """Two-character X王/X后 title with complete local title morphology."""
    end = i + 2
    if end > len(ctx.t) or any(ctx.consumed[i:end]):
        return None
    surface = ctx.t[i:end]
    suffix = surface[-1:]
    if surface not in ctx.corpus.ner or suffix not in {"王", "后"}:
        return None
    component = ctx.token_at(i)
    title_token = ctx.token_at(i + 1)
    previous = ctx.token_at(i - 1)
    repeated = ctx.t.find(surface, 0, i) >= 0 or ctx.t.find(surface, end) >= 0
    if (
        component is None
        or any(not "\u3400" <= char <= "\u9fff" for char in surface)
        or component.start != i
        or component.end != i + 1
        or component.pos != "PROPN"
        or "NameType=" not in component.tag
        or (
            suffix == "后"
            and surface[:1] not in CLEAN
        )
        or title_token is None
        or title_token.start != i + 1
        or title_token.end != end
        or title_token.pos != "NOUN"
        or (
            previous is not None
            and previous.start < i
            and previous.end > i
        )
        or (
            previous is not None
            and previous.end == i
            and previous.pos in {"PROPN", "NOUN", "ADJ"}
        )
        or any(
            ctx.t[left:end + right] in ctx.corpus.ner
            for left, right in ((i - 1, 0), (i - 2, 0), (i, 1), (i, 2))
            if left >= 0 and end + right <= len(ctx.t)
            and (left != i or right)
        )
        or not (
            repeated
            or _strict_person_frame(ctx, i, end)
            or _title_predicate_after(ctx, end)
            or ctx.t[i - 1:i] in NAMESTART
            or ctx.t[end:end + 1] in NAMESTART
        )
    ):
        return None
    return (i, end, surface, "model_ner_short_royal_title")


def _surface_has_local_surname_frame(ctx, surface):
    cursor = ctx.t.find(surface)
    while cursor >= 0:
        token = ctx.token_at(cursor)
        end = cursor + len(surface)
        if (
            token is not None
            and token.start == cursor
            and token.end == cursor + 1
            and token.pos == "PROPN"
            and "NameType=Sur" in token.tag
            and _strict_person_frame(ctx, cursor, end)
        ):
            return True
        cursor = ctx.t.find(surface, cursor + 1)
    return False


def rule_model_ner_local_surname_name(ctx, i):
    """Repeated model candidate supported by a local surname-headed occurrence."""
    max_length = min(ctx.corpus.ner_maxL, len(ctx.t) - i, 2)
    for length in range(max_length, 1, -1):
        end = i + length
        surface = ctx.t[i:end]
        tokens = ctx.tokens_for(i, end)
        if (
            surface not in ctx.corpus.ner
            or any(ctx.consumed[i:end])
            or not _is_xing_headed(surface)
            or any(not "\u3400" <= char <= "\u9fff" for char in surface)
            or surface.endswith(("氏", "后", "王", "公", "侯", "君", "卿", "妃"))
            or surface.endswith("者")
            or "之" in surface
            or surface in ctx.corpus.geo_names
            or surface in ctx.corpus.admin_places
            or surface in POLITY2
            or _local_polity_or_geo(ctx, i, end)
            or not _surface_has_local_surname_frame(ctx, surface)
            or not (
                _strict_person_frame(ctx, i, end)
            )
            or not tokens
            or tokens[-1].pos != "PROPN"
            or ctx.t[end:end + 1] in {
                "氏", "军", "兵", "部", "国", "后", "妃", "公", "王", "镇",
            }
            or any(
                title in surface
                for title in APPOINT_TITLES + OFFICE_TITLES
            )
            or _shi_guard(ctx.t, i)
            or _embedded_in_name_tokens(ctx, i, end)
            or (
                ctx.token_at(end) is not None
                and ctx.token_at(end).start == end
                and ctx.token_at(end).pos == "PROPN"
                and "NameType=" in ctx.token_at(end).tag
            )
            or any(
                ctx.t[i:candidate_end] in ctx.corpus.ner
                for candidate_end in range(end + 1, min(len(ctx.t), end + 3) + 1)
            )
        ):
            continue
        return (i, end, surface, "model_ner_local_surname_name")
    return None


def rule_model_ner_name(ctx, i):
    """Model-derived person candidate with occurrence-local structural evidence."""
    t, consumed = ctx.t, ctx.consumed
    max_length = min(ctx.corpus.ner_maxL, len(t) - i, 8)
    for length in range(max_length, 1, -1):
        end = i + length
        surface = t[i:end]
        if surface not in ctx.corpus.ner or any(consumed[i:end]):
            continue
        if (
            set(surface) & (NAMESTART | GLOSS_SEP)
            or _shi_guard(t, i)
            or _all_high_confidence_function_pos(ctx, i, end)
            or _local_nat_or_geo(ctx, i, end)
        ):
            continue
        title_shape = any(
            surface.endswith(suffix) and len(surface) > len(suffix)
            for suffix in MODEL_PERSON_TITLE_SUFFIXES
        )
        complete_name = _complete_person_pos(ctx, i, end)
        local_frame = _local_person_frame(ctx, i, end)
        local_given = (
            ctx.gspans.get(i) == end
            and local_frame
        )
        title_component_end = next(
            (
                end - len(suffix)
                for suffix in MODEL_PERSON_TITLE_SUFFIXES
                if surface.endswith(suffix) and len(surface) > len(suffix)
            ),
            None,
        )
        titled_person = (
            title_shape
            and title_component_end is not None
            and local_frame
            and (
                title_component_end - i >= 2
                or _local_personish_title_component(ctx, i, title_component_end)
            )
        )
        if complete_name or local_given or titled_person:
            return (i, end, surface, "model_ner_name")
    return None


def rule_model_ner_partial_pos(ctx, i):
    """Token-complete model candidate with strict syntax and surviving name POS."""
    def eligible(surface, end):
        if (
            surface in APPOINT_TITLES
            or surface in OFFICE_TITLES
            or any(
                title in surface and len(surface) > len(title)
                for title in APPOINT_TITLES + OFFICE_TITLES
            )
            or any(
                surface.startswith(role) and len(surface) > len(role)
                for role in PERSON_ROLE_PREFIXES
            )
        ):
            return False
        tokens = ctx.tokens_for(i, end)
        if not tokens or not _strict_person_frame(ctx, i, end):
            return False
        if any(
            token.pos in {"VERB", "AUX", "ADP", "SCONJ", "PRON"}
            and token.score is not None
            and token.score >= 0.75
            for token in tokens[1:-1]
        ):
            return False
        if sum(
            token.pos == "PROPN"
            and "NameType=Sur" in token.tag
            and not token.tag.startswith("I-")
            for token in tokens
        ) > 1:
            return False
        if any(
            token.text in GLOSS_TERM
            and token.pos == "NOUN"
            and token.score is not None
            and token.score >= 0.9
            for token in tokens[1:-1]
        ):
            return False
        if (
            surface.startswith("名")
            and ctx.t[i - 1:i] in {"赐", "更", "改"}
        ):
            return False
        following = ctx.token_at(end)
        if (
            following is not None
            and following.start == end
            and following.pos == "PROPN"
            and "NameType=" in following.tag
        ):
            return False
        first = tokens[0]
        last = tokens[-1]
        name_head = (
            first.pos == "PROPN"
            and "NameType=" in first.tag
            and first.score is not None
            and first.score >= 0.35
        )
        person_tail = (
            last.pos == "PROPN"
            and (
                "NameType=Giv" in last.tag
                or "NameType=Prs" in last.tag
            )
            and last.score is not None
            and last.score >= 0.5
        )
        strong_surname_head = (
            "NameType=Sur" in first.tag
            and first.score is not None
            and first.score >= 0.6
        )
        repeated_surface = (
            ctx.t.find(surface, 0, i) >= 0
            or ctx.t.find(surface, end) >= 0
        )
        repeated_person_prefix = (
            len(surface) > 2
            and surface[-1] in PERSON_REPORTING_VERBS
            and ctx.t[end:end + 1] in PERSON_SUBJECT_PRED
            and surface[:-1] in ctx.corpus.ner
            and (
                ctx.t.find(surface[:-1], 0, i) >= 0
                or ctx.t.find(surface[:-1], end) >= 0
            )
        )
        return name_head and person_tail and (
            strong_surname_head or repeated_surface
        ) and not repeated_person_prefix

    hit = _model_ner_surface(ctx, i, eligible)
    if hit is None:
        return None
    surface, end = hit
    return (i, end, surface, "model_ner_partial_pos")


GENEALOGY_PREFIXES = tuple(sorted((
    "其子", "长子", "次子", "少子", "幼子", "嫡子", "庶子", "嗣子", "生子",
    "养子", "义子", "兄子", "弟子", "从子", "族子",
    "其孙", "长孙", "次孙", "少孙", "幼孙", "兄孙", "弟孙", "从孙", "族孙",
    "其兄", "其弟", "从兄", "从弟", "族兄", "族弟",
    "兄", "弟",
), key=len, reverse=True))
GENEALOGY_RIGHT = (
    NAMESTART | PERSON_LEFT_VERBS | PERSON_RIGHT_PRED
    | set("而不大独等兵事节自于行继临淫懦顾说将令讨勒奉上尚娶袭封拜诣入出进守谋摄监已相夺"
          "所代部有统分嗣南妻举又居引用夜镇贼趋蒸异恨俱掌取仍权友闻皆驰非主同乳早幽谕"
          "往闚抚既应篡")
)
GENEALOGY_STATE_PHRASES = {
    "新昏", "尚幼", "幼弱", "年少", "年幼", "未冠", "无子",
}
GENEALOGY_PERSON_TITLES = {"莫离支"}
FOREIGN_NAME_SUFFIXES = {"汗"}
ROYAL_PERSON_TITLES = ("\u7687\u592a\u5b50", "\u592a\u5b50", "\u4e16\u5b50",
                       "\u7687\u5b50", "\u738b\u5b50")


def rule_foreign_suffix_name(ctx, i):
    """Compound surname + BIO given + a local foreign-name suffix.

    The suffix closes a model gap such as 拓跋沙漠汗, where 沙漠 is BIO-Giv but 汗 is
    tagged NOUN. This is a bounded morphological continuation, not length guessing.
    """
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    surname_len = 3 if t[i:i + 3] in COMPOUND3 else (
        2 if t[i:i + 2] in COMPOUND else 0
    )
    given_start = i + surname_len
    if not surname_len or given_start not in gset:
        return None
    given_end = ctx.gspans.get(given_start)
    if given_end is None:
        return None
    suffix = next((x for x in FOREIGN_NAME_SUFFIXES if t.startswith(x, given_end)), None)
    if suffix is None:
        return None
    end = given_end + len(suffix)
    if t[end:end + 1] not in GENEALOGY_RIGHT or any(consumed[i:end]):
        return None
    return (i, end, t[i:end], "foreign_suffix_name")


def rule_genealogy_given(ctx, i):
    """A kinship relation locally licenses a following POS-backed person name.

    One rule handles the complete POS-given span regardless of character count. The
    relation is excluded from the span. A reliable surname followed by POS.Giv is also
    admitted as a complete name (其子宋襄), without consulting the name KB. Personhood
    comes from the local genealogy structure, never from the corpus name KB.
    """
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    prefix = next((p for p in GENEALOGY_PREFIXES if t.startswith(p, i)), None)
    if prefix is None:
        return None
    if len(prefix) == 1 and i > 0 and t[i - 1] in "子兄弟":
        return None
    if prefix == "庶子" and i > 0 and t[i - 1] == "中":
        return None
    j = i + len(prefix)
    if j >= len(t):
        return None
    if j in gset:
        e = ctx.gspans.get(j)
        if e is None:
            e = j + 1
            while e < len(t) and e in gset and t[e] not in NAMESTART:
                e += 1
        if t[j:e] in GENEALOGY_PERSON_TITLES and e in ctx.gspans:
            j, e = e, ctx.gspans[e]
        while t[e:e + 1] not in GENEALOGY_RIGHT and e in ctx.gspans:
            e = ctx.gspans[e]
    else:
        surname_len = 3 if t[j:j + 3] in COMPOUND3 else (
            2 if t[j:j + 2] in COMPOUND else (1 if t[j:j + 1] in CLEAN else 0)
        )
        given_start = j + surname_len
        if not surname_len or given_start not in gset:
            return None
        e = ctx.gspans.get(given_start)
        if e is None:
            e = given_start + 1
            while e < len(t) and e in gset and t[e] not in NAMESTART:
                e += 1
    surf = t[j:e]
    nxt = t[e:e + 1] or "\u0001"
    if nxt not in GENEALOGY_RIGHT or any(consumed[j:e]):
        return None
    if set(surf) & NAMESTART:
        return None
    if any(t[e:e + title_len].endswith(tuple(JUE_HEAD))
           and e + title_len in gset for title_len in range(2, 5)):
        return None
    if t.startswith("将军", e) and e + 2 in gset:
        return None
    if _all_high_confidence_function_pos(ctx, j, e) or surf in GENEALOGY_STATE_PHRASES:
        return None
    if _shi_guard(t, j):
        return None
    return (j, e, surf, "gloss_kin")


def rule_presentative_person(ctx, i):
    """Fallback for the presentative construction `有 + person + 者`."""
    t, consumed = ctx.t, ctx.consumed
    if t[i - 1:i] != "有":
        return None
    start = i
    for end in range(min(len(t), start + 4), start, -1):
        if t[end:end + 1] != "者" or any(consumed[start:end]):
            continue
        surface = t[start:end]
        if set(surface) & (NAMESTART | GLOSS_SEP):
            continue
        tokens = ctx.tokens_for(start, end)
        complete_prs = bool(tokens) and all(
            token.tag.endswith("PROPN|NameType=Prs") for token in tokens
        )
        complete_giv = ctx.gspans.get(start) == end
        surname = tokens[0] if tokens else None
        surname_giv = (
            surname is not None
            and surname.start == start
            and surname.end == start + 1
            and surname.tag == "PROPN|NameType=Sur"
            and surname.score is not None
            and surname.score >= 0.5
            and ctx.gspans.get(surname.end) == end
        )
        if complete_prs or complete_giv or surname_giv:
            return (start, end, surface, "role_name")
    return None


def rule_person_naming_definition(ctx, i):
    """Person names introduced by productive 名/姓名…曰 syntax."""
    t, cs, consumed = ctx.t, ctx.corpus, ctx.consumed
    if t[i - 1:i] != "曰":
        return None

    ends = []
    giv_end = ctx.gspans.get(i)
    if giv_end is not None:
        ends.append(giv_end)
    morphology_ends = set()
    for end in range(i + 1, min(len(t), i + 4) + 1):
        tokens = ctx.tokens_for(i, end)
        if tokens and (
            all(token.tag.endswith("PROPN|NameType=Prs") for token in tokens)
            or all(
                token.tag.endswith("PROPN|NameType=Giv") for token in tokens
            )
        ):
            ends.append(end)
            morphology_ends.add(end)
    raw_left = t[max(0, i - 24):i - 1]
    explicit_single = (
        raw_left.endswith("名之")
        or any(
            raw_left.endswith("名" + relation)
            for relation in NAMING_PERSON_RELATIONS
        )
        or (
            raw_left.endswith("名")
            and any(cue in raw_left for cue in ("生子", "生男"))
        )
    )
    if not ends and explicit_single and t[i + 1:i + 2] in NAMESTART:
        ends.append(i + 1)
    end = max(ends, default=None)
    if (
        end is None
        or end > i + 4
        or any(consumed[i:end])
        or set(t[i:end]) & NAMESTART
    ):
        return None

    sentence_start = max(
        (t.rfind(separator, max(0, i - 40), i - 1) for separator in "。；！？\n"),
        default=-1,
    ) + 1
    left = t[sentence_start:i - 1]
    if "名" not in left:
        return None
    last_name = left.rfind("名")
    stem_tail = left[last_name:]
    recent = max(sentence_start, i - 30)
    earlier_person = any(consumed[recent:i - 1])

    explicit_relation = any(
        stem_tail.endswith("名" + relation)
        for relation in NAMING_PERSON_RELATIONS
    )
    pronoun_object = stem_tail.endswith("名之")
    self_naming = stem_tail.endswith("自名")
    implicit_rename = any(left.endswith(marker) for marker in NAMING_IMPLICIT_MARKERS)
    complete_person_morphology = giv_end == end or end in morphology_ends
    implicit_person = implicit_rename and (
        complete_person_morphology
        or earlier_person
        or any(role in left[-20:] for role in ("太子", "皇子", "帝", "臣", "生男", "生子"))
    )

    last_person_end = max(
        (
            offset + 1
            for offset in range(sentence_start, i - 1)
            if consumed[offset]
        ),
        default=-1,
    )
    target_gap = t[last_person_end:i - 1] if last_person_end >= 0 else ""
    before_target = t[sentence_start:last_person_end] if last_person_end >= 0 else ""
    prior_naming_chain = (
        "曰" in before_target
        and "名" in before_target[:before_target.rfind("曰")]
    )
    explicit_person_target = (
        last_person_end >= 0
        and len(target_gap) <= 3
        and set(target_gap) <= set("之子姓名")
        and ("名" in target_gap or prior_naming_chain)
    )
    explicit_role_target = any(
        role in left[-12:]
        for role in ("太子", "皇子", "兄子", "弟子", "其子", "之子")
    )
    birth_naming = (
        left.endswith("名")
        and any(cue in left[-16:] for cue in ("生子", "生男"))
    )
    previous_person_end = max(
        (
            offset + 1
            for offset in range(max(sentence_start, i - 30), i - 1)
            if consumed[offset]
        ),
        default=-1,
    )
    serial_prefix = t[previous_person_end:i - 1] if previous_person_end >= 0 else ""
    serial_old_start = previous_person_end + 1
    serial_old = t[serial_old_start:i - 1]
    serial_person_target = False
    for start in range(max(serial_old_start, i - 5), i - 1):
        tokens = ctx.tokens_for(start, i - 1)
        if not tokens:
            continue
        all_prs = all(
            token.tag.endswith("PROPN|NameType=Prs") for token in tokens
        )
        surname_given = (
            len(tokens) >= 2
            and
            tokens[0].tag.endswith("PROPN|NameType=Sur")
            and all(
                token.tag.endswith("PROPN|NameType=Giv")
                for token in tokens[1:]
            )
        )
        if all_prs or surname_given:
            serial_person_target = True
            break
    coordinated_naming = (
        previous_person_end >= 0
        and "名" in t[max(sentence_start, previous_person_end - 25):previous_person_end]
        and "曰" in t[max(sentence_start, previous_person_end - 25):previous_person_end]
        and (
            serial_prefix == "，"
            or (
                serial_prefix.startswith("，")
                and (
                    ctx.gspans.get(serial_old_start) == i - 1
                    or serial_person_target
                )
            )
        )
    )
    if not (
        explicit_relation
        or pronoun_object
        or self_naming
        or implicit_person
        or explicit_person_target
        or explicit_role_target
        or birth_naming
        or coordinated_naming
    ):
        return None
    return (i, end, t[i:end], "alias")


def rule_person_appellation(ctx, i):
    """Explicitly introduced appellation backed by current POS morphology."""
    t, consumed = ctx.t, ctx.consumed
    marker = next(
        (
            item for item in APPELLATION_MARKERS
            if t[max(0, i - len(item)):i] == item
        ),
        None,
    )
    alias_end = ctx.gspans.get(i) if marker is not None else None
    if alias_end is None and marker is not None:
        alias_end = next(
            (
                end for end in range(min(len(t), i + 4), i, -1)
                if _complete_person_pos(ctx, i, end)
            ),
            None,
        )
    giv_end = ctx.gspans.get(i) if marker == "字" else None
    if marker == "字" and t[i - 2:i - 1] == "用":
        giv_end = None
    end = max(
        (
            candidate for candidate in (alias_end, giv_end)
            if candidate is not None and candidate <= i + 4
        ),
        default=None,
    )
    if end is None or any(consumed[i:end]):
        return None
    surface = t[i:end]
    explicit_appellation = marker is not None and (
        alias_end == end or (marker == "字" and giv_end == end)
    )
    if not explicit_appellation or _shi_guard(t, i):
        return None
    if (
        set(surface) & NAMESTART
        or t[i - 1:i] == "《"
        or t[end:end + 1] == "诗"
    ):
        return None
    continuation = ctx.token_at(end)
    if (
        continuation is not None
        and continuation.start == end
        and "NameType=Prs" in continuation.tag
    ):
        return None

    predicate_start = end
    while predicate_start < len(t):
        token = ctx.token_at(predicate_start)
        if (
            token is None
            or token.start != predicate_start
            or token.pos != "ADV"
            or token.score is None
            or token.score < POS_FUNCTION_VETO_SCORE
        ):
            break
        predicate_start = token.end
    predicate = ctx.token_at(predicate_start)
    high_confidence_verb = (
        predicate is not None
        and predicate.start == predicate_start
        and predicate.pos in {"VERB", "AUX"}
        and predicate.score is not None
        and predicate.score >= POS_FUNCTION_VETO_SCORE
    )
    prev = t[i - 1:i] or "\u0001"
    subject_frame = prev in TITLE_SUBJECT_LEFT and high_confidence_verb
    object_frame = prev in PERSON_LEFT_VERBS or (
        prev == "与" and high_confidence_verb
    )

    left_polity = ctx.token_at(i - 2) if i >= 2 else None
    immediate_polity = ctx.token_at(i - 1) if i >= 1 else None
    polity_subject_frame = (
        immediate_polity is not None
        and immediate_polity.start == i - 1
        and immediate_polity.end == i
        and "NameType=Nat" in immediate_polity.tag
        and immediate_polity.score is not None
        and immediate_polity.score >= POS_FUNCTION_VETO_SCORE
        and high_confidence_verb
    )
    right_polity = ctx.token_at(end + 1)
    parallel_frame = (
        t[i - 1:i] == "之"
        and t[end:end + 1] == "、"
        and t[end + 2:end + 3] == "之"
        and left_polity is not None
        and left_polity.start == i - 2
        and left_polity.end == i - 1
        and "NameType=Nat" in left_polity.tag
        and right_polity is not None
        and right_polity.start == end + 1
        and right_polity.end == end + 2
        and "NameType=Nat" in right_polity.tag
    )

    complete_giv = ctx.gspans.get(i) == end
    local_person_syntax = subject_frame or object_frame or parallel_frame
    if not (
        explicit_appellation
        or polity_subject_frame
        or (
            local_person_syntax
            and (
                complete_giv
                or surface not in AMBIGUOUS_PERSONAL_TITLES
            )
        )
    ):
        return None
    return (i, end, surface, "alias")


def _title_predicate_after(ctx, start):
    cursor = start
    while cursor < len(ctx.t):
        token = ctx.token_at(cursor)
        if (
            token is None
            or token.start != cursor
            or token.pos != "ADV"
            or token.score is None
            or token.score < POS_FUNCTION_VETO_SCORE
        ):
            break
        cursor = token.end
    while cursor < len(ctx.t) and ctx.t[cursor] in TITLE_PREDICATE_MODIFIERS:
        cursor += 1
    if ctx.t[cursor:cursor + 1] in TITLE_SUBJECT_PREDICATES:
        return True
    token = ctx.token_at(cursor)
    return (
        token is not None
        and token.start == cursor
        and token.pos in {"VERB", "AUX"}
        and token.score is not None
        and token.score >= POS_FUNCTION_VETO_SCORE
    )


def _title_left_verb(ctx, start):
    token = ctx.token_at(start - 1)
    return (
        token is not None
        and token.end == start
        and token.pos in {"VERB", "AUX"}
        and token.score is not None
        and token.score >= POS_FUNCTION_VETO_SCORE
    )


def _explicit_title_context(ctx, start, title_end):
    right = ctx.t[title_end:title_end + 1]
    right_token = ctx.token_at(title_end)
    right_person = (
        right_token is not None
        and right_token.start == title_end
        and right_token.pos == "PROPN"
        and (
            "NameType=Giv" in right_token.tag
            or "NameType=Prs" in right_token.tag
        )
    )
    return (
        _title_left_verb(ctx, start)
        or _title_predicate_after(ctx, title_end)
        or right_person
        or right == "之"
        or ctx.t[max(0, start - 2):start].endswith(("封", "立", "拜", "谥", "号"))
    )


def _local_polity_or_geo(ctx, start, end):
    if end - start == 1 and ctx.t[start] in POLITY1:
        return True
    tokens = ctx.tokens_for(start, end)
    return (
        bool(tokens)
        and all(
            token.pos == "PROPN"
            and (
                "NameType=Nat" in token.tag
                or "NameType=Geo" in token.tag
            )
            and token.score is not None
            and token.score >= POS_FUNCTION_VETO_SCORE
            for token in tokens
        )
    )


def _local_nat_or_geo(ctx, start, end):
    tokens = ctx.tokens_for(start, end)
    return (
        bool(tokens)
        and any(
            "NameType=Nat" in token.tag or "NameType=Geo" in token.tag
            for token in tokens
        )
    )


def _local_nominal_title_component(ctx, start, end):
    tokens = ctx.tokens_for(start, end)
    return (
        bool(tokens)
        and all(token.pos in {"PROPN", "NOUN"} for token in tokens)
    )


def _local_personish_title_component(ctx, start, end):
    tokens = ctx.tokens_for(start, end)
    return (
        bool(tokens)
        and all(token.pos == "PROPN" for token in tokens)
        and not any(
            "NameType=Sur" in token.tag
            for token in tokens[1:]
        )
        and any(
            name_type in token.tag
            for token in tokens
            for name_type in ("NameType=Sur", "NameType=Giv", "NameType=Prs")
        )
    )


def _embedded_local_geo_left(ctx, start):
    token = ctx.token_at(start)
    return (
        token is not None
        and token.start < start < token.end
        and (
            "NameType=Nat" in token.tag
            or "NameType=Geo" in token.tag
        )
    )


def _earlier_local_temple_title(ctx, start, surface):
    offset = ctx.t.rfind(surface, 0, start)
    while offset >= 0:
        polity_start = offset - 1
        if polity_start >= 0 and _local_polity_or_geo(ctx, polity_start, offset):
            return True
        offset = ctx.t.rfind(surface, 0, offset)
    return False


def rule_title_appellation(ctx, i):
    """Title morphology plus occurrence-local evidence; no person KB admission."""
    t, consumed = ctx.t, ctx.consumed

    polity_token = ctx.token_at(i)
    if (
        polity_token is not None
        and polity_token.start == i
        and polity_token.pos == "PROPN"
        and "NameType=Nat" in polity_token.tag
        and polity_token.bio != "I"
    ):
        title_start = polity_token.end
        title = t[title_start:title_start + 2]
        title_tokens = ctx.tokens_for(title_start, title_start + 2)
        ruler_title = (
            title == "上皇"
            and len(title_tokens) == 2
            and all(token.pos == "NOUN" for token in title_tokens)
        )
        temple_title = (
            len(title_tokens) == 2
            and title.endswith(("宗", "祖"))
            and title_tokens[0].pos == "PROPN"
            and "NameType=Prs" in title_tokens[0].tag
            and title_tokens[1].pos == "NOUN"
        )
        end = title_start + 2
        if (ruler_title or temple_title) and not any(consumed[i:end]):
            return (i, end, t[i:end], "title_appellation")

    # A controlled polity plus a posthumous epithet and rank is an explicit person
    # title: 梁孝王, 齐悼惠王, 东魏昭王.
    for prefix_length in (2, 1):
        for epithet_length in (2, 1):
            suffix = t[i + prefix_length + epithet_length:i + prefix_length + epithet_length + 1]
            prefix = t[i:i + prefix_length]
            end = i + prefix_length + epithet_length + 1
            if (
                end > len(t)
                or suffix not in JUE_HEAD
                or not (
                    (prefix_length == 1 and prefix in POLITY1)
                    or (
                        prefix_length == 2
                        and prefix[0] in POLITY_PREFIX
                        and prefix[1] in POLITY1
                    )
                )
                or (
                    i > 0
                    and "\u3400" <= t[i - 1] <= "\u9fff"
                    and t[i - 1] not in NAMESTART
                    and not _title_left_verb(ctx, i)
                    and not (
                        ctx.token_at(i - 1) is not None
                        and ctx.token_at(i - 1).end == i
                        and ctx.token_at(i - 1).pos in FUNCTION_POS
                    )
                )
                or _embedded_local_geo_left(ctx, i)
                or not all(
                    char in FIEF_POSTHUMOUS_EPITHETS
                    for char in t[i + prefix_length:i + prefix_length + epithet_length]
                )
                or any(consumed[i:end])
            ):
                continue
            following = ctx.token_at(end)
            title_given = (
                following is not None
                and following.start == end
                and following.pos == "PROPN"
                and (
                    "NameType=Giv" in following.tag
                    or "NameType=Prs" in following.tag
                )
            )
            if not title_given:
                return (i, end, t[i:end], "title_appellation")

    # A bare temple title is licensed only by an earlier local polity+title form in
    # the same numbered section: 周世宗 ... 世宗.
    temple_title = t[i:i + 2]
    if (
        len(temple_title) == 2
        and temple_title[-1] in "宗祖"
        and not any(consumed[i:i + 2])
        and _earlier_local_temple_title(ctx, i, temple_title)
        and (
            t[i - 1:i] == "为" and t[i + 2:i + 3] == "所"
            or _title_left_verb(ctx, i)
            or _title_predicate_after(ctx, i + 2)
        )
    ):
        return (i, i + 2, temple_title, "title_appellation")

    # These are grammatical ruler-title forms, not identity entries. Numeral+世
    # remains syntax-gated because it also denotes a generation count.
    ruler = t[i:i + 2]
    ruler_shape = (
        ruler in LEXICAL_RULER_TITLES
        or ruler == "二世"
    )
    if ruler_shape:
        end = i + len(ruler)
        following = ctx.token_at(end)
        embedded_fullname = (
            following is not None
            and following.start == end
            and following.pos == "PROPN"
            and following.score is not None
            and following.score >= KNOWN_FULLNAME_POS_SCORE
            and (
                "NameType=Giv" in following.tag
                or "NameType=Prs" in following.tag
            )
        )
        ordinal_common_reading = (
            ruler == "二世"
            and (
                (t[i - 1:i] == "秦" and t[end:end + 2] == "即亡")
                or (
                    t[end:end + 1] in NAMESTART
                    and t[i - 1:i] not in PERSON_LEFT_VERBS
                )
            )
        )
        if (
            not any(consumed[i:end])
            and not embedded_fullname
            and not ordinal_common_reading
            and (
                _title_left_verb(ctx, i)
                or _title_predicate_after(ctx, end)
                or t[i - 1:i] in NAMESTART
                or (t[i - 1:i] == "为" and t[end:end + 1] == "所")
            )
        ):
            return (i, end, ruler, "title_appellation")

    # The explicit suffix proves personhood. Long suffixes license a two-character
    # title component directly; short ranks additionally need epithet morphology,
    # local person POS, or a selecting verb on the left.
    for length in (2,):
        component = t[i:i + length]
        suffix = next(
            (
                candidate
                for candidate in PERSON_TITLE_SUFFIXES
                if t.startswith(candidate, i + length)
            ),
            None,
        )
        if (
            len(component) != length
            or suffix is None
            or any(consumed[i:i + length])
            or not all("\u3400" <= char <= "\u9fff" for char in component)
            or _all_high_confidence_function_pos(ctx, i, i + length)
            or component[0] in TITLE_COMPONENT_BAD_HEAD
            or component[0] == "子"
            or _local_nat_or_geo(ctx, i, i + length)
            or _embedded_local_geo_left(ctx, i)
            or component in TITLE_NONPERSON_COMPONENTS
            or component in {"皇太", "太皇", "大长"}
        ):
            continue
        title_end = i + length + len(suffix)
        following = ctx.token_at(title_end)
        following_person = (
            following is not None
            and following.start == title_end
            and following.pos == "PROPN"
            and (
                "NameType=Giv" in following.tag
                or "NameType=Prs" in following.tag
            )
        )
        if (
            (
                suffix in INHERENT_PERSON_TITLE_SUFFIXES
                and (
                    component[-1] in TITLE_EPITHET_END
                    or _local_personish_title_component(ctx, i, i + length)
                    or (
                        suffix in {"可汗", "单于"}
                        and component in TITLE_NOMINAL_COMPONENTS
                        and _local_nominal_title_component(ctx, i, i + length)
                    )
                )
            )
            or (
                (
                    (
                        component[-1] in TITLE_EPITHET_END
                        and _local_personish_title_component(ctx, i, i + length)
                    )
                    or (
                        t[i - 1:i] in PERSON_LEFT_VERBS
                        and suffix == "王"
                        and _local_nominal_title_component(ctx, i, i + length)
                        and not following_person
                    )
                )
                and _explicit_title_context(ctx, i, title_end)
            )
        ) and not (
            suffix in {"可汗", "单于"}
            and (
                following_person
                or (
                    i > 0
                    and "\u3400" <= t[i - 1] <= "\u9fff"
                    and t[i - 1] not in PERSON_LEFT_VERBS
                    and t[max(0, i - 2):i] not in TITLE_NONPERSON_COMPONENTS
                    and component not in TITLE_EMBEDDED_COMPONENTS
                )
            )
        ) and not (
            suffix in INHERENT_PERSON_TITLE_SUFFIXES
            and suffix not in {"可汗", "单于"}
            and i > 0
            and "\u3400" <= t[i - 1] <= "\u9fff"
            and t[i - 1] not in NAMESTART
            and not _title_left_verb(ctx, i)
            and not (
                ctx.token_at(i - 1) is not None
                and ctx.token_at(i - 1).end == i
                and ctx.token_at(i - 1).pos in FUNCTION_POS
            )
        ) and not (
            suffix in {"王", "后", "公", "侯"}
            and i > 0
            and "\u3400" <= t[i - 1] <= "\u9fff"
            and t[i - 1] not in NAMESTART
            and t[i - 1] not in PERSON_LEFT_VERBS
            and not following_person
        ):
            return (i, i + length, component, "title_appellation")
    return None


def rule_explicit_title_frame(ctx, i):
    """Explicit succession and polity-ruler title frames."""
    t, consumed = ctx.t, ctx.consumed
    surface = t[i:i + 2]
    if any(consumed[i:i + 2]):
        return None
    if (
        t[i - 1:i] == "子"
        and t[i + 2:i + 3] == "立"
        and (
            _complete_person_pos(ctx, i, i + 1)
            or t[i + 2:i + 3] in PERSON_RIGHT_PRED
        )
    ):
        return (i, i + 2, surface, "alias")
    if not surface.endswith("公"):
        return None
    polity = ctx.token_at(i - 1) if i > 0 else None
    epithet = ctx.token_at(i)
    if not (
        polity is not None
        and polity.start == i - 1
        and polity.end == i
        and t[i - 1] in POLITY1
        and "NameType=Nat" in polity.tag
        and polity.score is not None
        and polity.score >= POS_FUNCTION_VETO_SCORE
        and epithet is not None
        and epithet.start == i
        and epithet.end == i + 1
        and epithet.tag.endswith("PROPN|NameType=Prs")
    ):
        return None
    return (i, i + 2, surface, "alias")


def rule_foreign_title_name(ctx, i):
    """A complete local component glued to the personal title 可汗 or 单于."""
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    for component_length in (3, 2, 1):
        component_end = i + component_length
        title = next(
            (
                candidate
                for candidate in ("\u53ef\u6c57", "\u5355\u4e8e")
                if t.startswith(candidate, component_end)
            ),
            None,
        )
        if title is None:
            continue
        title_end = component_end + len(title)
        component_tokens = ctx.tokens_for(i, component_end)
        if (
            any(consumed[i:title_end])
            or _surname_left_of(t, i)
            or _shi_guard(t, i)
            or not component_tokens
            or component_tokens[0].start != i
            or component_tokens[-1].end != component_end
            or component_tokens[0].bio == "I"
            or any(not "\u3400" <= char <= "\u9fff" for char in t[i:component_end])
            or _all_high_confidence_function_pos(ctx, i, component_end)
            or t[title_end:title_end + 1] in {"庭", "国", "國", "部", "军", "軍"}
        ):
            continue
        component_bio_unit = (
            component_tokens[0].bio == "B"
            and all(token.bio == "I" for token in component_tokens[1:])
            and all(
                left.end == right.start
                for left, right in zip(component_tokens, component_tokens[1:])
            )
        )
        following = ctx.token_at(title_end)
        following_name_end = None
        if (
            following is not None
            and following.start == title_end
            and following.pos == "PROPN"
            and (
                "NameType=Giv" in following.tag
                or "NameType=Prs" in following.tag
                or "NameType=Sur" in following.tag
            )
        ):
            following_name_end = following.end
            if following.bio == "B":
                cursor = following.end
                while cursor < len(t) and cursor - title_end < 3:
                    continuation = ctx.token_at(cursor)
                    if (
                        continuation is None
                        or continuation.start != cursor
                        or continuation.bio != "I"
                        or continuation.pos != "PROPN"
                        or "NameType=" not in continuation.tag
                    ):
                        break
                    following_name_end = continuation.end
                    cursor = continuation.end
        envoy_name = ctx.token_at(title_end + 1)
        envoy_designation = (
            t[title_end:title_end + 1] == "使"
            and not component_bio_unit
            and envoy_name is not None
            and envoy_name.start == title_end + 1
            and envoy_name.pos == "PROPN"
            and (
                "NameType=Giv" in envoy_name.tag
                or "NameType=Prs" in envoy_name.tag
                or "NameType=Sur" in envoy_name.tag
            )
        )
        if envoy_designation:
            continue
        complete_component = (
            ctx.gspans.get(i) == component_end
            or (
                len(component_tokens) == 1
                and component_tokens[0].end == component_end
                and component_tokens[0].pos == "PROPN"
                and "NameType=" in component_tokens[0].tag
            )
            or (
                component_bio_unit
                and all(token.pos == "PROPN" for token in component_tokens)
                and all("NameType=" in token.tag for token in component_tokens)
            )
        )
        regional_ruler = (
            component_length == 1
            and any(
                marker in component_tokens[0].tag
                for marker in ("Case=Loc", "NameType=Nat", "NameType=Geo")
            )
            and t[title_end:title_end + 1] in "弟兄父子"
            and t[i - 1:i] in PERSON_LEFT_VERBS
        )
        explicit_naming = t[max(0, i - 2):i] in {"号曰", "號曰"}
        explicit_frame = (
            explicit_naming
            or _strict_person_frame(ctx, i, title_end)
            or _title_predicate_after(ctx, title_end)
            or regional_ruler
        )
        legacy_exact_given = (
            component_length == 2
            and ctx.gspans.get(i) == component_end
            and i in gset
            and component_end not in gset
        )
        combined_title_name = (
            following_name_end is not None
            and component_bio_unit
            and (
                _title_predicate_after(ctx, following_name_end)
                or t[following_name_end:following_name_end + 1] in NAMESTART
            )
        )
        if following_name_end is not None and not combined_title_name:
            continue
        end = following_name_end or title_end
        if any(consumed[title_end:end]):
            continue
        if (
            (complete_component and explicit_frame)
            or (explicit_naming and 1 <= component_length <= 3)
            or legacy_exact_given
            or regional_ruler
            or combined_title_name
        ):
            return (i, end, t[i:end], "foreign_title_name")
    return None


def rule_royal_title_name(ctx, i):
    """Royal/heir title glued to a complete POS-given name."""
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    title = next((candidate for candidate in ROYAL_PERSON_TITLES
                  if t.startswith(candidate, i)), None)
    if title is None:
        return None
    name_start = i + len(title)
    name = t[name_start:name_start + 2]
    end = name_start + 2
    if ctx.gspans.get(name_start) != end or any(consumed[i:end]):
        return None
    if name_start not in gset or end in gset:
        return None
    if _shi_guard(t, name_start):
        return None
    # In 皇太子春秋鼎盛, 春秋 means age rather than a person's name.
    if name == "\u6625\u79cb" and t.startswith("\u9f0e\u76db", end):
        return None
    return (i, end, t[i:end], "royal_title_name")


def rule_corpus_given2(ctx, i):
    """Complete two-char POS-given span at a person-name boundary."""
    t, consumed = ctx.t, ctx.consumed
    surf = t[i:i + 2]
    if ctx.gspans.get(i) != i + 2 or any(consumed[i:i + 2]):
        return None
    if _all_high_confidence_function_pos(ctx, i, i + 2):
        return None
    if _shi_guard(t, i) or _foreign_title_followed_by_name(ctx, i + 2):
        return None
    # only at a name boundary (punct / appos) to curb mid-word collisions
    prev = t[i - 1] if i > 0 else "\u0001"
    if not (prev in NAMESTART or prev in APPOS_TAIL):
        return None
    return (i, i + 2, surf, "given2")


def _surname_left_of(t, i):
    return t[i - 1:i] in RSUR1 or t[i - 2:i] in COMPOUND or t[i - 3:i] in COMPOUND3


def rule_semantic_given2(ctx, i):
    """Embedded complete two-char POS-given span licensed by local person syntax.

    Unlike corpus_given2's punctuation/apposition boundary, this rule handles forms
    such as 封伯鲁、杀侠累、肥义曰、为师道所亲信、与望之有隙. POS·Giv supplies
    the name reading; immediate surname and trailing-Giv guards prevent splitting
    驷子阳 or a longer given name.
    """
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    surf = t[i:i + 2]
    if ctx.gspans.get(i) != i + 2 or any(consumed[i:i + 2]):
        return None
    if _shi_guard(t, i):
        return None
    prev = t[i - 1] if i > 0 else "\u0001"
    if prev in NAMESTART or prev in APPOS_TAIL:
        return None
    if i not in gset or i + 2 in gset or _surname_left_of(t, i):
        return None
    nxt = t[i + 2:i + 3]
    if (
        surf[-1] in PERSON_REPORTING_VERBS
        and nxt in PERSON_SUBJECT_PRED
        and i > 0
        and t[i - 1:i + 1] in ctx.corpus.ner
    ):
        return None
    person_frame = prev in PERSON_LEFT_VERBS or nxt in PERSON_RIGHT_PRED
    coord_frame = prev in COORD_HEAD and nxt in COORD_TAIL
    enumeration_frame = nxt == "\u7b49"
    passive_frame = prev == "\u4e3a" and nxt == "\u6240"
    association_frame = prev == "\u4e0e" and nxt == "\u6709"
    if not (person_frame or coord_frame or enumeration_frame
            or passive_frame or association_frame):
        return None
    return (i, i + 2, surf, "given2_sem")


def rule_office_alias2(ctx, i):
    """Office ending in 史 followed by a complete POS-given person surface.

    Reject a following surname because `刺史南阳朱穆` is office + place + person,
    whereas `刺史种暠` and `内史汲黯` place the person immediately after the office.
    """
    t, consumed = ctx.t, ctx.consumed
    surf = t[i:i + 2]
    if ctx.gspans.get(i) != i + 2 or any(consumed[i:i + 2]):
        return None
    if not any(t[max(0, i - len(title)):i] == title for title in OFFICE_SHI_TITLES):
        return None
    j = i + 2
    if t[j:j + 1] in RSUR1 or t[j:j + 2] in COMPOUND or t[j:j + 3] in COMPOUND3:
        return None
    if _shi_guard(t, i):
        return None
    return (i, i + 2, surf, "given2_office")


def rule_appointment(ctx, i):
    """`以 X 为 + office/title` appointment frame for a POS-backed person."""
    t, consumed = ctx.t, ctx.consumed
    surf = t[i:i + 2]
    if (
        not (
            ctx.gspans.get(i) == i + 2
            or _complete_person_pos(ctx, i, i + 2)
        )
        or any(consumed[i:i + 2])
    ):
        return None
    if t[i - 1:i] != "\u4ee5" or t[i + 2:i + 3] != "\u4e3a":
        return None
    if surf in APPOINT_TITLES:
        return None
    if not any(t.startswith(title, i + 3) for title in APPOINT_TITLES):
        return None
    if _shi_guard(t, i):
        return None
    return (i, i + 2, surf, "given2_appoint")


COMBINED_EVIDENCE_POLICIES = (
    E.CumulativeAdmissionPolicy(
        "cumulative-family-score",
        (
            E.FamilySupport(
                "jie_morphology",
                2,
                all_signals=frozenset({
                    "jie_person_morphology_anchor",
                    "jie_person_morphology_majority",
                }),
            ),
            E.FamilySupport(
                "local_anchor",
                2,
                all_signals=frozenset({"admitted_local_anchor"}),
            ),
            E.FamilySupport(
                "recurrence",
                1,
                all_signals=frozenset({"exact_local_recurrence"}),
            ),
            E.FamilySupport(
                "syntax",
                1,
                any_signals=frozenset({
                    "strict_person_frame",
                    "decisive_person_syntax",
                }),
            ),
            E.FamilySupport(
                "name_shape",
                1,
                all_signals=frozenset({
                    "surname_shape",
                    "local_surname_morphology",
                }),
            ),
            E.FamilySupport(
                "title_semantics",
                1,
                any_signals=frozenset({
                    "jie_person_title_anchor",
                    "person_title_shape",
                }),
            ),
            E.FamilySupport(
                "genealogy_semantics",
                1,
                all_signals=frozenset({"genealogy_name_anchor"}),
            ),
            E.FamilySupport(
                "translation",
                2,
                all_signals=frozenset({"translation_exact_identity"}),
            ),
        ),
        minimum_score=6,
        minimum_families=4,
        prerequisite_signals=frozenset({"model_ner_witness"}),
        conflict_penalties=(
            ("missing_person_morphology", 0),
            ("function_morphology", 1),
            ("geo_nat_morphology", 2),
        ),
    ),
    E.AdmissionPolicy(
        "inherent-title-appointment",
        frozenset({
            "inherent_person_title",
            "appointment_frame",
            "human_appointment_role",
        }),
    ),
    E.AdmissionPolicy(
        "ambiguous-title-appointment-pos",
        frozenset({
            "ambiguous_person_title",
            "appointment_frame",
            "human_appointment_role",
            "complete_person_pos",
        }),
    ),
    E.AdmissionPolicy(
        "ambiguous-title-appointment-translation",
        frozenset({
            "ambiguous_person_title",
            "appointment_frame",
            "human_appointment_role",
            "translation_exact_identity",
        }),
    ),
    E.AdmissionPolicy(
        "long-repeat-boundary-model",
        frozenset({
            "model_name_morphology",
            "exact_local_recurrence",
            "hard_name_boundary",
        }),
        prerequisite_signals=frozenset({
            "model_ner_witness",
            "long_surface",
        }),
    ),
    E.AdmissionPolicy(
        "soft-title-recurrence-syntax",
        frozenset({
            "jie_person_title_anchor",
            "exact_local_recurrence",
            "person_occurrence_syntax",
            "jie_person_morphology_majority",
        }),
        prerequisite_signals=frozenset({"model_ner_witness"}),
        allowed_soft_conflicts=frozenset({
            "function_morphology",
            "geo_nat_morphology",
            "missing_person_morphology",
        }),
    ),
    E.AdmissionPolicy(
        "soft-genealogy-recurrence-syntax",
        frozenset({
            "genealogy_name_anchor",
            "exact_local_recurrence",
            "person_occurrence_syntax",
            "jie_person_morphology_anchor",
        }),
        prerequisite_signals=frozenset({"model_ner_witness"}),
        allowed_soft_conflicts=frozenset({
            "function_morphology",
            "geo_nat_morphology",
            "missing_person_morphology",
        }),
    ),
    E.AdmissionPolicy(
        "soft-surname-recurrence-syntax",
        frozenset({
            "jie_person_morphology_anchor",
            "local_surname_morphology",
            "surname_shape",
            "exact_local_recurrence",
            "decisive_person_syntax",
        }),
        prerequisite_signals=frozenset({"model_ner_witness"}),
        allowed_soft_conflicts=frozenset({
            "geo_nat_morphology",
            "missing_person_morphology",
        }),
    ),
    E.AdmissionPolicy(
        "soft-surname-geo-decisive-syntax",
        frozenset({
            "jie_person_morphology_anchor",
            "surname_shape",
            "exact_local_recurrence",
            "decisive_person_syntax",
            "geo_nat_morphology",
        }),
        prerequisite_signals=frozenset({"model_ner_witness"}),
        allowed_soft_conflicts=frozenset({
            "geo_nat_morphology",
            "function_morphology",
            "missing_person_morphology",
        }),
    ),
    E.AdmissionPolicy(
        "soft-surname-geo-recurrence-syntax",
        frozenset({
            "jie_person_morphology_majority",
            "surname_shape",
            "exact_local_recurrence",
            "person_occurrence_syntax",
            "geo_nat_morphology",
        }),
        prerequisite_signals=frozenset({"model_ner_witness"}),
        allowed_soft_conflicts=frozenset({
            "geo_nat_morphology",
            "function_morphology",
            "missing_person_morphology",
        }),
    ),
    E.AdmissionPolicy(
        "soft-translation-recurrence-syntax",
        frozenset({
            "translation_exact_identity",
            "exact_local_recurrence",
            "person_occurrence_syntax",
            "jie_person_morphology_anchor",
        }),
        prerequisite_signals=frozenset({"model_ner_witness"}),
        allowed_soft_conflicts=frozenset({
            "function_morphology",
            "geo_nat_morphology",
            "missing_person_morphology",
        }),
    ),
)


def _combined_appointment_candidate(ctx, start, end):
    """Build evidence for `以 + candidate + 为 + human role/title`."""
    surface = ctx.t[start:end]
    candidate = E.Candidate(start=start, end=end, surface=surface)
    if ctx.t[start - 1:start] == "以" and ctx.t[end:end + 1] == "为":
        candidate.add("appointment_frame", "syntax")

    role = next(
        (
            title
            for title in HUMAN_APPOINT_TITLES
            if ctx.t.startswith(title, end + 1)
        ),
        None,
    )
    if role is not None:
        candidate.add("human_appointment_role", "role_semantics")

    suffix = next(
        (
            title
            for title in PERSON_TITLE_SUFFIXES
            if surface.endswith(title) and len(surface) > len(title)
        ),
        None,
    )
    if suffix in INHERENT_PERSON_TITLE_SUFFIXES:
        candidate.add("inherent_person_title", "title_semantics")
    elif suffix in {"王", "后", "公", "侯"}:
        candidate.add("ambiguous_person_title", "title_semantics")

    if _complete_person_pos(ctx, start, end):
        candidate.add("complete_person_pos", "morphology")
    translated = ctx.translation_fullnames.get(start)
    if translated is not None and translated["end"] == end:
        candidate.add("translation_exact_identity", "translation")

    if (
        not 2 <= len(surface) <= 6
        or any(not "\u3400" <= char <= "\u9fff" for char in surface)
    ):
        candidate.veto("invalid_surface")
    if any(ctx.consumed[start:end]):
        candidate.veto("overlap")
    if _shi_guard(ctx.t, start):
        candidate.veto("clan_suffix")
    if any(
        ctx.t.startswith(title, end)
        for title in FULLNAME_TITLE_CONTINUATIONS
    ):
        candidate.veto("office_continuation")
    if _occurrence_has_polity_frame(ctx, start, end):
        candidate.veto("local_polity_usage")
    return candidate


def _model_ner_witnesses(ctx):
    witnesses = []
    for start in range(len(ctx.t)):
        max_length = min(ctx.corpus.ner_maxL, len(ctx.t) - start, 8)
        for length in range(max_length, 1, -1):
            end = start + length
            surface = ctx.t[start:end]
            if (
                surface in ctx.corpus.ner
                and not set(surface) & (NAMESTART | GLOSS_SEP)
            ):
                witnesses.append((start, end))
    return witnesses


def _combined_candidate_lattice(ctx, cards):
    """Candidate union from model surfaces, POS spans, and translation mappings."""
    candidates = {}

    def add(start, end):
        if not 0 <= start < end <= len(ctx.t):
            return
        surface = ctx.t[start:end]
        if (
            not 2 <= len(surface) <= 8
            or set(surface) & (NAMESTART | GLOSS_SEP)
            or any(not "\u3400" <= char <= "\u9fff" for char in surface)
        ):
            return
        candidates.setdefault(
            (start, end),
            E.Candidate(start=start, end=end, surface=surface),
        )

    witnesses = _model_ner_witnesses(ctx)
    for start, end in witnesses:
        add(start, end)
    for start, end in ctx.gspans.items():
        add(start, end)
    for mapping in (
        tuple(ctx.translation_fullnames.values())
        + tuple(ctx.translation_mentions.values())
    ):
        add(mapping["start"], mapping["end"])

    admitted_surfaces = {
        card["surface"]
        for card in cards
        if card.get("surface")
    }
    for candidate in candidates.values():
        for witness_start, witness_end in witnesses:
            if (
                candidate.start == witness_start
                and candidate.end == witness_end
            ):
                candidate.add(
                    "model_ner_witness",
                    "model",
                    match="exact",
                    witness_start=witness_start,
                    witness_end=witness_end,
                )
        tokens = ctx.tokens_for(candidate.start, candidate.end)
        personal_tokens = [
            token
            for token in tokens
            if token.pos == "PROPN"
            and any(
                name_type in token.tag
                for name_type in (
                    "NameType=Sur",
                    "NameType=Giv",
                    "NameType=Prs",
                )
            )
        ]
        geo_nat = any(
            "NameType=Geo" in token.tag or "NameType=Nat" in token.tag
            for token in tokens
        )
        complete_person = (
            bool(tokens)
            and len(personal_tokens) == len(tokens)
            and not geo_nat
        )
        if complete_person:
            candidate.add("model_name_morphology", "model")
        else:
            candidate.missing("complete_person_morphology")
        if personal_tokens:
            candidate.add("partial_person_morphology", "model")
        else:
            candidate.missing("partial_person_morphology")
            candidate.conflict("missing_person_morphology")
        if geo_nat:
            candidate.add("geo_nat_morphology", "model")
            candidate.conflict("geo_nat_morphology")
        if _all_high_confidence_function_pos(
            ctx, candidate.start, candidate.end
        ):
            candidate.conflict("function_morphology")
        if ctx.t.count(candidate.surface) > 1:
            candidate.add("exact_local_recurrence", "local_recurrence")
        else:
            candidate.missing("exact_local_recurrence")
        if (
            ctx.t[candidate.start - 1:candidate.start] in NAMESTART
            or ctx.t[candidate.end:candidate.end + 1] in NAMESTART
        ):
            candidate.add("hard_name_boundary", "syntax")
        occurrence_syntax = (
            _strict_person_frame(ctx, candidate.start, candidate.end)
            or ctx.t[candidate.start - 1:candidate.start] in NAMESTART
            or ctx.t[candidate.end:candidate.end + 1] in NAMESTART
        )
        if occurrence_syntax:
            candidate.add("person_occurrence_syntax", "syntax")
        else:
            candidate.missing("person_occurrence_syntax")
        if _strict_person_frame(ctx, candidate.start, candidate.end):
            candidate.add("strict_person_frame", "syntax")
        decisive_syntax = (
            ctx.t[candidate.start - 1:candidate.start] in PERSON_LEFT_VERBS
            or ctx.t[candidate.end:candidate.end + 1] in PERSON_SUBJECT_PRED
            or (
                ctx.t[candidate.start - 1:candidate.start] in NAMESTART
                and _next_token_is_high_conf_verb(ctx, candidate.end)
            )
        )
        if decisive_syntax:
            candidate.add("decisive_person_syntax", "syntax")
        if _is_xing_headed(candidate.surface):
            candidate.add("surname_shape", "name_shape")
        if (
            tokens
            and tokens[0].start == candidate.start
            and tokens[0].pos == "PROPN"
            and "NameType=Sur" in tokens[0].tag
        ):
            candidate.add("local_surname_morphology", "model")
        if any(
            candidate.surface.endswith(suffix)
            and len(candidate.surface) > len(suffix)
            for suffix in MODEL_PERSON_TITLE_SUFFIXES
        ):
            candidate.add("person_title_shape", "title_semantics")
        if candidate.surface in admitted_surfaces:
            candidate.add("admitted_local_anchor", "local_anchor")
        if candidate.surface in ctx.jie_person_surfaces:
            candidate.add(
                "jie_person_morphology_anchor",
                "jie_morphology",
            )
        if candidate.surface in ctx.jie_partial_person_surfaces:
            candidate.add(
                "jie_partial_person_morphology_anchor",
                "jie_morphology",
            )
        if (
            candidate.surface
            in ctx.jie_person_morphology_majority_surfaces
        ):
            candidate.add(
                "jie_person_morphology_majority",
                "jie_morphology",
            )
        if candidate.surface in ctx.jie_person_title_surfaces:
            candidate.add("jie_person_title_anchor", "title_semantics")
        genealogy_anchor = any(
            (prefix + candidate.surface) in ctx.t
            for prefix in GENEALOGY_PREFIXES
            if len(prefix) > 1
        )
        if genealogy_anchor:
            candidate.add("genealogy_name_anchor", "genealogy_semantics")
        translated = ctx.translation_fullnames.get(candidate.start)
        if translated is not None and translated["end"] == candidate.end:
            candidate.add("translation_exact_identity", "translation")

        following = ctx.token_at(candidate.end)
        if len(candidate.surface) >= 3:
            candidate.add("long_surface", "span_shape")
        if any(ctx.consumed[candidate.start:candidate.end]):
            candidate.veto("overlap")
        if _shi_guard(ctx.t, candidate.start):
            candidate.veto("clan_suffix")
        if ctx.t[candidate.end:candidate.end + 1] == "氏":
            candidate.veto("clan_continuation")
        if candidate.surface.endswith("氏"):
            candidate.veto("clan_suffix")
        exact_nonperson_lexical_class = (
            candidate.surface in APPOINT_TITLES
            or candidate.surface in OFFICE_TITLES
            or candidate.surface in PERSON_ROLE_PREFIXES
            or candidate.surface in TITLE_NONPERSON_COMPONENTS
        )
        geographic_lexical_collision = (
            candidate.surface in ctx.corpus.admin_places
            or candidate.surface in ctx.corpus.geo_names
        ) and geo_nat and not personal_tokens
        if exact_nonperson_lexical_class or geographic_lexical_collision:
            candidate.veto("nonperson_lexical_class")
        previous_token = ctx.token_at(candidate.start - 1)
        if (
            not personal_tokens
            and tokens
            and tokens[0].bio == "I"
            and previous_token is not None
            and previous_token.end == candidate.start
            and ctx.t[candidate.start - 1:candidate.start]
            not in (PERSON_LEFT_VERBS | {"将"})
            and previous_token.bio == "B"
            and previous_token.pos == tokens[0].pos
        ):
            candidate.veto("incomplete_bio_left_continuation")
        if (
            not personal_tokens
            and (
                ctx.t.startswith("五品", candidate.end)
                or ctx.t.startswith("总", candidate.end)
                or ctx.t.startswith("皆", candidate.end)
                or ctx.t.startswith("酋", candidate.end)
            )
        ):
            candidate.veto("collective_role_continuation")
        if ctx.t[candidate.end:candidate.end + 1] in set(
            "一二三四五六七八九十百千万两"
        ):
            candidate.veto("numeric_continuation")
        if _has_office_continuation(ctx, candidate.end):
            candidate.veto("office_continuation")
        if (
            ctx.t.startswith("督", candidate.end)
            and following is not None
            and following.start == candidate.end
            and following.pos == "NOUN"
        ):
            candidate.veto("office_continuation")
        if (
            ctx.t[candidate.end:candidate.end + 1] in {"妃", "后"}
            and following is not None
            and following.start == candidate.end
            and following.pos == "NOUN"
        ):
            candidate.veto("title_continuation")
        if (
            candidate.surface in TRANSLATION_BARE_PERSON_TITLES
            and following is not None
            and following.start == candidate.end
            and following.pos == "PROPN"
            and any(
                name_type in following.tag
                for name_type in (
                    "NameType=Sur",
                    "NameType=Giv",
                    "NameType=Prs",
                )
            )
        ):
            candidate.veto("title_continuation")
        if _has_person_designation_right_continuation(
            ctx, candidate.start, candidate.end
        ):
            candidate.veto("title_continuation")
        if _has_polity_title_left_continuation(
            ctx, candidate.start, candidate.end
        ):
            candidate.veto("title_left_continuation")
        if _has_geo_title_right_continuation(ctx, candidate.end):
            candidate.veto("polity_title_continuation")
        if _has_location_office_right_continuation(
            ctx, candidate.start, candidate.end
        ):
            candidate.veto("location_office_continuation")
        if ctx.t[candidate.end:candidate.end + 1] == "寺":
            candidate.veto("nonperson_lexical_continuation")
        if (
            ctx.t[candidate.start - 1:candidate.start] == "被"
            and tokens
            and all(token.pos == "NOUN" for token in tokens)
        ):
            candidate.veto("nonperson_object_usage")
        if (
            ctx.t[candidate.start - 1:candidate.start] == "为"
            and ctx.t[candidate.end:candidate.end + 1] == "军"
        ):
            candidate.veto("collective_force_continuation")
        if _has_person_bio_left_continuation(ctx, candidate.start) and any(
            ctx.t[extension_start:candidate.end] in ctx.corpus.ner
            for extension_start in range(
                max(0, candidate.start - 2),
                candidate.start,
            )
        ):
            candidate.veto("longer_name_left_continuation")
        if (
            _has_person_bio_right_continuation(
                ctx, candidate.start, candidate.end
            )
            or _has_person_designation_right_continuation(
                ctx, candidate.start, candidate.end
            )
            or _has_repeated_model_extension(
                ctx, candidate.start, candidate.end
            )
        ) and any(
            ctx.t[candidate.start:extension_end] in ctx.corpus.ner
            for extension_end in range(
                candidate.end + 1,
                min(len(ctx.t), candidate.end + 2) + 1,
            )
        ):
            candidate.veto("longer_name_right_continuation")
        coordinated = ctx.token_at(candidate.end + 1)
        if (
            ctx.t[candidate.end:candidate.end + 1] == "、"
            and coordinated is not None
            and coordinated.start == candidate.end + 1
            and (
                "NameType=Geo" in coordinated.tag
                or "NameType=Nat" in coordinated.tag
            )
        ):
            candidate.veto("coordinated_polity")
        preceding = ctx.token_at(candidate.start - 1)
        if (
            ctx.t[candidate.start - 1:candidate.start] == "、"
            and ctx.t[candidate.end:candidate.end + 1] in {"、", "等"}
            and preceding is not None
            and preceding.end == candidate.start - 1
            and (
                "NameType=Geo" in preceding.tag
                or "NameType=Nat" in preceding.tag
            )
        ):
            candidate.veto("backward_coordinated_polity")
        if _occurrence_has_polity_frame(ctx, candidate.start, candidate.end):
            candidate.veto("local_polity_usage")
        if _surface_has_jie_collective_frame(ctx, candidate.surface):
            candidate.veto("jie_collective_usage")
        if _surface_has_supernatural_frame(ctx, candidate.surface):
            candidate.veto("supernatural_entity_context")
        if _occurrence_has_polity_frame(ctx, candidate.start, candidate.end):
            candidate.veto("jie_polity_usage")
        if ctx.t[candidate.end:candidate.end + 1] in {"众", "王"}:
            candidate.veto("collective_or_polity_continuation")
        if (
            ctx.t.startswith("并兴", candidate.end)
            or ctx.t.startswith("之道", candidate.end)
            or ctx.t.startswith("不备", candidate.end)
            or ctx.t.startswith("才能", candidate.end)
            or ctx.t.startswith("参用", candidate.end)
        ):
            candidate.veto("abstract_category_continuation")
        if any(
            ctx.t.startswith(suffix, candidate.end)
            for suffix in ("士卒", "精骑")
        ):
            candidate.veto("collective_force_continuation")
        if _has_person_bio_right_continuation(
            ctx, candidate.start, candidate.end
        ):
            candidate.veto("proper_name_continuation")
    return candidates.values()


def detect_combined_evidence(ctx, cards):
    """Admit candidates supported by independent evidence-family policies."""
    out = []
    marker = ctx.t.find("以")
    while marker >= 0:
        start = marker + 1
        limit = min(len(ctx.t), start + 7)
        end = ctx.t.find("为", start + 2, limit)
        if end >= 0:
            candidate = _combined_appointment_candidate(ctx, start, end)
            policy = E.decide(candidate, COMBINED_EVIDENCE_POLICIES)
            if ctx.evidence_audit is not None:
                audit_row = E.candidate_audit_metadata(
                    candidate,
                    COMBINED_EVIDENCE_POLICIES,
                    policy,
                )
                audit_row["candidate_source"] = "appointment"
                ctx.evidence_audit.append(audit_row)
            if policy is not None:
                for offset in range(start, end):
                    ctx.consumed[offset] = True
                out.append((
                    start,
                    end,
                    candidate.surface,
                    "combined_evidence",
                    E.audit_metadata(candidate, policy),
                ))
        marker = ctx.t.find("以", marker + 1)
    for candidate in sorted(
        _combined_candidate_lattice(ctx, cards),
        key=lambda row: (row.start, -(row.end - row.start)),
    ):
        if any(ctx.consumed[candidate.start:candidate.end]):
            continue
        policy = E.decide(candidate, COMBINED_EVIDENCE_POLICIES)
        if ctx.evidence_audit is not None:
            audit_row = E.candidate_audit_metadata(
                candidate,
                COMBINED_EVIDENCE_POLICIES,
                policy,
            )
            audit_row["candidate_source"] = "lattice"
            ctx.evidence_audit.append(audit_row)
        if policy is None:
            continue
        for offset in range(candidate.start, candidate.end):
            ctx.consumed[offset] = True
        out.append((
            candidate.start,
            candidate.end,
            candidate.surface,
            "combined_evidence",
            E.audit_metadata(candidate, policy),
        ))
    return out


def rule_struct_fuxing(ctx, i):
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    cp = 3 if t[i:i + 3] in COMPOUND3 else (2 if t[i:i + 2] in COMPOUND else 0)
    if not cp:
        return None
    g = 0
    while g < 2 and (i + cp + g) < len(t) and (i + cp + g) in gset:
        g += 1
    if g < 1 or any(consumed[i:i + cp + g]):
        return None
    surf = t[i:i + cp + g]
    if _all_high_confidence_function_pos(ctx, i, i + cp + g):
        return None
    return (i, i + cp + g, surf, "fuxing")


def rule_role_bio_name(ctx, i):
    """Human role immediately followed by a complete BIO given-name span.

    Classical prose often introduces surname-less religious or occupational names as
    `胡僧慧范`, `沙门法显`, or `尼法愿`. The role proves local personhood but remains
    outside the underline. Running this before corpus lexicon rules also prevents a
    role phrase such as `胡僧` from consuming the actual following name.
    """
    t, consumed = ctx.t, ctx.consumed
    role = next((item for item in PERSON_ROLE_PREFIXES if t.startswith(item, i)), None)
    if role is None:
        return None
    start = i + len(role)
    end = ctx.gspans.get(start)
    if end is None or not (1 <= end - start <= 3):
        return None
    if role == "尼" and t[i - 1:i] == "为":
        return None
    if any(consumed[i:end]):
        return None
    role_tokens = ctx.tokens_for(i, start)
    if not role_tokens or any(token.pos != "NOUN" for token in role_tokens):
        return None
    surface = t[start:end]
    if set(surface) & (NAMESTART | GLOSS_SEP):
        return None
    next_token = ctx.token_at(end)
    if next_token is not None and next_token.start == end and next_token.pos == "PROPN":
        return None
    return (start, end, surface, "role_name")


def rule_pos_fullname(ctx, i):
    """High-confidence POS surname followed by a complete BIO given-name span."""
    t, consumed = ctx.t, ctx.consumed
    surname = ctx.token_at(i)
    if surname is None or surname.start != i or surname.end != i + 1:
        return None
    if (
        surname.tag != "PROPN|NameType=Sur"
        or surname.score is None
        or surname.score < POS_FUNCTION_VETO_SCORE
        or t[i] in NAMESTART
    ):
        return None
    end = ctx.gspans.get(i + 1)
    if end is None or not (1 <= end - (i + 1) <= 2):
        return None
    given = t[i + 1:end]
    if t[i:i + 2] in COMPOUND or t[i:i + 3] in COMPOUND3:
        return None
    if any(consumed[i:end]):
        return None
    prev = t[i - 1:i] or "\u0001"
    numbered_left = _sec_num(prev) is not None
    polity_token = ctx.token_at(i - 1) if i > 0 else None
    polity_left = (
        prev in POLITY1
        and polity_token is not None
        and polity_token.start == i - 1
        and polity_token.end == i
        and "NameType=Nat" in polity_token.tag
        and polity_token.score is not None
        and polity_token.score >= POS_FUNCTION_VETO_SCORE
        and not (i > 1 and t[i - 2] in POLITY_NUM_PREFIX)
    )
    if numbered_left and t[i] in BLOCK1:
        return None
    if (
        (prev in BLOCK1 and not polity_left)
        or t[i - 2:i] in BLOCK2
        or _fuxing_left(t, i)
    ):
        return None
    role_left = any(t[max(0, i - len(role)):i] == role for role in PERSON_ROLE_PREFIXES)
    list_frame = prev == "以" and t[end:end + 1] == "、"
    admin_token = ctx.token_at(i - 1) if i > 0 else None
    local_geo_left = (
        admin_token is not None
        and admin_token.end == i
        and "Case=Loc" in admin_token.tag
        and "NameType=Geo" in admin_token.tag
        and admin_token.score is not None
        and admin_token.score >= POS_FUNCTION_VETO_SCORE
    )
    explicit_admin_geo_left = (
        admin_token is not None
        and admin_token.end == i
        and admin_token.text[-1:] in ADMIN_INTRO_TAIL
        and "Case=Loc" in admin_token.tag
        and "NameType=Geo" in admin_token.tag
    )
    ce_year = ctx.year_at(i)
    temporal_admin_left = False
    if ce_year is not None:
        for surface, years in ctx.corpus.admin_places.items():
            start = i - len(surface)
            if (
                start >= 0
                and ce_year in years
                and t[start:i] == surface
            ):
                tokens = ctx.tokens_for(start, i)
                if tokens and all(
                    token.pos == "PROPN" and "NameType=" in token.tag
                    for token in tokens
                ):
                    temporal_admin_left = True
                    break
    admin_left = local_geo_left or explicit_admin_geo_left or temporal_admin_left
    if not (
        numbered_left
        or polity_left
        or prev in NAMESTART
        or prev in APPOS_TAIL
        or prev in PERSON_LEFT_VERBS
        or prev in COORD_HEAD
        or role_left
        or list_frame
        or admin_left
    ):
        return None
    surface = t[i:end]
    if (
        _shi_guard(t, i)
        or any(t.startswith(title, end) for title in FULLNAME_TITLE_CONTINUATIONS)
    ):
        return None
    return (i, end, surface, "pos_fullname")


PERSON_POSSESSIVE_SUFFIXES = set("庙墓祠柩第")


def _next_token_is_high_conf_verb(ctx, start):
    token = ctx.token_at(start)
    return (
        token is not None
        and token.start == start
        and token.pos in {"VERB", "AUX"}
        and token.score is not None
        and token.score >= POS_FUNCTION_VETO_SCORE
    )


def _complete_person_pos(ctx, start, end):
    tokens = ctx.tokens_for(start, end)
    if not tokens or tokens[0].start != start or tokens[-1].end != end:
        return False
    if any(left.end != right.start for left, right in zip(tokens, tokens[1:])):
        return False
    if all(
        token.tag.endswith("PROPN|NameType=Prs")
        and token.score is not None
        and token.score >= KNOWN_FULLNAME_POS_SCORE
        for token in tokens
    ):
        return True
    surname = tokens[0]
    return (
        surname.tag == "PROPN|NameType=Sur"
        and surname.score is not None
        and surname.score >= POS_FUNCTION_VETO_SCORE
        and ctx.gspans.get(surname.end) == end
    )


def _high_confidence_name_pos(ctx, start, end):
    tokens = ctx.tokens_for(start, end)
    return (
        bool(tokens)
        and tokens[0].start == start
        and tokens[-1].end == end
        and all(left.end == right.start for left, right in zip(tokens, tokens[1:]))
        and all(
            token.pos == "PROPN"
            and (
                token.tag == "PROPN|NameType=Sur"
                or "NameType=Giv" in token.tag
            )
            and token.score is not None
            and token.score >= KNOWN_FULLNAME_POS_SCORE
            for token in tokens
        )
    )


def rule_known_fullname_pos(ctx, i):
    """Complete high-confidence POS name span, independent of person KB."""
    t, consumed = ctx.t, ctx.consumed
    for length in range(min(6, len(t) - i), 1, -1):
        surface = t[i:i + length]
        if any(consumed[i:i + length]):
            continue
        end = i + length
        if (
            _shi_guard(t, i)
            or (surface == "春秋" and t.startswith("鼎盛", end))
            or any(t.startswith(title, end) for title in FULLNAME_TITLE_CONTINUATIONS)
            or _foreign_title_followed_by_name(ctx, end)
        ):
            return None
        ce_year = ctx.year_at(i)
        if (
            ce_year is not None
            and ce_year in ctx.corpus.admin_places.get(surface, ())
        ):
            return None
        if _high_confidence_name_pos(ctx, i, end):
            return (i, end, surface, "pos_person_name")
    return None


def _person_object_end(ctx, start):
    for end in range(min(len(ctx.t), start + 4), start, -1):
        if _complete_person_pos(ctx, start, end) or ctx.gspans.get(start) == end:
            return end
    return None


def _coordinated_object_pair(ctx, start):
    first_end = _person_object_end(ctx, start)
    if first_end is None or ctx.t[first_end:first_end + 1] != "、":
        return None
    second_start = first_end + 1
    second_end = _person_object_end(ctx, second_start)
    if second_end is None:
        return None
    return first_end, second_start, second_end


def _nat_verb_before_coordinated_objects(ctx, start):
    nat = ctx.token_at(start)
    verb = ctx.token_at(start + 1)
    return (
        nat is not None
        and nat.start == start
        and nat.end == start + 1
        and "NameType=Nat" in nat.tag
        and nat.score is not None
        and nat.score >= POS_FUNCTION_VETO_SCORE
        and verb is not None
        and verb.start == start + 1
        and verb.end == start + 2
        and verb.pos == "VERB"
        and verb.score is not None
        and verb.score >= POS_FUNCTION_VETO_SCORE
        and _coordinated_object_pair(ctx, start + 2) is not None
    )


def rule_coordinated_person_object(ctx, i):
    """First of two independently POS-proven known people joined by `、`."""
    verb = ctx.token_at(i - 1) if i > 0 else None
    if not (
        verb is not None
        and verb.start == i - 1
        and verb.end == i
        and ctx.t[i - 1] in COORD_PERSON_VERBS
        and verb.pos == "VERB"
        and verb.score is not None
        and verb.score >= POS_FUNCTION_VETO_SCORE
    ):
        return None
    pair = _coordinated_object_pair(ctx, i)
    if pair is None:
        return None
    first_end, _, _ = pair
    if any(ctx.consumed[i:first_end]):
        return None
    surface = ctx.t[i:first_end]
    return (i, first_end, surface, "pos_person_name")


def rule_person_possessive(ctx, i):
    """POS-backed person surface used as the possessor of a personal site or object."""
    t, consumed = ctx.t, ctx.consumed
    for length in (3, 2):
        end = i + length
        surface = t[i:end]
        if t[end:end + 1] not in PERSON_POSSESSIVE_SUFFIXES:
            continue
        if any(consumed[i:end]):
            continue
        if _shi_guard(t, i):
            continue
        if not _complete_person_pos(ctx, i, end):
            continue
        return (i, end, surface, "pos_person_name")
    return None


def rule_struct_xingming(ctx, i):
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    c = t[i]
    if c not in CLEAN or _fuxing_left(t, i):
        return None
    if (t[i - 1] if i > 0 else "") in BLOCK1 or t[i - 2:i] in BLOCK2:
        return None
    if c in JUE_HEAD and not _jue_ok(t, i, gset):
        return None
    n = len(t)
    # 3-char novel
    if i + 2 < n and (i + 1) in gset and (i + 2) in gset and not any(consumed[i:i + 3]):
        surf = t[i:i + 3]
        if not _all_high_confidence_function_pos(ctx, i, i + 3):
            return (i, i + 3, surf, "xingming3")
    # 2-char novel
    if i + 1 < n and (i + 1) in gset and not any(consumed[i:i + 2]):
        if not (c == PU and t[i + 2:i + 3] == SHE):
            surf = t[i:i + 2]
            if not _all_high_confidence_function_pos(ctx, i, i + 2):
                return (i, i + 2, surf, "xingming2")
    return None


# priority order (higher first reserves the span)
RULES = [
    ("explicit_title_frame", "jie", rule_explicit_title_frame),
    ("royal_title_name", "jie", rule_royal_title_name),
    ("foreign_title_name", "jie", rule_foreign_title_name),
    ("surname_honorific", "jie", rule_surname_honorific),
    ("female_court_title", "jie", rule_female_court_title),
    ("foreign_suffix_name", "jie", rule_foreign_suffix_name),
    ("genealogy_given", "jie", rule_genealogy_given),
    ("office_fullname", "jie", rule_office_fullname),
    ("office_name", "jie", rule_office_name),
    ("pos_fullname", "jie", rule_pos_fullname),
    ("office_alias2", "jie", rule_office_alias2),
    ("appointment", "jie", rule_appointment),
    ("multifief_jue_name", "jie", rule_multifief_jue_name),
    ("surname_empress", "corpus", rule_surname_empress),
    ("known_title", "corpus", rule_known_title),
    ("princess_title", "jie", rule_princess_title),
    ("model_ner_fief_title", "jie", rule_model_ner_fief_title),
    ("model_ner_rank_title", "jie", rule_model_ner_rank_title),
    ("model_ner_temple_title", "jie", rule_model_ner_temple_title),
    ("model_ner_short_royal_title", "jie", rule_model_ner_short_royal_title),
    ("model_ner_title", "jie", rule_model_ner_title),
    ("model_ner_given_boundary", "jie", rule_model_ner_given_boundary),
    ("model_ner_predicate", "jie", rule_model_ner_predicate),
    ("model_ner_appos", "jie", rule_model_ner_appos),
    ("model_ner_local_surname_name", "jie", rule_model_ner_local_surname_name),
    ("model_ner_name", "jie", rule_model_ner_name),
    ("model_ner_partial_pos", "jie", rule_model_ner_partial_pos),
    ("title_appellation", "jie", rule_title_appellation),
    ("role_bio_name", "jie", rule_role_bio_name),
    ("corpus_lit3", "corpus", rule_corpus_lit3),
    ("empress_title", "jie", rule_empress_title),
    ("polity_appos", "jie", rule_polity_appos),
    ("block_appos", "jie", rule_block_appos),
    ("corpus_xing2", "corpus", rule_corpus_xing2),
    ("corpus_jue2", "corpus", rule_corpus_jue2),
    ("role", "corpus", rule_role),
    ("jue_name", "jie", rule_jue_name),
    ("polity_king", "corpus", rule_polity_king),
    ("struct_fuxing", "jie", rule_struct_fuxing),
    ("struct_xingming", "jie", rule_struct_xingming),
    ("corpus_given2", "corpus", rule_corpus_given2),
    ("person_possessive", "jie", rule_person_possessive),
    ("pos_known_fullname_appos", "jie", rule_pos_known_fullname_appos),
    ("person_naming_definition", "jie", rule_person_naming_definition),
    ("person_appellation", "jie", rule_person_appellation),
    ("presentative_person", "jie", rule_presentative_person),
    ("coordinated_person_object", "jie", rule_coordinated_person_object),
    # Existing original-text rules keep priority. Accepted mapped full names then
    # become ordinary anchors for the normal anaphora postpass.
    ("translation_fullname", "paragraph", rule_translation_fullname),
    ("translation_given", "paragraph", rule_translation_given),
    ("known_fullname_pos", "jie", rule_known_fullname_pos),
]

# ── gloss genealogy (jie, whole-text PREPASS) ────────────────────────────────
# Frames mined from all golden gloss spans + a corpus reader pass:
#   (a) 之REL     : SEP Subject ，Relative 之REL          -> tag Subject + Relative
#   (b) 谥曰X     : posthumous/alias name after 谥曰       -> tag X            (226x)
#   (c) [长次季幼庶]曰X : enumerated child after 曰         -> tag X            (57x)
# 之 is ALWAYS present in (a); overloaded terminals 族/后 require a trailing 也
# (else 之族。="wipe out the clan", 后="empress/after").
GLOSS_TERM = set("\u5b50\u5b59\u5f1f\u5144\u7236\u53d4\u6bcd\u58fb\u5a7f")  # 子孙弟兄父叔母壻婿
GLOSS_PREF = set("\u4ece\u65cf\u66fe\u7384\u5eb6\u5add")                    # 从族曾玄庶嫡 (need TERM/也 after)
GLOSS_NUM = set("\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341")  # 一..十 (之N世)
GLOSS_SEP = set("\uff0c\u3002\uff1b\u3001\uff01\uff1f\uff1a\u300c\u300d\u300e\u300f"
                "\uff08\uff09\u3014\u3015\u3000 \n\u0001\u25cf")
GLOSS_STOP = set("\u4e4b\u4e5f\u8005\u6240\u5176\u800c\u5219\u662f\u4e3a\u4ee5\u4e8e"
                 "\u5b50\u5b59\u5f1f\u5144\u7236\u53d4\u6bcd\u738b\u516c\u4faf\u5e1d"
                 "\u4e0a\u4e0b\u540e\u521d\u8c13\u89c1\u965b\u81e3\u541b\u592b\u59bb\u59be")
#  STOP = non-name honorifics/function words: 上下后初谓见陛臣君夫妻妾 + relations + 王公侯帝
GLOSS_ENUM = set("\u957f\u6b21\u5b63\u5e7c\u5eb6")   # 长次季幼庶 (before 曰)


def _is_gloss_marker(t, z):
    """True if t[z]=='之' starts a relational marker."""
    a = t[z + 1] if z + 1 < len(t) else ""
    if not a:
        return False
    if a in GLOSS_TERM:
        return True
    if a == "\u540e":                                  # 后 -> require 也
        return t[z + 2:z + 3] == "\u4e5f"
    if a in GLOSS_PREF:                                 # 从族曾玄庶嫡 -> TERM or 也 after
        nx = t[z + 2:z + 3]
        return nx in GLOSS_TERM or nx == "\u4e5f"
    if a in GLOSS_NUM:                                  # 之N世(孙)
        return t[z + 2:z + 3] == "\u4e16"
    return False


def _gloss_name_left(t, end, maxlen=2):
    """Maximal 1..maxlen char name ending just before `end`, bounded by a SEP.
    Stopword is only rejected at the name-START (leftmost) char, so names ending
    in 子/孙 (宣子, 桓子) are kept."""
    k = end
    while k > 0 and t[k - 1] not in GLOSS_SEP and (end - k) < maxlen:
        k -= 1
    if not (k == 0 or t[k - 1] in GLOSS_SEP):     # longer than maxlen -> not a bare name
        return None, None
    run = t[k:end]
    if not run or run[0] in GLOSS_STOP:
        return None, None
    return k, run


_RIGHT_BREAK = GLOSS_SEP | set("\u4e5f\u8005\u77e3\u7109\u4e4b")   # 也者矣焉之


def _gloss_name_right(t, start, maxlen=3):
    """Maximal 1..maxlen char name starting at `start`, bounded by SEP/particle."""
    n = len(t)
    if start >= n or t[start] in _RIGHT_BREAK or t[start] in GLOSS_STOP:
        return None, None
    k = start
    while k < n and t[k] not in _RIGHT_BREAK and (k - start) < maxlen:
        k += 1
    if k == start:
        return None, None
    return k, t[start:k]


def detect_gloss(t):
    """Return list of (start, end, surface, chunk_type) for genealogy names."""
    spans = []
    n = len(t)
    # (a) 之REL frame
    z = t.find("\u4e4b")
    while z != -1:
        if _is_gloss_marker(t, z):
            rs, rname = _gloss_name_left(t, z)
            if rname:
                spans.append((rs, z, rname, "gloss_rel"))
                if rs - 1 >= 0 and t[rs - 1] in "\uff0c\u3001":
                    ss, sname = _gloss_name_left(t, rs - 1)
                    if sname:
                        spans.append((ss, rs - 1, sname, "gloss_subj"))
        z = t.find("\u4e4b", z + 1)
    # (b) 谥曰X  (posthumous / alias name)
    p = t.find("\u8c25\u66f0")                          # 谥曰
    while p != -1:
        e, nm = _gloss_name_right(t, p + 2, maxlen=3)
        if nm:
            spans.append((p + 2, e, nm, "gloss_shi"))
        p = t.find("\u8c25\u66f0", p + 2)
    # (c) [长次季幼庶]曰X  (enumerated children)
    q = t.find("\u66f0")                                # 曰
    while q != -1:
        if q > 0 and t[q - 1] in GLOSS_ENUM:
            e, nm = _gloss_name_right(t, q + 1, maxlen=3)
            if nm:
                spans.append((q + 1, e, nm, "gloss_enum"))
        q = t.find("\u66f0", q + 1)
    return spans


PREPASS = [("gloss_geneal", "jie", detect_gloss)]


# ── jie-local anaphora (POSTPASS) ────────────────────────────────────────────
# After the anchor rules have tagged the full names / gloss aliases in THIS jie,
# tag later STANDALONE given-name occurrences of the same people. Pure jie-local
# coreference: a handle is only live if its full-name anchor appeared EARLIER in
# the SAME numbered section (one or more paragraphs; no juan/global fallback).
# POS·Giv gated for precision, so a bare
# 操 is only underlined inside a jie where 曹操 was already introduced.
def _handles_of(surface, ctype):
    """Given-name handles a full-name/gloss card contributes to the jie roster."""
    hs = set()
    bad = GLOSS_SEP | NAMESTART
    if ctype == "princess_title":
        return {"公主"}
    if surface in {"可汗", "单于"}:
        return set()
    if ctype.startswith("gloss"):
        if 1 <= len(surface) <= 2 and not (set(surface) & bad):
            hs.add(surface)                       # 瑶 / 智伯 / 襄子 as-is
    if ctype == "translation_fullname":
        surname_len = (
            3 if surface[:3] in COMPOUND3
            else 2 if surface[:2] in COMPOUND
            else 1 if surface[:1] in SUR_ALL
            else 0
        )
        given = surface[surname_len:] if surname_len else ""
        if 1 <= len(given) <= 2 and not (set(given) & bad):
            return {given}
    ruler_prefix_len = None
    if len(surface) >= 3 and surface[0] in POLITY1 and surface[1] == ZHU:
        ruler_prefix_len = 2
    elif (len(surface) >= 4 and surface[0] in POLITY_PREFIX
          and surface[1] in POLITY1 and surface[2] == ZHU):
        ruler_prefix_len = 3
    else:
        for polity in POLITY2:
            if surface.startswith(polity + ZHU):
                ruler_prefix_len = len(polity) + 1
                break
    if ruler_prefix_len is not None:
        given = surface[ruler_prefix_len:]
        if 1 <= len(given) <= 2 and not (set(given) & bad):
            hs.add(given)
    # 爵+名 may have been recognized by the higher-priority corpus rule (赵王虎 is
    # corpus_lit3), so identify the title-glued shape independently of chunk_type.
    if (len(surface) >= 3 and surface[0] in POLITY1 and surface[1] in JUE_HEAD):
        hs.add(surface[2:])
    elif (len(surface) >= 4 and surface[0] in POLITY_PREFIX
          and surface[1] in POLITY1 and surface[2] in JUE_HEAD):
        hs.add(surface[3:])
    title_end = max(surface.rfind(title) for title in JUE_HEAD)
    if 0 < title_end < len(surface) - 1:
        given = surface[title_end + 1:]
        if 1 <= len(given) <= 2 and not (set(given) & bad):
            hs.add(given)
            if len(given) == 2 and given[0] in SUR_ALL:
                hs.add(given[1:])
    if ctype == "title_name":
        for title in OFFICE_TITLES:
            if surface.startswith(title) and len(surface) > len(title):
                hs.add(surface[len(title):])
    if ctype == "role_name":
        hs.add(surface)
        return hs
    if ctype == "pos_fullname" and 2 <= len(surface) <= 3:
        hs.add(surface[1:])
        return hs
    if ctype == "appos_fullname" and 2 <= len(surface) <= 3:
        handle = surface[1:]
        if handle:
            hs.add(handle)
        return hs
    if ctype == "known_title":
        return hs
    if ctype == "title_appellation":
        return hs
    if ctype == "foreign_title_name":
        title_start = next(
            (
                surface.find(title)
                for title in ("可汗", "单于")
                if title in surface
            ),
            -1,
        )
        if title_start >= 0:
            trailing = surface[title_start + 2:]
            hs.add(trailing or surface[:title_start])
        return hs
    if ctype == "foreign_suffix_name":
        surname_len = 3 if surface[:3] in COMPOUND3 else 2
        hs.add(surface[surname_len:])
        return hs
    if ctype == "royal_title_name":
        title = next((candidate for candidate in ROYAL_PERSON_TITLES
                      if surface.startswith(candidate)), None)
        if title:
            hs.add(surface[len(title):])
        return hs
    g = None
    if surface[:3] in COMPOUND3:
        g = surface[3:]
    elif surface[:2] in COMPOUND:
        g = surface[2:]
    elif surface[:1] in SUR_ALL:
        g = surface[1:]
    if g and 1 <= len(g) <= 2 and not (set(g) & bad):
        hs.add(g)                                  # surname-stripped given: 操 / 亮 / 懿
    return hs


def _pos_handles_of_card(ctx, card):
    """Derive a surname-stripped handle from this card's complete POS morphology."""
    if card["chunk_type"] not in {
        "given2_office", "known_fullname_pos", "translation_fullname",
    } and not (
        card["chunk_type"] == "lit3" and len(card["surface"]) == 3
    ):
        return set()
    tokens = ctx.tokens_for(card["start"], card["end"])
    if len(tokens) < 2:
        return set()
    surname = tokens[0]
    if (
        surname.tag != "PROPN|NameType=Sur"
        or surname.score is None
        or surname.score < POS_FUNCTION_VETO_SCORE
    ):
        return set()
    given_start = surname.end
    if ctx.gspans.get(given_start) != card["end"]:
        return set()
    handle = ctx.t[given_start:card["end"]]
    if not (1 <= len(handle) <= 2) or set(handle) & (GLOSS_SEP | NAMESTART):
        return set()
    return {handle}


STATE_SUR = set("\u66f9\u97e9\u9b4f\u8d75\u695a\u71d5\u79e6\u5434\u8d8a\u6881\u9648"
                "\u5b8b\u536b\u90d1\u8521\u8bb8\u6ede\u859b\u5468\u5510")  # state names also used as surnames
RSUR1 = SUR_ALL | STATE_SUR                # single-char surnames for backtrack recovery


def detect_anaphora(ctx, cards):
    """POSTPASS: jie-local standalone-given tagging. Two parts:

    1. SEMANTIC BACKTRACK (anchor repair): for a 1- or 2-char POS·Giv candidate
       with no tagged anchor, look BACK in the same jie for an earlier
       `surname+candidate` full name the anchor rules missed (e.g. 曹操 — 曹 is
       ambiguous with a state/common word). If found, RECOVER the full name and
       register the candidate as its handle. Recurrence count is NOT an admission
       gate: one syntactically supported anaphora is enough.
    2. FORWARD RESOLUTION: tag later standalone occurrences of a given only when
       its (tagged or recovered) full-name anchor started EARLIER in the same jie.

    Scope is the whole numbered 节 (block): `ctx.t` is the block's paragraphs joined
    by a hard separator, so a bare 起 in one paragraph resolves to 吴起 introduced in
    a sibling paragraph of the same 节.
    """
    t, gset, consumed = ctx.t, ctx.gset, ctx.consumed
    n = len(t)
    roster = {}                                    # handle -> earliest anchor start
    roster_sources = collections.defaultdict(list)
    translation_roster = set()
    fallback_sources = collections.defaultdict(list)
    card_starts = {c["start"] for c in cards}
    card_spans = {(c["start"], c["end"]) for c in cards}
    for c in cards:
        handles = _handles_of(c["surface"], c["chunk_type"])
        handles.update(_pos_handles_of_card(ctx, c))
        if c["chunk_type"] == "translation_fullname":
            translation_roster.update(handles)
        for h in handles:
            s = c["start"]
            roster_sources[h].append((s, c["surface"]))
            if h not in roster or s < roster[h]:
                roster[h] = s
        if c["chunk_type"] not in {
            "role", "empress_title", "title_appellation", "princess_title",
            "female_court_title", "foreign_title_name",
        }:
            surface = c["surface"]
            if surface.endswith(("可汗", "单于")):
                continue
            for length in (2, 1):
                if len(surface) <= length:
                    continue
                handle = surface[-length:]
                if set(handle) & (NAMESTART | GLOSS_SEP):
                    continue
                fallback_sources[handle].append((c["start"], surface))
    strict_fallback_handles = set()
    two_char_fallback_handles = set()
    for handle, sources in fallback_sources.items():
        distinct_sources = {surface for _, surface in sources}
        if len(distinct_sources) != 1 or handle in roster:
            continue
        earliest = min(start for start, _ in sources)
        source = next(iter(distinct_sources))
        roster[handle] = earliest
        roster_sources[handle].extend(sources)
        if len(handle) == 1:
            strict_fallback_handles.add(handle)
        else:
            two_char_fallback_handles.add(handle)
    external_by_handle = collections.defaultdict(list)
    for anchor in ctx.translation_anchors:
        handle = anchor["handle"]
        if (
            len(handle) >= 2
            and handle[-1] in TRANSLATION_OVEREXTENSION_TAIL | {"子"}
            and handle[:-1] in roster
        ):
            continue
        external_by_handle[handle].append(anchor)
    out = []
    # ── (1) semantic backtrack: recover missed full-name anchors ─────────────
    candidates = collections.defaultdict(list)
    for i in range(n):
        if i in gset and not consumed[i]:
            candidates[t[i]].append(i)                       # 1-char given
            if i + 1 < n and (i + 1) in gset and not consumed[i + 1]:
                candidates[t[i:i + 2]].append(i)             # 2-char disyllabic given
    for tok in sorted(candidates, key=len, reverse=True):    # 2-char before 1-char
        if tok in roster:
            continue
        if tok in {"可汗", "单于"}:
            continue
        if set(tok) & (NAMESTART | GLOSS_SEP):
            continue
        positions = candidates[tok]
        rec = None
        q = t.find(tok)                            # scan ALL occurrences of tok,
        while q != -1:                             # incl. ones POS missed
            for slen in (3, 2, 1):
                j = q - slen
                if j < 0:
                    continue
                sur = t[j:q]
                is_sur = (slen == 1 and sur in RSUR1) or \
                         (slen == 2 and sur in COMPOUND) or \
                         (slen == 3 and sur in COMPOUND3)
                if not is_sur or any(consumed[j:q + len(tok)]):
                    continue
                prev = t[j - 1] if j > 0 else "\u0001"
                if not (prev in NAMESTART or prev in APPOS_TAIL):
                    continue
                e = q + len(tok)
                if t[e:e + 1] in JUE_HEAD:         # 归[赧王] is title text, not 归赧
                    continue
                # Semantic relation, not a frequency gate: this full-name candidate
                # must precede at least one independent POS·Giv occurrence of `tok`.
                # An occurrence still glued to another surname is another full name,
                # not an anaphora.
                has_later_anaphora = False
                for p in positions:
                    if p <= q:
                        continue
                    surname_left = False
                    for plen in (3, 2, 1):
                        ps = p - plen
                        if ps < 0:
                            continue
                        left = t[ps:p]
                        if (plen == 1 and left in RSUR1) or \
                                (plen == 2 and left in COMPOUND) or \
                                (plen == 3 and left in COMPOUND3):
                            surname_left = True
                            break
                    if not surname_left:
                        has_later_anaphora = True
                        break
                if not has_later_anaphora:
                    continue
                rec = (j, q + len(tok))
                if t[rec[0]:rec[1]] in {"可汗", "单于"}:
                    rec = None
                    continue
                break
            if rec:
                break
            q = t.find(tok, q + 1)
        if rec:
            s, e = rec
            for k in range(s, e):
                consumed[k] = True
            out.append((s, e, t[s:e], "fullname_bt"))
            roster[tok] = s
    # ── (2) forward resolution ───────────────────────────────────────────────
    if not roster and not external_by_handle:
        return out
    handles = sorted(
        (
            h
            for h in set(roster) | set(external_by_handle)
            if not (set(h) & (NAMESTART | GLOSS_SEP))
        ),
        key=len,
        reverse=True,
    )                                                # greedy longest first
    i = 0
    while i < n:
        if consumed[i]:
            i += 1
            continue
        hit = None
        for h in handles:
            L = len(h)
            active_external = [
                anchor
                for anchor in external_by_handle.get(h, ())
                if anchor["start"] <= i < anchor["end"]
            ]
            if len({
                anchor["identity_surface"] for anchor in active_external
            }) > 1:
                continue
            normal_anchor = h in roster and roster[h] < i
            if (
                (normal_anchor or active_external)
                and t[i:i + L] == h
                and not any(consumed[i:i + L])
            ):
                previous_token = ctx.token_at(i - 1) if i > 0 else None
                translation_assisted_handle = bool(active_external) or h in translation_roster
                derived_fallback_handle = (
                    h in strict_fallback_handles | two_char_fallback_handles
                )
                if (translation_assisted_handle or derived_fallback_handle) and (
                    any(
                        i >= prefix_len
                        and (
                            ctx.gspans.get(i - prefix_len) == i + L
                            or (
                                previous_token is not None
                                and previous_token.start == i - prefix_len
                                and previous_token.end >= i + L
                                and previous_token.pos == "PROPN"
                            )
                        )
                        for prefix_len in (3, 2, 1)
                    )
                    or (
                        previous_token is not None
                        and previous_token.end == i
                        and previous_token.tag == "PROPN|NameType=Sur"
                        and (
                            previous_token.text in CLEAN
                            or previous_token.text in COMPOUND
                            or previous_token.text in COMPOUND3
                        )
                    )
                    or (
                        derived_fallback_handle
                        and previous_token is not None
                        and previous_token.start < i
                        and previous_token.end >= i + L
                        and previous_token.pos == "PROPN"
                    )
                ):
                    continue
                title_continuation = t[i + L:i + L + 1] in JUE_HEAD
                next_token = ctx.token_at(i + L)
                current_token = ctx.token_at(i)
                name_continuation = (
                    translation_assisted_handle
                    and (
                        (
                            next_token is not None
                            and next_token.start == i + L
                            and next_token.pos == "PROPN"
                            and "NameType=" in next_token.tag
                        )
                        or (
                            current_token is not None
                            and current_token.start == i
                            and current_token.end > i + L
                            and current_token.pos == "PROPN"
                            and "NameType=" in current_token.tag
                        )
                        or (
                            current_token is not None
                            and current_token.start == i
                            and current_token.end == i + L
                            and current_token.pos in FUNCTION_POS
                            and any(
                                t[i:candidate_end] in ctx.corpus.ner
                                for candidate_end in range(
                                    i + L + 1, min(len(t), i + L + 3) + 1
                                )
                            )
                        )
                    )
                )
                if name_continuation:
                    continue
                active_sources = list(roster_sources[h])
                active_sources.extend(
                    (
                        anchor["anchor_start"],
                        anchor["identity_surface"],
                    )
                    for anchor in active_external
                )
                embedded_fullname = False
                for anchor_start, source in active_sources:
                    if anchor_start >= i:
                        continue
                    if not source.endswith(h):
                        continue
                    for prefix_len in range(1, len(source) - L + 1):
                        if i < prefix_len:
                            continue
                        extension_start = i - prefix_len
                        extension = t[extension_start:i + L]
                        if (
                            source.endswith(extension)
                            and (
                                extension in ctx.corpus.ner
                                or ctx.gspans.get(extension_start) == i + L
                            )
                        ):
                            embedded_fullname = True
                            break
                    if embedded_fullname:
                        break
                if embedded_fullname:
                    continue
                containing_token = ctx.token_at(i)
                if any(
                    span_start < i and span_end >= i + L
                    for span_start, span_end in ctx.gspans.items()
                    if (
                        ctx.token_at(span_start) is not None
                        and "NameType=Sur" in ctx.token_at(span_start).tag
                    )
                ) or (
                    containing_token is not None
                    and containing_token.start < i
                    and containing_token.end >= i + L
                ):
                    continue
                if derived_fallback_handle and any(
                    t[extension_start:extension_end] in ctx.corpus.ner
                    for extension_start in range(max(0, i - 3), i)
                    for extension_end in range(
                        i + L,
                        min(len(t), extension_start + ctx.corpus.ner_maxL) + 1,
                    )
                    if extension_end > i + L
                ):
                    continue
                previous_name_token = ctx.token_at(i - 1)
                if (
                    containing_token is not None
                    and containing_token.start == i
                    and containing_token.bio == "I"
                    and previous_name_token is not None
                    and previous_name_token.end == i
                    and previous_name_token.bio == "B"
                    and previous_name_token.pos == containing_token.pos
                    and "NameType=Sur" in previous_name_token.tag
                ):
                    continue
                pos_ok = i in gset
                prev = t[i - 1] if i > 0 else "\u0001"
                nxt = t[i + L:i + L + 1]
                zhi_pred = nxt == "\u4e4b" and t[i + L + 1:i + L + 2] in PERSON_ZHI_PRED
                coord = prev in COORD_HEAD and nxt in COORD_TAIL
                appointment = prev == "\u4ee5" and nxt == "\u4e3a"
                possessive = L >= 2 and nxt in PERSON_POSSESSIVE_SUFFIXES
                candidate_tokens = ctx.tokens_for(i, i + L)
                token = candidate_tokens[0] if len(candidate_tokens) == 1 else None
                token_covers_handle = bool(candidate_tokens)
                subject_predicate = (
                    prev in NAMESTART
                    and nxt in PERSON_SUBJECT_PRED
                )
                follow_object = (
                    subject_predicate
                    and nxt == "从"
                    and t[i + L + 1:i + L + 2] in PERSON_FOLLOW_OBJECT
                )
                predicate_pos_ok = (
                    subject_predicate
                    and (
                        (
                            next_token is not None
                            and next_token.start == i + L
                            and next_token.pos == "VERB"
                        )
                        or follow_object
                    )
                )
                object_frame = (
                    prev in PERSON_LEFT_VERBS and nxt in PERSON_OBJECT_TAIL
                )
                object_predicate = ctx.token_at(i + L)
                object_predicate_frame = (
                    L == 1
                    and prev in FUNCTION_HANDLE_OBJECT_VERBS
                    and object_predicate is not None
                    and object_predicate.start == i + L
                    and object_predicate.pos in {"VERB", "AUX"}
                    and object_predicate.score is not None
                    and object_predicate.score >= POS_FUNCTION_VETO_SCORE
                )
                subject_cursor = i + L
                subject_modifier = ctx.token_at(subject_cursor)
                if (
                    t[subject_cursor:subject_cursor + 1] in "不将窃"
                    and subject_modifier is not None
                    and subject_modifier.start == subject_cursor
                    and subject_modifier.pos in {"ADV", "AUX"}
                    and subject_modifier.score is not None
                    and subject_modifier.score >= POS_FUNCTION_VETO_SCORE
                ):
                    subject_cursor = subject_modifier.end
                function_subject_predicate = ctx.token_at(subject_cursor)
                unique_handle_source = len({
                    source
                    for _, source in active_sources
                    if source.endswith(h)
                }) == 1
                function_subject_frame = (
                    L == 1
                    and token is not None
                    and token.pos == "ADV"
                    and h not in FUNCTION_HANDLE_SUBJECT_VETO
                    and unique_handle_source
                    and prev in NAMESTART
                    and function_subject_predicate is not None
                    and function_subject_predicate.start == subject_cursor
                    and function_subject_predicate.pos in {"VERB", "AUX"}
                    and function_subject_predicate.score is not None
                    and function_subject_predicate.score >= POS_FUNCTION_VETO_SCORE
                    and (
                        subject_cursor > i + L
                        or t[subject_cursor:function_subject_predicate.end]
                        in FUNCTION_HANDLE_SUBJECT_PREDICATES
                    )
                )
                translation_function_subject_frame = (
                    bool(active_external)
                    and L == 1
                    and token is not None
                    and token.pos == "ADV"
                    and h not in FUNCTION_HANDLE_SUBJECT_VETO
                    and unique_handle_source
                    and prev in NAMESTART
                    and next_token is not None
                    and next_token.start == i + L
                    and next_token.pos in {"VERB", "AUX"}
                    and next_token.score is not None
                    and next_token.score >= POS_FUNCTION_VETO_SCORE
                )
                function_word = (
                    token is not None
                    and token.pos in FUNCTION_POS
                    and token.score is not None
                    and token.score >= POS_FUNCTION_VETO_SCORE
                    and not object_frame
                    and not object_predicate_frame
                    and not function_subject_frame
                    and not translation_function_subject_frame
                    and (
                        (token.pos == "AUX" and not follow_object)
                        or not predicate_pos_ok
                    )
                )
                adjacent_person_after_function = (
                    L == 1
                    and i + 1 in card_starts
                    and token is not None
                    and token.pos in FUNCTION_POS
                    and token.score is not None
                    and token.score >= POS_FUNCTION_VETO_SCORE
                )
                syntax_ok = (
                    token_covers_handle
                    and not function_word
                    and h not in BLOCK1
                    and (
                        object_frame
                        or object_predicate_frame
                        or subject_predicate
                        or function_subject_frame
                        or translation_function_subject_frame
                        or zhi_pred
                        or coord
                        or appointment
                        or possessive
                    )
                )
                structural_syntax_ok = (
                    token_covers_handle
                    and not function_word
                    and (
                        object_frame
                        or object_predicate_frame
                        or subject_predicate
                        or function_subject_frame
                        or translation_function_subject_frame
                        or zhi_pred
                        or coord
                        or appointment
                        or possessive
                    )
                )
                strict_fallback_frame = (
                    h not in strict_fallback_handles
                    or (
                        unique_handle_source
                        and pos_ok
                        and (
                            structural_syntax_ok
                            or
                            nxt in PERSON_RIGHT_PRED
                            or prev in PERSON_LEFT_VERBS
                            or (
                                next_token is not None
                                and next_token.start == i + L
                                and next_token.pos in {"VERB", "AUX"}
                                and next_token.score is not None
                                and next_token.score >= POS_FUNCTION_VETO_SCORE
                            )
                        )
                    )
                )
                local_two_char_frame = (
                    h in two_char_fallback_handles
                    and unique_handle_source
                    and not function_word
                    and token_covers_handle
                    and (
                        prev in PERSON_LEFT_VERBS
                        or nxt in PERSON_RIGHT_PRED
                        or subject_predicate
                        or object_frame
                        or coord
                        or appointment
                        or possessive
                        or (
                            next_token is not None
                            and next_token.start == i + L
                            and next_token.pos in {"VERB", "AUX"}
                            and next_token.score is not None
                            and next_token.score >= POS_FUNCTION_VETO_SCORE
                        )
                    )
                )
                allowed_title_continuation = (
                    not title_continuation
                )
                if (
                    (
                        (
                            not derived_fallback_handle
                            and (pos_ok or syntax_ok)
                        )
                        or (
                            L == 1
                            and h in strict_fallback_handles
                            and strict_fallback_frame
                        )
                        or local_two_char_frame
                    )
                    and strict_fallback_frame
                    and allowed_title_continuation
                    and not adjacent_person_after_function
                ):
                    quoted_low_confidence = (
                        token_covers_handle
                        and token is not None
                        and token.is_giv
                        and token.score is not None
                        and token.score < 0.5
                        and prev in NAMESTART
                        and nxt in "「『"
                    )
                    if quoted_low_confidence:
                        continue
                    if L >= 2 and derived_fallback_handle:
                        bio_end = ctx.gspans.get(i)
                        if bio_end is not None and bio_end > i + L:
                            continue
                    # Reject a one-char handle only when the next character belongs to
                    # the same BIO entity. Separate adjacent entities may be a name
                    # followed by a predicate (遂；/弘遂), not a longer given name.
                    if L == 1 and i + 1 < n and not consumed[i + 1]:
                        bio_end = ctx.gspans.get(i)
                        same_entity = bio_end is not None and bio_end > i + 1
                        legacy_contiguous = bio_end is None and (i + 1) in gset
                        embedded_anchor = not normal_anchor and (i + 1) in gset and any(
                            anchor_start < i and (
                                source.endswith(t[i:i + 2])
                                or (i > 0 and source.endswith(t[i - 1:i + 1]))
                            )
                            for anchor_start, source in active_sources
                        )
                        local_title_form = (i + 1) in gset and any(
                            title + form in t[:i]
                            for title in JUE_HEAD
                            for form in (
                                t[i:i + 2],
                                t[i - 1:i + 1] if i > 0 else "",
                            )
                            if len(form) == 2
                        )
                        longer_surface = t[i:bio_end] if same_entity else ""
                        longer_claimed = (
                            bool(longer_surface)
                            and (
                                (i, bio_end) in card_spans
                                or longer_surface in roster
                                or _local_nat_or_geo(ctx, i, bio_end)
                            )
                        )
                        anchored_person_frame = (
                            structural_syntax_ok
                            or (
                                unique_handle_source
                                and (
                                    nxt in PERSON_RIGHT_PRED
                                    or (
                                        next_token is not None
                                        and next_token.start == i + L
                                        and next_token.pos in {"VERB", "AUX"}
                                        and next_token.score is not None
                                        and next_token.score >= POS_FUNCTION_VETO_SCORE
                                    )
                                )
                            )
                        )
                        bio_context_frame = (
                            unique_handle_source
                            and (
                                i + 1 in card_starts
                                or (
                                    prev in NAMESTART
                                    and
                                    same_entity
                                    and _next_token_is_high_conf_verb(ctx, bio_end)
                                )
                            )
                        )
                        if (
                            (
                                same_entity
                                and (
                                    (
                                        longer_claimed
                                        and not bio_context_frame
                                    )
                                    or not (
                                        anchored_person_frame or bio_context_frame
                                    )
                                )
                            )
                            or legacy_contiguous
                            or embedded_anchor
                            or (local_title_form and not bio_context_frame)
                        ):
                            continue
                    provenance = None
                    ordinary_sources = [
                        (anchor_start, source)
                        for anchor_start, source in active_sources
                        if anchor_start < i and source.endswith(h)
                    ]
                    if not active_external and len({
                        source for _, source in ordinary_sources
                    }) == 1:
                        provenance = min(ordinary_sources)
                    hit = (
                        h,
                        L,
                        bool(active_external) and not normal_anchor,
                        provenance,
                    )
                    break
        if hit:
            h, L, translation_only, provenance = hit
            for k in range(i, i + L):
                consumed[k] = True
            result = (
                i,
                i + L,
                h,
                "translation_anaphora" if translation_only else "anaphora",
            )
            if provenance is not None:
                anchor_start, anchor_surface = provenance
                result += ({
                    "anchor_start": anchor_start,
                    "anchor_surface": anchor_surface,
                },)
            out.append(result)
            i += L
            continue
        i += 1
    return out


def detect_semantic_given2(ctx, cards):
    """Last-resort embedded KB aliases after anchored anaphora has had priority."""
    out = []
    for i in range(len(ctx.t) - 1):
        hit = rule_semantic_given2(ctx, i)
        if not hit:
            continue
        s, e, surf, ctype = hit
        for k in range(s, e):
            ctx.consumed[k] = True
        out.append((s, e, surf, ctype))
    return out


_EXACT_PROPAGATION_EXCLUDED_ANCHORS = {
    "role", "empress_title", "title_appellation", "model_ner_partial_pos",
}


def _model_title_suffix(surface):
    return next(
        (
            suffix
            for suffix in MODEL_PERSON_TITLE_SUFFIXES + EXTRA_MODEL_PERSON_TITLE_SUFFIXES
            if surface.endswith(suffix) and len(surface) > len(suffix)
        ),
        None,
    )


def _embedded_in_name_tokens(ctx, start, end):
    left = ctx.token_at(start - 1)
    right = ctx.token_at(end)
    return (
        left is not None
        and left.end == start
        and left.pos == "PROPN"
        and "NameType=" in left.tag
    ) or (
        right is not None
        and right.start == end
        and right.pos == "PROPN"
        and "NameType=" in right.tag
    )


def _strong_exact_anchor(ctx, card, *, titles):
    if titles:
        return card["chunk_type"] in {
            "alias", "female_court_title", "known_title", "local_title_anchor",
            "model_ner_fief_title", "model_ner_name", "model_ner_rank_title",
            "model_ner_temple_title", "model_ner_title", "surname_honorific",
        }
    return (
        _complete_person_pos(ctx, card["start"], card["end"])
        or card["chunk_type"] in {
            "appos_fullname", "foreign_suffix_name", "foreign_title_name",
            "genealogy_given", "gloss_enum", "gloss_rel", "gloss_subj",
            "multifief_jue_name", "pos_fullname",
            "royal_title_name", "translation_fullname", "xing2_appos",
        }
    )


def _detect_exact_local_surface(ctx, cards, *, titles):
    """Propagate an admitted exact model-NER surface within this numbered section."""
    anchors = {
        card["surface"]
        for card in cards
        if card["chunk_type"] not in _EXACT_PROPAGATION_EXCLUDED_ANCHORS
        and len(card["surface"]) >= 2
        and card["surface"] in ctx.corpus.ner
        and bool(_model_title_suffix(card["surface"])) == titles
        and _strong_exact_anchor(ctx, card, titles=titles)
    }
    trusted_title_anchors = {
        card["surface"]
        for card in cards
        if card["chunk_type"] in {"female_court_title", "surname_honorific"}
    } if titles else set()
    relaxed_anchors = set()
    if not titles:
        for card in cards:
            fullname_source = (
                card["chunk_type"] in {
                    "appos_fullname", "known_fullname_pos", "model_ner_name",
                    "pos_fullname", "translation_fullname", "xingming2", "xingming3",
                }
                and _is_xing_headed(card["surface"])
                and _model_title_suffix(card["surface"]) is None
            )
            for handle in (
                _handles_of(card["surface"], card["chunk_type"])
                if fullname_source
                else ()
            ):
                if (
                    len(handle) >= 2
                    and len(card["surface"]) > len(handle)
                    and handle in ctx.corpus.ner
                    and handle not in MODEL_PERSON_TITLE_SUFFIXES
                    and (
                        ctx.t.find(handle, 0, card["start"]) >= 0
                        or ctx.t.find(handle, card["end"]) >= 0
                    )
                ):
                    relaxed_anchors.add(handle)
            if (
                card["chunk_type"] == "translation_fullname"
                and len(card["surface"]) >= 2
                and card["surface"] in ctx.corpus.ner
                and not _occurrence_has_polity_frame(
                    ctx, card["start"], card["end"]
                )
            ):
                relaxed_anchors.add(card["surface"])
        anchors.update(relaxed_anchors)
    out = []
    for surface in sorted(anchors, key=lambda value: (-len(value), value)):
        suffix = _model_title_suffix(surface)
        if titles and (
            suffix == "子"
            or surface in TITLE_NONPERSON_COMPONENTS
            or surface in APPOINT_TITLES
            or surface in OFFICE_TITLES
        ):
            continue
        start = ctx.t.find(surface)
        while start >= 0:
            end = start + len(surface)
            tokens = ctx.tokens_for(start, end)
            previous = ctx.token_at(start - 1)
            relaxed = surface in relaxed_anchors
            if (
                not any(ctx.consumed[start:end])
                and tokens
                and (
                    titles
                    or relaxed
                    or any(
                        token.pos == "PROPN"
                        and "NameType=" in token.tag
                        and token.score is not None
                        and token.score >= 0.5
                        for token in tokens
                    )
                )
                and not (set(surface) & (NAMESTART | GLOSS_SEP))
                and not _shi_guard(ctx.t, start)
                and (relaxed or not _all_high_confidence_function_pos(ctx, start, end))
                and (titles or not _local_nat_or_geo(ctx, start, end))
                and (titles or not _embedded_in_name_tokens(ctx, start, end))
                and not _has_person_bio_left_continuation(ctx, start)
                and not _has_person_bio_right_continuation(ctx, start, end)
                and not _has_repeated_model_extension(ctx, start, end)
                and (
                    _complete_person_pos(ctx, start, end)
                    or _has_person_name_before_rank_title(ctx, start, end)
                    or (
                        not _has_office_continuation(ctx, end)
                        and not _has_person_designation_right_continuation(
                            ctx, start, end
                        )
                    )
                )
                and not _has_polity_title_left_continuation(ctx, start, end)
                and not _has_geo_title_right_continuation(ctx, end)
                and not _has_location_office_right_continuation(
                    ctx, start, end
                )
                and not _occurrence_has_polity_frame(ctx, start, end)
                and not _surface_has_jie_collective_frame(ctx, surface)
                and not (
                    previous is not None
                    and previous.end == start
                    and previous.pos == "VERB"
                    and previous.score is not None
                    and previous.score >= 0.8
                    and ctx.t[start - 1:start] not in PERSON_LEFT_VERBS
                    and surface not in trusted_title_anchors
                )
                and ctx.t[end:end + 2] not in {"上流", "下流"}
                and (
                    titles
                    or relaxed
                    or ctx.t[start - 1:start] in NAMESTART
                    or ctx.t[end:end + 1] in NAMESTART
                )
                and not any(
                    ctx.t[left:end + right] in ctx.corpus.ner
                    and not (
                        surface in trusted_title_anchors
                        and left == start
                        and right > 0
                        and _all_high_confidence_function_pos(
                            ctx, end, end + right
                        )
                    )
                    for left, right in ((start - 1, 0), (start - 2, 0), (start, 1), (start, 2))
                    if left >= 0 and end + right <= len(ctx.t)
                    and (left != start or right)
                )
                and (
                    not titles
                    or (
                        ctx.t[end:end + 1]
                        not in {"王", "公", "侯", "君", "卿", "子", "妃", "后"}
                        and _title_component_is_relaxed_proper(
                            ctx, start, end - len(suffix)
                        )
                    )
                )
            ):
                for offset in range(start, end):
                    ctx.consumed[offset] = True
                out.append((
                    start,
                    end,
                    surface,
                    "local_exact_title" if titles else "local_exact_surface",
                ))
            start = ctx.t.find(surface, start + 1)
    return out


def detect_local_exact_surface(ctx, cards):
    return _detect_exact_local_surface(ctx, cards, titles=False)


def detect_local_exact_title(ctx, cards):
    return _detect_exact_local_surface(ctx, cards, titles=True)


def detect_local_title_anchor(ctx, cards):
    """Recover a title only from an earlier admitted same-jie textual anchor."""
    out = []
    anchors = [
        card for card in cards
        if card["chunk_type"] not in {
            "role", "empress_title", "title_appellation", "local_title_anchor",
        }
    ]
    max_length = min(ctx.corpus.ner_maxL, 8)
    for i in range(len(ctx.t)):
        if ctx.consumed[i]:
            continue
        for length in range(max_length, 1, -1):
            end = i + length
            if end > len(ctx.t) or any(ctx.consumed[i:end]):
                continue
            surface = ctx.t[i:end]
            suffix = next(
                (
                    candidate for candidate in MODEL_PERSON_TITLE_SUFFIXES
                    if surface.endswith(candidate) and len(surface) > len(candidate)
                ),
                None,
            )
            if (
                suffix is None
                or surface not in ctx.corpus.ner
                or surface in TITLE_NONPERSON_COMPONENTS
                or surface in APPOINT_TITLES
                or set(surface) & (NAMESTART | GLOSS_SEP)
                or _shi_guard(ctx.t, i)
                or _all_high_confidence_function_pos(ctx, i, end)
            ):
                continue
            tokens = ctx.tokens_for(i, end)
            if (
                not tokens
                or tokens[0].start != i
                or tokens[-1].end != end
                or any(left.end != right.start for left, right in zip(tokens, tokens[1:]))
            ):
                continue
            earlier_sources = {
                card["surface"]
                for card in anchors
                if card["start"] < i
                and (
                    card["chunk_type"] == "model_ner_title"
                    or _complete_person_pos(ctx, card["start"], card["end"])
                    or _complete_model_name_tokens(ctx, card["start"], card["end"])
                )
                and (
                    card["surface"] == surface
                    or (
                        len(card["surface"]) > len(surface)
                        and card["surface"].endswith(surface)
                    )
                )
            }
            earlier_text_repeat = ctx.t.find(surface, 0, i) >= 0
            baseline_frame = (
                _title_left_verb(ctx, i)
                or _title_predicate_after(ctx, end)
                or (
                    ctx.t[i - 1:i] == "为"
                    and ctx.t[end:end + 1] == "所"
                )
            )
            strict_repeat_frame = (
                _strict_person_frame(ctx, i, end)
                or _title_predicate_after(ctx, end)
                or (
                    ctx.t[i - 1:i] == "为"
                    and ctx.t[end:end + 1] == "所"
                )
            )
            exact_source = surface in earlier_sources
            suffix_source = bool(earlier_sources) and not exact_source
            component_end = end - len(suffix)
            component = surface[:-len(suffix)]
            left_token = ctx.token_at(i - 1)
            right_token = ctx.token_at(end)
            embedded_in_larger_name = any(
                ctx.t[max(0, i - left):min(len(ctx.t), end + right)] in ctx.corpus.ner
                for left, right in ((1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3))
                if i - left >= 0 and end + right <= len(ctx.t)
            ) or (
                left_token is not None
                and left_token.end == i
                and left_token.pos in {"PROPN", "NOUN", "ADJ", "NUM", "DET"}
            ) or (
                right_token is not None
                and right_token.start == end
                and right_token.pos == "PROPN"
                and (
                    "NameType=Giv" in right_token.tag
                    or "NameType=Prs" in right_token.tag
                )
            )
            repeat_source = exact_source or (
                earlier_text_repeat and suffix != "子"
            )
            baseline_admit = (
                bool(earlier_sources)
                and baseline_frame
                and not _local_nat_or_geo(ctx, i, end)
            )
            repeated_title_admit = (
                strict_repeat_frame
                and not embedded_in_larger_name
                and suffix != "子"
                and not (suffix == "侯" and len(component) > 2)
                and not (
                    len(component) == 1
                    and component in {"王", "公", "侯", "君", "卿", "子"}
                )
                and ctx.t[end:end + 1]
                not in {"王", "公", "侯", "君", "卿", "子", "妃", "后"}
                and _title_component_is_relaxed_proper(ctx, i, component_end)
                and (repeat_source or suffix_source)
            )
            if not baseline_admit and not repeated_title_admit:
                continue
            for offset in range(i, end):
                ctx.consumed[offset] = True
            out.append((i, end, surface, "local_title_anchor"))
            break
    return out


def detect_pos_given_local_frame(ctx, cards):
    """Last-resort strict POS-given recovery after anchored anaphora."""
    out = []
    for i in range(len(ctx.t)):
        if ctx.consumed[i]:
            continue
        hit = rule_pos_given_local_frame(ctx, i)
        if hit is None:
            continue
        start, end, surface, chunk_type = hit
        for offset in range(start, end):
            ctx.consumed[offset] = True
        out.append((start, end, surface, chunk_type))
    return out


def detect_local_exact_given(ctx, cards):
    """Propagate a proven ordinary-anaphora lineage within the same section."""
    source_surfaces = collections.defaultdict(set)
    for card in cards:
        if (
            card["chunk_type"] != "anaphora"
            or len(card["surface"]) != 1
            or not card.get("anchor_surface")
            or card.get("anchor_start", card["start"]) >= card["start"]
        ):
            continue
        source_surfaces[card["surface"]].add(card["anchor_surface"])
    out = []
    for handle, sources in source_surfaces.items():
        if len(sources) != 1 or handle in BLOCK1:
            continue
        start = ctx.t.find(handle)
        while start >= 0:
            end = start + 1
            token = ctx.token_at(start)
            if (
                not ctx.consumed[start]
                and token is not None
                and token.start == start
                and token.end == end
                and token.is_giv
                and "\u3400" <= handle <= "\u9fff"
                and "NameType=Nat" not in token.tag
                and "NameType=Geo" not in token.tag
                and ctx.t[start - 1:start] in NAMESTART | PERSON_LEFT_VERBS
                and (
                    _strict_person_frame(ctx, start, end)
                    or _title_predicate_after(ctx, end)
                )
            ):
                ctx.consumed[start] = True
                out.append((start, end, handle, "local_exact_given"))
            start = ctx.t.find(handle, start + 1)
    return out


POSTPASS = [
    ("combined_evidence", "jie", detect_combined_evidence),
    ("local_exact_surface", "jie", detect_local_exact_surface),
    ("local_exact_title", "jie", detect_local_exact_title),
    ("jie_anaphora", "jie", detect_anaphora),
    ("semantic_given2", "jie", detect_semantic_given2),
    ("pos_given_local_frame", "jie", detect_pos_given_local_frame),
    ("local_exact_given", "jie", detect_local_exact_given),
    ("local_title_anchor", "jie", detect_local_title_anchor),
]
# Evaluated presets (see files/rules_report.md).
#   CORE  = high-precision set: 88.8% precision-proxy, 56.9% golden-lit recall,
#           only 550 non-NER FP. The safe Stage-1 default.
#   RECALL = CORE + given2 + xingming: 70.9% recall but precision-proxy 78.6%;
#           relies on Stage-2 confident-owner-or-drop to clean the tail.
PRESET_CORE = {"corpus_lit3", "corpus_xing2", "role", "jue_name", "struct_fuxing"}
PRESET_RECALL = (
    {r[0] for r in RULES}
    | {p[0] for p in PREPASS}
    | {p[0] for p in POSTPASS}
)
ALL_RULES = {r[0] for r in RULES} | {p[0] for p in PREPASS} | {p[0] for p in POSTPASS}


class Ctx:
    __slots__ = (
        "t", "gset", "gspans", "tokens", "_token_by_offset", "corpus", "consumed",
        "juan", "sec", "para_id", "ce", "_year_ranges", "translation_anchors",
        "translation_fullnames", "translation_mentions",
        "jie_person_surfaces", "jie_partial_person_surfaces",
        "jie_person_morphology_majority_surfaces",
        "jie_person_title_surfaces",
        "evidence_audit",
    )

    def __init__(
        self, t, gset, corpus, juan, sec, para_id, ce, gspans=(), tokens=(),
        year_ranges=(), jie_person_surfaces=(),
        jie_partial_person_surfaces=(),
        jie_person_morphology_majority_surfaces=(),
        jie_person_title_surfaces=(),
        evidence_audit=None,
    ):
        self.t, self.gset, self.corpus = t, gset, corpus
        self.gspans = {}
        for start, end in gspans:
            if 0 <= start < end <= len(t):
                self.gspans[start] = max(end, self.gspans.get(start, end))
        self.tokens = tuple(tokens)
        self._token_by_offset = {}
        for token in self.tokens:
            if 0 <= token.start < token.end <= len(t):
                for offset in range(token.start, token.end):
                    self._token_by_offset[offset] = token
        self.consumed = [False] * len(t)
        self.juan, self.sec, self.para_id, self.ce = juan, sec, para_id, ce
        self._year_ranges = tuple(year_ranges)
        self.translation_anchors = ()
        self.translation_fullnames = {}
        self.translation_mentions = {}
        self.jie_person_surfaces = frozenset(jie_person_surfaces)
        self.jie_partial_person_surfaces = frozenset(
            jie_partial_person_surfaces
        )
        self.jie_person_morphology_majority_surfaces = frozenset(
            jie_person_morphology_majority_surfaces
        )
        self.jie_person_title_surfaces = frozenset(
            jie_person_title_surfaces
        )
        self.evidence_audit = evidence_audit

    def token_at(self, offset):
        return self._token_by_offset.get(offset)

    def tokens_for(self, start, end):
        tokens = []
        cursor = start
        while cursor < end:
            token = self.token_at(cursor)
            if token is None or token.start != cursor or token.end > end:
                return ()
            tokens.append(token)
            cursor = token.end
        return tuple(tokens) if cursor == end else ()

    def year_at(self, offset):
        if self.ce is not None:
            return self.ce
        return next(
            (
                year
                for start, end, year in self._year_ranges
                if start <= offset < end
            ),
            None,
        )


def _sec_num(mt):
    if not mt:
        return None
    o = ord(mt[0])
    if 0x2460 <= o <= 0x2473:
        return o - 0x2460 + 1
    if 0x3251 <= o <= 0x325F:
        return o - 0x3251 + 21
    if 0x32B1 <= o <= 0x32BF:
        return o - 0x32B1 + 36
    return None


def detect_para(ctx, enabled):
    out = []
    n = len(ctx.t)
    # whole-text prepass rules (look-ahead patterns like genealogy) run first and
    # reserve their spans in `consumed` before the positional lexicon scan.
    for pname, pscope, pfn in PREPASS:
        if pname not in enabled:
            continue
        for (s, e, surf, ctype) in pfn(ctx.t):
            if s < 0 or e > n or any(ctx.consumed[s:e]):
                continue
            for k in range(s, e):
                ctx.consumed[k] = True
            out.append({"juan": ctx.juan, "sec": ctx.sec, "para_id": ctx.para_id,
                        "start": s, "end": e, "surface": surf, "chunk_type": ctype,
                        "rule": pname, "scope": pscope, "ce_year": ctx.ce})
    i = 0
    while i < n:
        if ctx.consumed[i]:
            i += 1
            continue
        hit = None
        for name, scope, fn in RULES:
            if name not in enabled:
                continue
            r = fn(ctx, i)
            if r:
                hit = (r, name, scope)
                break
        if hit:
            (s, e, surf, ctype), rname, scope = hit
            for k in range(s, e):
                ctx.consumed[k] = True
            out.append({"juan": ctx.juan, "sec": ctx.sec, "para_id": ctx.para_id,
                        "start": s, "end": e, "surface": surf, "chunk_type": ctype,
                        "rule": rname, "scope": scope, "ce_year": ctx.ce})
            i = e
            continue
        i += 1
    # postpass rules (jie anaphora) use the anchor cards emitted above, over the
    # whole block text.
    for pname, pscope, pfn in POSTPASS:
        if pname not in enabled:
            continue
        for result in pfn(ctx, out):
            s, e, surf, ctype = result[:4]
            card = {"juan": ctx.juan, "sec": ctx.sec, "para_id": ctx.para_id,
                    "start": s, "end": e, "surface": surf, "chunk_type": ctype,
                    "rule": pname, "scope": pscope, "ce_year": ctx.ce}
            if len(result) == 5:
                card.update(result[4])
            out.append(card)
    return out


SEC_SEP = "\n"   # hard boundary between paragraphs of a block (in NAMESTART & GLOSS_SEP)


def _blocks_of(paras):
    """Group paragraphs into numbered 节 (jie) blocks. A paragraph whose first char
    is a circled number opens a new block; following UNNUMBERED paragraphs belong to
    it (carry-over) until the next number. Returns list of (sec, [paras]).

    When ANAPHORA_BLOCK is off, every paragraph is its own block (paragraph scope)."""
    psec, sec = [], None
    for para in paras:
        sn = _sec_num(para.get("main", "") or "")
        if sn is not None:
            sec = sn
        psec.append(sec)
    if not ANAPHORA_BLOCK:
        return [(s, [p]) for p, s in zip(paras, psec)]
    blocks = []
    for para, s in zip(paras, psec):
        sn = _sec_num(para.get("main", "") or "")
        if sn is not None or not blocks:
            blocks.append((s, [para]))             # new numbered block (or leading block)
        else:
            blocks[-1][1].append(para)             # carry-over into current block
    return blocks


def _jie_person_morphology_surfaces(paras, giv, corpus):
    """Exact surfaces with person morphology in this numbered section."""
    complete_surfaces = set()
    partial_surfaces = set()
    complete_counts = collections.Counter()
    geo_nat_counts = collections.Counter()
    for para in paras:
        text = para.get("main", "") or ""
        evidence = giv.get(para.get("id"), set())
        tokens = tuple(getattr(evidence, "tokens", ()))
        by_start = {token.start: token for token in tokens}
        for start in range(len(text)):
            cursor = start
            candidate_tokens = []
            while cursor < min(len(text), start + 8):
                token = by_start.get(cursor)
                if token is None:
                    break
                candidate_tokens.append(token)
                cursor = token.end
                surface = text[start:cursor]
                if (
                    len(surface) >= 2
                    and surface in corpus.ner
                ):
                    personal = [
                        item
                        for item in candidate_tokens
                        if item.pos == "PROPN"
                        and any(
                            name_type in item.tag
                            for name_type in (
                                "NameType=Sur",
                                "NameType=Giv",
                                "NameType=Prs",
                            )
                        )
                    ]
                    if personal:
                        partial_surfaces.add(surface)
                    if all(
                        item.pos == "PROPN"
                        and any(
                            name_type in item.tag
                            for name_type in (
                                "NameType=Sur",
                                "NameType=Giv",
                                "NameType=Prs",
                            )
                        )
                        and "NameType=Geo" not in item.tag
                        and "NameType=Nat" not in item.tag
                        for item in candidate_tokens
                    ):
                        complete_surfaces.add(surface)
                        complete_counts[surface] += 1
                    if any(
                        "NameType=Geo" in item.tag
                        or "NameType=Nat" in item.tag
                        for item in candidate_tokens
                    ):
                        geo_nat_counts[surface] += 1
    majority_surfaces = {
        surface
        for surface, count in complete_counts.items()
        if count > geo_nat_counts[surface]
    }
    return (
        frozenset(complete_surfaces),
        frozenset(partial_surfaces),
        frozenset(majority_surfaces),
    )


def _jie_person_title_surfaces(paras, giv, corpus):
    """Exact POS-given spans introduced with a title in this section."""
    surfaces = set()
    for para in paras:
        text = para.get("main", "") or ""
        evidence = giv.get(para.get("id"), set())
        spans = tuple(getattr(evidence, "spans", ()))
        span_starts = {start for start, _ in spans}
        for start, end in spans:
            surface = text[start:end]
            if len(surface) < 2 or surface not in corpus.ner:
                continue
            for title in ("可汗", "单于", "公主"):
                if not text.startswith(title, end):
                    continue
                if end + len(title) not in span_starts:
                    surfaces.add(surface)
                break
    return frozenset(surfaces)


def detect_juan(
    juan_no,
    paras,
    giv,
    corpus,
    enabled=None,
    scan_notes=False,
    translation_evidence=None,
    evidence_audit=None,
):
    if enabled is None:
        enabled = {r[0] for r in RULES}
    out = []
    for bsec, bparas in _blocks_of(paras):
        # Assemble the 节 as ONE text: every rule (anchors, gloss prepass, anaphora
        # postpass) runs over the whole block. SEC_SEP is a hard boundary so no name
        # spans a paragraph join, but the anaphora roster / semantic backtrack see
        # the full block. Spans are mapped back to per-paragraph offsets afterwards.
        parts, blk_gset, blk_gspans, blk_tokens = [], set(), [], []
        blk_year_ranges, blk_translation_anchors = [], []
        blk_translation_fullnames, blk_translation_mentions, pmap, off = [], [], [], 0
        blk_translation_jie_anchors = []
        for para in bparas:
            t = para.get("main", "") or ""
            pmap.append((off, off + len(t), para))
            blk_year_ranges.append((off, off + len(t), para.get("ce_year")))
            evidence = giv.get(para.get("id"), set())
            for gp in evidence:
                if 0 <= gp < len(t):
                    blk_gset.add(off + gp)
            for start, end in getattr(evidence, "spans", ()):
                if 0 <= start < end <= len(t):
                    blk_gspans.append((off + start, off + end))
            for token in getattr(evidence, "tokens", ()):
                if 0 <= token.start < token.end <= len(t):
                    blk_tokens.append(token.shifted(off))
            if translation_evidence:
                for identity in translation_evidence.get(para.get("id"), ()):
                    if not identity.get("eligible_anchor"):
                        continue
                    identity_surface = str(identity.get("identity_surface", ""))
                    strict_identity = all(
                        candidate.get("eligible")
                        for candidate in identity.get("candidates", ())
                    )
                    for candidate in identity.get("candidates", ()):
                        if not candidate.get("eligible"):
                            continue
                        start, end = int(candidate["start"]), int(candidate["end"])
                        if not (0 <= start < end <= len(t)):
                            continue
                        row = {
                            "start": off + start,
                            "end": off + end,
                            "surface": str(candidate["surface"]),
                            "identity_surface": identity_surface,
                            "strict_identity": strict_identity,
                            "mode": candidate.get("transfer_mode"),
                        }
                        mode = candidate.get("transfer_mode")
                        if (
                            mode == "exact"
                            and str(candidate.get("normalized_surface", ""))
                            == identity_surface
                        ):
                            blk_translation_fullnames.append(row)
                        elif mode in {"anchor_given", "title_given"}:
                            blk_translation_mentions.append(row)
                    handles = set(identity.get("handles", ()))
                    handles.update(
                        _handles_of(identity_surface, "translation_identity")
                    )
                    eligible_starts = [
                        off + int(candidate["start"])
                        for candidate in identity.get("candidates", ())
                        if candidate.get("eligible")
                    ]
                    for handle in handles:
                        if (
                            not 1 <= len(handle) <= 2
                            or set(handle) & (NAMESTART | GLOSS_SEP)
                        ):
                            continue
                        blk_translation_anchors.append(
                            {
                                "start": off,
                                "end": off + len(t),
                                "anchor_start": off - 1,
                                "identity_surface": identity_surface,
                                "handle": handle,
                            }
                        )
                        if eligible_starts:
                            blk_translation_jie_anchors.append(
                                {
                                    "start": min(eligible_starts),
                                    "end": None,
                                    "anchor_start": min(eligible_starts),
                                    "identity_surface": identity_surface,
                                    "handle": handle,
                                }
                            )
            parts.append(t)
            off += len(t) + len(SEC_SEP)
        blocktext = SEC_SEP.join(parts)
        (
            jie_person_surfaces,
            jie_partial_person_surfaces,
            jie_person_morphology_majority_surfaces,
        ) = _jie_person_morphology_surfaces(bparas, giv, corpus)
        jie_person_title_surfaces = _jie_person_title_surfaces(
            bparas, giv, corpus
        )
        for anchor in blk_translation_jie_anchors:
            anchor["end"] = len(blocktext)
        blk_translation_anchors.extend(blk_translation_jie_anchors)
        ctx = Ctx(
            blocktext, blk_gset, corpus, juan_no, bsec, None, None,
            gspans=blk_gspans,
            tokens=blk_tokens,
            year_ranges=blk_year_ranges,
            jie_person_surfaces=jie_person_surfaces,
            jie_partial_person_surfaces=jie_partial_person_surfaces,
            jie_person_morphology_majority_surfaces=(
                jie_person_morphology_majority_surfaces
            ),
            jie_person_title_surfaces=jie_person_title_surfaces,
            evidence_audit=[] if evidence_audit is not None else None,
        )
        ctx.translation_anchors = tuple(blk_translation_anchors)
        by_start = collections.defaultdict(list)
        for candidate in blk_translation_fullnames + blk_translation_mentions:
            by_start[candidate["start"]].append(candidate)
        for start, same_start in by_start.items():
            if len({
                candidate["identity_surface"] for candidate in same_start
            }) != 1:
                continue
            strict_local_owner = all(
                candidate["strict_identity"] for candidate in same_start
            )
            for modes, target in (
                ({"exact"}, ctx.translation_fullnames),
                ({"anchor_given", "title_given"}, ctx.translation_mentions),
            ):
                eligible = [
                    candidate
                    for candidate in same_start
                    if candidate["mode"] in modes
                ]
                if not eligible:
                    continue
                selected = max(
                    eligible,
                    key=lambda row: row["end"] - row["start"],
                )
                selected["strict_local_owner"] = strict_local_owner
                target[start] = selected
        for c in detect_para(ctx, enabled):
            s = c["start"]
            for a, b, para in pmap:                # re-stamp to the containing paragraph
                if a <= s < b:
                    c["start"] -= a
                    c["end"] -= a
                    c["para_id"] = para.get("id")
                    c["ce_year"] = para.get("ce_year")
                    break
            c["field"] = "main"
            out.append(c)
        if evidence_audit is not None:
            for row in ctx.evidence_audit:
                s = row["start"]
                for a, b, para in pmap:
                    if a <= s < b:
                        row["start"] -= a
                        row["end"] -= a
                        row["para_id"] = para.get("id")
                        row["ce_year"] = para.get("ce_year")
                        row["juan"] = juan_no
                        row["sec"] = bsec
                        break
                else:
                    continue
                evidence_audit.append(row)
        if scan_notes:
            # Hu Sanxing commentary: same rules, offsets into each paragraph's notes.
            # Notes stay paragraph-local (they carry no POS·Giv cache and no roster).
            for para in bparas:
                ntext = "".join(n.get("text", "") or "" for n in para.get("notes", []))
                if not ntext:
                    continue
                nctx = Ctx(ntext, set(), corpus, juan_no, bsec, para.get("id"),
                           para.get("ce_year"))
                for c in detect_para(nctx, enabled):
                    c["field"] = "notes"
                    out.append(c)
    if translation_evidence:
        baseline = detect_juan(
            juan_no,
            paras,
            giv,
            corpus,
            enabled=enabled,
            scan_notes=scan_notes,
            translation_evidence=None,
        )
        for original in baseline:
            overlaps = [
                assisted
                for assisted in out
                if (
                assisted["para_id"] == original["para_id"]
                and assisted.get("field") == original.get("field")
                and assisted["start"] < original["end"]
                and original["start"] < assisted["end"]
                )
            ]
            if not overlaps:
                out.append(original)
                continue
            if all(
                original["start"] <= assisted["start"]
                and assisted["end"] <= original["end"]
                for assisted in overlaps
            ) and any(
                original["start"] < assisted["start"]
                or assisted["end"] < original["end"]
                for assisted in overlaps
            ):
                overlap_ids = {id(assisted) for assisted in overlaps}
                out[:] = [row for row in out if id(row) not in overlap_ids]
                out.append(original)
        out.sort(key=lambda row: (
            row["para_id"], row.get("field", "main"), row["start"], row["end"],
            row["surface"], row["chunk_type"],
        ))
    return out
