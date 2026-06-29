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
    "卢水胡",  # 匈奴别部 (ethnic group), not a person — 卢(姓)+水胡
    "杜姥宅",  # 建康地名 (a place), not a person — 杜(姓)-headed glue
    "魏博留",  # 魏博留后 (藩镇官) truncation, not a person
    "于智",   # 明于智略 — "明于智(谋)", not the surname 于
    "公亮",   # 杞公亮 — 爵(公)-glue, the person is 宇文亮
    # RC-1 (wave-4 audit) — surname + common char that jieba mislabels nr; each spread
    # across many 卷 as a junk card. All are idioms / collective titles, never one person.
    "莫如", "莫知", "莫肯", "莫能",   # 莫 + 助动词: 莫如此 / 莫知其 / 莫能御
    "伏惟",   # 伏惟 — "I respectfully submit" (memorial formula)
    "谢恩",   # 谢恩 — to give thanks for imperial grace
    "蒙恩",   # 蒙恩 — to receive grace
    "杜绝",   # 杜绝 — to put a stop to
    "顾望",   # 顾望 — to look about / hold back
    "殷勤",   # 殷勤 — attentive / earnest
    "余生",   # 余生 — one's remaining years
    "苗裔",   # 苗裔 — descendants
    "胡骑",   # 胡骑 — Hu cavalry (collective), not a person
    "符瑞",   # 符瑞 — auspicious omen
    "布陈",   # 布陈 = 布阵 — to deploy in battle array
    "纪纲",   # 纪纲 — the bonds of governance / discipline
    "王公", "王侯", "王府",          # collective titles / an institution, not one person
    # RC-2c (wave-4 tail scan) — bad generated 省称 tails: offices/collectives/mis-strips.
    "龙骧",   # 龙骧将军 (an office), not a person — 慕容X 为 龙骧(将军)
    "诸吕",   # 诸吕 — the Lü clan collectively (封诸吕为王)
    "菩萨",   # 菩萨 — Buddhist term; 尉迟菩萨 amplifies it corpus-wide
    "王安",   # 韩王安 = 韩王·名安 (封号+单名), tail 王安 is a mis-strip of 韩
    "王政",   # 秦王政 = 秦王·名政 (嬴政), tail 王政 is a mis-strip of 秦
    "吕王嘉",  # 封号 glue: 吕王(吕氏封王)+名嘉 mis-segmentation (卷013); the real 吕嘉
              # (南越相) and 王嘉 (汉相) both already have their own cards.
    # RC-1 (R21 dup-card audit) — 姓/伪姓 + verb/quantifier/role-title phrase that jieba's
    # nr tagger mislabels as a name; each shipped as junk cards spread across many 卷.
    # NB: 徐有功 (唐 法官, 有功 = given name) is a REAL person — do NOT blacklist by tail.
    "于公有",   # 于公有… — 于公(廷尉)+有 (had…)
    "伏精骑",   # 伏精骑 — 伏(set ambush)+精骑 (elite cavalry)
    "周天子", "唐天子", "戴天子",   # 天子 role-title glued to a dynasty/surname char
    "布大喜",   # 布大喜 — 布(吕布)+大喜
    "王大夫",   # 王大夫 — 王+大夫 (官名), not a person
    "王大怒",   # 王大怒 — 王+大怒
    "王诸子",   # 王诸子 — 王+诸子 (the king's sons, collective)
    "莫能明",   # 莫能明 — "none could understand it"
    "费巨万",   # 费巨万 — 费(spend)+巨万 (a vast sum)
    "马大呼",   # 马大呼 — 马+大呼
    # RC-1 (R22 卷200 historian audit) — confirmed non-person surfaces (no 字/bio,
    # same false surface recurs across 卷). NB: 王弘(刘宋 字休元)/王郎(汉 王昌)/李崇(北魏)
    # are REAL — their 卷200 issues are 卷-local 封号/截断, handled elsewhere, NOT here.
    "温恭",   # 容貌温恭 — adjective (gentle & respectful)
    "班赐",   # 班赐有差 — to distribute/bestow (verb)
    "黎明",   # 黎明遂倾隋室 — daybreak (time word)
    "马尽",   # 马尽，人自相食 — "the horses ran out"
    "马韦",   # 太子洗马韦季方 — 洗马(官)+韦(姓) mis-glue
    "殷国",   # 微子去而殷国以亡 — the Yin state, not a person
    "鼠尼施",  # 别部鼠尼施 — a 突厥 sub-tribe, not an individual
    "熊津",   # 熊津江口/都督/城 — a 百济 place (Korean Ungjin)
    "陆浑",   # 上畋于陆浑 — a place (陆浑县/戎)
    # RC-1 (R23 multi-卷 historian audit: 卷20/110/220) — confirmed non-person surfaces
    # with NO real homograph anywhere in the 403BC–959AD corpus. NB: 王足(北魏将)/司马尚
    # (战国赵将)/王崇(汉)/王乌(汉使)/王德(晋太守)/王尚 ARE real — deferred as 卷-local 封号/截断.
    # — 官号/将军号/侯号 fragments —
    "五利",   # 五利将军 (栾大的封号), not a name
    "伏波",   # 伏波将军 (office), 路博德/马援 are the persons
    "楼船",   # 楼船将军 (office) / 水军部队
    "梁侯",   # 将梁侯 — 侯号 fragment
    "蔡侯",   # 临蔡侯 — 侯号 fragment
    "胡帅",   # 离石胡帅/西河胡帅 — "胡人首领" (a title), 呼延铁/张崇 are the persons
    # — 部族 / 地名 / 国名 —
    "同罗",   # 曳落河、同罗 — a 突厥 tribe / war-band
    "库莫奚",  # 袭库莫奚 — a 部族 (the Kumo Xi)
    "郁林",   # 郁林郡 — a commandery
    "梅岭",   # 屯豫章、梅岭 — a mountain/place
    "倪塘",   # 斩于倪塘 — an execution ground (place)
    "杨口",   # 至杨口 — a place
    "蔡洲",   # 回军蔡洲 — a place / garrison
    "钱唐",   # 钱唐杜子恭 — a place (钱塘), 杜子恭 is the person
    # — 谥号 / 称号 fragments —
    "忠节",   # 谥曰忠节 — a posthumous name (颜杲卿's 谥)
    "献哀",   # 献哀太子策 — 谥号 modifier, 慕容策 is the person
    # — 礼服 / 典籍 / 财物 —
    "皮弁",   # 侍中儒者皮弁、搢绅 — ceremonial cap (attire)
    "王制",   # 《王制》— a 礼记 chapter title
    "罗锦",   # 率罗锦万匹 — silk goods
    "贺表",   # 更为群臣贺表 — a congratulatory memorial (document)
    # — 跨词误切 / common phrases —
    "王至",   # 广平王至 / 王...至 — 王(爵)+至(arrive), never a name
    "王素",   # 赵王素怨 / 王素无意 — 王+素(always), cross-word
    "钱益",   # 铸钱益少 — 钱+益(more), cross-word
    "陈东",   # 伏精骑于陈东 — 阵(陈)东, battlefield bearing
    "陈前",   # 立于陈前 — 阵(陈)前, battlefield bearing
    "陈力",   # 为之陈力 — to exert effort (verb)
    "顾恋",   # 顾恋妻子 — to be attached to (verb)
    "马少",   # 军中马少 — "horses are few"
    "胡虏",   # 与胡虏战 — collective term for the enemy/Hu
    "廉直",   # 以廉直稍迁 — upright (adjective)
    "莫应",   # 天下莫应 — "none responded"
    # RC-1 (R25 name-boundary audit) — 2-char surfaces surfaced by the 单姓双名
    # truncation scan that are actually place / 官号 / 藩镇 / 国名 / verb phrases jieba
    # mislabels nr; each a junk card across several 卷. Verified: NONE has a dated/字
    # brief (no real homograph). NB: 魏相(汉丞相)/王乌(汉使)/魏公·景穆·闵孝(封号/谥号) were
    # checked and DELIBERATELY kept — real persons or 尊号 handled by rc4/rc5.
    # — 地名 / 郡名 —
    "陈留",   # 陈留郡 / 陈留太守 — a commandery (place)
    "魏兴",   # 魏兴郡 / 魏兴太守 — a commandery (place)
    "魏东",   # 魏东徐州 / 魏东豫州 / 魏东荆州 — 魏(国)+东(方位)+州, not a person
    # — 官号 / 藩镇 / 禁军号 —
    "魏仪",   # 魏仪同三司 — 魏(国)+仪同三司 (官名)
    "夏绥",   # 夏绥(银)节度使 — a 藩镇, not a person
    "陈许",   # 陈许(蔡)节度使 — a 藩镇, not a person
    "龙虎",   # 左/右龙虎统军 — 禁军号, not a person
    "龙武",   # 左/右龙武统军 — 禁军号, not a person
    "莫离",   # 莫离支 (高句丽 막리지, 官名) truncation — not a person
    # — 国名 / 跨词误切 —
    "吴越",   # 吴越国 / 吴越王 (钱镠) — a state, not a person
    "吴大",   # 吴大赦 — 吴(国)+大赦, never a person
    "钱万",   # 钱万缗 — 一万钱 (money amount), not the surname 钱
    "寇魏",   # 寇魏边 — 寇(raid)+魏(国)边, a verb phrase
    "布满",   # 布满山谷 — "spread over" (verb), not a person
    # RC-1 (R26 Phase-2 historian audit: 卷050 东汉 / 卷145 南齐-北魏 / 卷265 唐末) —
    # confirmed non-person surfaces. Each: (a) flagged by the historian audit with text
    # evidence AND (b) verified to carry ONLY auto-minted "见于卷NNN" briefs spread across
    # many dynasties (汉→唐→后晋) — the signature of a recurring false surface, never one
    # dated/字 person. NB single-char anaphora junk (见/诏/诸/律/废/怒/勇/监) and real
    # homographs (王平 三国蜀汉) deliberately excluded; 封号+名/谥号/错链 → findings (rc4/rc5).
    # — 地名 —
    "霍丘",   # 军于霍丘 — a county (place)
    "狄道",   # 陇西郡始还狄道 — a place
    "温宿",   # 姑墨、温宿 — 西域国名 (place)
    "蒋桥",   # 进屯蒋桥 — a place
    "林园",   # 送华林园 — 华林园 mid-cut (place)
    "王庭",   # 车师前王庭 — 单于/court 处所, not the surname 王
    "石洞",   # 赤石洞蛮酋 — 赤石洞 place fragment
    "高句",   # 高句骊与濊貊 — 高句骊 truncation (国名/部族)
    # — 官号片段 —
    "散骑",   # 散骑常侍 — an office
    "林监",   # 羽林监敦煌范绍 — 羽林监 mid-cut (官名)
    "王傅",   # 和王傅张廷范 — 王府傅 (官名), not a person
    # — 国名/朝代字 + 后字 跨词误切 —
    "魏散骑",  # 魏散骑常侍赵修 — 魏(国)+散骑常侍
    "魏诏",   # 魏诏营缮国学 — 魏(国)+诏(decree)
    "魏诸",   # 魏诸军粮运绝 — 魏(国)+诸(all)
    "魏顿",   # 魏顿丘匡公穆亮 — 魏(国)+顿丘…
    "魏平",   # 魏平阳平公丕 — 魏(国)+平阳…
    "魏发冀",  # 魏发冀、定、瀛 — 魏(国)+发(dispatch)冀州…
    "唐祚",   # 欲延唐祚 — 唐(国)+祚 (dynastic fortune)
    "唐恩",   # 世受唐恩 — 唐(国)+恩 (imperial grace)
    # — 典籍 / 年号 —
    "梁律",   # 为《梁律》— a law code (book title)
    "景明",   # 景明中，崇为太乐令 — a 年号 (北魏景明), not a person
    # — 普通词 / 动宾 / 形容 —
    "寇钞",   # 共入寇钞 — to raid & plunder (verb)
    "布德",   # 宣威布德 — to spread virtue (verb phrase)
    "周旋",   # 周旋民间 — to move about / deal with (verb)
    "伏见",   # 臣伏见诏书 — memorial formula ("I respectfully observe")
    "伏辜",   # 石显伏辜 — to be punished for one's guilt (verb)
    "陈列",   # 为之陈列军事 — to set out / arrange (verb)
    "莫言",   # 若肉食莫言 — "say nothing of…" (verb phrase)
    "余怒",   # 柳璨余怒所注 — remaining anger (noun)
    "林木",   # 无林木可为栅 — trees (common noun)
    "沈醉",   # 乃饮沈醉 — 沉醉 dead-drunk (state)
    "沈思",   # 澄沈思良久 — 沉思 to ponder (verb)
    "沈废",   # 坐沈废累年 — 沉废 to be cast aside (state)
    "沈勇",   # 性沈勇 — 沉勇 deep & brave (adjective)
    "王苟",   # 王苟先下寿州 — 王+苟先 (cross-word)
    # RC-1 (R27 lookback-pass audit) — 3-char garbage surfaces _surname_of admits
    # (国名/复姓 head + 官名/虚词 tail). Surfaced when the lookback pass would have
    # amplified them into adjacent 卷. Verified: only auto "见于卷NNN" briefs, no real
    # homograph (李斯/王猛 are the 秦丞相; 百里 名人 are named in full elsewhere).
    "秦丞相",  # 秦(国)+丞相(官) — never one person
    "百里于",  # 百里(复姓)+于(虚词) — a fragment, not a name
    # RC-1 (R27 国名+兴/衰 disambiguation) — 国名/朝代字 + 兴 is a dynastic-fortune
    # phrase (自唐兴以来 = "since the Tang arose") OR a 地名 (唐兴县, 台州), NOT the
    # 唐(姓)+兴 person. Strategy: the surface 唐兴 is dominated corpus-wide by
    # 「唐兴以来」(phrase ×6) and the 台州 county (破/拔/屯/守唐兴, 唐兴军/南谷); the one
    # real person (卷114 广武将军唐兴) is never captured here, so blacklisting the
    # surface loses no person. Generalises to 汉兴/晋兴/宋兴-type 国名兴 phrases.
    "唐兴",   # 唐兴以来 (phrase) / 唐兴县 (place), never 唐(姓)+兴 as a card
    # RC-1 (R29 rolling historian audit: 卷080 西晋 / 卷180 隋 / 卷280 五代) — confirmed
    # non-person surfaces. Each verified to carry ONLY auto "见于卷NNN" briefs spread
    # across multiple dynasties (the recurring-false-surface signature) with NO 字/dated
    # bio and NO real-person homograph in 403BC–960AD. Real homographs flagged by the
    # audit (陈声/樊建/张尚/孙愼/吴平/陈乐/萧琮/张彦 etc.) were DELIBERATELY kept — those are
    # 卷-local 截断/错链, handled elsewhere, never blacklisted by surface.
    # — 谥号 / 尊号 / 封爵 片段 —
    "景献",   # 景献(皇后) 谥号片段 (晋), not a person
    "章安侯",  # 章安侯 — 封爵号 (孙奋 etc. 封章安侯); the person is named in full elsewhere
    "德孝",   # 「(高祖圣文章武)明德孝(皇帝)」— 后晋高祖石敬瑭 尊号片段, not a person
    # — 形容 / 动宾 / 普通词 —
    "吉祥",   # 祥瑞(adj/noun) — 见于卷077→154 (魏→梁), never one person
    "费损",   # 费损 — to waste/expend (verb), 见于卷073→159 (魏→梁)
    "屈膝",   # 屈膝 — to kneel/submit (verb), 宋→陈→后晋
    "纪功",   # 纪功 = 记功 record-merit (verb), 魏→宋→隋→唐→后晋
    "徐图",   # 徐图(之) — to plan slowly (verb), 隋→唐→后唐→后汉, no real 徐图 pre-960
    # — 地名 / 城郭 / 渡口 —
    "郭内",   # 郭内 — inside the outer wall (城郭+内), not 郭(姓)+内
    "蒲津",   # 蒲津(关) — a Yellow-River ferry/place, 梁→隋→唐 (×4)
    # — 部族 / 突厥官号 / 军队 / 府名 —
    "薛延陀",  # 薛延陀 — a 铁勒 tribe (Xueyantuo), 隋→唐 (×10卷), not a person
    "仆骨",   # 仆骨 — a 铁勒 部 (Pugu); the surname 仆固 (仆固怀恩) is a different surface
    "俟斤",   # 俟斤 — a 突厥 title (irkin), not a person; 万俟(复姓) is a different surface
    "胡寇",   # 胡寇 — "the Hu raiders" (collective enemy), 汉→晋→陈→唐→后晋 (7 cards)
    "唐府",   # 唐府 — 后唐 的官府/府 (国名+府), not a person
    "唐骑",   # 唐骑 — 唐军骑兵 (国名+骑), not a person
    # R29 round 2 (卷030 西汉 / 卷120 南朝宋-北魏 / 卷230 唐德宗) — same discipline.
    # Real homographs kept: 李实/赵有/武安/王梁/马周/发(姬发)/梁/权 — 卷-local 截断/单名.
    # — 谥号 / 时代 / 城名 片段 —
    "忠烈",   # 忠烈 — 谥号 (颜杲卿/张巡 谥), 唐→宋→梁, not one person
    "惠文",   # 惠文 — 谥号 (秦惠文王/赵惠文王), 周→唐→宋
    "魏初",   # 魏初 — "early 魏" 时代短语 (国名+初), not a person
    "统万",   # 统万(城) — 赫连勃勃 的都城 (a place), not a person
    # — 官名 / 官署 / 藩镇 / 亲属称谓 —
    "莫府",   # 莫府 = 幕府 (military HQ), 汉→秦, not 莫(姓)+府
    "昭义",   # 昭义(军) — a 唐 藩镇, not a person
    "傅相",   # 傅相 — 太傅/丞相 (官), 宋→汉→魏, not 傅(姓)+相
    "顾命",   # 顾命(大臣) — deathbed-charge (官义名词), 宋→晋→隋→魏
    "尹李齐",  # 尹李齐 — 京兆尹(官)+李齐 误连; the person 李齐 has his own surface
    "王舅",   # 王舅 — "the king's maternal uncle" (亲属称谓), not a person
    # — 军器 / 形容 / 动宾 / 成语 —
    "刁斗",   # 刁斗 — a night-watch clapper (军中器具), not 刁(姓)+斗
    "赫赫",   # 赫赫 — grand/illustrious (叠词形容), never a name
    "章奏",   # 章奏 — to submit a memorial (上章奏), 唐→汉, not 章(姓)+奏
    "蒙尘",   # 蒙尘 — (天子)蒙尘 = the sovereign flees (成语), 汉→晋→梁→唐 (6 cards)
    "屈卿",   # 屈卿 — "to condescend to you" (动宾), not 屈(姓)+卿
    "石柱",   # 石柱 — a stone pillar / place, 唐→宋, not 石(姓)+柱
    # — 国名/朝代字 + 后字 跨词误切 (cf. 王至/王素/王苟) —
    "王欲",   # 王欲… — 王(爵/姓)+欲 (wishes to), 周→梁→宋→后梁, never a name
    "王发",   # 王发… — 王(爵/姓)+发 (dispatches), 汉→宋, cross-word (发=姬发 is single-char)
    # RC-1 (R33 卷250 historian audit) — 两林蛮 is a 西南 部族 (the Liang-Lin Man); the
    # model false-fired a 林蛮 card inside 两林蛮. No 林蛮 person in the corpus.
    "林蛮",   # 掠两林蛮 / 两林蛮开门纳之 — 两林(蛮)部 fragment, not 林(姓)+蛮
    # Wave 36b (reader-agent audit of 卷035–232) — confirmed non-person auto surfaces, each
    # carrying only auto briefs across multiple 卷/dynasties with NO real-person homograph.
    # — 普通词 / 形容 / 成语 —
    "英杰",   # 天资英杰 / 英杰之才 / 招延英杰 — heroes·talents (普通名词), 14 cards, never a person
    "布野",   # 牛马布野 / 百姓布野 — spread across the fields (成语), not 布(姓)+野
    "沈密",   # 性沈密 / 重厚沈密 — 沉静缜密 (形容性格), not 沈(姓)+密; bare 密 = 「密(secretly)」
    # — 地名 / 城戍 / 渡口 —
    "石梁",   # 克…石梁(二戍) / 退保石梁 / 拔石梁 — a 戍/fortress place, not 石(姓)+梁
    "钱溪",   # 钱溪江岸 / 趣钱溪 / 攻钱溪 — a 长江 battle site (place), 19 cards, not 钱(姓)+溪
    # — 部族 / 跨词 —
    "索头",   # 索头郁鞠 / 加索头乞得 — 索头(鲜卑)=拓跋部 nickname (a 部族), the person is the tail
    "胡来",   # 胡来攻 / 胡来转近 — 胡(虏)来 (the Hu came, verb) or 去胡来王 title, not 胡(姓)+来
    # — [实名/指代] + 动词 误粘 (cf. 帅-tail in bad_auto_surface) —
    "温至",   # 温至金乡 / 温至枋头 / 温至长安 — 桓温 至(arrived at), not 温(姓)+至
    "温屯",   # 温屯故太极殿前 / 温屯武阳 — 桓温 屯(stationed at), not 温(姓)+屯
    "景犹",   # 景犹欲全其一子 / 景犹有船 — 侯景 犹(still·yet), not 景(姓)+犹; bare 犹 = 「still」
    # Wave 39 (reader-agent audit of 卷001–034, early W. Han) — confirmed non-person auto
    # surfaces. 战国 narrative is dense with [国名/王]+动词 slices the char-NER mis-cards.
    # Rejected from this batch (REAL people, deliberately kept): 魏相 (宣帝丞相), 王尚
    # (谯王·尚之 封号 mis-slice), 章武 (章武王·融) — those need targeted handling, not a ban.
    # — [国名/王] + 动词 战国叙事切片 —
    "秦败", "魏败", "秦自", "秦合", "秦益", "秦强", "赵会", "赵壁",
    "王崩", "王恐", "王必", "王怒", "王召", "王许", "王齐", "王实", "魏筑",
    # — 动宾 / 动词 —
    "支秦",   # 韩、魏不能支秦 — to resist Qin
    "谢汉",   # 上书谢汉 — to apologize/thank Han
    "寇狄",   # 匈奴寇狄(道) — to raid 狄道, cross-word
    "解鞍",   # 皆下马解鞍 — to unsaddle
    # — 地名 / 城郭 / 台榭 / 封邑 —
    "中山",   # 攻中山 / 略中山之地 / 中山君奔齐 — the STATE 中山 (ruler is 中山君/王)
    "广武",   # 军广武 / 渡广武(水) / 临广武间 — 楚汉相争 的地名
    "桑丘",   # 齐伐燕取桑丘 — place
    "杜平",   # 魏惠王会于杜平 — place
    "邢丘",   # 秦拔魏邢丘 — place
    "章华",   # 起章华之台 — 楚 的台榭 (palace terrace)
    "隆虑",   # 隆虑公主 — 封邑/汤沐邑 fragment, the person is the named 公主/子
    # — 普通词 / 成语 / 形容 / 部族 —
    "郑声",   # 放郑声，郑声淫 — the music of 郑 (郑卫之声)
    "韩卢",   # 走韩卢而搏蹇兔 — 韩国的名犬 (proverbial hound), not 韩(姓)+卢
    "英俊",   # 英俊宜可得矣 — talented men (cf. 英杰)
    "温良",   # 仁爱温良者 — gentle·kind (形容)
    "汤功",   # 嘉延寿、汤功 — 陈汤之「功」(merit), not a person
    "骆越",   # 骆越之人 — a 越 部族 (ethnonym)
    # — 官名 / 复姓 / 爵号 glue —
    "太史",   # 智果别族于太史 — 官/复姓 origin, not a standalone person (太史慈 is 3-char)
    "赵相",   # 赵相张耳/贯高/周昌/虞卿 — 赵国(的)相 (title) glued before the named person
    "侯莽",   # 新都侯莽 — 新都侯·王莽 mis-slice; 王莽 is carded in full elsewhere
}

# RC-6/RC-3c — 3-char 鲜卑/胡 复姓 (clan names). On their own they are a CLAN, not
# a person (阿史那 = the Türk royal house), so a bare occurrence is dropped; but as
# the head of a longer name (阿史那社尔) the model SHOULD keep them — handled in
# load_model_ner_people (bare-clan drop only).
COMPOUND3: set[str] = {
    "阿史那", "阿史德", "破六韩", "纥豆陵", "费也利", "吐谷浑", "是云", "没鹿回",
}

# 复姓 that double as offices (大司马, 司空, 司徒, 太史令): the model glues these onto a
# following 姓名, so a surname-headed tail signals office-glue rather than one person.
_TITLE_SURNAMES: set[str] = {"司马", "司空", "司徒", "太史"}


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
    # [名]帅 — 帅 = 统帅/将兵 (to lead / commander), a role-verb the char-NER glues onto a
    # real leading name: 越帅甲士 / 桓帅众 / 刘胡帅步卒 / 景帅骑 / 王帅银枪都. Never a personal
    # given name in this corpus (every 帅-tail surface is such a misparse), so a 姓-headed
    # form ending in 帅 is role-glue, not one referent. (帅 as a leading surname — 帅范 — is
    # untouched: this only fires on a trailing 帅.)
    if len(s) <= 4 and s[-1] == "帅":
        return True
    if s.endswith("太子") or s.endswith("世子"):
        return True
    if len(s) <= 3 and s.endswith("氏"):  # 王氏 / 窦氏 clan reference
        return True
    # 姓 + 官/后妃称号: 曹尚书 / 梁贵人 / 赵倢伃 — a person referred to by office or
    # consort rank, not a fixed personal name (the model glues 姓 onto the title).
    if len(s) >= 3 and any(s.endswith(suf) for suf in _ROLE_SUFFIX):
        return True
    # 司马 / 司空 / 司徒 / 太史 双为官与复姓: 当其后缀本身是姓首时即官+名 glue（司马·高颎、
    # 司马·任约），非一人 → 丢弃。此前仅在 model NER 路径施加，jieba/导读 路径漏网，故上提到
    # 共用 guard，全路径生效。
    for _off in _TITLE_SURNAMES:
        if len(s) >= len(_off) + 2 and s.startswith(_off) and s[len(_off)] in CLEAN_SURNAMES:
            return True
    return False


# 官职 / 后妃称号 that the char-level model glues onto a leading 姓 (曹·尚书, 梁·贵人).
# None is ever a personal given name, so a 姓-prefixed form ending in one is title-glue.
_ROLE_SUFFIX = ("尚书", "仆射", "刺史", "将军", "太守", "长史", "贵人", "婕妤", "倢伃",
                "昭仪", "婉仪", "夫人", "留后", "都督", "中郎", "常侍", "录事")


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
    "费连", "莫多", "厍狄", "贺若",
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
    # 官名 glue (司马 = 官 here, not 姓): 高/任 are AMBIGUOUS surnames so the
    # CLEAN_SURNAMES office-glue guard skips them — route the glued surface to the
    # real person instead (卷174 historian audit).
    "司马高颎": "高颎",
    "司马任约": "任约",
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
        # 复姓 (鲜卑/匈奴/胡 compound surnames: 尉迟/达奚/乙弗/豆卢/贺娄/叱列/淳于…) are
        # structurally invisible to jieba's word segmenter, so 复姓-headed names fall
        # through the jieba path entirely (尉迟惇/达奚儒/豆卢勣…). The char-level model
        # keeps the compound intact → recover them HERE. Single-姓-headed names stay
        # deferred to jieba, which screens 2-char common-word noise via its surname gate.
        cpfx = surf[:2] if (len(surf) >= 3 and surf[:2] in COMPOUND) else (
            surf[:3] if (len(surf) >= 4 and surf[:3] in COMPOUND3) else None)
        is_compound_headed = cpfx is not None
        is_single_surname = surname_headed(surf) and not is_compound_headed
        # 司马 / 司空 / 司徒 / 太史 are offices as well as 复姓 (大司马 …), so the model glues
        # them onto a following full name (司马·刘文静, 司马·杨统). When the tail after such a
        # title-surname is itself surname-headed, it is office-glue, not one person → drop.
        if cpfx in _TITLE_SURNAMES:
            tail = surf[len(cpfx):]
            if len(tail) >= 2 and tail[0] in CLEAN_SURNAMES:
                continue
        if surf in tails:
            continue
        if prefix_blocked(surf):
            continue
        # Frequency / eligibility floor by surface class:
        #  - 复姓-headed: any n — the distinctive compound surname disambiguates.
        #  - single-姓-headed: len≥3 and n≥2 — 3+char 姓名 the char-level model segments
        #    correctly where jieba mis-splits (崔弘度/席毗罗); 2-char single-姓 stays jieba's
        #    job, since its surname gate already screens 2-char common-word noise.
        #  - non-姓 model surface: original n≥3 or (len≥4 & n≥2).
        if is_compound_headed:
            ok = True
        elif is_single_surname:
            ok = len(surf) >= 3 and n >= 2
        else:
            ok = n >= 3 or (len(surf) >= 4 and n >= 2)
        if not ok:
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


# Curated jieba mid-name truncations: a short card is the SAME person as a full 姓名,
# but the truncated surface (达奚[长]儒 → 达奚儒, [陈慧]纪 wrongly read as 陈纪) yielded a
# separate card. Each entry restricts the fold to the 卷 where the truncation occurs,
# so unrelated 同名 cards (the 汉 陈纪 of 卷014/059) are never touched. Seeded from the
# 卷174 historian audit (Wave 3/5 backlog); extend as new truncations surface.
MANUAL_TRUNC_MERGE: dict[str, tuple[str, set[int]]] = {
    "达奚儒": ("达奚长儒", {174}),
    "陈纪":   ("陈慧纪", {174}),
}


def merge_truncations(people):
    """Fold curated truncated-name cards into their full 姓名, restricted to the listed
    卷. Mutates `people` in place; returns the merge count."""
    by_canon: dict[str, list] = collections.defaultdict(list)
    for p in people:
        by_canon[p["canonical_name"]].append(p)
    merged = 0
    drop_ids: set[str] = set()
    for short, (full, juan_filter) in MANUAL_TRUNC_MERGE.items():
        hosts = by_canon.get(full)
        if not hosts:
            continue
        host = hosts[0]
        for p in by_canon.get(short, ()):
            if p["id"] in drop_ids or not (set(p["juans"]) & juan_filter):
                continue
            for j in p["juans"]:
                if j not in host["juans"]:
                    host["juans"].append(j)
            for m in p.get("match", []):
                if m not in host["match"]:
                    host["match"].append(m)
            if short not in host["match"]:
                host["match"].append(short)
            host["juans"] = sorted(set(host["juans"]))
            drop_ids.add(p["id"])
            merged += 1
    if drop_ids:
        people[:] = [p for p in people if p["id"] not in drop_ids]
    return merged


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


def _given_tail(full: str):
    """The bare given-name tail of a full 姓名 after stripping a leading 姓:
    a 2-char COMPOUND prefix, else a single CLEAN 姓. Returns the tail only when it
    is exactly 2 Han chars (single-char tails are validator-banned as too ambiguous;
    3-char tails are rare and risky). Ambiguous surnames (于/方/白…) are deliberately
    NOT stripped here — only their pre-existing fragments are folded by merge_anaphora."""
    if not _HAN_ONLY.match(full):
        return None
    if len(full) == 4 and full[:2] in COMPOUND:
        tail = full[2:]
    elif len(full) == 3 and full[:2] in COMPOUND:
        # 复姓 + 单名 (夏侯孜, 诸葛亮, 司马懿): given is ONE char — there is no valid
        # 2-char tail. Stripping only the first 姓 char would cross the 复姓 boundary
        # (夏侯孜 → 侯孜). The single given char is handled by _given_single/anaphora.
        return None
    elif len(full) == 3 and full[0] in CLEAN_SURNAMES:
        tail = full[1:]
    else:
        return None
    return tail if len(tail) == 2 else None


# Wave 5 P2 — single given-chars too common as ordinary 文言 words/verbs to ever
# resolve as a bare 省称, even behind the anchor + bigram guards. Their full-name
# forms still match as aliases; only the single-char anaphora is withheld.
ANAPHORA_CHAR_EXCLUDE = set(
    "信温安明进通顺和良善真立成道德文武政平正定大小公侯王主君臣民贤忠孝义礼智素")


def _given_single(full: str):
    """The single-char given name of a 姓+名 (杨坚→坚, 尉迟迥→迥, 宇文招→招), used for
    Wave 5 single-char 省称回指. Returns the char ONLY when the given name is exactly
    one Han char after stripping a CLEAN single 姓 or a 2-char COMPOUND 复姓; ambiguous
    surnames (于/方/白…) are excluded (their leading char is too often a common word)."""
    if not _HAN_ONLY.match(full):
        return None
    if len(full) == 3 and full[:2] in COMPOUND:
        g = full[2:]
    elif len(full) == 2 and full[0] in CLEAN_SURNAMES:
        g = full[1:]
    else:
        return None
    return g if len(g) == 1 else None


def build_anaphora_rules(people, allowed):
    """Wave 5 P2 — per-卷 set of single-char 省称 CANDIDATE given-chars
    {juan: {char, ...}}.

    A given-char is a candidate for a 卷 when at least one person whose full name
    appears there owns it via _given_single (杨坚→坚), minus the common-word exclude
    list. WHICH person a bare char resolves to is decided at build time by the
    nearest preceding full-name antecedent (build_persons.extract_anaphora).

    COLLISION GUARD: a char is suppressed for a 卷 when ≥2 DISTINCT people there have
    a canonical name ending in that char — detected by endswith, NOT by _given_single,
    so it catches collisions even when one party's surname is outside our list (元胄 &
    宇文胄 both end 胄 → 胄 suppressed in that 卷, never mis-bound to the recognised one).
    Precision-first: an ambiguous given-char is dropped rather than guessed."""
    cand: dict[int, set] = collections.defaultdict(set)
    endswith: dict[int, dict] = collections.defaultdict(lambda: collections.defaultdict(set))
    for p in people:
        cn = p["canonical_name"]
        g = _given_single(cn)
        last = cn[-1] if (len(cn) >= 2 and _HAN_ONLY.match(cn)) else None
        for j in p["juans"]:
            if j not in allowed:
                continue
            if g and g not in ANAPHORA_CHAR_EXCLUDE:
                cand[j].add(g)
            if last:
                endswith[j][last].add(p["id"])
    out: dict[int, set] = {}
    n_admitted = 0
    for j, chars in cand.items():
        keep = {c for c in chars if len(endswith[j].get(c, ())) < 2}
        if keep:
            out[j] = keep
            n_admitted += len(keep)
    return out, n_admitted


def generate_anaphora_tails(people):
    """RC-2c (generative 省称) — propose each full 姓名's bare 2-char given-name tail
    (韦孝宽→孝宽, 李德林→德林, 司马消难→消难) as a 卷-local match surface, EVEN when no
    NER candidate for the bare form exists (the gap merge_anaphora cannot fill).

    Disambiguation is delegated to the per-卷 collision step: a generated tail is kept
    only where exactly ONE person in the 卷 owns it, and dropped wherever two or more do
    — so a short form shared by two actors in a 卷 (the case the audit flagged) is never
    mislinked, it is simply left un-underlined. Common words/titles are screened by
    bad_auto_surface. Conservative: 2-char tails only, never single-char."""
    added = 0
    for p in people:
        tail = _given_tail(p["canonical_name"])
        if not tail or bad_auto_surface(tail):
            continue
        if tail in p["match"]:
            continue
        p["match"].append(tail)
        if tail not in p.setdefault("names", []):
            p["names"].append(tail)
        added += 1
    return added


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


# ── Wikipedia/Wikidata period VERIFICATION (wiki_enrich.py) ──────────────────
# wiki_person_info.json maps a verified-person surface to {qid, title, desc,
# extract}. Wikipedia is used ONLY to verify/disambiguate a surface — never as
# card CONTENT (card brief/identity is book-derived; see assembly §2a). The
# biographical-brief generators were retired with the dewiki-content cleanup;
# what remains is the period gate (_wiki_year/_wiki_out_of_period) that flags
# post-959 / modern namesakes (资治通鉴 ends 959 CE) for the lifespan-gate
# disambiguation layer.
def _load_wiki_info():
    f = Path(__file__).resolve().parent / "wiki_person_info.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


WIKI_INFO = _load_wiki_info()

_LATE_TOKENS = (
    "明朝", "明代", "清朝", "清代", "元朝", "元代", "宋朝", "北宋", "南宋", "宋代",
    "金朝", "民國", "民国", "中华民国", "中華民國", "中华人民共和国", "中国大陆",
    "中國大陸", "現代", "现代", "當代", "当代",
)


def _cjk_len(s):
    return sum(1 for ch in s if '\u3400' <= ch <= '\u9fff')


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
    trunc_merged = merge_truncations(people)

    # 2d. RC-2c generative 省称 — propose each full 姓名's 2-char given-name tail as a
    #     match surface (recall for short forms no NER candidate proposed). The collision
    #     step below is the disambiguator: ambiguous tails (≥2 owners in a 卷) are dropped.
    anaphora_generated = generate_anaphora_tails(people)

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

    # Wave 5 P2 — single-char 省称 candidate-char set per 卷 (consumed by
    # build_persons; the bare char resolves to its nearest full-name antecedent).
    anaphora_rules, anaphora_admitted = build_anaphora_rules(people, allowed)

    return people, dict(rules), anaphora_rules, {"ambiguous_dropped": dropped,
                                 "glue_bound": glue_bound,
                                 "glue_missing": glue_missing,
                                 "anaphora_merged": anaphora_merged,
                                 "anaphora_dropped": anaphora_dropped,
                                 "anaphora_generated": anaphora_generated,
                                 "anaphora_char_admitted": anaphora_admitted}
