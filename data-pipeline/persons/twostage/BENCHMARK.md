# Agent 1 benchmark

本页记录当前 `rules.py` Agent 1 相对生产 v1 的可重复兼容性 benchmark。最新机器可读
结果见 [`benchmark-latest.json`](benchmark-latest.json)。

## 运行方法

在仓库根目录执行：

```powershell
data-pipeline\.venv-ner\Scripts\python.exe -X utf8 `
  data-pipeline\persons\twostage\benchmark.py `
  --json data-pipeline\persons\twostage\benchmark-latest.json
```

只做快速局部检查时，可以指定卷号：

```powershell
data-pipeline\.venv-ner\Scripts\python.exe -X utf8 `
  data-pipeline\persons\twostage\benchmark.py --juans 23 69 97
```

可选译文 evidence 必须显式指定，且不覆盖正式 `benchmark-latest.json`：

```powershell
data-pipeline\.venv-ner\Scripts\python.exe -X utf8 `
  data-pipeline\persons\twostage\benchmark.py `
  --juans 27 37 45 150 `
  --translation-evidence-dir C:\temp\translation-evidence `
  --json C:\temp\translation-evidence-benchmark.json
```

全量运行会读取已有 POS·Giv cache；若正文变化或 cache 不存在，`pos_giv.py` 会自动重建
相应卷的 cache。当前 warm-cache 全 294 卷通常约需一分钟。

## 固定口径

- **Agent 1：** `twostage/rules.py`，`PRESET_RECALL`，所有规则以带圈编号节为作用域。
- **v1 reference：** `web/public/text/persons/mentions/juan_NNN.json` 中
  `source == "main"` 的生产 span。
- **匹配：** 必须是同一卷、同一 paragraph id，且半开字符区间 `[start,end)` 重叠。
- **正文范围：** 全 294 卷的 main text；注文不参加本次比较。
- **Agent 2：** 尚未开发，不参加 benchmark。
- **旧 `persons-v2/`：** 是过时的 v1 ADD-only union，不是当前 Agent 1 输出，禁止作为
  v2 benchmark 输入。

生产 v1 不是独立人工金标。它既漏掉 `丞相斯`、`其弟乙` 等真实人物表达，也含有卷级
回指和普通词误标。因此：

- `v1 coverage` 只表示 Agent 1 对生产行为的兼容覆盖，不等同于真实 recall；
- `v1 overlap proxy` 只表示 Agent 1 span 与 v1 重叠的比例，不等同于 precision；
- `Agent1 nonoverlap-v1` 不能直接算作 false positive，必须人工抽样。

译文 evidence 仍默认关闭，不进入下方正式 baseline。显式开启后的全 294 卷实验结果
如下；机器可读输出应另存，不能覆盖 `benchmark-latest.json`。

| 指标 | 默认 baseline | translation-assisted |
|---|---:|---:|
| v1 coverage | 123,821 / 128,596 (96.287%) | 124,473 / 128,596 (96.794%) |
| v1-only gap | 4,775 | 4,123 |
| assisted gap closure | - | 652 / 4,775 (13.65%) |
| alias coverage | 85,228 / 88,619 (96.174%) | 85,647 / 88,619 (96.646%) |
| anaphora coverage | 33,886 / 35,261 (96.101%) | 34,117 / 35,261 (96.756%) |
| Agent 1 spans | 172,497 | 174,386 |
| v1 overlap proxy | 72.022% | 71.661% |

translation-assisted 输出比默认输出净多 1,920 个 span。译文路径仍只产生
`translation_fullname`、`translation_anaphora`，以及由新完整姓名锚点触发的普通
`anaphora`；默认路径不读取 translation evidence。

precision 必须与 v1 overlap proxy 分开报告。此前 translation 路径的固定样本结果不能
外推为本轮称号 precision。本轮先逐条审计 360 个不与 v1 重叠的候选称号：331 个明确
成立、2 个非人物、27 个属于更长称号的尾部组件。收紧后删除两个非人物和无局部授权的
长外族称号切尾；最终相对 fullname-anchor baseline 新增的 294 个 non-v1 称号中，
291 个来自明确成立集合，另外 3 个是按既定组件政策保留的 `成康/贤文/成靖`。

## 最新全量结果

运行时间：2026-07-18 UTC；Python 3.11.4；294 卷。结果 JSON 同时记录：

- rule-bundle SHA-256：`88681ccbc4673a25ce92783eafe07912080a5d06c478b507f0eba42bed4cae2f`
- `admin-places.json` SHA-256：`b6849b571ae31041ea362bb1d2a9c689a61da7e081aa5460cc10e162e1bd5370`

后者使时序行政区证据变化不会被误当成“同一规则”的 benchmark。

| 指标 | 结果 |
|---|---:|
| v1 main spans | 128,596 |
| Agent 1 覆盖 v1 | 123,821 |
| **v1 coverage** | **96.287%** |
| v1-only | 4,775 |
| Agent 1 spans | 172,497 |
| Agent 1 spans overlapping v1 | 124,235 |
| Agent 1 spans not overlapping v1 | 48,262 |
| v1 overlap proxy | 72.022% |

两个 overlap 数不是一一对应计数：当两个系统的跨度切分不同，一个 span 可能与多个 span
重叠，所以 `123,139` 与 `123,419` 可以不同。

### v1 coverage by kind

| v1 kind | 覆盖 | 总数 | coverage | v1-only |
|---|---:|---:|---:|---:|
| alias | 85,228 | 88,619 | 96.174% | 3,391 |
| anaphora | 33,886 | 35,261 | 96.101% | 1,375 |
| feng | 355 | 363 | 97.796% | 8 |
| gloss | 1,535 | 1,536 | 99.935% | 1 |
| role | 2,817 | 2,817 | 100.000% | 0 |
| **ALL** | **123,821** | **128,596** | **96.287%** | **4,775** |

完整的 Agent 1 `chunk_type` 数量、重叠数和 overlap proxy 保存在
[`benchmark-latest.json`](benchmark-latest.json)。每次修改规则后，应重新执行同一命令并
审查 JSON diff；新增规则还必须抽查 non-overlap 样例，不能只优化相对 v1 的数字。

### Agent 1 全局人物 KB 移除

Agent 1 不再读取 `people.json`。原 `lit3/xing2/given2/name_all/canonical_names/jue2/
personal_titles/personal_appellations` admission 与 identity-count uniqueness 逻辑已经删除；
`STOP_GIVEN` 和 `seed.bad_auto_surface` 也不再参与规则。旧 `corpus_*` rule id 目前仅作为
benchmark provenance 的兼容名称，执行内容已经改为完整 POS/BIO、局部人物句法和
model-derived NER corroboration。

Step 1 仍可使用姓氏形态、官职、亲属、谓词、爵位、标点等语言学类别，以及时序地名
veto；这些集合不包含 person identity。人物 KB 只允许 Agent 2 做 identity binding。
相对上一 KB-backed baseline，default v1 compatibility 从 98.247% 降至 95.076%，这是
移除全局姓名 surface oracle 后公开记录的代价，不能用 Step 2 KB 回填到 Step 1。

### KB-free gap top-4

对 6,332 个 KB-free baseline gap 做结构聚类后，优先处理四个可由当前 occurrence
证据证明的 family：严格人物谓词 823、称号形态 711、完整多字 given 194、官职/角色/
并列 apposition。规则只使用 identity-free model NER、完整 POS/BIO token、局部句法和
边界，不使用人物 hardcode、反例 blacklist、stop-word admission 或全局人物 KB。

宽版 prototype 曾把 `武宗疾/韦述议` 等谓词尾部吞入姓名，已拒绝。最终 predicate 与
appos 必须由首尾完整、连续、高置信 `PROPN·NameType` tokens 覆盖；predicate 还拒绝官职
类别，title 要求完整称号组件边界，given 要求完整多字 BIO-Giv 和硬边界。最终 default
相对 95.076% baseline 恢复 304 个 v1 gap（+0.236pp），spans +897；overlap proxy
73.704%→73.491%。translation-assisted 恢复 207 个 gap（+0.161pp）。

按最终 provenance，`model_ner_predicate` 8,491/10,556 与 v1 重叠（80.438% proxy），
`model_ner_given` 16,000/20,041（79.836%），`model_ner_appos` 2,905/3,767（77.117%）。
`model_ner_title` 为 671/1,558（43.068%）；该低 proxy 不能当 precision，因为固定
non-v1 样本中大量是 v1 漏标的 `田公/朱夫人/武王/武侯/郭太后`，但仍单独保留风险标记。

### POS-Giv local frame 与同节称号 anchor

下一轮关闭两个 residual family。`pos_given_local_frame` 在普通规则和同节 anaphora
之后才运行，只接收完整高置信 POS-Giv 与严格人物主宾谓词；同时拒绝功能词、地名、姓氏
后的截断、右侧姓名 continuation，以及可向右扩成 model NER 人名的单字，因此不会把
`葛从周` 错切成 `[从]周`。`local_title_anchor` 只允许同节更早的完整高置信姓名或
`model_ner_title` card 授权同一称号或其完整后缀；当前 occurrence 仍须 model NER、
完整 token 边界和人物谓词。

相对 top-4 baseline，default 恢复 287 个 v1 gap（+0.224pp），assisted 恢复 172 个
（+0.133pp）。最终 `pos_given_local_frame` 产生 2,666 spans，其中 341 与 v1 重叠；
固定 SHA-256 样本 240/240 均为人物。`local_title_anchor` 产生 97 spans，其中 42 与
v1 重叠；全部 55 个 non-v1 occurrence 已逐条检查，均为真实人物称号或姓名省称。
因此两者的低 v1 proxy 记录为生产 v1 漏标现象，不解释为 precision。

### 重复称号与公主称号

残余 gap 中最大的重复 family 是 `沛公/项王/平原君/武安君/信陵君`。新增的
`local_title_anchor` 分支仍要求 model NER、完整 token 边界、同节更早同形文本、
完整 PROPN 称号组件和当前严格人物句法；它允许封地被 POS 标成 Geo、姓氏置信度偏低，
但拒绝功能/数量组件、长姓名内部切尾、连续爵号和 `子` 尾的高歧义 fallback。全量新增
305 个 geometry、无移除，关闭 132 个 v1 gap；304 个是逐条确认的人物称号，剩余
`乌洛侯使者` 是 model/POS 无法仅凭局部形态区分的已知歧义，不以 surface blacklist
修补。

`princess_title` 不依赖 model NER 或人物 KB：完整 BIO 称号组件后接两个独立 NOUN
token `公 + 主` 即构成人物称号；BIO 的 `I` 起点会向左补齐一个被 POS 错分的首字，
同时拒绝标点和 suffix 切尾。规则产生 135 个完整称号，逐条审计 135/135 为人物；
它另触发 18 个同节 bare `公主` 回指。相对重复称号阶段，新增 153、移除 65，65 个
移除全部由更完整的公主称号 span 替换。两项合计把正式 default coverage 从
95.536% 提高到 95.673%，gap 5,741→5,564；assisted coverage 为 96.113%，gap 4,998。

### 同节 exact propagation 与 partial-POS 修复

本轮关闭三个 residual family，均不使用人物 hardcode、surface blacklist、stop-word
admission 或全局人物 KB：

1. `local_exact_surface` 由同节任一可信的同形完整姓名 card 支持当前 occurrence，但当前
   occurrence 仍须 model NER、完整 token/姓名硬边界和人物 NameType signal。
2. `local_exact_title` 对同形 `X公/X王/X君/X侯` 分流处理，要求完整 PROPN 称号组件，
   并拒绝 `子` 高歧义 fallback、连续爵号、官职和姓名内部切尾。
3. `model_ner_partial_pos` 修复 model NER 内只有部分 POS 人名成分幸存的姓名；要求严格
   人物句法、完整 token 边界、NameType 首部和 Giv/Prs 尾部，并拒绝角色、官职、亲属、
   命名语法及 reporting verb 被吞入姓名。

exact propagation 只传播完全相同的完整 surface，不是身份绑定；它可以使用同节后文
同形 occurrence，但较短 handle 的 anaphora 仍严格要求更早 anchor。宽版 prototype 曾
放大 `柔然/胥靡/玉皇` 等弱 anchor，并生成 `臣光/乐羊伐/拜祜/童之白` 等句法 overmerge；
这些均由通用 anchor、边界、continuation、role/office/kinship 和 reporting-verb guards
移除，没有加入 surface 例外。

最终 default provenance 为：`local_exact_surface` 72（58 与 v1 重叠、14 non-v1），
`local_exact_title` 115（64/51），`model_ner_partial_pos` 321（248/73）。138 个 non-v1
occurrence 已逐条审计。相对上一正式阶段，default 关闭 107 个 v1 gap，coverage
95.673%→95.756%；assisted 关闭 78 个 gap，96.113%→96.174%。

### Assisted 高频称号与省称

本轮针对 assisted residual 的高频真实 family 增加三类 model-NER + occurrence-local
结构规则：Geo 封号 `X君/X公/X侯`、rank title `X伯`、庙号 `X宗`。称号必须满足完整
POS/BIO 边界和当前人物语境；封号还拒绝地点后缀及左、右侧长名截断，庙号拒绝 Geo/Nat
组件、爵号 continuation 和移动到地点的语法。已授权称号可在同节做 exact-title 传播。

Translation handle 另增加 forward-only same-jie anchor：从本节最早 eligible translated
candidate 起生效，不反向覆盖此前文字；功能词形单字仍要求独立的高置信谓词结构。宽版
单字传播 prototype 会误标 `主上/通议大夫/何谓/劝进/坚守/定太子`，已整体删除。

高频 family 定向关闭结果：`沛公` 45/45、`智伯` 28/28、`庄宗` 22/24、`密` 9/43；
`垂` 0/38。`垂` 的宽版收益依赖不安全的通用单字传播，因此没有保留。五个 family
合计关闭 104/178。全量 assisted 相对上一正式结果新增覆盖 292 个 v1 span，
96.174%→96.401%，gap 4,920→4,628；default 新增覆盖 263 个，
95.756%→95.961%。规则 SHA-256 为
`4611fe69ca0c7f93335bc94449aaef565557e233bc0792219ddb20ddab7b5c40`。

### 受控多字 surface 与 translation exact admission

本轮从 4,628 个 assisted residual 中选择十个高频多字 family，但没有按 surface 硬补。
新增四个通用 gate：

1. `X王/X后` 必须是完整二字 model-NER title、独立 `PROPN + NOUN` token，并有重复、
   人物句法或硬边界；拒绝左侧长名截断。
2. `X公主` 在无 BIO component 时，只允许同节重复的完整二字 component，并拒绝人物选择
   动词被吞入 title。
3. translation exact mapping 可在完整 token 边界下，由同节另一 occurrence 的人物
   NameType 或称号形态授权；干支、官职、地点/族群、军镇 continuation 和 polity frame
   全部否决。
4. 较长的已接纳姓氏全名可贡献完整双字 handle，供 `local_exact_surface` 在同节传播；
   称号、政权/族群 frame 和更长姓名内部不参与。

定向关闭为 `太平公主` 15/23、`启民` 9/19、`项王` 11/14、`韦后` 7/13、
`用之` 10/13、`颉利` 2/15、`盖吴` 1/12，合计 55/146。`异人/可足浑/安禄山`
剩余 occurrence 无法仅靠通用局部证据与普通词、族名或地名稳定区分，未强行接纳。

全量审计先后删除了 `丁巳/辛酉` 干支、`室韦/柔然/高句丽` polity、`遣长公主`
动词吞并及复姓 clan-prefix prototype；最终新增 delta 中这些高风险 family 为零。
相对上一正式 baseline，default 覆盖增加 380，95.961%→96.256%；assisted 增加 497，
96.401%→96.788%。最终规则 SHA-256 为
`80b5c97a24e07af63668c68f5b8a696665e57c1a3add09769459f161a4232219`。

### 独立证据族组合 admission

新增 `evidence.py` 作为 candidate admission 层。它不把多个相关规则或模型字段简单相加，
而由显式 policy 要求不同 evidence family 同时成立，并让结构 veto 保持绝对优先。
首个 `inherent-title-appointment` policy 要求：

1. `X` 具有固有人物称号后缀；
2. 当前句法是 `以 X 为 Y`；
3. `Y` 属于受控人类角色/官职。

因此 `以太平公主为女官` 的 `太平公主` 可在没有 BIO 与同节重复时 admission；
`遣长公主` 和 `以太平公主为此事` 不成立。歧义 `X王/X后/X公/X侯` 还必须加入完整人物
POS 或 paragraph-local translation exact identity 第四族。全 294 卷 default 与 assisted
delta 均为新增 1、移除 0；新增 span 与 v1 重叠。default gap 4,814→4,813，
assisted gap 4,131→4,130。

`rules_sha256` 从本轮起表示稳定的 rule-bundle hash（`rules.py + evidence.py`），避免只改
组合 policy 却不改变 corpus provenance。当前 prototype hash 为
`88681ccbc4673a25ce92783eafe07912080a5d06c478b507f0eba42bed4cae2f`。

随后加入 candidate lattice prototype：model surface、POS/BIO span 与 translation mapping
可提供候选，`long-repeat-boundary-model` 组合 model-name morphology、同节重复和硬边界。
全量 default 相对上一轮新增 42 个 geometry、移除 2 个短 anaphora geometry；两处移除分别
由完整 `贺拔岳/斛斯椿` 取代。default 新增覆盖 38 个 v1 span，assisted 新增覆盖 7 个。
本轮主要收益来自同一 exact span 上的弱证据联合，而非 fuzzy boundary；后续设计应转向
rule witness 软化。该 prototype 保存用于后续对照，不视为最终 cutover。

### 称号与庙号 schema

`title_appellation` 统一处理语法化君主称号、同节局部引入的庙号简称、明确尾衔称号
组件和正式封谥称号。它的 Step-1 admission 不读取人物 KB；人物 KB 只留给 Step 2
identity binding。原独立审计的 16 条 title gap 已关闭 **16/16**，包括 `始皇`、`二世`、
`主父`、`世宗`、`太穆`、`日逐`、`太和`、`天亲`、`宣简` 和 `梁孝王`。

相对 fullname-anchor assisted 输出，本轮新增 465 个 geometry、移除 120 个、净增 345；
v1 gap 减少 56。457 个新增 geometry 由 `title_appellation` 产生。最终 294 个不与 v1
重叠的新增称号已纳入上述逐条审计；120 个移除中 118 个有替代 span，另外 2 个没有替代。

全量审计专门拒绝了 `禁锢二世` 的代数义、`秦二世即亡` 的世代义、`其武侯岭` 的
地名片段、`中山简王` 内的错误 `[山简]王`、`乐成靖王` 内的错误 `[成靖王]`，
以及政权/部族 `突厥/柔然/匈奴` 等被误作称号组件。短称号也不会抢占 `主父偃`；
`天亲/宣简` 则按称号组件政策保留。

### 同节 anchor / POS / translation gate 收紧

本轮按顺序处理三个 residual family：

1. 已 admission 的更早 card 可派生唯一的 1–2 字 suffix handle；二字 handle 要求
   当前 POS/token 完整性和局部人物句法，单字 handle 还必须有 POS·Giv 和受控句法 frame。
2. BIO 把 `胜非、寄父、弘遂、陶公` 等误并为更长 entity 时，只在局部语法能够证明
   单字 handle 独立时穿透；完整姓名、称号 continuation 和外族名内部切尾仍优先拒绝。
3. translation exact/given candidate 先合并检查同起点 owner 唯一性；新的无 POS admission
   只允许 paragraph-local strict owner 下的 `为 X 所` 或句界后人物谓词 frame。T6 exact
   fullname gate 未放宽。

本节记录的是移除全局人物 KB 之前的 close-123 历史阶段；其当时结果不再是当前 baseline。

中间 prototype 曾包含审计样例驱动的 surface 黑名单和短语例外；这些已全部删除。
当前新增路径不查询全局人物 KB，不使用 stop-word admission，也不列举人物名、handle
或反例 surface。它只读取同节已 admission card、POS/BIO、token 词类/置信度、句法类别、
爵位 continuation 和 paragraph-local translation owner。`温王/子路/倾耳/武艺/侯史吴`
只作为回归断言，不出现在规则实现中。

当前保证已经扩展到整个 `rules.py` Agent 1：人物 surface KB、identity-derived uniqueness、
`STOP_GIVEN` 与全局 surface blacklist 引用均为零。此声明不包含仍在并行存在的旧生产
pipeline，也不限制 Agent 2 使用人物 KB。

### 时序行政区证据

全 294 卷正文生成 5,438 个高置信 Geo entity、32 个 fallback dynasty-period、21 个
fallback surface。当前位置的完整高置信 Geo 可直接证明籍贯左边界；POS 失败时，fallback
只在同纪至少两个不同全名支持、且当前 CE 年份确实有证据时生效。注文不参加挖掘，也不把
首次至末次出现推断成连续有效期。

相对上一正式阶段，新增规则恢复 135 个 v1 span（alias +69、anaphora +66），lost 0。
新增的 159 个全名锚点中，158 个来自当前位置 Geo，1 个来自时序 fallback
（`陈国何夔`）；91 个锚点不与 v1 重叠。159 个锚点及其 142 个新增同节省称已逐条审查，
均为真实人物表达。代表恢复包括：

- `山阳[满宠]……[宠]收治之`
- `泰山[于禁]……[禁]数其罪`
- `平原[祢衡]……[衡]骂辱操`
- `陈国[何夔]……[夔]常蓄毒药`

审查同时发现“新锚点抢先把后续完整姓名切成给名”的优先级问题。现在只有完整
`semantic_given2` 可成立时才让完整姓名优先，保留 `立[曲嘉]为王`，而
`礼、[嘉]还高昌` 仍是省称。该通用修复另将 `义纵/荣晦/俱延/丘堆/度律` 和
`宗均/宗犀/伊愼/牟羽` 九处后续完整姓名从单字尾恢复为二字全名；未改变 span 总数，
v1 lost 仍为 0。

`build_admin_places.py` 的 JSON 不含运行时间戳，同一输入可 byte-for-byte 重建。
`retag.py` 默认先重建该证据，再输出 294 卷不带身份的 occurrence cards；本次正式输出
共 169,422 条，并在 manifest 中记录两个 SHA-256。

### 全名左边界补充

本轮另将两个高置信 `Sur + 完整 BIO-Giv` 左边界逐项跑完 294 卷并审查：

| 规则 | 直接全名锚点 | 下游/替换 span | v1 gained | v1 lost |
|---|---:|---:|---:|---:|
| 编号圈号后全名 | 23 | 38 | 57 | 0 |
| 高置信单字政权后全名 | 28 | 4 | 11 | 0 |

编号圈号只提供边界，不覆盖候选首字本身的歧义否决，因此拒绝
`①韩申不害卒 → [韩申]不害`。政权规则要求前一 token 恰为一个高置信
`NameType=Nat` 单字政权，不能只看前一字符；另拒绝数词限定区域
`三吴[却籍]者`，以及姓名候选后继续构成官职的
`蜀[左匡圣]马步都指挥使`。代表恢复包括
`①[衞鞅]`、`秦[衞鞅]`、`秦[商鞅]`、`秦[白起]`、`吴[潘濬]……[濬]还武昌`。

### 显式人物句法

四项通用句法规则也分别以全 294 卷 stage 审查：

| 规则 | 几何新增 | 几何移除 | v1 gained | v1 lost |
|---|---:|---:|---:|---:|
| 历史窄规则 `臣/吏 + 有 + person + 者` | 4 | 0 | 4 | 0 |
| 通用 presentative `有 + person + 者` | 7 | 0 | 2 | 0 |
| 历史窄规则 `号曰/告/攻/见 + 唯一 X君` | 4 | 0 | 4 | 0 |
| 全局唯一 `X君` + 当前节独立人物句法 | 8 | 0 | 3 | 0 |
| 不限后缀的通用人物称名 | 12 | 0 | 1 | 0 |
| 通用 `名/姓名...曰` 人物命名 | 39 | 1 | 0 | 0 |
| 显式继承/政权公号 | 5 | 0 | 3 | 0 |
| 受控谓词 + person `、` person | 17 | 3 | 14 | 1 |

presentative 不再要求前文恰为 `臣/吏`：全书共有 41 个完整人物形态候选，35 个已被
更高优先级姓名规则覆盖；fallback 新增 `正先、于谨、并韶、史窣干、白可久、禅奴利`
六个真实人物锚点，并由 `正先` 新增一个同节省称。原窄规则恢复的
`檀子、盼子、黔夫、种首` 均保留。

`person_appellation` 不再要求以 `君` 结尾。它先从全书安全定义结构建立称名词典：
称名必须是只属于一个 KB 人物的 alias，并实际出现在 `字/号曰/更名曰/自名曰/更其名曰`
等结构中；`字 + 完整 BIO-Giv` 还允许 KB 暂无该字。人物角色通称和
`尊/庙/国/年/谥 + 号曰` 不进入词典。

每次出现仍须在当前节独立满足以下一种证据：

- `字/号曰/名曰/谓之/称之/称为/号为/名为 + 称名`；
- 左边是句界、转折词或亲属词，右边经可选副词后为高置信动词；
- 人物宾语结构，或 `政权之称名、政权之人物` 对称并列；
- 单字政权 + 称名 + 高置信人物谓词。

泛称 `长君/嗣君/郎君/王君` 不能只靠词典，必须另有完整 BIO-Giv 或更强政权主语结构；
更长 KB 姓名和书名均否决。取消 `君` 限制后的 stage 新增
`犀首、季、大孙、仲、养、叔达、处道、显宗、也咥、上交、孺赟×2`，
12 个均人工确认，无几何移除。此前 8 个 `X君` 新增全部保留。
这不是跨节传播：例如 `而[商君]尤称刻薄` 由全局称名词典和本句
`称名 + ADV + VERB` 共同证明；`孺赟` 的后续出现则只由同节定义锚定。显式 title frame
恢复 `子[嗣君]立`、`齐[简公]`、`宋[景公]` 等。并列宾语要求受控对象引介词，
且顿号两边都必须是 KB 已知、POS 独立证明的完整人物；它恢复
`魏用[犀首]、张仪`、`举[皋陶]、伊尹`、`征[何武]、师丹` 等 17 处。
两个移除是短 span `[永]/[房]` 被完整 `[谷永]/[京房]` 取代；唯一 v1 loss
是已人工确认的假人物 `[魏用]`，其中 `魏` 是政权、`用` 是动词。

`person_naming_definition` 把 `名...曰` 从固定前缀表改为人物目标语法。全书 121 个
含 `名...曰` 的结构中，同时存在大量同形非人物命名（官名、军号、宫殿、器物、谥法）。
新规则只接受可证明的人物目标：`姓名`、`名之`、`名其子/兄子`、人物改名，或已进入
人物命名列表的并列项；右侧仍须完整 Prs/Giv、唯一 KB 人名，或在强人物结构中具有
明确单字边界。完整 294 卷几何审计新增 39 个真实人物名，移除 1 个被完整 `存贤`
替代的截断 `[贤]`，没有非人物新增。新增包括 `弘、骜、默/沈/浑/深、𡙇/垂、
瑛/琮/玙、存信/存进/存贤/存孝、宗训/宗谨、知诰、元坦`。这些位置均是 v1 欠标，
所以正式 v1 coverage 数值不变；这也说明该 benchmark 不能代替独立 precision 审计。

### 官职全名与功能词省称

本轮针对最高缺口卷中反复出现的两类问题，分别完成了全 294 卷独立 stage：

| 规则 | v1 gained | alias | anaphora | 几何新增 | 几何移除 |
|---|---:|---:|---:|---:|---:|
| `pos_known_fullname_appos` | 168 | 67 | 101 | 254 | 81 |
| 严格功能词单字 handle | 36 | 25 | 11 | 68 | 0 |
| **合计** | **204** | **92** | **112** | **322** | **81** |

`pos_known_fullname_appos` 要求候选是高置信 POS 姓氏 + 完整给名形态、属于已知人物
surface，并且左侧是人物身份、官职或人物选择谓词。它以 fallback 优先级运行，避免把
约 15,000 个已经由成熟规则识别的 span 改写 provenance。81 个移除主要是边界修正，
例如 `[刘敬往] → [刘敬]`、`[罗尚求] → [罗尚]`、`[阎式诣] → [阎式]`。

功能词规则只处理已有更早同节全名锚点的单字 handle，并要求该 handle 在当前节只有一个
全名来源。准入结构限于 `劝/从/遣 + handle + 高置信谓词`，或严格句界后的
`handle + 人物谓词`；`当` 等模型高置信功能词另有否决。曾测试的宽泛“锚点 +
主语谓词”版本新增 332 个 span，但含大量 `非/不/何/遂` 假人物，已完全撤销。最终阶段
32 个不在 v1 中的新增逐条审查，均为真实人物省称。

本轮按规则逐项跑完 294 卷并审查 delta：身份词 + 完整 BIO 姓名、高置信
`Sur + BIO-Giv` 全名、card-local POS handle、POS 证明的三字全名 handle、完整 token
功能词冲突否决，以及人物 + `庙/墓/祠/柩/第` 领属结构。相对上一正式基线，v1 coverage
增加 1,216（alias +342、anaphora +874），没有用卷级或跨节 roster。

完整 POS 冲突否决净移除 340 个 span；主要是 `甲子`、`必欲`、`自为`、`莫相` 等模型以
高置信功能词解释的同形误标。它失去的 7 个 v1 span 经逐条检查均为普通词。人物领属规则
先用完整 `NameType=Prs` 或 `Sur + BIO-Giv` 证明直接姓名；低 POS 的双字省称仍须有更早
同节锚点。因此恢复 `比干庙/比干墓/崇训墓`，而不把 `承宗庙`、`高第`、`公卿第舍`
一类普通结构纳入。

随后新增三项高缺口锚点规则，每项均独立跑完 294 卷并与前一阶段比较：

| 规则 | v1 gained | v1 lost | 代表恢复 |
|---|---:|---:|---|
| 政权主号 handle | 239 | 0 | `燕主垂 → 垂`、`秦主登 → 登`、`魏主嗣 → 嗣` |
| 完整 Geo 封地 + 爵 + 姓名 | 112 | 0 | `淮南王生 → 生`、`临淄王隆基 → 隆基` |
| 行政区介绍 + POS 全名 | 34 | 0 | `东郡京房 → 房`、`蜀郡何武 → 武` |

三项合计相对上一正式结果增加 385 个 v1 span（alias +41、anaphora +324、
feng +20），无兼容性回退。多字封地规则要求完整高置信 Geo BIO 实体、受控
`王/公/侯`、高置信完整 BIO-Giv，并让已有姓氏全名规则优先。行政区规则已由本页上方的
时序三层证据取代旧的“必须以行政后缀结尾”限制。

POS cache 已全部升级到 v3：294 卷保存 3,126,326 个完整 token 的原文、半开 offset、
UPOS、完整 tag、BIO、置信度、句界、模型 revision 和正文 hash；旧 `gset/giv_spans`
由这些 token 派生。只切换 v2 → v3、不修改规则时，正式 benchmark 与上一基线完全一致，
证明 cache 迁移本身没有改变输出。

用当前规则分别运行真实 BIO span 和旧 cache 的连续-offset 推导 span，v1 coverage 均为
123,065。BIO 仅改变 2 个谱系边界：`宽信代贤 → 宽信`、`莫离支任武 → 任武`，均修正了
旧 offset 合并造成的过长 span；其余拆开的外族名由相邻 BIO span + 右语法边界恢复。
`foreign_suffix_name` 另以局部形态规则补全 `拓跋沙漠汗`：`沙漠` 是 BIO-Giv，`汗` 是
外族姓名后缀；规则不查询人物 KB，也不使用固定姓名长度。

本轮省称审查新增四项同节局部规则：单字候选只在下一字属于**同一 BIO entity** 时拒绝；
同节锚点可由 `遣/命/引/闻/问/怒/使/从/纳/听/拒/救` 等人物主语谓词补足无 POS 场景；
官职 + 已知全名（`司马班超`、`军司马班勇`）优先建立正确锚点；已存在的
`高阳王隆/范阳王德/湘东王彧` 等 title-person span 可贡献爵号后的给名 handle。后者新增
的大量 non-v1 省称经样例审查多为 v1 欠标的真实人物回指；同时拒绝 handle 后紧接
`王/公/侯`，避免 `建[信]侯` 一类 title 内部误切。

完整 POS 接入后，省称规则以 token span、UPOS 和置信度校验无 Giv 的句法恢复，不再按
具体字符特判 `云/可/当`。例如 `可=AUX(0.991)`、`当=ADV(0.945)` 且后续
`从=ADP` 时拒绝；`敢=AUX` 但局部结构为 `敢从上` 时由人物跟随句法保留；低置信度
Giv 后直接接引号时拒绝。四项规则的 counterfactual precision audit 结果：

| 规则 | 全量新增 | v1 overlap | 人工审查 |
|---|---:|---:|---:|
| BIO entity boundary | 39 | 33 | 39 / 39 |
| person-subject predicates | 93 | 52 | 93 / 93 |
| office + full name | 103 | 93 | 103 / 103 |
| title-person handle | 1,451 | 61 | 确定性抽样 300 / 300 |

前三项逐条全审；title-person 使用按 span key 的固定 SHA-256 顺序抽取 300 条，因而可重复。
共审查 535 条，未发现剩余 false positive。该人工结果仍不是独立全语料 gold precision。
