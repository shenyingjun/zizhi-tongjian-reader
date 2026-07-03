# LLM 人名标注缓存（durable annotation cache）

> ## ⚠️ 重要指令 — 永不删除（NEVER DELETE）
>
> **这些 `juan_*.jsonl` 缓存文件是全书 294 卷一次性人工 + LLM 校补的成果，绝对不能删除、重置或覆盖。**
>
> - **永不删除** `data-pipeline/persons/llm_annotations/juan_*.jsonl` —— 任何清理、重构、重命名都必须保留它们。
> - **永不重跑 LLM** 去重新生成缓存。全书 294 卷已完成（001–294 无缺口），重跑既昂贵又会丢失已校对的人工标注。
> - 若流水线格式变化，**迁移**这些文件，不要重建。
> - 这条指令是永久性的，除非仓库所有者明确书面要求，否则任何 agent / 脚本都不得移除本文件或缓存。
>
> _Rationale: this cache exists specifically to avoid re-running the LLM again and again. Deleting it destroys hand-verified annotations across all 294 卷 and forces an expensive, lossy re-run._

这个目录是**流水线与 LLM 的混合层**。目标：LLM 对全书人名做**一次性**校补，
结果写成缓存，`build_persons.py` 每次构建时直接读取，**永不重复调用 LLM**。

## 为什么这样设计

卷251 的 pipeline-vs-LLM 对照实验证明：确定性流水线与 LLM 在**精度**上等同
（≈100%，0 误标），LLM 只在**召回**上多出约 18 个百分点——都是官衔连写、
NER 漏掉的孤例人名（如 元密 8 次、卢望回）。因此混合策略是：

- **LLM 只负责“发现人名 + 声明本卷省称”**（detection）。
- **偏移量、指代枚举、消歧仍由流水线确定性完成**（offsets / anaphora）。

LLM 给出人名表面形（及别名），流水线在该卷正文里精确定位。LLM 从不产生坐标。

## 文件格式（JSONL，v3 — FULL CAST）

每卷一个文件 `juan_NNN.jsonl`（NNN = 三位卷号，如 `juan_253.jsonl`）。
UTF-8，**一行一个 JSON 对象**（一个人），`#` 开头的行是注释。

**v3 起，每个文件是本卷的“全体人物名单”（full cast）**，而不仅是 LLM 相对流水线的
增量。这样缓存本身即是**自洽的模型级 ground truth**：不依赖流水线当前状态，流水线
改动也不会使它失效，且可直接据它测流水线的精度/召回（漏标 = 在名单但流水线没有；
误标 = 反之）。

```json
{"name":"康道伟","aliases":["道伟"],"role":"高品·宦官使者","confidence":"high","carded":false,"evidence":"遣高品康道伟赍敕书抚慰之"}
```

字段：

- **name**（必填）：2–4 汉字的**完整姓名**，必须以已知姓氏 / 复姓开头。只写全名，
  不写官职、不写称号。→ 成为卡片 `canonical_name`。
- **aliases**（可选）：该人在**本卷**出现的省称 / 别名（如 道伟→康道伟、汉璋→柳汉璋）。
  → 仅注册进**本卷** `RULES`，指向同一张卡。别名是姓氏前缀规则的豁免项（可不以姓氏开头），
  但仍须在本卷正文出现、且不与其它卡冲突。**per-卷 作用域**：省称只在声明它的那一卷生效
  （勋=庞勋 只在该语境成立）。
- **role**（可选）：官职 / 身份。→ 写进卡片 `brief`/`identity`，替代泛泛的“见于卷NNN”。
- **confidence**：`high` / `med`（目前只采纳 high；med 供人工复核）。
- **carded**（v3 新增）：`true` = 流水线本卷已识别此人（名单的既有部分）；
  `false` = LLM 新增、流水线漏掉。构建时忽略此字段，仅用于审计与混合分析。
- **evidence**（可选）：正文上下文片段，便于审计。

> 兼容性：`build_persons.py` 的加载器忽略 `carded` 及任何未知字段，且 LLM 召回层
> 会跳过已建卡的名字，所以 full-cast 文件对构建是 drop-in 兼容的——只有 `carded:false`
> 的新增项会真正建卡。

## veto 记录（JSONL，v4 — 精度否决层）

除人物行外，文件可含 **veto 记录**，用于抑制流水线在**本卷**误标的表面串（非人名的
文言词/官衔·动词边界截断/部族名等）。这是**只删不增**的精度手段，作用域限于声明它的
那一卷——LLM 从未审计的卷不受影响，故"宁缺勿错"的精度承诺由构造保证。

```json
{"type":"veto","surface":"胡可","reason":"文言「岂能」，非人名"}
```

字段：

- **type**（必填）：固定为 `"veto"`。（等价写法：任意行带 `"veto": true`。）
- **surface**（必填）：要在**本卷**抑制的表面串。构建时该串的**所有** mention
  （不论 alias/anaphora/role/gloss/feng 或胡注）都被丢弃；若某卡因此在全书零登场，
  该卡不再进入 `people.json`。
- **reason**（可选）：判为非人名 / 边界错的依据，便于审计。

> 消费点：`build_persons.py` 的 `_load_llm_veto(juan)` 读取本表；人物行的
> `_load_llm_ann` 铸卡加载器**忽略** veto 行（无 `name`），故二者可同文件共存。
> 边界截断（孙武→孙武开、石真若→石真若留）在本层先 veto 掉截断串，正确的完整名
> 由后续 **binding** 层（v4 recall）重新绑定。

## binding 记录（JSONL，v4 — 召回绑定层）

除人物行与 veto 行外，文件可含 **binding 记录**，把确定性扫描漏掉的**封号 / 官职 /
省称**表面串（吴王、辽西王农、文泰）在**本卷**登记为它所指的真实姓名（canonical）。
偏移仍由流水线确定：构建时扫描 `surface` 命中并逐条校验 `text[start:end]==surface`，
LLM 只负责“表面串→人物”这一映射。这样能召回**只以封号出现、正文从不写本名**的人物
（吴王→刘濞，刘濞在卷016从未字面出现）与语境省称（文泰→曲文泰）。

```json
{"type":"binding","surface":"吴王","canonical":"刘濞","dynasty":"汉","role":"吴王，七国之乱首","evidence":"para6-31 皆指刘濞"}
{"type":"binding","surface":"赵王","canonical":"刘遂","dynasty":"汉","role":"赵王，七国之一","para_range":[7,26],"evidence":"赵王有罪…赵王引兵还邯郸"}
```

字段：

- **type**（必填）：固定为 `"binding"`。
- **surface**（必填）：要在**本卷**登记的封号/官职/省称串。必须在本卷正文出现，
  否则跳过（精度优先）。同一卷若该串已被占用则不覆盖（`setdefault`，无冲突）。
- **canonical**（必填）：该串所指人物的真实姓名。优先解析到既有卡；否则要求以真实
  姓/复姓打头且为 2–4 汉字时**新建**卡（`见于卷NNN（LLM 校补）` 简介，dynasty/role
  取自本记录或卷元数据）。canonical 同时被登记为卷内表面串，使 rc4 titleglue 能一致
  地绑定「封号+名」诸形（范阳王德/琅邪王德/阳平王德→慕容德），不受目标卡是否有生卒
  年影响。
- **dynasty / role**（可选）：新建卡时写入朝代与一句话身份（书内依据）。
- **para_range**（可选）：`[lo,hi]`（含端点，段 id 即 0 基段序）。用于**同卷轮换的
  封号**——赵王在七国之乱诸段指刘遂，景帝子受封后诸段指刘彭祖；胶西王卬（段7–25）与
  刘端（段31）分属两人。带 `para_range` 的绑定只在该段窗口内生效（emission 段级 overlay），
  不做整卷登记。

### 单字省称 → 省称/anaphora 通道（精度关键）

`surface` 为**单个汉字**的 binding（如 `卬→刘卬`、`戣→孔戣`）**不**走整卷别名登记，而是
喂入**受门控的 anaphora 通道**：构建时把该字登记为本卷可用省称（`minted_admit`）并把它锚
定到 LLM 指定的人物（`minted_anchor` + `llm_anchor`），再由确定性 anaphora pass 逐处判定
是否绑定。原因是单字极易撞词——`遂`多为副词“于是”、`农`见于弘农/务农、`隆`见于姚熙隆——
盲目整卷别名会是精度灾难。省称门控（antecedent 存在、左邻非姓、`COMMON_BIGRAMS`、须有人称
语境 `_ANAPHORA_RIGHT/LEFT`/亲属、生卒在范围）仍逐处把关；LLM 只提供“此字→此人”的锚点。
故只 seed **干净的**单字（卬→刘卬），刻意**不** seed 遂/农/隆。

> 单字锚点为**整卷权威**：不同于 gloss 家世锚点会被分节标记（①②…）逐节清空，LLM 单字
> 省称锚点在每次分节重置后**重新注入**（`llm_anchor`），故 016 卬 在跨节的 p25「卬等」
> 仍能绑定到刘卬。

## card 记录（JSONL，v4 — 卡片修缮层）

除以上各类外，文件可含 **card 记录**，对既有卡做**仅元数据**的修缮（不动任何 span/偏移，
故精度承诺不受影响）。对应审计的三类卡片问题：朝代误标、占位简介、同人多卡。

```json
{"type":"card","canonical":"慕容德","dynasty":"后燕","brief":"后燕宗室，慕容垂弟，封范阳王，后自立为南燕主。"}
{"type":"card","canonical":"魏其侯","merge_into":"窦婴","evidence":"魏其侯窦婴，同一人"}
```

字段：

- **type**（必填）：固定为 `"card"`。
- **canonical**（必填）：目标卡的真实姓名，按名解析到离本卷最近的一张卡。
- **dynasty**（可选）：朝代重标（十六国实为后燕/后秦者初见于晋卷会被误标为晋；拓跋珪→北魏）。
  同时刷新 `era_hint`。
- **brief**（可选）：**书内一句话**简介，替换 `见于卷NNN` 占位（须为通鉴叙事事实，非维基文本）。
- **merge_into**（可选）：把本卡并入 `merge_into` 命名的幸存卡（`_merge_xref_card`：合并
  juans/names/match、把 RULES 表面串改指幸存 pid、清理 `given_of`/`canon_to_pids`/锚点）。
  **仅用于证据确凿的异名同人**（魏其侯≡窦婴、万纪≡权万纪省姓）；跨代同名（姚兴@108 后秦 vs
  @162 梁、袁盎跨魏宋）为**同形异人**，保持分立。

> 消费点：`build_persons.py` 的 `_load_llm_card(juan)`；在 xref 合并之后、emission 之前
> apply（先并卡再改朝代/简介，使幸存卡承接全部下游 mention）。铸卡加载器忽略 card 行（无 `name`）。
- **para_range**（可选）：`[lo,hi]` 闭区间段落 id。用于**卷内改封**的封号（赵王在
  前段=刘遂、改封段后=刘彭祖；胶西王=刘卬 而非后段的刘端）。带此字段者不进整卷 RULES，
  而登记进 `_PARA_BINDINGS`，由发射循环按段落 id 做范围叠加，越界绝不误绑（精度优先）。
- **evidence**（可选）：书内佐证，便于审计。

> 消费点：`build_persons.py` 的 `_load_llm_binding(juan)`（在 `build_gloss_cards`
> 内消费）。整卷绑定写入 `RULES[juan]`，段落绑定写入 `_PARA_BINDINGS[juan]`。人物行
> 的 `_load_llm_ann` 忽略 binding 行（无 `name`），三类记录（person / veto / binding）
> 同文件共存。发射统计见构建日志 `llm recall-binding: N whole-卷 + M para-scoped …`。

## 构建时如何被消费（build_persons.py 的护栏）

`build_gloss_cards` 里的 **LLM-annotation recall tier** 通过 `_load_llm_ann(juan)`
读取本目录（优先 JSONL，回退旧版 TSV）。每条断言：

1. **name** 须 2–4 汉字、以 `SURNAMES | AMBIGUOUS_SURNAMES | LLM_EXTRA_SURNAMES` 或
   复姓开头；通过全部非人名护栏（`_strong_ok`：未被占用 / 非常用词 / 非官衔）；
   且**在本卷正文出现**（precision-first）才建卡。
2. **aliases** 逐个校验：在本卷出现、未被 `canon_to_pids` 或其它新卡占用、非常用词，
   才注册进 `RULES[juan]`（`setdefault` 保持 collision-free，不覆盖既有映射）。
   ≥2 字的省称由别名扫描直接标注（如 汉璋→柳汉璋）；Wave 44 前置门控保证省称
   不会抢在全名之前触发。

生成的卡片 brief 标注「（LLM 校补）」，可与程序自动卡区分。

## 如何一次性生成全书缓存

主用法是**在会话内**由 LLM 直接逐卷标注（$0）。批量 API 方案见同目录
`../run_llm_pass.py`（OpenAI 兼容 endpoint，断点续跑，全书一次约 $40–70）。

当前已种子化（v3 full cast）：`juan_250` `juan_251` `juan_252` `juan_253`
`juan_254` `juan_255`。回填/合并工具见同目录 `_fullcast.py`。
