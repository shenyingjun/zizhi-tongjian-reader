# 两阶段人物下划线规范

> 状态：`twostage/` 目标架构规范。当前仍与旧生产管线并存；本文件定义正在验证、
> 将来替代旧管线的语义。实现以 `rules.py`、`tagger.py`、`identify.py` 和本规范为准。
> 可重复运行方法和最新结果见 [`BENCHMARK.md`](BENCHMARK.md)。

## 1. 目标与基本原则

人物处理分成两个独立问题：

1. **Agent 1 / Tagger：哪里应当画人物下划线。**
2. **Agent 2 / Identifier：这个下划线指向哪个人物。**

二者不得混在同一条规则中。Agent 1 可以在暂时不知道人物身份时画线；Agent 2
可以延后合并身份，但不能删除 Agent 1 已确认的人物提及。

总原则：

- **规则优先，不做实例补丁。** `赵王虎`、`大将军光`、`弟亮`只是验证例；
  实现必须覆盖对应的通用结构。
- **语义证据优先，不以频次准入。** 姓名结构、官职、爵称、亲属关系、同节先行
  锚点和句法角色决定是否画线；出现次数只用于排序、审计和选择人工样本。
- **精确率优先。** 缺一条线优于把普通词错误画成人物。
- **Step 1 不读人物 KB。** Agent 1 只使用原文、POS/BIO、model-derived NER、局部
  translation evidence 和非 identity 的语言学类别；`people.json` 只属于 Agent 2。
- **可解释。** 每个 occurrence 必须记录 `rule`、`scope` 和跨度来源。
- **不剧透。** 展示层仍按当前位置裁剪人物经历；本规范不改变该约束。

## 2. 术语

### 2.1 文本层级

| 术语 | 定义 |
|---|---|
| **全书 / corpus** | 全 294 卷文本及非 identity 的语言学/模型证据。 |
| **卷 / juan** | 原书卷次。可作为年代、文件和批处理单位，但**不是省称解析作用域**。 |
| **段落 / paragraph** | JSON `paragraphs[]` 中的一项，拥有独立 `id` 和局部字符偏移。 |
| **节 / jie** | 一个带圈编号段落（①②…）开始，加上其后所有未编号段落，直到下一个带圈编号。一个节包含一个或多个 paragraph。 |

带圈编号会在新的纪年重新从①开始，因此节的先后关系按正文顺序判断，不能把
“节号数字”当作卷内全局编号。

### 2.2 标注与身份

| 术语 | 定义 |
|---|---|
| **span** | 原文中的字符区间 `[start, end)`。 |
| **Occurrence Card** | Agent 1 输出的一次人物提及；包含 span、surface、rule、scope，但可没有 person id。 |
| **Person Card** | Agent 2 合并后的身份实体，可聚合多个 occurrence。 |
| **全名锚点** | 可证明某个给名属于人物的完整姓名，如 `曹操`、`诸葛亮`。 |
| **全名等价锚点** | 虽非姓+名，但局部语法已明确指向一个人物，如 `赵王虎`、`大将军光`、`左将军桀`、`弟亮`。 |
| **给名 handle** | 从锚点抽出的后续省称形式，如 `赵王虎 → 虎`、`诸葛亮 → 亮`。 |
| **省称 / anaphora** | 已有先行人物锚点后，仅以给名再次出现，如 `曹操……操曰`。 |
| **回指解析** | 为省称寻找更早的锚点。当前 Agent 1 只做确定性的**同节向后解析**。 |

## 3. 作用域约束

每条 Agent 1 规则只能具有以下作用域之一：

- **corpus**：只允许使用姓氏、官职、亲属、谓词、爵位等语言学类别或模型生成的
  identity-free NER evidence；不得使用人物 surface/identity KB。
- **jie**：使用当前编号节内的结构、POS、谱系和先后关系。

禁止：

- 以整个卷建立省称 roster；
- 从上一节继承省称锚点；
- 用后文全名反向绑定前文的较短省称；同节同形完整 surface 的 exact propagation
  不是身份回指，可使用该 surface 在节内任一位置的可信 occurrence；
- 以 `people.json` 中是否存在某个 surface 或 identity 决定是否画线。

离线译文 NER 是**最终输出必须加载的补充 identity evidence**。不加载译文的输出只用于
规则消融、定位增量和调试，不作为正式结果，也不单独汇报覆盖指标。译文 evidence 须满足：

- evidence 只在其 canonical paragraph 内生效，不跨 paragraph、节或卷；
- exact full-identity mapping 只有在姓名边界和 title guards 全部通过时才可输出
  `translation_fullname`；通常要求完整原文人物 POS/BIO。若 POS 失败，只允许完整 token
  边界下由同节另一 occurrence 的人物 NameType 或称号形态授权，并应用干支、官职、地点/
  族群、军镇 continuation、polity frame 和现代 NER 尾部过伸 guards；
- mapped given 只有在精确 occurrence offset、唯一 owner 和受控原文人物句法全部通过时，
  才可输出 `translation_anaphora`；
- 获准的 `translation_fullname` 在普通 anaphora postpass 前成为 anchor；其 handle 必须
  是完整 given，不能把 `綦公顺`、`高文集` 错拆成 `顺`、`集`；
- `flagged_partial_person_span`、POS 冲突等风险 mapping 不得授权 anchor；
- loader 必须以 `manifest.json` 中的逐卷 SHA-256 校验 evidence 文件，不能只记录
  manifest hash 而消费未经认证的卷文件；
- 译文未识别人物绝不能删除或改变原文已有 card；
- assisted 输出必须单调保留所有未与新 span 重叠的原文 card；若新 span 被原文完整 card
  严格包含，保留原文完整 card，禁止 suffix 截断；
- 未显式传入 evidence 时，原文规则输出必须保持确定性，但该路径不是最终生产输出。

`detect_juan()` 可以按卷读取文件，但必须先通过 `_blocks_of()` 划分节，再将同节
paragraph 用硬边界 `"\n"` 拼接后运行规则。输出时恢复到原 paragraph id 和局部偏移。

## 4. Agent 1：人物 span 规则

规则共享一个 `consumed[]`。高优先级规则先占用 span，低优先级规则不得重叠覆盖。
除此之外，`combined_evidence` postpass 可先生成未占用 candidate，再由多个**独立证据族**
共同 admission。它不按规则命中次数或模型分数相加：同一模型派生的 BIO、POS、NER
属于相关证据，每个 family 最多贡献一次。严格 policy 继续要求显式 signal 组合；累计
policy 则对 family-level support 加权并扣除 soft conflict penalty。两者都要求任一 hard
veto 优先于全部正向证据。输出 card 保存 `evidence_policy`、`evidence_families` 和
`evidence_signals`，使组合决策可逐条复核。

### 4.1 姓名与模型/结构规则

| 规则 | 作用域 | 证据 | 输出示例 |
|---|---|---|---|
| `corpus_lit3` | jie | 兼容 rule id；执行内容为长度≥3的完整高置信人物 POS/BIO span | `诸葛亮` |
| `corpus_xing2` | jie | 兼容 rule id；执行内容为完整二字 `Sur + Giv` POS/BIO span | `曹操` |
| `model_ner_name` | jie | model-derived NER candidate + 当前完整人物 POS，或完整 Giv + 严格人物句法；不含 identity | `魏斯` |
| `model_ner_predicate` | jie | model NER + 首尾完整、连续、高置信 `PROPN·NameType` tokens + 严格人物谓词；拒绝官职 surface | `信曰` 中的 `信` |
| `model_ner_title` | jie | model NER + 完整姓名/称号 token 边界 + 人物称号后缀 + 句界或严格人物句法 | `田公`、`郭太后` |
| `model_ner_given` | jie | model NER + 完整多字 BIO-Giv span + 姓名硬边界 | `统叶护` |
| `model_ner_appos` | jie | model NER + 首尾完整、连续、高置信 `PROPN·NameType` tokens + 官职/人物角色/并列左边界 | `刺史张氏` 中的 `张氏` |
| `model_ner_partial_pos` | jie | model NER + 严格人物句法 + 完整 token 边界；首 token 保留 NameType、末 token 为 Giv/Prs，并由可靠姓首或同节 exact repeat 支持；拒绝功能/动词、官职/角色/亲属前缀、右侧姓名延续、多姓首及 reporting-verb 尾部 | `昭奚恤`、`逸豆归` |
| `model_ner_short_royal_title` | jie | 完整二字 model-NER `X王/X后` + 独立 `PROPN + NOUN` tokens + 重复/人物句法/硬边界；拒绝左侧更长姓名和右侧 continuation | `项王`、`韦后` |
| `model_ner_local_surname_name` | jie | 当前严格人物句法中的完整二字姓氏头 model-NER name；同节另有姓首 token 证明，拒绝政权/地名、clan/title continuation 和更长姓名 | `盖吴` |
| `princess_title` | jie | 完整 BIO 称号组件 + 独立 NOUN tokens `公 + 主`；可补齐一个被 POS 错分但由后续 `I` 证明的首字；无 BIO component 时仅接纳同节重复的完整二字 component，并拒绝动词吞并；不查 NER/人物 KB | `太平公主`、`安乐公主`、`全公主` |
| `pos_given_local_frame` | jie | 最后 fallback：完整高置信 POS-Giv + 严格当前人物句法；在 anaphora 后执行，并拒绝姓氏后切尾、右侧姓名 continuation 和可扩展 model NER 人名 | `昌为人`、`召拜式为中郎` 中的省称 |
| `local_title_anchor` | jie | model NER 称号 + 完整 token 边界 + 当前人物谓词；由同节更早高置信 card 授权，或由更早同形文本 + 完整 PROPN 称号组件 + 更严格当前句法授权；拒绝功能/数量组件、姓名切尾和连续爵号 | `赵襄子…襄子弗与`、`沛公…沛公引兵` |
| `local_exact_surface` | jie | 同节任一可信同形姓名 card + 当前 model NER、完整 token/姓名硬边界及人物 NameType signal；较长的已接纳姓氏全名还可贡献完整双字 handle；拒绝称号来源、政权/族群 frame、嵌入更长姓名、动词/功能结构和地理 continuation | 同节重复的完整姓名或双字 given |
| `local_exact_title` | jie | 同节任一可信同形称号 card + 当前完整 PROPN 称号组件和边界；拒绝高歧义 `子` fallback、连续爵号、官职及嵌套姓名 | 同节重复的 `X公/X王/X君/X侯` |
| `block_appos` | jie | BLOCK 称谓/关系 + 完整人物 POS + 右边界 | `平阳侯曹参` 中的 `曹参` |
| `struct_fuxing` | jie | 复姓 + POS·Giv | `慕容垂` |
| `struct_xingming` | jie | 可靠单姓 + POS·Giv | 新发现姓名 |
| `pos_fullname` | jie | 高置信 `PROPN·Sur` + 完整 BIO-Giv + 人物边界；边界可由圈号、严格 POS 政权或当前 Geo/时序行政区证明 | `①衞鞅`、`秦商鞅`、`东郡京房`、`山阳满宠` |
| `known_fullname_pos` | jie | 兼容 rule id；完整连续的高置信 `Sur/Giv/Prs` span，并保留称号延伸和时序地名护栏 | `帝以满宠都督扬州` |
| `role_bio_name` | jie | 身份词 + 完整 BIO-Giv；身份词留在线外 | `胡僧[慧范]`、`尼[法静]` |
| `corpus_given2` | jie | 兼容 rule id；完整双字 POS·Giv + 姓名边界 | `道济` |
| `semantic_given2` | jie | 完整双字 POS·Giv + 人物句法框架 | `劝望之`、`为师道所亲信`、`与望之有隙` |
| `genealogy_given` | jie | 明确谱系前缀 + 连续 POS·Giv 姓名 + 右语法边界；不查 KB | `弟亮`、`其子仁果` 中的姓名 |
| `foreign_title_name` | jie | 完整 1–3 字 BIO component + `可汗/单于`；紧邻完整人物 BIO 时合并 title+name | `佗钵可汗`、`柔然可汗阿那瓌` |
| `foreign_suffix_name` | jie | 复姓 + BIO-Giv + 外族姓名后缀 + 右语法边界；不查 KB | `拓跋沙漠汗` |
| `royal_title_name` | jie | `太子/世子/皇子/王子` + 完整双字 POS·Giv | `太子承乾` |
| `surname_honorific` | jie | 完整姓氏 morphology + `公/君/侯/卿/郎` + 人物谓词、称呼或同位语 | `张公`、`荀卿`、`萧郎` |
| `female_court_title` | jie | 严格命名/人物 frame 下的 `X夫人`，或姓氏 + 一字 appellation + `妃` | `华阳夫人`、`萧淑妃` |
| `person_possessive` | jie | 完整人物 POS，或先行同节双字 handle + `庙/墓/祠/柩/第` | `比干庙`、`崇训墓` |
| `pos_known_fullname_appos` | jie | fallback：高置信 `Sur + 完整 Giv`，且左侧为人物身份、官职或人物选择谓词 | `司徒刘敬`、`刺史罗尚` |
| `multifief_jue_name` | jie | 完整 Geo BIO 封地 + `王/公/侯` + 高置信完整 BIO-Giv | `淮南王生`、`临淄王隆基` |
| `presentative_person` | jie | fallback：`有 + 完整 Prs/Giv 或 Sur+Giv + 者`，左侧领域/领属短语不限词 | `有于谨者`、`同郡有并韶者` |
| `person_naming_definition` | jie | 通用人物命名语法：`姓名`、`名之`、`名其子/兄子`、人物改名及同一命名列表的并列项；拒绝军号、官名、宫殿、器物和谥法 | `更名曰垂`、`名其兄子曰默，曰沈`、`王茂权名曰宗训` |
| `person_appellation` | jie | `字/号曰/名曰/谓之` 等显式定义 + 完整 POS/BIO appellation | `字季`、`号曰犀首` |
| `explicit_title_frame` | jie | `子 + X君 + 立`，或高置信政权 + POS-Prs 谥号 + `公` | `子嗣君立`、`齐简公` |
| `title_appellation` | jie | 仅用称号形态、当前 POS/句法和同节前文：语法化君主称号；局部引入的庙号简称；明确尾衔前的称号组件；受控政权 + 谥号 + 爵位 | `始皇`、`周世宗…世宗`、`太穆神皇后` 中的 `太穆`、`梁孝王` |
| `coordinated_person_object` | jie | 受控对象引介词 + 两个独立 POS 证明的人物，中间为 `、` | `魏用犀首、张仪` |
| `combined_evidence` | jie | candidate 上的独立证据族按严格组合或累计 family score 联合；同 family 只计一次，soft conflict 显式扣分，hard veto 永远优先 | `以太平公主为女官`、重复省称 `襄子`、Geo-misclassified `安禄山` |

`corpus_xing2/corpus_given2` 是为 benchmark provenance 保留的旧名称，不再表示 corpus
人物词典。它们使用完整 token span 作冲突否决：候选若完全由高置信
功能词组成且没有人物主语句法，则不画线。该规则用于拒绝 `甲子/必欲/自为/莫相`，
而不是把“缺少 POS·Giv”当作普遍否决条件。

组合层目前还定义了两条更严格的歧义短称号 policy：`X王/X后/X公/X侯` 除 appointment
句法与人类角色外，还必须由完整人物 POS 或 paragraph-local translation exact identity
提供第四个独立 family。不能因为两个相关模型字段同时成立就降低该门槛。

candidate lattice 汇集 model surface、POS/BIO span 和可选 translation mapping，但正式
graded policy 只接受 exact model witness geometry。证据状态分为：

- **matched signal**：policy 可计入的正向证据；
- **missing signal**：没有观察到，不自动否决，可由其他独立 family 补偿；
- **soft conflict**：例如当前 occurrence 被标成 Geo/Nat；只有显式列出该 conflict 的
  policy 才能 admission；
- **hard veto**：clan/office continuation、政权用法、数量 continuation、较长姓名延续、
  标点/重叠等；任何票数都不能覆盖。

hard veto 必须描述当前 occurrence，不能因为同一 surface 在本节另一处像政权或族群，就
否决当前人物动作语境。例外仅限本节明确把 exact surface 定义为群体的结构，如
`X数千骑/数万骑`、`谓之 X` 后又募集/集合 `X`，或同时出现 `X将军/X军` 的单位简称。
`部/国/族` 与攻击对象等 polity frame 仍须结合当前 token morphology 和局部政权句法；
当前完整人物 morphology 不能被另一 occurrence 覆盖。

BIO continuation 只有在相邻 token 真正连续时才是 hard veto。标点即使被误标为 `I`，
以及 `将` 等谓词被误标成姓名 `B`，都不能传播 continuation。当前 occurrence 若确实位于
较长 BIO/NER 名称内部仍必须拒绝，例如 `高句丽` 内的 `高句`、`乙咄陆` 内的 `咄陆`；
同节重复的外族全名只在明确的协调 `可汗/单于` 结构中辅助 continuation，不能因
`马通军`、`杨郎何` 等 noisy model surface 截断人物名。

office/title continuation 同样区分姓名与称号组件。`史良娣` 中的 `史良`、`慕容镇军`
中的 `慕容镇`、`武平君畔` 中的 `武平` 是不完整 geometry；`车犂单于`、`始毕可汗`
中的前段则是完整姓名。人物姓名在 `赐/封/拜 + 姓名 + 王/公/侯` 中也可保留；完整人物
称号后的 `长子/长女` 是亲属结构，不是 office continuation。

`cumulative-family-score` 是 family-level 累计 policy，不是 raw rule-count voting：

- exact `model_ner_witness` 是 prerequisite，本身不增加多个模型票；
- 同节 exact surface 的完整人物 morphology majority 与同节已接纳 exact surface 是较强
  family，各权重 2；不得从同卷其他节导入 morphology；
- 同节 recurrence 最多计 1 分；strict/decisive 当前人物句法、
  当前 surname morphology 支持的姓氏形态、称号、谱系各权重 1；
- translation exact identity 权重 2；最终生产路径必须加载经 manifest 校验的
  translation evidence；
- `geo_nat_morphology` 扣 2，`function_morphology` 扣 1，缺少当前人物 morphology 只记为
  missing，不单独否决；
- 总分至少 6，且至少四个不同 support family，才可 admission。

累计 policy 不能传播既有错误 anchor。标点硬边界不算 syntax family；候选若出现前向或
后向政权并列、直接 `X数千骑/数万骑`、`谓之 X` 且又按群体募集/集合、BIO 中段切尾、
官职/称号 continuation，或当前是 `并兴/之道/不备/才能/参用` 等抽象类别 continuation，
均产生 hard veto。由此同一累计门槛仍恢复 `世民/乾归/暮末/恐热` 等当前 occurrence
可独立佐证的短称，同时拒绝 `铁勒/室韦/山棚/文武/罗门`，不使用 surface blacklist。

同一模型派生的 POS、BIO、NER 不算三个独立 family。Agent 1 不生成也不读取同卷
recurrence/morphology surface summary；recurrence、morphology anchor、title/genealogy
anchor、polity/collective veto 都必须在当前节内，并来自同一 exact surface。Geo-conflict
路径通常还要求完整 person morphology 在同节占多数，或当前 occurrence 有 decisive
person syntax。Title witness 必须是 POS-Giv span 紧邻 `可汗/单于/公主`，若 title 后又
紧邻另一个姓名 span，则前段按政权修饰语处理。Genealogy witness 只接受多字亲属引介，
不用单字 `兄/弟` 的宽松 substring。

`long-repeat-boundary-model` 保留为 exact-span 对照：完整 model-name morphology、同节
exact recurrence、长度至少三字及硬边界同时成立。全量实验确认 span containment/shift
没有稳定增益，因此不进入正式 graded admission。

`local_exact_surface/local_exact_title` 只传播完全相同的完整文本跨度，不把两个不同
surface 绑定为同一身份。它们因此可以使用同节后文的同形可信 card；`曹操…操曰` 这类
较短 handle 的回指仍只能由更早 anchor 授权，不能借后文全名反向传播。

### 4.2 全名等价锚点

这些规则不仅画线，还必须向当前节的 roster 提供给名 handle。

#### 爵号/国号 + 给名

形式：

```text
[方位前缀]? + 政权 + 王/公/侯 + 给名
```

例：

- `赵王虎 → 虎`
- `秦王坚 → 坚`
- `魏王操 → 操`

画线覆盖完整人物表达，如 `[赵王虎]`。即使高优先级 POS/NER 规则先占用它，
`_handles_of()` 仍须识别其 title-glued 结构并产出 `虎`。

#### 官职 + 给名

已支持的官职前缀包括：

```text
大将军、骠骑将军、车骑将军、卫/衞将军、
前/后/左/右将军、将军、
大司马、司徒、司空、丞相、相国、太尉、太傅、太师
```

例：

- `[大将军光] → 光`
- `[左将军桀] → 桀`

官职后的给名必须通过 POS·Giv 和姓名尾边界检查，不能把标点吃入姓名。

#### POS 证明的全名 handle

`pos_fullname` 和兼容 rule id `known_fullname_pos` 可直接贡献去掉 POS 姓氏后的
handle。既有 `given2_office` 和三字 `pos_person_name` 只有在 card-local token 明确构成高置信
`Sur + 完整 BIO-Giv` 时，才补充同样 handle；例如 `敬晖 → 晖`、`武三思 → 三思`、
`韦月将 → 月将`。handle 仍只向后作用于同一编号节，不跨节或跨卷传播。

译文 virtual anchor 是更窄的例外：它只在 evidence 所属 paragraph 内为同一个
resolver 提供 owner，绝不继承到同节其他 paragraph。由该路径独立恢复的 card 使用
`chunk_type=translation_anaphora`；安全 admitted 的完整姓名使用
`chunk_type=translation_fullname`。后者可在正常 postpass 中触发普通 `anaphora`，
三类 provenance 必须分开审计。

圈号可作为 `pos_fullname` 的左边界，但不能覆盖候选首字自身的政权/称谓歧义，
因此 `①衞鞅` 成立而 `①韩申不害卒` 不会误切成 `韩申`。政权左边界必须由恰好覆盖
前一字的高置信 `NameType=Nat` token 证明，不能把 `广汉` 等地名尾字当作政权。
数词限定的区域表达（如 `三吴`）不提供该边界；候选后若立即继续为
`马步都指挥使/都指挥使`，候选属于官职修饰语而非姓名。

`政权 + 主 + 给名` 也是受控全名等价锚点，例如 `燕主垂 → 垂`、
`秦主登 → 登`、`魏主嗣 → 嗣`。政权必须来自闭集单字政权名（可带
`北/南/东/西/后` 前缀）或 `契丹`；不能把任意 `X主Y` 当作姓名。

多字封地称谓只在封地是完整 `Case=Loc|NameType=Geo` BIO 实体时成立，后接
`王/公/侯` 和高置信完整 BIO-Giv。若爵号后已经是既有姓氏全名，交给更高优先级姓名
规则，避免改变已证明的姓名边界。无 BIO-Giv 时，仅允许单字名后紧接
`幼/少/长/壮/老/年` 等状态词。该锚点可产生爵号后给名 handle。

行政区介绍采用三层证据，行政区本身不画入人物 span：

1. 姓名前紧邻当前出现的完整、高置信 `Case=Loc|NameType=Geo` BIO 实体；不要求
   地名以行政后缀结尾，因此可识别 `山阳满宠`、`泰山于禁`、`平原祢衡`。
2. 保留原有兼容路径：紧邻带 Geo tag 且以 `郡/州/县/國/国/邑` 结尾的 token。
   该路径不要求 0.9 高分，避免丢失 `蜀郡任叡` 一类较低置信但结构明确的证据。
3. 若当前位置 POS 把行政区误标成姓名，可查询 `admin-places.json` 的时序 fallback。
   fallback 必须在同一纪中至少由两个不同的 POS 证明全名支持，并且只在语料中
   **实际出现过的 CE 年份**生效；不把首末年份之间推断成连续有效期。正文与注文分离，
   注文中的跨朝代沿革不能继承正文事件年份。由此恢复 `陈国何夔`，但不把普通 `柱国`
   当地名。

行政区证据由 `build_admin_places.py` 对全 294 卷确定性重建。`retag.py` 默认先重建它，
再加载当前规则；benchmark 与 retag manifest 均记录该文件的 SHA-256。

#### 亲属称谓 + 给名

当前通用形式包括：

```text
弟/兄/其弟/其兄/从弟/从兄/族弟/族兄 + 姓名
其子/长子/次子/少子/幼子/嫡子/庶子/兄子/从子/嗣子/生子等 + 姓名
其孙/兄孙/弟孙/从孙/族孙等 + 姓名
```

例：

```text
弟[亮]已失身于人
其子[仁果]进围宁州
```

只画人物姓名，不画关系词：`弟亮` 只画 `亮`，`其子仁果` 只画 `仁果`。同一条规则
使用 POS 模型保存的完整 BIO span，不按姓名字符数采用不同 personhood 原则，也不设置
一至三字上限。全名也可由可靠姓氏 + POS·Giv span 构成，因此 `其子宋襄` 即使 POS 只标
`襄`，仍由局部结构画出完整 `[宋襄]`。两类均须后接谓词、并列边界或标点，且标点是不可
跨越的绝对上界；无法证明边界时宁可不标。姓名同时成为当前节的谱系锚点。该规则只依赖
局部 genealogy evidence，不查询全局 person KB，也不调用全局 surface blacklist。

BIO 模型偶尔会把一个外族名拆成相邻实体 span。谱系规则只在前一 span 尚未到达有效右
语法边界时继续到下一 BIO span，因此恢复 `叱支乙拔` 等完整姓名，而不会把
`宽信代贤` 合并。官名 + 姓名的局部结构也要排除，例如 `其子莫离支任武` 只画
`[任武]`。另有受控形态补全：`拓跋 + 沙漠(BIO-Giv) + 汗(NOUN)` 画为
`[拓跋沙漠汗]`；`汗` 是明确外族姓名后缀，不是任意向右猜长。

### 4.3 谱系注释

现有谱系预扫描还包括：

- `X，Y之子/孙/弟/兄/父……也`
- `谥曰X`
- `长/次/季/幼/庶曰X`

谱系结构可以证明 personhood；身份与姓氏补全由 Agent 2 进一步处理。

### 4.4 称谓与封爵

- `role` 识别受控政权称谓，如 `吴主`、`魏主`、`契丹主`。
- `polity_king` 识别 `汉王`、`吴王` 等本身已确立 personhood 的王号；身份留给 Agent 2。
- `jue_name` 识别“政权/封号 + 王公侯 + 给名”。
- `multifief_jue_name` 识别完整多字 Geo 封地后的“王公侯 + 姓名”，并建立同节 handle。
- `corpus_jue2` 是兼容 rule id；当前要求 model-derived NER corroboration、爵位形态、
  人名 POS 和局部人物句法，不再按 identity 数量准入。
- `known_title` 保留为无输出兼容 shim；长称谓由 `title_appellation/model_ner_name`
  依据形态和当前 occurrence 证据处理。
- `title_appellation` 补足四类称号：
  - `始皇/主父/二世` 是受控语法形式，仍要求当前人物主语/宾语句法，拒绝
    `禁锢二世` 这类代数义；
  - 庙号简称必须由同一节更早的 `政权 + X宗/祖` 局部文本引入，如
    `周世宗……世宗`；
  - 两字称号组件必须紧接 `皇后/太后/长公主/可汗/单于/王/后/公/侯`
    等明确尾衔，并由谥号形态、当前人物 POS 或选择谓词证明，如
    `太穆神皇后`、`日逐王`；
  - 受控政权字 + 1–2 字谥号 + `王/公/侯` 构成正式称号，如 `梁孝王`。
  该规则的 Step-1 admission 不读取人物 KB 或 canonical identity；人物 KB 只可在
  Step 2 绑定 identity。这里的受控集合只描述称号语法、政权和明确非人物类别。
  规则同时拒绝高置信地名、政权/部族组件、长外族称号切尾、功能词跨界和更长地名
  内部片段，不抢占完整 `foreign_title_name/model_ner_name`，也不产生给名 handle。
- `foreign_title_name` 接受完整 1–3 字 BIO component + `可汗/单于`。称号后若紧邻
  完整人物 BIO，则合并整个 title+name，如 `柔然可汗阿那瓌`、`突骑施可汗苏禄`；
  envoy designation、功能词 component、部/国/军 continuation 和不完整姓名 continuation
  均拒绝。无后接姓名时产生 component handle；合并形态只产生后接姓名 handle。
- `surname_honorific` 只接受完整姓氏 morphology + `公/君/侯/卿/郎`，并要求当前
  occurrence 的人物谓词、称呼或同位语；`公主`、更长姓名和 office continuation 拒绝。
- `female_court_title` 只接受严格命名/人物 frame 下的 `X夫人`，或完整姓氏 +
  一字 appellation + `妃`；普通 `夫人/妃` 不据此准入。
- `royal_title_name` 将 `太子承乾/世子方等/皇子弗陵` 作为完整 title-glued 姓名；
  `皇太子春秋鼎盛` 中的 `春秋` 是年龄表达，受语义保护而不标。
- `empress_title` 识别 `太后/皇后`；`surname_empress` 只在 surname POS、model-derived
  NER 和局部称谓句法共同成立时识别 `贾后/独孤后`。称谓成立即可画线，不等待身份绑定。

## 5. 同节向后省称规则

省称算法必须严格遵守：

> 若发现一个疑似省称，只向前查找同一节内更早的全名或全名等价锚点；
> 找到则标注锚点和省称，找不到就不标。

具体流程：

1. 对整个节运行谱系预扫描和姓名/称谓规则，产生锚点。
2. 从锚点提取给名 handle，并记录该锚点在节内的最早位置。
3. 对没有现成锚点的 POS·Giv candidate，可以在**同一节内向前**回找被基础规则
   漏掉的 `姓 + candidate`；找到后补标该全名并建立 handle。一次有充分结构证据
   的回指已经成立，不要求 candidate 出现 ≥N 次。
4. 从左到右扫描省称候选。只有满足以下条件才画线：
   - 相同 handle 的锚点位置严格早于候选；
   - 锚点和候选属于同一个 jie；
   - 候选头字符通过 POS·Giv；
   - span 尚未被其他规则消费；
   - 单字候选的下一字若属于同一个 BIO·Giv entity，默认拒绝；只有局部语法明确证明
     下一字是亲属/领属成分、谓词或受控修饰语时才穿透，例如 `寄父、胜非、弘遂`；
   - 无 POS 时，必须由同节先行锚点和受控人物谓词共同证明；当前受控扩展包括
     `遣/命/引/闻/问/怒/使/从/纳/听/拒/救`；
   - 若单字 handle 被 POS 标成 ADV/AUX 等功能词，只允许两个更窄的覆盖：
     `劝/从/遣 + handle + 高置信谓词`，或严格句界后的
     `handle + 人物谓词`；handle 在本节必须只有一个更早全名来源，`当` 明确否决；
   - handle 后紧接 `王/公/侯` 时默认拒绝；只开放 `免 + handle + 侯` 和
     `handle + 公 + 明确谓词`，避免把 `温王` 切成 `[温]王`；
   - 若当前位置与锚点证明的姓氏前缀重新组成完整姓名，并且完整二字姓名规则可成立，
    保留完整姓名而不先消费其给名；例如保留 `立[曲嘉]为王`，同时仍允许
    `礼、[嘉]还高昌`。
   - 已 admission card 的唯一二字 suffix 可以补作局部 handle，但目标必须有完整
    POS/token 支持和人物句法；单字 fallback 还必须有 POS·Giv。普通词、泛称、完整姓名
    内部和称号 continuation 不能仅凭 suffix 相同进入 roster。
   - fallback/BIO 新路径不得使用人物 surface 黑白名单、stop-word admission、审计样例
    例外或全局人物 KB；反例只进入测试断言，不进入生产判断。

Translation-assisted admission 还要求同一原文起点的 exact/given candidates 合并后只有
一个 identity owner。given 的无 POS admission 只开放 paragraph-local strict owner 下的
`为 X 所` 和句界后 `X + 人物谓词`。exact fullname 的 T6 gate 仅在完整 token 边界、
model-NER surface，以及同节人物 NameType 或正式称号形态共同成立时放宽；干支、官职、
地点/族群、军镇 continuation 和 polity frame 均否决。该 evidence 仍是 Step 1 的局部
occurrence 证据，不执行 Step 2 identity binding。

同一节内还允许 forward-only translation anchor：每个 identity/handle 从本节最早的
eligible translated candidate 起生效，到节末结束；不能反向授权更早 occurrence。
单字 handle 即使有该 anchor，若 POS 判为功能词，也必须由句界和紧邻高置信
VERB/AUX 独立证明；位于更长 PROPN/NER 姓名内部时拒绝。

官职 + POS 证明的全名可建立锚点，例如 `[班超]` in `假司马班超`、`[班勇]` in
`军司马班勇`。已经被 Agent 1 识别的 title-person span 若形如
`…王/公/侯 + 1–2 字给名`，其爵号后部分也进入同节 roster，例如
`高阳王隆 → 隆`、`范阳王德 → 德`、`湘东王彧 → 彧`。

Model NER 称号另有三个 KB-free schema：

- `model_ner_fief_title`：完整 Geo component + `君/公/侯`，并要求当前称号句法；
- `model_ner_rank_title`：完整 `X伯` rank title，拒绝左右更长姓名；
- `model_ner_temple_title`：完整 `X宗`，要求庙号/君主语境，拒绝 Geo/Nat、爵号及地点
  continuation。
- `model_ner_short_royal_title`：完整二字 `X王/X后`，要求 title token 形态和局部人物
  evidence，拒绝长名内部切尾。

这些 schema 不把 `伯/宗` 加入旧的全局 title suffix 集合。BIO component 必须从
非 `I-` token 起始；地点后缀和移动到地点的结构均拒绝。已 admission 的完整称号可由
`local_exact_title` 在同节传播。Pipeline 不注册通用 `local_exact_given`：唯一例外是
已经 admission 的普通 `anaphora` card 保留其唯一、更早、同节完整锚点 provenance，
后续 exact 单字还必须有完整 POS·Giv 与严格人物句法。仅凭同字、POS·Giv 和句首谓词
仍不能传播。

明确禁止：

- `卷60` 前节出现 `曹操`，后节“宁、原俱以操尚称”中的 `操`不能据此标为曹操；
- `卷97` 某一节出现 `石虎`，不能自动授权整卷所有 `虎`；
- 后文出现 `慕容垂`，不能反向标注此前另一节的 `垂`。

正确例：

```text
同一①内：[赵王虎]享群臣……[虎]命射之……[虎]曰……
同一节内：[霍光]……[光]白封……
同一节内：[大将军光]……[光]纳其言……
```

## 6. “封禁姓氏”与边界护栏

“封禁姓氏”不是说这些字永远不能作姓，而是说它们同时高度像国名、爵位或普通词，
不能在缺乏结构证据时自由触发姓名规则。

- `BLOCK1`：齐、韩、汉、魏、赵、楚、燕、秦、吴、曹……以及王、侯、公、帝、
  后等高歧义字。
- `BLOCK2`：中山、常山、太子、公子、王子等高风险双字左环境。
- `STATE_SUR`：既是国名又可能是真姓的子集，用于受控的同节全名回溯。

目的：

- 防止把 `秦攻燕`、`汉王` 等普通国名/称谓切成人名；
- 同时允许有充分证据的 `曹操`、`吴起`、`赵王虎` 成为锚点。

因此护栏应被结构证据“有条件穿透”，不能简单删除。

## 7. Agent 2：身份合并

Agent 2 接收 Agent 1 的 occurrence cards，负责回答“是谁”。

### 7.1 合并方向

由安全到风险逐层扩大：

1. 同一节内的确定关系；
2. 同卷内的唯一身份；
3. 相邻卷/时代窗口内的唯一身份；
4. 胡注交叉引用、姓名变体、谱系、LLM 审核等额外信号；
5. 与已有 KB person card 对齐。

### 7.2 安全原则

- **只合并，不因身份未决而删除下划线。**
- 不确定时保留多个 singleton occurrence/person cards，即宁可 under-merge。
- 同名跨时代、同给名多人、谥号复用时不得强并。
- Agent 2 可以解决 `坚` 同时可能对应苻坚、庭坚的问题；Agent 1 不应靠卷级频率猜。
- 人物 KB 只在 Agent 2 使用；Agent 1 不读取它。

## 8. 评估口径

### 8.1 正确匹配

所有 span 评估必须同时约束：

- 同一卷；
- 同一 paragraph id；
- 字符区间重叠。

不能把不同 paragraph 的局部 offset 展平后比较，否则相同数字偏移会产生伪命中。

### 8.2 当前基线

正式可重复 benchmark 只测量 Translation-assisted 最终输出，由 `benchmark.py` 生成；
未加载译文的 default 路径仅作消融诊断，不作为正式指标。生产 v1 中逐 geometry 审核确认
的错误 reference 记录在 `benchmark-reference-exclusions.jsonl`；benchmark 和 candidate
audit 共用同一 loader，并验证 geometry 仍存在且正文 surface 未漂移。不得使用 Agent 1
规则自动过滤 reference，也不得把排除数记作规则 recovery。完整命令、输入定义和最新 JSON
见 [`BENCHMARK.md`](BENCHMARK.md)。当前 jie-only assisted audited 结果为
124,864 / 128,330 = **97.299%**，剩余 audited v1 gap 为 3,466；raw 兼容诊断口径为
124,866 / 128,596 = 97.099%，gap 3,730。以下表格保留最初
paragraph scope → jie scope 的历史对照：

除明确标为 raw compatibility、default ablation 或 attribution 的诊断表外，文档、报告和
对外结论中的 “coverage”“gap”“final benchmark” 均只能指 audited Translation-assisted
口径。

| v1 kind | paragraph scope | jie scope |
|---|---:|---:|
| alias | 92.9% | 94.3% |
| anaphora | 68.5% | 90.1% |
| role | 100.0% | 100.0% |
| gloss | 99.4% | 99.5% |
| feng | 88.7% | 89.3% |
| **ALL** | **86.5%** | **93.3%** |

最新 Agent 1 输出 175,399 spans，其中 125,324 与 v1 重叠，overlap proxy 为 71.451%。
这些 proxy 不是独立 precision；`太后/皇后`、政权王号、`丞相斯`、`其弟乙` 等真实表达
在 v1 中大量欠标，必须结合人工例审。

### 8.3 相对 audited 生产 v1 的未覆盖 span

这是“audited v1 有、Agent 1 没有”的集合，不等同于 3,466 个 Agent 1 错误：尚未审计
的 v1 仍可能包含卷级绑定和可疑误标。

| v1 kind | 未覆盖 |
|---|---:|
| alias | 2,368 |
| anaphora | 1,091 |
| feng | 6 |
| gloss | 1 |
| role | 0 |

此前缺口的详细分桶已被本轮规则改变，不能继续当作当前计数。最新阶段结果显示：身份词和
POS 全名补齐 alias；POS-derived handle 只在更早同节全名之后恢复省称；高置信功能词冲突
会反向否决 `甲子/必欲/自为/莫相` 等同形词。政权主号、多字封地爵号和时序行政区介绍
补齐了一批会产生连锁省称缺口的锚点。最新逐卷全类别报告已从
`benchmark-latest.json` 同规则输出重新生成。

## 9. 当前已知缺口

1. **嵌入式双字人物词项：** 剩余最大 alias 桶仍需新的通用人物句法，不能取消边界。
2. **POS 漏标：** 某些真实裸给名未被 POS·Giv 接受，导致同节有锚点仍漏线。
3. **谱系枚举边界：** `及弟叔璠、叔道` 等并列结构尚未完全覆盖。
4. **同给名多人：** 不能由 Agent 1 用卷级频率解决，应交给 Agent 2。
5. **复用爵称：** `智伯/于公` 等不能仅靠字面词典绑定身份。
6. **golden 欠标：** title/kinship/person-appellation 类需要持续人工抽样校准。

## 10. 后续计划

### Phase A：稳定 Agent 1

- 为全名等价锚点增加精确、可复用的结构规则，不添加实例特判；
- 完成王/公/侯、官职、亲属并列等规则的语料级评估；
- 对每条规则分别记录新增数、golden overlap 和人工误例；
- 保持严格同节向后省称，不再引入卷级 fallback。

### Phase B：完成 Agent 2

- 以 occurrence card 为原子实现同节→同卷→时代窗口的保守合并；
- 加入胡注、谱系、变体和已有 KB 作为合并证据；
- 系统处理同名、同给名、复姓、谥号复用；
- 保证无法合并的 occurrence 仍保留下划线。

### Phase C：迁移与删除旧管线

- shadow 输出与当前生产输出逐 span 比较；
- 分类所有 LOST / GAINED / REBIND；
- 完成全 294 卷 build、validator 和前端抽查；
- 切换唯一入口后删除旧 `build_persons.py` 路径、旧 v1/v2 数据切换及废弃脚本。

## 11. 修改规则的验收要求

任何新规则都必须：

1. 写成结构规则，不硬编码用于说明的具体人物；
2. 明确 `corpus` 或 `jie` scope；
3. 保留 rule provenance；
4. 针对目标例和相邻反例验证；
5. 跑全 294 卷同 paragraph 评估；
6. 抽查高频 FP，区分真误标与 golden 欠标；
7. 不引入跨节或反向省称；
8. 不破坏输出 offset、去重和 `validate_persons.py` 不变量。
9. 不以固定出现次数作为 personhood 硬门槛；频次只能辅助评估和排查。
