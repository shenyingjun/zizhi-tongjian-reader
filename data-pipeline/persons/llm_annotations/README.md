# LLM 人名标注缓存（durable annotation cache）

这个目录是**流水线与 LLM 的混合层**。目标：LLM 对全书人名做**一次性**校补，
结果写成缓存，`build_persons.py` 每次构建时直接读取，**永不重复调用 LLM**。

## 为什么这样设计

卷251 的 pipeline-vs-LLM 对照实验证明：确定性流水线与 LLM 在**精度**上等同
（≈100%，0 误标），LLM 只在**召回**上多出约 18 个百分点——都是官衔连写、
NER 漏掉的孤例人名（如 元密 8 次、卢望回）。因此混合策略是：

- **LLM 只负责“发现人名 + 声明本卷省称”**（detection）。
- **偏移量、指代枚举、消歧仍由流水线确定性完成**（offsets / anaphora）。

LLM 给出人名表面形（及别名），流水线在该卷正文里精确定位。LLM 从不产生坐标。

## 文件格式（JSONL，v2）

每卷一个文件 `juan_NNN.jsonl`（NNN = 三位卷号，如 `juan_253.jsonl`）。
UTF-8，**一行一个 JSON 对象**（一个人），`#` 开头的行是注释：

```json
{"name":"康道伟","aliases":["道伟"],"role":"高品·宦官使者","confidence":"high","evidence":"遣高品康道伟赍敕书抚慰之"}
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
- **evidence**（可选）：正文上下文片段，便于审计。

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

当前已种子化：`juan_250` `juan_251` `juan_252` `juan_253`。
