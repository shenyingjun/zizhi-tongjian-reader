# LLM 人名标注缓存（durable annotation cache）

这个目录是**流水线与 LLM 的混合层**。目标：LLM 对全书人名做**一次性**校补，
结果写成缓存，`build_persons.py` 每次构建时直接读取，**永不重复调用 LLM**。

## 为什么这样设计

卷251 的 pipeline-vs-LLM 对照实验证明：确定性流水线与 LLM 在**精度**上等同
（≈100%，0 误标），LLM 只在**召回**上多出约 18 个百分点——都是官衔连写、
NER 漏掉的孤例人名（如 元密 8 次、卢望回）。因此混合策略是：

- **LLM 只负责“发现人名”**（detection）。
- **偏移量、指代枚举、消歧仍由流水线确定性完成**（offsets / anaphora）。

LLM 给出人名表面形，流水线在该卷正文里用 longest-first 精确定位。LLM 从不产生坐标。

## 文件格式

每卷一个文件 `juan_NNN.tsv`（NNN = 三位卷号，如 `juan_251.tsv`）。
UTF-8，制表符分隔，一行一个人名：

```
姓名<TAB>置信度<TAB>证据片段
```

- **姓名**：2–4 个汉字的完整姓名（必须以已知姓氏 / 复姓开头）。只写全名，
  不写单字省称、不写官职、不写称号。
- **置信度**：`high` / `med`（目前只采纳 high；med 供人工复核）。
- **证据片段**：该人名在正文中的一小段上下文，便于审计。可留空。
- `#` 开头的行是注释。

## 构建时如何被消费（build_persons.py 的护栏）

`build_gloss_cards` 里的 **LLM-annotation recall tier** 读取本目录。每个 LLM 断言
的表面形，**只有在同时满足**下列条件时才建卡：

1. 2–4 汉字，且以 `SURNAMES` 或 `COMPOUND`（复姓）开头；
2. 通过全部非人名护栏（`bad_auto_surface` / `COMMON_WORD_NONPERSON` /
   `COMPOUND` 常用词 / `_TITLE_BANNED` / 未被其它卡占用）；
3. **该表面形在该卷正文中确实出现**（precision-first：宁缺毋滥）。

生成的卡片 brief 标注「（LLM 校补）」，可与程序自动卡区分。

## 如何一次性生成全书缓存

见同目录 `../run_llm_pass.py`：给定一个 OpenAI 兼容的 API endpoint 与 key，
它会分批把 294 卷正文喂给模型，按上面的格式写入本目录。断点续跑（已缓存的卷跳过）。
全书一次约 $40–70（分批），跑完即长期缓存，除非语料或提示词变更否则不再重跑。

当前已种子化：`juan_251.tsv`（对照实验里 LLM 独有的召回项）。
