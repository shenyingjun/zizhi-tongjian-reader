# Two-stage person-underlining pipeline

本目录是人物下划线目标管线，当前仍与旧生产管线并存。

- [`SPEC.md`](SPEC.md)：术语、节级作用域、Agent 1 / Agent 2 边界和规则规范。
- [Copilot instructions](../../../.github/copilot-instructions.md)：规则修改、targeted
  验证、全量 rebuild 和 precision 审计的默认操作门槛与经验。
- [`rules.py`](rules.py)：当前开发中的 Agent 1 Tagger；所有规则按编号节运行。
- [`BENCHMARK.md`](BENCHMARK.md)：相对生产 v1 的正式可重复 benchmark、运行命令和最新结果。
- [`benchmark.py`](benchmark.py)：全 294 卷 benchmark runner。
- [`benchmark-latest.json`](benchmark-latest.json)：最新机器可读结果。
- [`build_admin_places.py`](build_admin_places.py)：从全书正文和 POS cache 重建带年份的
  行政区证据。
- [`retag.py`](retag.py)：用当前规则输出不带人物身份的 occurrence cards，不覆盖生产 v1。
- [`translation_evidence.py`](translation_evidence.py)：把离线译文 NER/mapping 输出转换为
  带 canonical paragraph hash 的可选 identity evidence；不保存译文正文。

当前 **v1** 指 `web/public/text/persons/` 中的生产输出；**v2** 指当前 Agent 1 Tagger
及未来 Agent 2 Identifier 的目标管线。Agent 2 尚未完成，因此目前只能评估 Agent 1 的
画线跨度，不能声称已有完整 v2 数据集。`web/public/text/persons-v2/` 是旧的 ADD-only
union 产物，已不能代表这里的最新实现。

`stage1.py`、`stage2.py`、`run.py` 和 `build_v2.py` 属于较早的 shadow harness；其历史
AGREE/RECOVER/LOST 数字不再作为当前 benchmark。新规则改动统一通过 `benchmark.py`
复跑，并按 `SPEC.md` 的人工抽样要求审查。

最终输出必须加载 Translation evidence。先生成 evidence，再在仓库根目录重建行政区证据
并重标全 294 卷：

```powershell
data-pipeline\.venv-ner\Scripts\python.exe -X utf8 `
  data-pipeline\persons\twostage\translation_evidence.py `
  --mapping-json C:\temp\mapping-v3.json `
  --output-dir C:\temp\translation-evidence

data-pipeline\.venv-ner\Scripts\python.exe -X utf8 `
  data-pipeline\persons\twostage\retag.py `
  --translation-evidence-dir C:\temp\translation-evidence `
  --output-dir C:\temp\ztj-agent1-final
```

`retag.py` 默认先重建 `admin-places.json`；反复调试规则时可加
`--skip-admin-rebuild`，也可用 `--juans 62 141` 只处理指定卷。输出目录包含每卷 JSON
和 `manifest.json`，两者都记录 `rules.py` 与行政区证据的 SHA-256。

未传 `--translation-evidence-dir` 的路径只用于规则消融和调试，不作为最终输出，也不汇报
正式覆盖率。需要针对部分卷验证时可运行：

```powershell
data-pipeline\.venv-ner\Scripts\python.exe -X utf8 `
  data-pipeline\persons\twostage\retag.py `
  --juans 27 37 45 150 `
  --skip-admin-rebuild `
  --translation-evidence-dir C:\temp\translation-evidence `
  --output-dir C:\temp\ztj-agent1-translation
```

该 evidence 只在同 paragraph 生效。exact fullname 与 mapped given 分别须通过完整
POS/BIO/边界 gate 和受控原文句法 gate；获准 fullname 会成为后续普通 anaphora anchor。
translation 路径不得删除未重叠的原文 card，也不得用 suffix 替换原文完整姓名。
loader 会按 manifest 的逐卷 SHA-256 拒绝被替换或损坏的 evidence。正式 benchmark 只测量
Translation-assisted 全 294 卷最终输出；结果和人工 precision 审计见 `BENCHMARK.md`。
