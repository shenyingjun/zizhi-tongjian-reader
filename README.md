# 资治通鉴 Reader

A simplified-Chinese, offline-first PWA for reading 《资治通鉴》(胡三省音注), with:

- 卷内按年导航 (year navigation inside each 卷)
- 简体中文 (Simplified Chinese, converted from Wikisource Traditional)
- 胡三省音注 inline, collapsible
- **Hover any person → see their prior appearances in 通鉴** (filtered to events at or before the current reading year, no spoilers)
- Full-text search
- Installable on desktop & mobile, works offline

## Repo layout

```
web/             React + Vite + TypeScript PWA (the reader)
data-pipeline/   Python pipeline that builds the static corpus + person index
plan.md          (in session workspace) implementation plan
```

## Status

Early scaffolding. See `plan.md` in the session workspace for phase breakdown.

## License

Code: MIT (planned). Text content: derived from Wikisource (CC-BY-SA / PD).
