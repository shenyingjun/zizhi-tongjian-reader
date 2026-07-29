"use strict";

const state = {
  index: null, juan: null, phase: "blind", payload: null, jie: 0,
  annotations: [], decisions: {}, noteDecisions: {}, pending: null,
  focusedCandidate: null,
};
const $ = id => document.getElementById(id);
const ROLE_LABELS = {
  random: "随机基准",
  rules_v1_disagreement: "规则-v1分歧",
  rare_pattern_challenge: "罕见结构挑战",
};
const PHASE_LABELS = {
  blind: "盲标",
  assisted: "Copilot 辅助审核",
  recall: "候选补漏",
  role_audit: "角色复核",
};

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

function setStatus(message, error = false) {
  $("status").textContent = message;
  $("status").className = error ? "error" : "";
}

async function loadIndex() {
  state.index = await api("/api/index");
  const nav = $("juans");
  nav.replaceChildren();
  for (const row of state.index.juans) {
    const button = document.createElement("button");
    const role = row.role ? (ROLE_LABELS[row.role] || row.role) : "";
    button.textContent = `卷 ${row.juan}${role ? ` · ${role}` : ""}`;
    button.onclick = () => loadTask(
      row.juan, row.initial_phase || state.phase);
    button.dataset.juan = row.juan;
    nav.append(button);
  }
  const first = state.index.juans[0];
  await loadTask(first.juan, first.initial_phase || "blind");
}

async function loadTask(juan, phase) {
  try {
    await flushSave();
    const payload = await api(`/api/task?juan=${juan}&phase=${phase}`);
    state.juan = juan; state.phase = phase; state.payload = payload;
    state.jie = rememberedJie(juan, phase, payload.task.jies.length);
    state.annotations = structuredClone(payload.state.annotations || []);
    state.decisions = structuredClone(payload.state.decisions || {});
    state.noteDecisions = structuredClone(payload.state.note_decisions || {});
    state.pending = null; state.focusedCandidate = null;
    restoreDraft(juan, phase, payload.locked);
    $("phase").value = phase;
    for (const button of $("juans").children)
      button.classList.toggle("active", Number(button.dataset.juan) === juan);
    populateJies(); render();
    setStatus(
      `卷 ${juan} · ${PHASE_LABELS[phase]}` +
      ` · 已保存 ${state.annotations.length} 个跨度` +
      `${payload.locked ? " · 已锁定" : ""}`);
  } catch (error) {
    setStatus(error.message, true);
    if (phase !== "blind") $("phase").value = "blind";
  }
}

function draftKey(juan = state.juan, phase = state.phase) {
  return `ml-poc-draft:v4:${juan}:${phase}`;
}

function persistDraft() {
  if (!state.payload || state.payload.locked) return;
  localStorage.setItem(draftKey(), JSON.stringify({
    geometry_version: 4,
    annotations: state.annotations,
    decisions: state.decisions,
    note_decisions: state.noteDecisions,
  }));
}

function restoreDraft(juan, phase, locked) {
  const key = draftKey(juan, phase);
  const raw = localStorage.getItem(key);
  if (!raw || locked) {
    if (locked) localStorage.removeItem(key);
    return;
  }
  try {
    const draft = JSON.parse(raw);
    if (draft.geometry_version !== 4) {
      localStorage.removeItem(key);
      return;
    }
    if (confirm(
        `发现卷${juan}未保存的本地草稿（${draft.annotations.length} 个跨度），是否恢复？`)) {
      state.annotations = draft.annotations;
      state.decisions = draft.decisions || {};
      state.noteDecisions = draft.note_decisions || {};
      setStatus(`卷${juan}已恢复本地草稿，正在重新保存`);
      scheduleSave();
    } else {
      localStorage.removeItem(key);
    }
  } catch {
    localStorage.removeItem(key);
  }
}

function populateJies() {
  const select = $("jie"); select.replaceChildren();
  state.payload.task.jies.forEach((jie, index) => {
    const option = document.createElement("option");
    option.value = index;
    option.textContent = `节 ${index + 1}${jie.jie_number ? `（${jie.jie_number}）` : ""}`;
    select.append(option);
  });
  select.value = String(state.jie);
}

function positionKey(juan = state.juan, phase = state.phase) {
  return `ml-poc-jie:${juan}:${phase}`;
}

function rememberedJie(juan, phase, count) {
  const saved = Number(localStorage.getItem(positionKey(juan, phase)));
  return Number.isInteger(saved) && saved >= 0 && saved < count ? saved : 0;
}

function selectJie(index) {
  const count = state.payload.task.jies.length;
  state.jie = Math.max(0, Math.min(index, count - 1));
  state.pending = null;
  localStorage.setItem(positionKey(), String(state.jie));
  $("jie").value = String(state.jie);
  render();
  $("previous-jie").disabled = state.jie === 0;
  $("next-jie").disabled = state.jie === count - 1;
}

function lastTaggedJie() {
  if (!state.annotations.length) {
    setStatus(`卷${state.juan}尚无已保存标注`);
    return;
  }
  const annotatedParagraphs = new Set(
    state.annotations.map(row => row.para_id));
  let last = 0;
  state.payload.task.jies.forEach((jie, index) => {
    if (jie.segments.some(row => annotatedParagraphs.has(row.para_id)))
      last = index;
  });
  selectJie(last);
}

function sliceCodepoints(text, start, end) {
  return [...text].slice(start, end).join("");
}

function paragraphText(jie, segment) {
  return sliceCodepoints(
    jie.text, segment.assembled_start, segment.assembled_end);
}

function annotationAt(paraId, offset) {
  return state.annotations.some(row =>
    row.para_id === paraId && row.start <= offset && offset < row.end);
}

function annotationIndexAt(paraId, offset) {
  return state.annotations.findIndex(row =>
    row.para_id === paraId && row.start <= offset && offset < row.end);
}

function currentCandidates() {
  if (state.phase === "blind") return [];
  const visibleParagraphs = new Set(
    state.payload.task.jies[state.jie].segments.map(row => row.para_id));
  return state.payload.review.candidates.filter(
    row => visibleParagraphs.has(row.para_id));
}

function unresolvedCandidateAt(paraId, offset) {
  return currentCandidates().some(row =>
    !state.decisions[row.id] && row.para_id === paraId &&
    row.start <= offset && offset < row.end);
}

function render() {
  const jie = state.payload.task.jies[state.jie];
  $("review-workspace").classList.toggle(
    "recall-layout", state.phase !== "blind");
  const host = $("text"); host.replaceChildren();
  for (const segment of jie.segments) {
    const paragraph = document.createElement("div"); paragraph.className = "paragraph";
    const text = paragraphText(jie, segment);
    [...text].forEach((char, offset) => {
      const span = document.createElement("span");
      span.className = "char";
      span.dataset.paraId = segment.para_id;
      span.dataset.offset = offset;
      if (annotationAt(segment.para_id, offset)) span.classList.add("tagged");
      if (unresolvedCandidateAt(segment.para_id, offset))
        span.classList.add("candidate-unresolved");
      if (state.focusedCandidate &&
          state.focusedCandidate.para_id === segment.para_id &&
          state.focusedCandidate.start <= offset &&
          offset < state.focusedCandidate.end)
        span.classList.add("candidate-focused");
      if (state.pending && state.pending.para_id === segment.para_id &&
          state.pending.start === offset) span.classList.add("pending");
      span.textContent = char;
      span.onclick = () => selectChar(segment.para_id, offset, text);
      paragraph.append(span);
    });
    host.append(paragraph);
  }
  renderAnnotations();
  renderRecall();
  $("save").disabled = state.payload.locked;
  $("complete").disabled = state.payload.locked;
  $("accept-rest").hidden = state.phase === "blind";
  $("accept-rest").disabled = state.payload.locked;
  $("previous-jie").disabled = state.jie === 0;
  $("next-jie").disabled =
    state.jie === state.payload.task.jies.length - 1;
}

function selectChar(paraId, offset, text) {
  if (state.payload.locked) return;
  const existingIndex = annotationIndexAt(paraId, offset);
  if (!state.pending && existingIndex >= 0) {
    const existing = state.annotations[existingIndex];
    if (confirm(`删除已有标注「${existing.surface}」并重新标注吗？`))
      removeAnnotation(existingIndex);
    return;
  }
  if (!state.pending || state.pending.para_id !== paraId) {
    state.pending = {para_id: paraId, start: offset};
    render(); return;
  }
  const start = Math.min(state.pending.start, offset);
  const end = Math.max(state.pending.start, offset) + 1;
  const overlap = state.annotations.some(row =>
    row.para_id === paraId && start < row.end && row.start < end);
  if (overlap) { setStatus("跨度与已有标注重叠", true); state.pending = null; render(); return; }
  state.annotations.push({
    para_id: paraId,
    start,
    end,
    surface: sliceCodepoints(text, start, end),
    status: "person",
    note: "",
  });
  state.pending = null; state.annotations.sort((a,b) => a.para_id-b.para_id || a.start-b.start);
  render(); scheduleSave();
}

function removeAnnotation(index) {
  const [removed] = state.annotations.splice(index, 1);
  if (state.phase !== "blind") {
    for (const candidate of state.payload.review.candidates) {
      if (candidate.para_id === removed.para_id &&
          candidate.start === removed.start &&
          candidate.end === removed.end)
        state.decisions[candidate.id] = "reject";
    }
  }
  state.pending = null;
  render();
  scheduleSave();
}

function renderAnnotations() {
  const host = $("annotations"); host.replaceChildren();
  const visibleParagraphs = new Set(
    state.payload.task.jies[state.jie].segments.map(row => row.para_id));
  const visibleIndexes = [...state.annotations.keys()].filter(
    index => visibleParagraphs.has(state.annotations[index].para_id));
  for (const index of visibleIndexes.reverse()) {
    const row = state.annotations[index];
    const div = document.createElement("div"); div.className = "annotation";
    div.textContent = `段 ${row.para_id} [${row.start},${row.end}) ${row.surface} `;
    const remove = document.createElement("button"); remove.textContent = "删除";
    remove.disabled = state.payload.locked;
    remove.onclick = () => removeAnnotation(index);
    div.append(remove); host.append(div);
  }
}

function renderRecall() {
  const panel = $("recall-panel");
  panel.hidden = state.phase === "blind";
  if (panel.hidden) return;
  $("review-title").textContent =
    state.phase === "role_audit"
      ? "特定角色复核"
      : state.phase === "assisted" ? "Copilot 候选审核" : "候选补漏";
  const visibleParagraphs = new Set(
    state.payload.task.jies[state.jie].segments.map(row => row.para_id));
  const visibleCandidates = currentCandidates();
  const unresolved = visibleCandidates.filter(row => !state.decisions[row.id]);
  $("recall-counts").textContent =
    `本节待审 ${unresolved.length} / 候选 ${visibleCandidates.length}。`;
  const showResolved = $("show-resolved").checked;
  const candidates = $("candidates"); candidates.replaceChildren();
  for (const candidate of visibleCandidates.filter(
      row => showResolved || !state.decisions[row.id])) {
    const div = document.createElement("div");
    div.className = `candidate ${state.decisions[candidate.id] || ""}`;
    if (state.focusedCandidate?.id === candidate.id)
      div.classList.add("focused");
    div.onclick = event => {
      if (event.target.tagName !== "BUTTON") focusCandidate(candidate);
    };
    div.append(document.createTextNode(
      `段 ${candidate.para_id} [${candidate.start},${candidate.end}) ${candidate.surface} `));
    const channels = document.createElement("span"); channels.className = "channels";
    channels.textContent = candidate.channels.join(" + "); div.append(channels);
    div.append(candidateContext(candidate));
    for (const decision of ["accept", "reject", "unsure"]) {
      const button = document.createElement("button"); button.textContent = decision;
      button.disabled = state.payload.locked;
      button.onclick = () => decide(candidate, decision);
      div.append(button);
    }
    candidates.append(div);
  }
  const notes = $("notes"); notes.replaceChildren();
  for (const note of (state.payload.review.note_evidence || []).filter(
      row => visibleParagraphs.has(row.para_id))) {
    const div = document.createElement("div"); div.className = "note";
    div.textContent = `段 ${note.para_id} 正文偏移 ${note.after}：注文提及「${note.surface}」（不自动生成跨度）`;
    notes.append(div);
  }
}

function decide(candidate, decision) {
  const exact = state.annotations.some(row =>
    row.para_id === candidate.para_id &&
    row.start === candidate.start && row.end === candidate.end);
  if (decision === "accept" && !exact) {
    const overlaps = state.annotations.filter(row =>
      row.para_id === candidate.para_id &&
      candidate.start < row.end && row.start < candidate.end);
    if (overlaps.length) {
      const oldSurfaces = overlaps.map(row => `「${row.surface}」`).join("、");
      if (!confirm(
          `候选「${candidate.surface}」与已有标注 ${oldSurfaces} 重叠。` +
          "是否用候选边界替换已有标注？")) {
        delete state.decisions[candidate.id];
        setStatus("未修改；请选择 reject / unsure，或先手工纠正边界");
        render();
        return;
      }
      const overlapGeometry = new Set(
        overlaps.map(row => `${row.para_id}:${row.start}:${row.end}`));
      state.annotations = state.annotations.filter(row =>
        !overlapGeometry.has(`${row.para_id}:${row.start}:${row.end}`));
    }
    state.annotations.push({
      para_id: candidate.para_id,
      start: candidate.start,
      end: candidate.end,
      surface: candidate.surface,
      status: "person",
      note: "",
    });
  }
  state.decisions[candidate.id] = decision;
  if (decision !== "accept") {
    state.annotations = state.annotations.filter(row =>
      !(row.para_id === candidate.para_id &&
        row.start === candidate.start && row.end === candidate.end));
  }
  render(); scheduleSave();
}

function acceptRestInCurrentJie() {
  if (state.phase === "blind" || state.payload.locked) return;
  const unresolved = currentCandidates().filter(
    candidate => !state.decisions[candidate.id]);
  if (!unresolved.length) {
    setStatus("当前节没有未处理候选");
    return;
  }
  if (!confirm(`接受当前节其余 ${unresolved.length} 个未处理候选吗？`))
    return;
  let accepted = 0;
  let rejectedForCorrection = 0;
  for (const candidate of unresolved) {
    const exact = state.annotations.some(row =>
      row.para_id === candidate.para_id &&
      row.start === candidate.start && row.end === candidate.end);
    const overlaps = state.annotations.some(row =>
      row.para_id === candidate.para_id &&
      candidate.start < row.end && row.start < candidate.end);
    if (overlaps && !exact) {
      state.decisions[candidate.id] = "reject";
      rejectedForCorrection += 1;
      continue;
    }
    if (!exact) {
      state.annotations.push({
        para_id: candidate.para_id,
        start: candidate.start,
        end: candidate.end,
        surface: candidate.surface,
        status: "person",
        note: "",
      });
    }
    state.decisions[candidate.id] = "accept";
    accepted += 1;
  }
  state.annotations.sort(
    (a, b) => a.para_id - b.para_id || a.start - b.start);
  render();
  scheduleSave();
  const correctionMessage = rejectedForCorrection
    ? `；${rejectedForCorrection} 个与人工修正重叠，已保留修正并拒绝原候选`
    : "";
  setStatus(`已接受 ${accepted} 个候选${correctionMessage}`);
}

function paragraphFor(paraId) {
  const jie = state.payload.task.jies[state.jie];
  const segment = jie.segments.find(row => row.para_id === paraId);
  if (!segment) return "";
  return paragraphText(jie, segment);
}

function candidateContext(candidate) {
  const text = paragraphFor(candidate.para_id);
  const length = [...text].length;
  const radius = 14;
  const before = sliceCodepoints(
    text, Math.max(0, candidate.start - radius), candidate.start);
  const selected = sliceCodepoints(text, candidate.start, candidate.end);
  const after = sliceCodepoints(
    text, candidate.end, Math.min(length, candidate.end + radius));
  const context = document.createElement("div");
  context.className = "candidate-context";
  context.append(document.createTextNode(before));
  const mark = document.createElement("mark"); mark.textContent = selected;
  context.append(mark, document.createTextNode(after));
  return context;
}

function focusCandidate(candidate) {
  state.focusedCandidate = candidate;
  render();
  const char = document.querySelector(
    `.char[data-para-id="${candidate.para_id}"][data-offset="${candidate.start}"]`);
  char?.scrollIntoView({behavior: "smooth", block: "center"});
}

function nextUnresolvedJie() {
  if (state.phase === "blind") return;
  const unresolvedParagraphs = new Set(
    state.payload.review.candidates
      .filter(row => !state.decisions[row.id])
      .map(row => row.para_id));
  const count = state.payload.task.jies.length;
  for (let step = 1; step <= count; step++) {
    const index = (state.jie + step) % count;
    if (state.payload.task.jies[index].segments.some(
        row => unresolvedParagraphs.has(row.para_id))) {
      selectJie(index);
      return;
    }
  }
  setStatus(`卷${state.juan}没有剩余待审候选`);
}

let saveTimer = null;
let saveQueue = Promise.resolve();
let latestSave = Promise.resolve();
function scheduleSave() {
  persistDraft();
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => save().catch(() => {}), 350);
}

async function save() {
  if (!state.payload || state.payload.locked) return;
  clearTimeout(saveTimer);
  saveTimer = null;
  const snapshot = {
    juan: state.juan,
    phase: state.phase,
    annotations: structuredClone(state.annotations),
    decisions: structuredClone(state.decisions),
    noteDecisions: structuredClone(state.noteDecisions),
  };
  const request = saveQueue.then(async () => {
    await api("/api/save", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        juan: snapshot.juan,
        phase: snapshot.phase,
        annotations: snapshot.annotations,
        decisions: snapshot.decisions,
        note_decisions: snapshot.noteDecisions,
        geometry_version: 4,
      }),
    });
    localStorage.removeItem(draftKey(snapshot.juan, snapshot.phase));
    if (state.juan === snapshot.juan && state.phase === snapshot.phase) {
      setStatus(
        `卷 ${snapshot.juan} · 已保存 ${snapshot.annotations.length} 个跨度`);
    }
  });
  saveQueue = request.catch(() => {});
  latestSave = request;
  try {
    await request;
  } catch (error) {
    setStatus(error.message, true);
    throw error;
  }
}

async function flushSave() {
  if (saveTimer !== null) {
    await save();
  } else {
    await latestSave;
  }
}

async function complete() {
  if (!confirm("完成后本阶段将永久锁定。确定吗？")) return;
  try {
    await flushSave();
    const completedJuan = state.juan;
    const completedPhase = state.phase;
    await api("/api/complete", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        juan: completedJuan,
        phase: completedPhase,
        geometry_version: 4,
      }),
    });
    state.index = await api("/api/index");
    const completedRow = state.index.juans.find(
      row => row.juan === completedJuan);
    if (completedPhase === "assisted") {
      const next = state.index.juans.find(
        row => row.mode === "assisted" && !row.assisted_complete);
      await loadTask(
        next?.juan || completedJuan,
        next?.initial_phase || "assisted");
      return;
    }
    if (completedPhase === "blind" &&
        completedRow?.mode === "blind_anchor") {
      const next = state.index.juans.find(row => row.mode === "assisted");
      await loadTask(next.juan, next.initial_phase);
      return;
    }
    const nextPhase = completedPhase === "blind"
      ? "recall"
      : completedPhase === "recall" ? "role_audit" : "role_audit";
    await loadTask(completedJuan, nextPhase);
  } catch (error) { setStatus(error.message, true); }
}

$("phase").onchange = event => loadTask(state.juan, event.target.value);
$("jie").onchange = event => selectJie(Number(event.target.value));
$("previous-jie").onclick = () => selectJie(state.jie - 1);
$("next-jie").onclick = () => selectJie(state.jie + 1);
$("last-tagged-jie").onclick = lastTaggedJie;
$("accept-rest").onclick = acceptRestInCurrentJie;
$("show-resolved").onchange = renderRecall;
$("next-unresolved").onclick = nextUnresolvedJie;
$("save").onclick = save;
$("complete").onclick = complete;
loadIndex().catch(error => setStatus(error.message, true));
