"use strict";

const GEOMETRY_VERSION = 1;
const state = {
  index: null,
  taskId: null,
  payload: null,
  annotations: [],
  humanDecisions: {},
  pending: null,
  focusedCandidate: null,
  localRevision: 0,
};
const $ = id => document.getElementById(id);

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

function sliceCodepoints(text, start, end) {
  return [...text].slice(start, end).join("");
}

function paragraphText(jie, segment) {
  return sliceCodepoints(
    jie.text, segment.assembled_start, segment.assembled_end);
}

function taskRow(taskId = state.taskId) {
  return state.index.tasks.find(row => row.task_id === taskId);
}

function effectiveDecisions() {
  const initial = state.payload.state.expanded_full_union
    ? {}
    : state.payload.review.initial_decisions;
  return {...initial, ...state.humanDecisions};
}

function requiredIds() {
  return new Set(state.payload.state.required_ids);
}

function unresolvedCandidates() {
  const required = requiredIds();
  return state.payload.review.candidates.filter(
    candidate => required.has(candidate.id) &&
      !state.humanDecisions[candidate.id]);
}

async function loadIndex(preferredTaskId = null) {
  state.index = await api("/api/production-review/index");
  const select = $("task");
  select.replaceChildren();
  for (const row of state.index.tasks) {
    const option = document.createElement("option");
    option.value = row.task_id;
    option.textContent =
      `${row.position}/${row.total} · 卷${row.juan} 节${row.jie_number ?? row.jie_index}` +
      `${row.complete ? " · 已完成" : row.expanded_full_union ? " · 已扩大" : ""}`;
    select.append(option);
  }
  if (!state.index.tasks.length) {
    setStatus("复核包中没有任务。", true);
    return;
  }
  const firstOpen = state.index.tasks.find(row => !row.complete);
  await loadTask(
    preferredTaskId || firstOpen?.task_id || state.index.tasks[0].task_id);
}

async function loadTask(taskId) {
  try {
    await flushSave();
    state.payload = await api(
      `/api/production-review/task?task_id=${encodeURIComponent(taskId)}`);
    state.taskId = taskId;
    state.annotations = structuredClone(state.payload.state.annotations);
    state.humanDecisions =
      structuredClone(state.payload.state.human_decisions);
    state.pending = null;
    state.focusedCandidate = null;
    state.localRevision = 0;
    $("task").value = taskId;
    render();
  } catch (error) {
    setStatus(error.message, true);
  }
}

function annotationIndexAt(paraId, offset) {
  return state.annotations.findIndex(row =>
    row.para_id === paraId && row.start <= offset && offset < row.end);
}

function exactAnnotation(candidate) {
  return state.annotations.some(row =>
    row.para_id === candidate.para_id &&
    row.start === candidate.start &&
    row.end === candidate.end);
}

function candidateAtGeometry(annotation) {
  return state.payload.review.candidates.find(candidate =>
    candidate.para_id === annotation.para_id &&
    candidate.start === annotation.start &&
    candidate.end === annotation.end);
}

function render() {
  const row = taskRow();
  const task = state.payload.task;
  const jie = task.jies[0];
  $("task-title").textContent =
    `卷 ${row.juan} · jie_index ${row.jie_index}` +
    `${row.jie_number == null ? "" : ` · 第 ${row.jie_number} 节`}`;
  $("expansion").hidden = !state.payload.state.expanded_full_union;
  const host = $("text");
  host.replaceChildren();
  for (const segment of jie.segments) {
    const paragraph = document.createElement("div");
    paragraph.className = "paragraph";
    const text = paragraphText(jie, segment);
    [...text].forEach((char, offset) => {
      const span = document.createElement("span");
      span.className = "char";
      span.dataset.paraId = segment.para_id;
      span.dataset.offset = offset;
      if (annotationIndexAt(segment.para_id, offset) >= 0)
        span.classList.add("tagged");
      if (unresolvedCandidates().some(candidate =>
          candidate.para_id === segment.para_id &&
          candidate.start <= offset && offset < candidate.end))
        span.classList.add("candidate-unresolved");
      if (state.focusedCandidate &&
          state.focusedCandidate.para_id === segment.para_id &&
          state.focusedCandidate.start <= offset &&
          offset < state.focusedCandidate.end)
        span.classList.add("candidate-focused");
      if (state.pending &&
          state.pending.para_id === segment.para_id &&
          state.pending.start === offset)
        span.classList.add("pending");
      span.textContent = char;
      span.onclick = () => selectChar(segment.para_id, offset, text);
      paragraph.append(span);
    });
    host.append(paragraph);
  }
  renderAnnotations();
  renderCandidates();
  $("save").disabled = state.payload.locked;
  $("complete").disabled = state.payload.locked;
  const position = state.index.tasks.findIndex(
    item => item.task_id === state.taskId);
  $("previous-task").disabled = position === 0;
  $("next-task").disabled = position === state.index.tasks.length - 1;
  setStatus(
    `进度 ${state.index.complete}/${state.index.total} · ` +
    `本题待审 ${unresolvedCandidates().length}` +
    `${state.payload.locked ? " · 已锁定" : ""}`);
}

function selectChar(paraId, offset, text) {
  if (state.payload.locked) return;
  const existingIndex = annotationIndexAt(paraId, offset);
  if (!state.pending && existingIndex >= 0) {
    const existing = state.annotations[existingIndex];
    if (confirm(`删除标注「${existing.surface}」吗？`))
      removeAnnotation(existingIndex);
    return;
  }
  if (!state.pending || state.pending.para_id !== paraId) {
    state.pending = {para_id: paraId, start: offset};
    render();
    return;
  }
  const start = Math.min(state.pending.start, offset);
  const end = Math.max(state.pending.start, offset) + 1;
  if (state.annotations.some(row =>
      row.para_id === paraId && start < row.end && row.start < end)) {
    state.pending = null;
    setStatus("新跨度与已有标注重叠", true);
    render();
    return;
  }
  state.annotations.push({
    para_id: paraId,
    start,
    end,
    surface: sliceCodepoints(text, start, end),
  });
  state.annotations.sort(
    (a, b) => a.para_id - b.para_id || a.start - b.start);
  state.pending = null;
  render();
  scheduleSave();
}

function removeAnnotation(index) {
  const [removed] = state.annotations.splice(index, 1);
  const candidate = candidateAtGeometry(removed);
  if (candidate) state.humanDecisions[candidate.id] = "reject";
  state.pending = null;
  render();
  scheduleSave();
}

function renderAnnotations() {
  const host = $("annotations");
  host.replaceChildren();
  for (const [index, annotation] of state.annotations.entries()) {
    const div = document.createElement("div");
    div.className = "annotation";
    div.textContent =
      `段 ${annotation.para_id} [${annotation.start},${annotation.end}) ` +
      `${annotation.surface} `;
    const remove = document.createElement("button");
    remove.textContent = "删除";
    remove.disabled = state.payload.locked;
    remove.onclick = () => removeAnnotation(index);
    div.append(remove);
    host.append(div);
  }
}

function candidateContext(candidate) {
  const jie = state.payload.task.jies[0];
  const segment = jie.segments.find(
    row => row.para_id === candidate.para_id);
  const text = paragraphText(jie, segment);
  const radius = 16;
  const context = document.createElement("div");
  context.className = "candidate-context";
  context.append(document.createTextNode(sliceCodepoints(
    text, Math.max(0, candidate.start - radius), candidate.start)));
  const mark = document.createElement("mark");
  mark.textContent = candidate.surface;
  context.append(mark, document.createTextNode(sliceCodepoints(
    text, candidate.end, Math.min([...text].length, candidate.end + radius))));
  return context;
}

function renderCandidates() {
  const decisions = effectiveDecisions();
  const required = requiredIds();
  const unresolved = unresolvedCandidates();
  $("counts").textContent =
    `待审 ${unresolved.length} / 当前要求 ${required.size} / ` +
    `union ${state.payload.review.candidates.length}。`;
  const showResolved = $("show-resolved").checked;
  const host = $("candidates");
  host.replaceChildren();
  for (const candidate of state.payload.review.candidates.filter(candidate =>
      showResolved || (required.has(candidate.id) &&
        !state.humanDecisions[candidate.id]))) {
    const decision = decisions[candidate.id] || "";
    const card = document.createElement("div");
    card.className = `candidate ${decision}`;
    card.onclick = event => {
      if (event.target.tagName !== "BUTTON") focusCandidate(candidate);
    };
    card.append(document.createTextNode(
      `段 ${candidate.para_id} [${candidate.start},${candidate.end}) ` +
      `${candidate.surface} `));
    const channels = document.createElement("span");
    channels.className = "channels";
    channels.textContent =
      `${candidate.channels.join(" + ")} · ${candidate.confidence}`;
    card.append(channels);
    if (candidate.review_reason) {
      const reason = document.createElement("div");
      reason.className = "candidate-reason";
      reason.textContent = candidate.review_reason;
      card.append(reason);
    }
    card.append(candidateContext(candidate));
    for (const choice of ["accept", "reject"]) {
      const button = document.createElement("button");
      button.textContent = choice === "accept" ? "接受" : "拒绝";
      button.disabled = state.payload.locked;
      button.onclick = () => decide(candidate, choice);
      card.append(button);
    }
    host.append(card);
  }
}

function decide(candidate, decision) {
  const exact = exactAnnotation(candidate);
  if (decision === "accept" && !exact) {
    const overlaps = state.annotations.filter(row =>
      row.para_id === candidate.para_id &&
      candidate.start < row.end && row.start < candidate.end);
    if (overlaps.length) {
      if (!confirm(`用候选「${candidate.surface}」替换重叠的人工跨度吗？`))
        return;
      for (const overlap of overlaps) {
        const old = candidateAtGeometry(overlap);
        if (old) state.humanDecisions[old.id] = "reject";
      }
      const overlapKeys = new Set(overlaps.map(
        row => `${row.para_id}:${row.start}:${row.end}`));
      state.annotations = state.annotations.filter(row =>
        !overlapKeys.has(`${row.para_id}:${row.start}:${row.end}`));
    }
    state.annotations.push({
      para_id: candidate.para_id,
      start: candidate.start,
      end: candidate.end,
      surface: candidate.surface,
    });
  }
  if (decision === "reject") {
    state.annotations = state.annotations.filter(row =>
      !(row.para_id === candidate.para_id &&
        row.start === candidate.start && row.end === candidate.end));
  }
  state.humanDecisions[candidate.id] = decision;
  render();
  scheduleSave();
}

function focusCandidate(candidate) {
  state.focusedCandidate = candidate;
  render();
  document.querySelector(
    `.char[data-para-id="${candidate.para_id}"]` +
    `[data-offset="${candidate.start}"]`)
    ?.scrollIntoView({behavior: "smooth", block: "center"});
}

let saveTimer = null;
let saveQueue = Promise.resolve();

function scheduleSave() {
  state.localRevision += 1;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => save().catch(() => {}), 350);
}

async function save() {
  if (!state.payload || state.payload.locked) return;
  clearTimeout(saveTimer);
  saveTimer = null;
  const snapshot = {
    task_id: state.taskId,
    annotations: structuredClone(state.annotations),
    human_decisions: structuredClone(state.humanDecisions),
    geometry_version: GEOMETRY_VERSION,
    revision: state.localRevision,
  };
  const request = saveQueue.then(() => api(
    "/api/production-review/save", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(snapshot),
    }));
  saveQueue = request.catch(() => {});
  try {
    const saved = await request;
    if (state.taskId === snapshot.task_id &&
        state.localRevision === snapshot.revision) {
      state.payload.state = saved;
      state.humanDecisions = structuredClone(saved.human_decisions);
      render();
    }
  } catch (error) {
    setStatus(error.message, true);
    throw error;
  }
}

async function flushSave() {
  if (saveTimer !== null) await save();
  else await saveQueue;
}

async function complete() {
  if (!confirm("完成后本题将永久锁定。确定吗？")) return;
  try {
    await flushSave();
    await api("/api/production-review/complete", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        task_id: state.taskId,
        geometry_version: GEOMETRY_VERSION,
      }),
    });
    const completed = state.taskId;
    await loadIndex(completed);
    const next = state.index.tasks.find(row => !row.complete);
    if (next) await loadTask(next.task_id);
  } catch (error) {
    setStatus(error.message, true);
  }
}

function adjacentTask(delta) {
  const index = state.index.tasks.findIndex(
    row => row.task_id === state.taskId);
  const target = state.index.tasks[index + delta];
  if (target) loadTask(target.task_id);
}

function nextUnresolvedTask() {
  const current = state.index.tasks.findIndex(
    row => row.task_id === state.taskId);
  for (let step = 1; step <= state.index.tasks.length; step++) {
    const row = state.index.tasks[
      (current + step) % state.index.tasks.length];
    if (!row.complete) {
      loadTask(row.task_id);
      return;
    }
  }
  setStatus("所有任务均已完成。");
}

function nextHumanTask() {
  const current = state.index.tasks.findIndex(
    row => row.task_id === state.taskId);
  for (let step = 1; step <= state.index.tasks.length; step++) {
    const row = state.index.tasks[
      (current + step) % state.index.tasks.length];
    if (!row.complete && row.required > 0) {
      loadTask(row.task_id);
      return;
    }
  }
  setStatus("没有剩余需要人工复核的任务。");
}

$("task").onchange = event => loadTask(event.target.value);
$("previous-task").onclick = () => adjacentTask(-1);
$("next-task").onclick = () => adjacentTask(1);
$("next-unresolved").onclick = nextUnresolvedTask;
$("next-human").onclick = nextHumanTask;
$("show-resolved").onchange = renderCandidates;
$("save").onclick = () => save().catch(() => {});
$("complete").onclick = complete;
loadIndex().catch(error => setStatus(error.message, true));
