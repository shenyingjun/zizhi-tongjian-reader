"use strict";

const state = {payloads: [], stopped: false};
const $ = id => document.getElementById(id);

async function api(url, body = null) {
  const response = await fetch(url, body === null ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.statusText);
  return value;
}

function codepoints(text, start, end) {
  return [...text].slice(start, end).join("");
}

function paragraphMap(jie) {
  return new Map(jie.segments.map(segment => [
    segment.para_id,
    codepoints(jie.text, segment.assembled_start, segment.assembled_end),
  ]));
}

function candidateContext(candidate, paragraphs) {
  const text = paragraphs.get(candidate.para_id);
  const radius = 22;
  const host = document.createElement("div");
  host.append(document.createTextNode(codepoints(
    text, Math.max(0, candidate.start - radius), candidate.start)));
  const mark = document.createElement("mark");
  mark.textContent = candidate.surface;
  host.append(mark, document.createTextNode(codepoints(
    text, candidate.end, Math.min([...text].length, candidate.end + radius))));
  return host;
}

function setError(error) {
  $("status").textContent = error.message;
  $("status").className = "error";
}

function payload(taskId) {
  return state.payloads.find(row => row.task.task_id === taskId);
}

function firstUnresolved() {
  for (const row of state.payloads) {
    const candidate = row.task.candidates.find(
      item => !row.state.decisions[item.candidate_id]);
    if (candidate) return {row, candidate};
  }
  return null;
}

function addButton(host, text, action, className = "") {
  const button = document.createElement("button");
  button.textContent = text;
  button.className = className;
  button.onclick = () => action().catch(setError);
  host.append(button);
}

async function initialCandidate(taskId, candidateId, label) {
  const decision = await api("/api/safe-negative-audit/initial", {
    task_id: taskId, candidate_id: candidateId, label,
  });
  payload(taskId).state.decisions[candidateId] = decision;
  if (label === "exclude_from_negative_training") state.stopped = true;
  render();
  if (!state.stopped) {
    requestAnimationFrame(() => {
      document.querySelector(".candidate.active")?.scrollIntoView({
        block: "center", behavior: "smooth",
      });
    });
  }
}

async function revealTask(taskId) {
  const result = await api("/api/safe-negative-audit/reveal-task", {
    task_id: taskId,
  });
  const row = payload(taskId);
  row.revealed_rationales = result.judgments;
  for (const decision of Object.values(row.state.decisions))
    decision.rationales_revealed = true;
  render();
  document.getElementById(`jie-${taskId}`)?.scrollIntoView({block: "start"});
}

async function excludeCandidate(taskId, candidateId) {
  const decision = await api("/api/safe-negative-audit/final", {
    task_id: taskId,
    candidate_id: candidateId,
    label: "exclude_from_negative_training",
  });
  payload(taskId).state.decisions[candidateId] = decision;
  state.stopped = true;
  render();
}

async function confirmTask(taskId) {
  const completed = await api("/api/safe-negative-audit/confirm-task", {
    task_id: taskId,
  });
  payload(taskId).state = completed;
  render();
}

async function revealAll() {
  const result = await api("/api/safe-negative-audit/reveal-all", {});
  for (const row of state.payloads) {
    row.revealed_rationales = result.judgments[row.task.task_id];
    for (const decision of Object.values(row.state.decisions))
      decision.rationales_revealed = true;
  }
  render();
  window.scrollTo({top: 0, behavior: "smooth"});
}

async function confirmAll() {
  await api("/api/safe-negative-audit/confirm-all", {});
  for (const row of state.payloads) {
    for (const decision of Object.values(row.state.decisions))
      decision.final = "not_person";
    row.state.complete = true;
  }
  render();
}

function renderCandidate(row, candidate, paragraphs, active) {
  const decision = row.state.decisions[candidate.candidate_id];
  const card = document.createElement("div");
  card.className =
    `candidate ${decision?.final ? "done" : ""} ${active ? "active" : ""}`;
  const metadata = document.createElement("div");
  metadata.textContent =
    `段 ${candidate.para_id} [${candidate.start},${candidate.end})`;
  card.append(metadata, candidateContext(candidate, paragraphs));
  if (!decision) {
    const prompt = document.createElement("p");
    prompt.textContent = "高亮精确跨度是否是人名出现？";
    card.append(prompt);
    addButton(card, "明确不是人名", () => initialCandidate(
      row.task.task_id, candidate.candidate_id, "not_person"));
    addButton(card, "是人名或边界可疑（排除）", () => initialCandidate(
      row.task.task_id, candidate.candidate_id,
      "exclude_from_negative_training"), "danger");
  } else if (decision.rationales_revealed) {
    for (const judgment of
         row.revealed_rationales[candidate.candidate_id] || []) {
      const rationale = document.createElement("div");
      rationale.className = "rationale";
      rationale.textContent = `${judgment.model}: ${judgment.rationale}`;
      card.append(rationale);
    }
    if (!decision.final) {
      addButton(card, "改为排除", () => excludeCandidate(
        row.task.task_id, candidate.candidate_id), "danger");
    } else {
      const result = document.createElement("strong");
      result.textContent = decision.final === "not_person"
        ? "已完成：不是人名" : "已排除；整轮停止";
      card.append(result);
    }
  } else if (decision.final === "exclude_from_negative_training") {
    const result = document.createElement("strong");
    result.textContent = "已排除；整轮停止";
    card.append(result);
  } else {
    const waiting = document.createElement("em");
    waiting.textContent = "初判已锁定；请完成本节其余初判。";
    card.append(waiting);
  }
  return card;
}

function renderTask(row, active) {
  const section = document.createElement("section");
  section.className = "jie";
  section.id = `jie-${row.task.task_id}`;
  const title = document.createElement("h2");
  title.textContent =
    `卷 ${row.task.juan} · jie_index ${row.task.jie_index}`;
  section.append(title);
  const paragraphs = paragraphMap(row.task.jie);
  for (const segment of row.task.jie.segments) {
    const paragraph = document.createElement("div");
    paragraph.className = "paragraph";
    paragraph.textContent = paragraphs.get(segment.para_id);
    section.append(paragraph);
  }
  for (const candidate of row.task.candidates) {
    section.append(renderCandidate(
      row, candidate, paragraphs,
      active?.row === row && active?.candidate === candidate));
  }
  const decisions = row.state.decisions;
  const allInitial = row.task.candidates.every(
    candidate => decisions[candidate.candidate_id]);
  const allRevealed = allInitial && row.task.candidates.every(
    candidate => decisions[candidate.candidate_id].rationales_revealed);
  const unfinished = allRevealed && row.task.candidates.some(
    candidate => !decisions[candidate.candidate_id].final);
  if (!row.state.complete && !state.stopped && allInitial && !allRevealed) {
    addButton(section, "展开本节全部 AI 理由", () => revealTask(
      row.task.task_id));
  }
  if (!row.state.complete && !state.stopped && unfinished) {
    addButton(section, "维持本节全部判断并锁定", () => confirmTask(
      row.task.task_id));
  }
  return section;
}

function render() {
  const active = firstUnresolved();
  const host = $("tasks");
  host.replaceChildren(...state.payloads.map(row => renderTask(row, active)));
  const total = state.payloads.reduce(
    (sum, row) => sum + row.task.candidates.length, 0);
  const decided = state.payloads.reduce(
    (sum, row) => sum + Object.values(row.state.decisions).filter(
      decision => decision.final).length, 0);
  $("progress").textContent =
    ` · ${decided}/${total}` +
    (active ? " · 快捷键 N：不是人名，X：排除" : "");
  $("status").textContent = state.stopped
    ? "发现排除项：Revision 9 已 fail-closed 停止。" : "";
  $("status").className = state.stopped ? "stopped" : "";
  const allInitial = state.payloads.every(row =>
    row.task.candidates.every(
      candidate => row.state.decisions[candidate.candidate_id]));
  const allRevealed = allInitial && state.payloads.every(row =>
    row.task.candidates.every(
      candidate =>
        row.state.decisions[candidate.candidate_id].rationales_revealed));
  const allComplete = state.payloads.every(row => row.state.complete);
  $("reveal-all").hidden =
    state.stopped || !allInitial || allRevealed || allComplete;
  $("confirm-all").hidden =
    state.stopped || !allRevealed || allComplete;
}

async function loadAll() {
  const result = await api("/api/safe-negative-audit/all");
  state.payloads = result.payloads;
  state.stopped = result.stopped;
  render();
}

document.addEventListener("keydown", event => {
  if (event.repeat || event.ctrlKey || event.altKey || event.metaKey) return;
  const active = firstUnresolved();
  if (!active || state.stopped) return;
  if (event.key.toLowerCase() === "n")
    initialCandidate(
      active.row.task.task_id, active.candidate.candidate_id,
      "not_person").catch(setError);
  if (event.key.toLowerCase() === "x")
    initialCandidate(
      active.row.task.task_id, active.candidate.candidate_id,
      "exclude_from_negative_training").catch(setError);
});
$("reveal-all").onclick = () => revealAll().catch(setError);
$("confirm-all").onclick = () => confirmAll().catch(setError);

loadAll().catch(setError);
