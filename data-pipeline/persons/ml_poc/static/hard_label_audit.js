"use strict";

const state = {payloads: [], pending: false};
const labels = [
  ["E", "exact_person", "精确人名"],
  ["W", "wrong_boundary", "有人名但边界错误"],
  ["N", "not_person", "非个人名"],
  ["U", "uncertain", "不确定"],
];
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

function unresolved() {
  for (const row of state.payloads) {
    for (const candidate of row.task.candidates) {
      if (!row.state.decisions[candidate.candidate_id]) return {row, candidate};
    }
  }
  return null;
}

function setError(error) {
  $("status").textContent = error.message;
  $("status").className = "error";
}

async function decide(row, candidate, label) {
  if (state.pending) return;
  state.pending = true;
  render();
  try {
    const result = await api("/api/hard-label/decision", {
      task_id: row.task.task_id,
      candidate_id: candidate.candidate_id,
      label,
    });
    row.state = result;
  } finally {
    state.pending = false;
    render();
    requestAnimationFrame(() => {
      document.querySelector(".candidate.active")?.scrollIntoView({
        block: "center", behavior: "smooth",
      });
    });
  }
}

function candidateCard(row, candidate, paragraphs, active) {
  const card = document.createElement("div");
  const done = row.state.decisions[candidate.candidate_id];
  card.className = `candidate ${done ? "done" : ""} ${active ? "active" : ""}`;
  const text = paragraphs.get(candidate.para_id);
  const context = document.createElement("div");
  context.append(document.createTextNode(codepoints(
    text, Math.max(0, candidate.start - 35), candidate.start)));
  const mark = document.createElement("mark");
  mark.textContent = candidate.surface;
  context.append(mark, document.createTextNode(codepoints(
    text, candidate.end, Math.min([...text].length, candidate.end + 35))));
  card.append(context);
  if (done) {
    const result = document.createElement("strong");
    result.textContent = `已锁定：${done}`;
    card.append(result);
  } else {
    for (const [key, label, title] of labels) {
      const button = document.createElement("button");
      button.textContent = `${key} · ${title}`;
      button.disabled = state.pending;
      button.onclick = () => decide(row, candidate, label).catch(setError);
      card.append(button);
    }
  }
  return card;
}

function render() {
  const active = unresolved();
  const sections = state.payloads.map(row => {
    const section = document.createElement("section");
    section.className = "jie";
    const title = document.createElement("h2");
    title.textContent = `卷 ${row.task.juan} · jie ${row.task.jie_index}`;
    section.append(title);
    const paragraphs = paragraphMap(row.task.jie);
    for (const segment of row.task.jie.segments) {
      const paragraph = document.createElement("div");
      paragraph.className = "paragraph";
      paragraph.textContent = paragraphs.get(segment.para_id);
      section.append(paragraph);
    }
    for (const candidate of row.task.candidates) {
      section.append(candidateCard(
        row, candidate, paragraphs,
        active?.row === row && active?.candidate === candidate));
    }
    return section;
  });
  $("tasks").replaceChildren(...sections);
  const total = state.payloads.reduce(
    (sum, row) => sum + row.task.candidates.length, 0);
  const decided = state.payloads.reduce(
    (sum, row) => sum + Object.keys(row.state.decisions).length, 0);
  $("progress").textContent = ` · ${decided}/${total}`;
  $("status").textContent = decided === total ? "全部初判已锁定。" : "";
}

document.addEventListener("keydown", event => {
  if (event.repeat || event.ctrlKey || event.altKey || event.metaKey) return;
  const active = unresolved();
  if (!active || state.pending) return;
  const selected = labels.find(([key]) => key.toLowerCase() === event.key.toLowerCase());
  if (selected) decide(active.row, active.candidate, selected[1]).catch(setError);
});

api("/api/hard-label/all").then(result => {
  state.payloads = result.payloads;
  render();
}).catch(setError);
