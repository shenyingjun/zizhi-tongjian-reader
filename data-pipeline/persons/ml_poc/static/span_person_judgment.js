"use strict";

const state = {index: null, taskId: null, payload: null};
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

function context(candidate, paragraphs) {
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

async function refreshIndex(preferred = state.taskId) {
  state.index = await api("/api/safe-negative-audit/index");
  const select = $("task");
  select.replaceChildren();
  for (const row of state.index.tasks) {
    const option = document.createElement("option");
    option.value = row.task_id;
    option.textContent =
      `${row.position}/${row.total} · 卷${row.juan} jie ${row.jie_index}` +
      ` · ${row.decided}/${row.candidates}${row.complete ? " · 已锁定" : ""}`;
    select.append(option);
  }
  const open = state.index.tasks.find(row => !row.complete);
  await loadTask(preferred || open?.task_id || state.index.tasks[0].task_id);
}

async function loadTask(taskId) {
  state.taskId = taskId;
  state.payload = await api(
    `/api/safe-negative-audit/task?task_id=${encodeURIComponent(taskId)}`);
  $("task").value = taskId;
  render();
}

function addButton(host, text, action, className = "") {
  const button = document.createElement("button");
  button.textContent = text;
  button.className = className;
  button.onclick = async () => {
    try {
      await action();
      await refreshIndex(state.taskId);
    } catch (error) {
      $("status").textContent = error.message;
      $("status").className = "error";
    }
  };
  host.append(button);
}

function render() {
  const task = state.payload.task;
  const decisions = state.payload.state.decisions;
  const paragraphs = paragraphMap(task.jie);
  $("title").textContent = `卷 ${task.juan} · jie_index ${task.jie_index}`;
  $("text").replaceChildren(...task.jie.segments.map(segment => {
    const div = document.createElement("div");
    div.className = "paragraph";
    div.textContent = paragraphs.get(segment.para_id);
    return div;
  }));
  const host = $("candidates");
  host.replaceChildren();
  for (const candidate of task.candidates) {
    const decision = decisions[candidate.candidate_id];
    const card = document.createElement("div");
    card.className = `candidate ${decision?.final ? "done" : ""}`;
    card.append(context(candidate, paragraphs));
    const metadata = document.createElement("div");
    metadata.textContent =
      `段 ${candidate.para_id} [${candidate.start},${candidate.end})`;
    card.prepend(metadata);
    if (!decision) {
      const prompt = document.createElement("p");
      prompt.textContent = "先独立判断：高亮精确跨度是否是人名出现？";
      card.append(prompt);
      addButton(card, "明确不是人名", () => api(
        "/api/safe-negative-audit/initial", {
          task_id: state.taskId,
          candidate_id: candidate.candidate_id,
          label: "not_person",
        }));
      addButton(card, "是人名或边界可疑（排除）", () => api(
        "/api/safe-negative-audit/initial", {
          task_id: state.taskId,
          candidate_id: candidate.candidate_id,
          label: "exclude_from_negative_training",
        }), "danger");
    } else if (!decision.rationales_revealed &&
               decision.final !== "exclude_from_negative_training") {
      addButton(card, "锁定初判并展开 AI 理由", () => api(
        "/api/safe-negative-audit/reveal", {
          task_id: state.taskId, candidate_id: candidate.candidate_id,
        }));
    } else if (decision.rationales_revealed && !decision.final) {
      for (const judgment of
           state.payload.revealed_rationales[candidate.candidate_id]) {
        const div = document.createElement("div");
        div.className = "rationale";
        div.textContent = `${judgment.model}: ${judgment.rationale}`;
        card.append(div);
      }
      addButton(card, "维持：不是人名", () => api(
        "/api/safe-negative-audit/final", {
          task_id: state.taskId,
          candidate_id: candidate.candidate_id,
          label: "not_person",
        }));
      addButton(card, "改为排除", () => api(
        "/api/safe-negative-audit/final", {
          task_id: state.taskId,
          candidate_id: candidate.candidate_id,
          label: "exclude_from_negative_training",
        }), "danger");
    } else {
      const result = document.createElement("strong");
      result.textContent = decision.final === "not_person"
        ? "已完成：不是人名" : "已排除；整轮停止";
      card.append(result);
    }
    host.append(card);
  }
  const row = state.index.tasks.find(item => item.task_id === state.taskId);
  $("progress").textContent =
    `总进度 ${state.index.decided}/${state.index.total_candidates}`;
  $("status").textContent = state.index.stopped
    ? "发现排除项：Revision 9 已 fail-closed 停止。" : "";
  $("status").className = state.index.stopped ? "stopped" : "";
  $("complete").disabled = row.complete || state.index.stopped;
  const position = state.index.tasks.indexOf(row);
  $("previous").disabled = position === 0;
  $("next").disabled = position === state.index.tasks.length - 1;
}

$("task").onchange = () => loadTask($("task").value);
$("previous").onclick = () => {
  const row = state.index.tasks.find(item => item.task_id === state.taskId);
  loadTask(state.index.tasks[state.index.tasks.indexOf(row) - 1].task_id);
};
$("next").onclick = () => {
  const row = state.index.tasks.find(item => item.task_id === state.taskId);
  loadTask(state.index.tasks[state.index.tasks.indexOf(row) + 1].task_id);
};
$("complete").onclick = async () => {
  try {
    await api("/api/safe-negative-audit/complete", {task_id: state.taskId});
    await refreshIndex();
  } catch (error) {
    $("status").textContent = error.message;
    $("status").className = "error";
  }
};

refreshIndex().catch(error => {
  $("status").textContent = error.message;
  $("status").className = "error";
});
