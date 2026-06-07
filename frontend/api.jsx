/* api.jsx — bridges the UI to the Flask backend.
   Toggle via window.ALUMNI_CONFIG (set in index.html):
     { useApi: true, apiBase: '' }
   When useApi is false, everything falls back to the local engine (engine.jsx)
   so the file still runs as a standalone demo. */

function cfg() { return window.ALUMNI_CONFIG || { useApi: false, apiBase: "" }; }
function base() {
  const configured = cfg().apiBase || "";
  return configured.replace(/\/$/, "");
}

/* ---- build the UI's dataset object from /api/upload's summary ---- */
function dsFromSummary(filename, datasetId, summary) {
  const columns = summary.column_names || [];
  const types = summary.column_types || {};
  const missing = summary.missing_values || {};
  const preview = summary.preview || [];
  const meta = {};
  for (const c of columns) {
    const dt = String(types[c] || "").toLowerCase();
    let type = "text";
    if (/int|float|number|decimal|double/.test(dt)) type = "num";
    else if (/date|time/.test(dt)) type = "date";
    const currency = preview.some(r => String(r[c] ?? "").includes("$"));
    const year = /year/.test(c.toLowerCase()) && type === "num";
    meta[c] = { name: c, type, currency, year, missing: missing[c] || 0 };
  }
  return {
    name: filename, dataset_id: datasetId, columns, meta, rows: preview,
    rows_n: summary.rows || 0, cols_n: summary.columns || columns.length,
    totalMissing: Object.values(missing).reduce((a, b) => a + (b || 0), 0),
  };
}

function datasetIdFromUpload(data) {
  return data.dataset_id || (data.metadata && data.metadata.dataset_id) || "";
}

function pickDisplayColsApi(ds, sortCol) {
  const text = ds.columns.filter(c => ds.meta[c] && ds.meta[c].type === "text");
  const nameCol = text.find(c => /name/i.test(c)) || text[0] || ds.columns[0];
  const extra = ds.columns.filter(c => c !== nameCol && c !== sortCol).slice(0, 2);
  return [nameCol, ...extra, sortCol].filter((v, i, a) => a.indexOf(v) === i);
}

function cleanText(value) {
  if (value == null) return "";
  return String(value).replace(/<[^>\n]*>/g, "").replace(/\u0000/g, "").trim();
}
function normalizeStructuredAnswer(answer, fallbackText) {
  let raw = answer;
  if (raw && raw.answer && typeof raw.answer === "object") raw = raw.answer;

  if (!raw || typeof raw !== "object") {
    const summary = cleanText(fallbackText || raw || "I could not format that response.");
    return { title: "", summary, blocks: summary ? [{ type: "markdown", content: summary }] : [], followups: [] };
  }

  const summary = cleanText(raw.summary || fallbackText || "");
  const normalized = {
    title: cleanText(raw.title || ""),
    summary,
    blocks: [],
    followups: Array.isArray(raw.followups) ? raw.followups.map(cleanText).filter(Boolean).slice(0, 4) : [],
  };

  const blocks = Array.isArray(raw.blocks) ? raw.blocks : [];
  for (const block of blocks.slice(0, 8)) {
    if (!block || typeof block !== "object") continue;
    if (block.type === "markdown") {
      const content = cleanText(block.content || "");
      if (content) normalized.blocks.push({ type: "markdown", content });
    } else if (block.type === "table") {
      const columns = Array.isArray(block.columns) ? block.columns.map(cleanText).filter(Boolean).slice(0, 12) : [];
      if (!columns.length) continue;
      const rows = Array.isArray(block.rows) ? block.rows.slice(0, 20).map(row => {
        if (Array.isArray(row)) return columns.map((_, i) => cleanText(row[i] ?? ""));
        if (row && typeof row === "object") return columns.map(c => cleanText(row[c] ?? ""));
        return [cleanText(row), ...columns.slice(1).map(() => "")];
      }) : [];
      normalized.blocks.push({ type: "table", title: cleanText(block.title || ""), columns, rows, caption: cleanText(block.caption || "") });
    } else if (block.type === "metrics") {
      const items = Array.isArray(block.items) ? block.items.slice(0, 8).map(item => ({
        label: cleanText(item && item.label),
        value: cleanText(item && item.value),
      })).filter(item => item.label || item.value) : [];
      if (items.length) normalized.blocks.push({ type: "metrics", items });
    } else if (block.type === "ranked_list") {
      const items = Array.isArray(block.items) ? block.items.slice(0, 10).map(item => ({
        label: cleanText(item && item.label),
        value: cleanText(item && item.value),
        description: cleanText(item && item.description),
      })).filter(item => item.label || item.value || item.description) : [];
      if (items.length) normalized.blocks.push({ type: "ranked_list", title: cleanText(block.title || ""), items });
    }
  }

  if (!normalized.blocks.length && summary) normalized.blocks.push({ type: "markdown", content: summary });
  return normalized;
}
function opLabel(operation) {
  if (!operation || !operation.type) return null;
  if (operation.type === "group_by_aggregate") return `group_by · ${operation.aggregation}`;
  if (operation.type === "top_rows") return `top_rows · ${operation.ascending ? "asc" : "desc"}`;
  return operation.type;
}
function adaptAnswer(operation, result, answer, ds, answerText) {
  const structured = normalizeStructuredAnswer(answer, answerText);
  return {
    op: opLabel(operation),
    kind: "structured",
    text: structured.summary,
    answer: structured,
    operation,
    result,
  };
}
function adaptLocalAnswer(message) {
  const text = message && message.text ? message.text : "I could not format that response.";
  return {
    ...(message || {}),
    kind: "structured",
    text,
    answer: normalizeStructuredAnswer(null, text),
  };
}

/* ---- network calls ---- */
async function apiUpload(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(base() + "/api/upload", { method: "POST", body: fd });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Upload failed (${res.status})`);
  const datasetId = datasetIdFromUpload(data);
  if (!datasetId) throw new Error("Upload succeeded, but the response did not include dataset_id.");
  return dsFromSummary(data.filename, datasetId, data.summary || {});
}
async function apiPreview(datasetId) {
  if (!datasetId) throw new Error("Cannot load preview because dataset_id is missing.");
  const res = await fetch(base() + `/api/datasets/${encodeURIComponent(datasetId)}/preview`);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Preview failed (${res.status})`);
  const columns = data.columns || data.column_names || [];
  const rows = data.rows || data.preview || [];
  const preview = {
    dataset_id: data.dataset_id || datasetId,
    filename: data.filename,
    columns,
    rows,
    row_count: data.row_count,
    column_count: data.column_count,
  };
  if (data.missing_count != null) preview.missing_count = data.missing_count;
  if (data.data_types || data.column_types) preview.data_types = data.data_types || data.column_types;
  if (data.missing_values) preview.missing_values = data.missing_values;
  return preview;
}
async function apiSummary(datasetId) {
  if (!datasetId) throw new Error("Cannot load summary because dataset_id is missing.");
  const res = await fetch(base() + `/api/datasets/${encodeURIComponent(datasetId)}/summary`);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Summary failed (${res.status})`);
  return data;
}
async function apiAsk(ds, question) {
  const res = await fetch(base() + "/api/ask", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: ds.dataset_id, question }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) return adaptAnswer(null, null, null, ds, data.error || `Request failed (${res.status})`);
  return adaptAnswer(data.operation, data.result, data.answer, ds, data.answer_text);
}

/* ---- local fallback (engine.jsx) ---- */
function localLoad(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = e => { try { const p = window.profile(window.parseCSV(e.target.result)); p.name = file.name; resolve(p); } catch (err) { reject(err); } };
    r.onerror = () => reject(new Error("Could not read file"));
    r.readAsText(file);
  });
}

window.Alumni = {
  isApi: () => !!cfg().useApi,
  load(file) { return cfg().useApi ? apiUpload(file) : localLoad(file); },
  preview(dsOrId) {
    const datasetId = typeof dsOrId === "string" ? dsOrId : dsOrId && dsOrId.dataset_id;
    if (cfg().useApi && datasetId) return apiPreview(datasetId);
    const ds = typeof dsOrId === "string" ? null : dsOrId;
    return Promise.resolve({ dataset_id: datasetId, columns: ds ? ds.columns : [], rows: ds ? ds.rows.slice(0, 10) : [] });
  },
  summary(dsOrId) {
    const datasetId = typeof dsOrId === "string" ? dsOrId : dsOrId && dsOrId.dataset_id;
    if (cfg().useApi && datasetId) return apiSummary(datasetId);
    return Promise.resolve(null);
  },
  ask(ds, q) { return cfg().useApi ? apiAsk(ds, q) : Promise.resolve(window.ask(ds, q)).then(adaptLocalAnswer); },
};
