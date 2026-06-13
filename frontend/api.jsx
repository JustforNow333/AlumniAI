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

function cleanText(value) {
  if (value == null) return "";
  return String(value).replace(/<[^>\n]*>/g, "").replace(/\u0000/g, "").trim();
}
function normalizeColumnKey(value) {
  return cleanText(value).toLowerCase().replace(/[^a-z0-9]+/g, "");
}
function canonicalDisplayColumn(column) {
  const key = normalizeColumnKey(column);
  if (key === "firstname" || key === "givenname") return "First Name";
  if (key === "lastname" || key === "surname" || key === "familyname") return "Last Name";
  if (key === "occupation" || key === "jobtitle" || key === "job" || key === "role" || key === "position") return "Occupation";
  if (key === "employer" || key === "company" || key === "organization" || key === "organisation" || key === "workplace") return "Employer";
  if (isLinkedInColumn(column)) return "LinkedIn URL";
  return cleanText(column);
}
function isLinkedInColumn(column) {
  const key = normalizeColumnKey(column);
  return key === "linkedin" || key === "linkedinurl" || key === "linkedinprofile" || key === "linkedinprofileurl";
}
function linkedInHref(value) {
  const text = cleanText(value);
  if (!text) return "";
  if (/^https?:\/\//i.test(text)) return text;
  if (/^www\./i.test(text)) return `https://${text}`;
  if (/linkedin\.com/i.test(text)) return `https://${text}`;
  return "";
}
function isDebugColumn(column) {
  return [
    "matchreason",
    "rawmatchreason",
    "score",
    "internalscore",
    "matchedterms",
    "confidence",
    "classificationreason",
    "uncertaintyreason",
    "modelreason",
    "internalreason",
    "matchedcolumn",
    "matchedterm",
    "classification",
  ].includes(normalizeColumnKey(column));
}
function isPeopleResult(result) {
  return !!(result && result.intent === "people_filter" && result.entity === "alumni");
}
function peopleVisibleColumns(result) {
  const columns = Array.isArray(result && result.visible_columns) && result.visible_columns.length
    ? result.visible_columns
    : ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL"];
  return columns.map(canonicalDisplayColumn).filter(Boolean);
}
function findColumn(columns, desired) {
  const desiredKey = normalizeColumnKey(canonicalDisplayColumn(desired));
  return columns.find(column => {
    const key = normalizeColumnKey(column);
    const canonicalKey = normalizeColumnKey(canonicalDisplayColumn(column));
    return key === desiredKey || canonicalKey === desiredKey;
  });
}
function sanitizeTableBlock(block, result) {
  const debugMode = !!(cfg().debug || cfg().debugMode);
  const columns = Array.isArray(block.columns) ? block.columns.map(cleanText).filter(Boolean).slice(0, 12) : [];
  if (!columns.length) return null;
  const rows = Array.isArray(block.rows) ? block.rows.slice(0, 100) : [];

  if (!isPeopleResult(result)) {
    const kept = debugMode ? columns : columns.filter(column => !isDebugColumn(column));
    const indices = kept.map(column => columns.indexOf(column));
    return {
      type: "table",
      title: cleanText(block.title || ""),
      columns: kept,
      rows: rows.map(row => {
        if (Array.isArray(row)) return indices.map(i => cleanText(row[i] ?? ""));
        if (row && typeof row === "object") return kept.map(c => cleanText(row[c] ?? ""));
        return [cleanText(row), ...kept.slice(1).map(() => "")];
      }),
      caption: cleanText(block.caption || ""),
    };
  }

  const visible = peopleVisibleColumns(result);
  const resolved = visible.map(header => ({ header, source: findColumn(columns, header) })).filter(item => item.source);
  if (!resolved.length) return null;
  const indices = resolved.map(item => columns.indexOf(item.source));
  return {
    type: "table",
    title: cleanText(block.title || ""),
    columns: resolved.map(item => item.header),
    rows: rows.map(row => {
      if (Array.isArray(row)) return indices.map(i => cleanText(row[i] ?? ""));
      if (row && typeof row === "object") return resolved.map(item => cleanText(row[item.source] ?? row[item.header] ?? ""));
      return resolved.map((_, i) => i === 0 ? cleanText(row) : "");
    }),
    caption: cleanText(block.caption || ""),
  };
}
function peopleMetricsBlock(result) {
  if (!isPeopleResult(result)) return null;
  const total = result.total_matches;
  if (total == null) return null;
  const items = [{ label: cleanText(result.answer_label || "Alumni matching criteria"), value: cleanText(total) }];
  if (result.displayed_count != null && Number(result.displayed_count) !== Number(total)) {
    items.push({ label: "Showing", value: cleanText(result.displayed_count) });
  }
  if (result.uncertain_count) {
    items.push({ label: "Uncertain not counted", value: cleanText(result.uncertain_count) });
  }
  if (result.adjacent_count) {
    items.push({ label: "Adjacent not counted", value: cleanText(result.adjacent_count) });
  }
  if (result.adjacent_included_count) {
    items.push({ label: "Adjacent included", value: cleanText(result.adjacent_included_count) });
  }
  return { type: "metrics", items };
}
function sanitizeStructuredAnswer(answer, result) {
  const peopleMetrics = peopleMetricsBlock(result);
  let replacedMetrics = false;
  const blocks = [];
  for (const block of answer.blocks || []) {
    if (!block || typeof block !== "object") continue;
    if (block.type === "table") {
      const sanitized = sanitizeTableBlock(block, result);
      if (sanitized && sanitized.columns.length) blocks.push(sanitized);
    } else if (block.type === "metrics" && peopleMetrics) {
      if (!replacedMetrics) {
        blocks.push(peopleMetrics);
        replacedMetrics = true;
      }
    } else {
      blocks.push(block);
    }
  }
  if (peopleMetrics && !replacedMetrics) blocks.unshift(peopleMetrics);
  return { ...answer, blocks };
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
      const normalizedTable = sanitizeTableBlock(block, null);
      if (normalizedTable) normalized.blocks.push(normalizedTable);
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
function buildAskResponsePayload(data, structured, operation, result, answerText, ds) {
  const payload = {
    dataset_id: cleanText((data && data.dataset_id) || (ds && ds.dataset_id) || ""),
    question: cleanText(data && data.question),
    answer: structured,
    answer_text: cleanText(answerText || (structured && structured.summary) || ""),
    operation: operation || null,
    result: result || null,
  };
  if (data && data.analysis_intent) payload.analysis_intent = data.analysis_intent;
  if (data && data.analysis_plan) payload.analysis_plan = data.analysis_plan;
  if (data && Array.isArray(data.operation_results)) payload.operation_results = data.operation_results;
  return payload;
}
function adaptAnswer(operation, result, answer, ds, answerText, data) {
  const structured = sanitizeStructuredAnswer(normalizeStructuredAnswer(answer, answerText), result);
  const responsePayload = buildAskResponsePayload(data || {}, structured, operation, result, answerText, ds);
  return {
    op: opLabel(operation),
    kind: "structured",
    text: structured.summary,
    answer: structured,
    operation,
    result,
    response_payload: responsePayload,
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
function normalizeDatasetEntry(entry) {
  if (!entry || typeof entry !== "object") return null;
  const datasetId = entry.dataset_id || "";
  if (!datasetId) return null;
  return {
    dataset_id: datasetId,
    display_name: cleanText(entry.display_name || entry.original_filename || "Untitled dataset"),
    original_filename: cleanText(entry.original_filename || ""),
    stored_filename: entry.stored_filename || "",
    uploaded_at: entry.uploaded_at || "",
    row_count: entry.row_count != null ? Number(entry.row_count) || 0 : null,
    column_count: entry.column_count != null ? Number(entry.column_count) || 0 : null,
    columns: Array.isArray(entry.columns) ? entry.columns : [],
    file_type: entry.file_type || "",
    status: entry.status === "missing" ? "missing" : "ready",
  };
}
async function apiDatasets() {
  const res = await fetch(base() + "/api/datasets");
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Could not load datasets (${res.status})`);
  const list = Array.isArray(data.datasets) ? data.datasets : Array.isArray(data) ? data : [];
  return list.map(normalizeDatasetEntry).filter(Boolean);
}
async function apiRenameDataset(datasetId, displayName) {
  if (!datasetId) throw new Error("Cannot rename because dataset_id is missing.");
  const res = await fetch(base() + `/api/datasets/${encodeURIComponent(datasetId)}`, {
    method: "PATCH", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ display_name: displayName }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Rename failed (${res.status})`);
  return normalizeDatasetEntry(data);
}
async function apiDeleteDataset(datasetId) {
  if (!datasetId) throw new Error("Cannot delete because dataset_id is missing.");
  const res = await fetch(base() + `/api/datasets/${encodeURIComponent(datasetId)}`, { method: "DELETE" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Delete failed (${res.status})`);
  return data;
}
/* ---- saved insights (manually saved answer snapshots; not history) ---- */
function defaultInsightTitle(question) {
  const text = String(question || "").replace(/\s+/g, " ").trim().replace(/[?.!\s]+$/, "");
  if (!text) return "Saved insight";
  if (text.length <= 80) return text;
  const clipped = text.slice(0, 80).replace(/\s+\S*$/, "").trim();
  return (clipped || text.slice(0, 80)) + "…";
}
function insightTextFromAnswer(answer, fallbackText) {
  // Flatten a structured answer into the plain-text snapshot stored with the
  // insight: summary plus markdown/metrics content, never debug fields.
  if (!answer || typeof answer !== "object") return cleanText(fallbackText || "");
  const parts = [];
  if (answer.summary) parts.push(cleanText(answer.summary));
  for (const block of Array.isArray(answer.blocks) ? answer.blocks : []) {
    if (!block || typeof block !== "object") continue;
    if (block.type === "markdown") {
      const content = cleanText(block.content);
      if (content && content !== parts[0]) parts.push(content);
    } else if (block.type === "metrics") {
      const lines = (block.items || [])
        .map(item => item && (item.label || item.value) ? `${cleanText(item.label)}: ${cleanText(item.value)}` : "")
        .filter(Boolean);
      if (lines.length) parts.push(lines.join("\n"));
    } else if (block.type === "table") {
      const rows = Array.isArray(block.rows) ? block.rows.length : 0;
      if (rows) parts.push(`Table: ${rows} row${rows === 1 ? "" : "s"} (${(block.columns || []).join(", ")})`);
      if (block.caption) parts.push(cleanText(block.caption));
    }
  }
  const text = parts.filter(Boolean).join("\n\n");
  return text || cleanText(fallbackText || "");
}
function normalizeInsightResponsePayload(payload, fallbackText) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return null;
  const result = payload.result || (Array.isArray(payload.operation_results) ? payload.operation_results[0] : null);
  const answer = sanitizeStructuredAnswer(
    normalizeStructuredAnswer(payload.answer, payload.answer_text || fallbackText),
    result
  );
  return {
    ...payload,
    answer,
    answer_text: cleanText(payload.answer_text || answer.summary || fallbackText || ""),
    operation: payload.operation || null,
    result: result || null,
  };
}
function normalizeInsightEntry(entry) {
  if (!entry || typeof entry !== "object") return null;
  const insightId = entry.insight_id || "";
  if (!insightId) return null;
  const answer = typeof entry.answer === "string" ? entry.answer : cleanText(entry.answer_text || entry.answer || "");
  return {
    id: entry.id || insightId,
    insight_id: insightId,
    dataset_id: entry.dataset_id || "",
    dataset_filename: cleanText(entry.dataset_filename || entry.dataset_name_snapshot || "Unknown dataset"),
    dataset_name_snapshot: cleanText(entry.dataset_name_snapshot || entry.dataset_filename || "Unknown dataset"),
    dataset_status: entry.dataset_status === "deleted" ? "deleted" : "ready",
    title: cleanText(entry.title || "") || defaultInsightTitle(entry.question),
    question: cleanText(entry.question || ""),
    answer,
    answer_text: answer,
    response_payload: normalizeInsightResponsePayload(entry.response_payload, answer),
    created_at: entry.created_at || "",
    updated_at: entry.updated_at || "",
    tags: Array.isArray(entry.tags) ? entry.tags.map(cleanText).filter(Boolean) : [],
    metadata: entry.metadata && typeof entry.metadata === "object" ? entry.metadata : {},
  };
}
async function apiInsights(datasetId) {
  const query = datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : "";
  const res = await fetch(base() + "/api/insights" + query);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Could not load saved insights (${res.status})`);
  const list = Array.isArray(data.insights) ? data.insights : Array.isArray(data) ? data : [];
  return list.map(normalizeInsightEntry).filter(Boolean);
}
async function apiGetInsight(insightId) {
  if (!insightId) throw new Error("Cannot load insight because insight_id is missing.");
  const res = await fetch(base() + `/api/insights/${encodeURIComponent(insightId)}`);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Could not load insight (${res.status})`);
  return normalizeInsightEntry(data);
}
async function apiSaveInsight({ dataset_id, title, question, answer, tags, response_payload }) {
  if (!dataset_id) throw new Error("Cannot save an insight without an active dataset.");
  if (!String(question || "").trim()) throw new Error("Cannot save an insight without the original question.");
  if (!String(answer || "").trim()) throw new Error("Cannot save an insight without a completed answer.");
  const body = { dataset_id, question, answer, title: title || defaultInsightTitle(question) };
  if (Array.isArray(tags) && tags.length) body.tags = tags;
  if (response_payload && typeof response_payload === "object" && !Array.isArray(response_payload)) {
    body.response_payload = response_payload;
  }
  const res = await fetch(base() + "/api/insights", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Save failed (${res.status})`);
  return normalizeInsightEntry(data);
}
async function apiRenameInsight(insightId, title) {
  if (!insightId) throw new Error("Cannot rename because insight_id is missing.");
  const res = await fetch(base() + `/api/insights/${encodeURIComponent(insightId)}`, {
    method: "PATCH", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Rename failed (${res.status})`);
  return normalizeInsightEntry(data);
}
async function apiDeleteInsight(insightId) {
  if (!insightId) throw new Error("Cannot delete because insight_id is missing.");
  const res = await fetch(base() + `/api/insights/${encodeURIComponent(insightId)}`, { method: "DELETE" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Delete failed (${res.status})`);
  return data;
}
async function apiAsk(ds, question) {
  const res = await fetch(base() + "/api/ask", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: ds.dataset_id, question }),
  });
  const data = await res.json().catch(() => ({}));
  const payloadData = { ...data, question: data.question || question };
  if (!res.ok) return adaptAnswer(null, null, null, ds, data.error || `Request failed (${res.status})`, payloadData);
  return adaptAnswer(data.operation, data.result, data.answer, ds, data.answer_text, payloadData);
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
  datasets() { return cfg().useApi ? apiDatasets() : Promise.resolve([]); },
  renameDataset(datasetId, displayName) {
    if (!cfg().useApi) return Promise.reject(new Error("Dataset library requires API mode."));
    return apiRenameDataset(datasetId, displayName);
  },
  deleteDataset(datasetId) {
    if (!cfg().useApi) return Promise.reject(new Error("Dataset library requires API mode."));
    return apiDeleteDataset(datasetId);
  },
  insights(datasetId) { return cfg().useApi ? apiInsights(datasetId) : Promise.resolve([]); },
  insight(insightId) {
    if (!cfg().useApi) return Promise.reject(new Error("Saved insights require API mode."));
    return apiGetInsight(insightId);
  },
  saveInsight(payload) {
    if (!cfg().useApi) return Promise.reject(new Error("Saved insights require API mode."));
    return apiSaveInsight(payload || {});
  },
  renameInsight(insightId, title) {
    if (!cfg().useApi) return Promise.reject(new Error("Saved insights require API mode."));
    return apiRenameInsight(insightId, title);
  },
  deleteInsight(insightId) {
    if (!cfg().useApi) return Promise.reject(new Error("Saved insights require API mode."));
    return apiDeleteInsight(insightId);
  },
  helpers: {
    canonicalDisplayColumn,
    isLinkedInColumn,
    linkedInHref,
    isDebugColumn,
    defaultInsightTitle,
    insightTextFromAnswer,
  },
  _test: {
    canonicalDisplayColumn,
    isLinkedInColumn,
    linkedInHref,
    isDebugColumn,
    sanitizeStructuredAnswer,
    normalizeInsightResponsePayload,
    buildAskResponsePayload,
    normalizeDatasetEntry,
    normalizeInsightEntry,
    defaultInsightTitle,
    insightTextFromAnswer,
  },
};
