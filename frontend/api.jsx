/* api.jsx — bridges the UI to the Flask backend.
   Toggle via window.ALUMNI_CONFIG (set in index.html):
     { useApi: true, apiBase: 'http://localhost:5000' }
   When useApi is false, everything falls back to the local engine (engine.jsx)
   so the file still runs as a standalone demo. */

function cfg() { return window.ALUMNI_CONFIG || { useApi: false, apiBase: "" }; }
function base() { return (cfg().apiBase || "").replace(/\/$/, ""); }

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

function pickDisplayColsApi(ds, sortCol) {
  const text = ds.columns.filter(c => ds.meta[c] && ds.meta[c].type === "text");
  const nameCol = text.find(c => /name/i.test(c)) || text[0] || ds.columns[0];
  const extra = ds.columns.filter(c => c !== nameCol && c !== sortCol).slice(0, 2);
  return [nameCol, ...extra, sortCol].filter((v, i, a) => a.indexOf(v) === i);
}

/* ---- map backend {operation, result, answer} → a UI message ---- */
function adaptAnswer(operation, result, answer, ds) {
  const type = operation && operation.type;
  try {
    if (type === "group_by_aggregate") {
      const agg = operation.aggregation, groupCol = operation.group_col, valueCol = operation.value_col;
      const rows = Object.entries(result || {}).map(([k, v]) => ({ key: k, value: Number(v) }));
      const currency = agg !== "count" && ds.meta[valueCol] && ds.meta[valueCol].currency;
      return { op: `group_by · ${agg}`, kind: "group", text: answer, result: { rows, groupCol, valueCol, agg, currency } };
    }
    if (type === "top_rows") {
      return { op: `top_rows · ${operation.ascending ? "asc" : "desc"}`, kind: "top", text: answer,
        result: { rows: result.rows || [], sortCol: operation.sort_col, asc: !!operation.ascending, showCols: pickDisplayColsApi(ds, operation.sort_col) } };
    }
    if (type === "correlation") {
      return { op: "correlation", kind: "correlation", text: answer,
        result: { c1: result.col1, c2: result.col2, r: result.correlation, n: result.rows_used } };
    }
    if (type === "summarize_column") {
      const r = result;
      if (r.type === "numeric")
        return { op: "summarize_column", kind: "colsummary", text: answer,
          result: { kind: "numeric", col: r.column, count: r.count, mean: r.mean, median: r.median, min: r.min, max: r.max, sum: r.sum, sd: r.standard_deviation } };
      if (r.type === "date")
        return { op: "summarize_column", kind: "colsummary", text: answer,
          result: { kind: "date", col: r.column, earliest: r.earliest, latest: r.latest, count: (ds.rows_n - (r.missing_count || 0)) } };
      const top = Object.entries(r.top_values || {});
      return { op: "summarize_column", kind: "colsummary", text: answer,
        result: { kind: "categorical", col: r.column, unique: r.unique_count, top: top.length ? top : [["—", 0]] } };
    }
    if (type === "summarize_dataframe") {
      const mv = result.missing_values || {};
      const per = Object.entries(mv).map(([col, n]) => ({ col, n })).filter(x => x.n > 0).sort((a, b) => b.n - a.n);
      return { op: "summarize · missing", kind: "missing", text: answer, result: { per, total: result.total_missing_values || 0 } };
    }
    if (type === "analysis_error") {
      return { op: null, kind: "help", text: answer || (result && result.error) || "I couldn't run that one." };
    }
  } catch (e) {
    return { op: null, kind: "help", text: answer || "Got an answer but couldn't render the detail." };
  }
  // operation == null → backend answered in prose only
  return { op: null, kind: "help", text: answer };
}

/* ---- network calls ---- */
async function apiUpload(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(base() + "/api/upload", { method: "POST", body: fd });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Upload failed (${res.status})`);
  return dsFromSummary(data.filename, data.dataset_id, data.summary || {});
}
async function apiAsk(ds, question) {
  const res = await fetch(base() + "/api/ask", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: ds.dataset_id, question }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) return { op: null, kind: "help", text: data.error || `Request failed (${res.status})` };
  return adaptAnswer(data.operation, data.result, data.answer, ds);
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
  ask(ds, q) { return cfg().useApi ? apiAsk(ds, q) : Promise.resolve(window.ask(ds, q)); },
};
