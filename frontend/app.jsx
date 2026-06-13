/* app.jsx — Alumni AI interactive prototype.
   Upload → profile → chat. Uses engine.jsx (window.ask/parseCSV/profile) +
   kit.jsx (Icon, Brand, TypePill) + theme.css. */
const { useState, useRef, useEffect, useCallback } = React;

/* ---------- text helpers ---------- */
function boldify(t) {
  return String(t).split(/(\*\*[^*]+\*\*)/g).map((s, i) =>
    s.startsWith("**") && s.endsWith("**")
      ? <strong key={i}>{s.slice(2, -2)}</strong>
      : <React.Fragment key={i}>{s}</React.Fragment>);
}
function previewVal(raw) {
  if (raw === "" || raw == null) return <span style={{ color: "var(--text-3)" }}>-</span>;
  return String(raw);
}
function uiTypeFromDtype(dtype) {
  const dt = String(dtype || "").toLowerCase();
  if (/int|float|number|decimal|double/.test(dt)) return "num";
  if (/date|time/.test(dt)) return "date";
  return "text";
}
function sumMissing(missingValues = {}) {
  return Object.values(missingValues).reduce((a, b) => a + (Number(b) || 0), 0);
}
function mergeDatasetPreview(ds, preview) {
  if (!preview) return ds;
  const columns = (preview.columns && preview.columns.length ? preview.columns : ds.columns) || [];
  const rows = preview.rows || ds.rows || [];
  const dataTypes = preview.data_types || {};
  const missingValues = preview.missing_values || {};
  const meta = { ...(ds.meta || {}) };

  for (const c of columns) {
    const current = meta[c] || {};
    const type = dataTypes[c] ? uiTypeFromDtype(dataTypes[c]) : (current.type || "text");
    meta[c] = {
      ...current,
      name: c,
      type,
      missing: missingValues[c] != null ? Number(missingValues[c]) || 0 : (current.missing || 0),
    };
  }

  return {
    ...ds,
    name: ds.name || preview.filename,
    columns,
    meta,
    rows,
    rows_n: preview.row_count != null ? preview.row_count : ds.rows_n,
    cols_n: preview.column_count != null ? preview.column_count : (columns.length || ds.cols_n),
    totalMissing: preview.missing_count != null ? preview.missing_count : sumMissing(missingValues) || ds.totalMissing,
  };
}
function dsFromDatasetEntry(entry) {
  const columns = Array.isArray(entry.columns) ? entry.columns : [];
  const meta = {};
  for (const c of columns) meta[c] = { name: c, type: "text", currency: false, year: false, missing: 0 };
  return {
    name: entry.display_name || entry.original_filename || "Untitled dataset",
    dataset_id: entry.dataset_id,
    columns,
    meta,
    rows: [],
    rows_n: entry.row_count || 0,
    cols_n: entry.column_count != null ? entry.column_count : columns.length,
    totalMissing: 0,
    status: entry.status || "ready",
  };
}
function formatUploadDate(iso) {
  if (!iso) return "";
  try {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return String(iso);
    return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  } catch (e) {
    return String(iso);
  }
}

/* ---------- answer renderers ---------- */
function answerText(value) {
  if (value === "" || value == null) return <span style={{ color: "var(--text-3)" }}>-</span>;
  return String(value);
}
function linkedInHref(value) {
  if (window.Alumni && window.Alumni.helpers && window.Alumni.helpers.linkedInHref) {
    return window.Alumni.helpers.linkedInHref(value);
  }
  const text = String(value || "").trim();
  if (!text) return "";
  if (/^https?:\/\//i.test(text)) return text;
  if (/^www\./i.test(text)) return `https://${text}`;
  if (/linkedin\.com/i.test(text)) return `https://${text}`;
  return "";
}
function isLinkedInColumn(column) {
  return !!(window.Alumni && window.Alumni.helpers && window.Alumni.helpers.isLinkedInColumn && window.Alumni.helpers.isLinkedInColumn(column));
}
function tableCellValue(column, value) {
  if (isLinkedInColumn(column)) {
    const href = linkedInHref(value);
    if (value === "" || value == null) return "";
    if (!href) return answerText(value);
    return <a href={href} target="_blank" rel="noreferrer" style={{ color: "var(--primary)", fontWeight: 700 }}>LinkedIn</a>;
  }
  return answerText(value);
}
function normalizeAnswerForRender(answer, fallbackText = "") {
  if (answer && answer.answer && typeof answer.answer === "object") answer = answer.answer;
  if (!answer || typeof answer !== "object") {
    const summary = String(fallbackText || answer || "");
    return { title: "", summary, blocks: summary ? [{ type: "markdown", content: summary }] : [], followups: [] };
  }
  return {
    title: String(answer.title || ""),
    summary: String(answer.summary || fallbackText || ""),
    blocks: Array.isArray(answer.blocks) ? answer.blocks : [],
    followups: Array.isArray(answer.followups) ? answer.followups.filter(Boolean).slice(0, 4) : [],
  };
}
function MarkdownBlock({ content }) {
  const chunks = String(content || "").split(/\n{2,}/).filter(Boolean);
  return (
    <div className="prose" style={{ margin: 0 }}>
      {chunks.map((chunk, i) => {
        const lines = chunk.split(/\n/).filter(Boolean);
        const isList = lines.every(line => /^\s*[-*]\s+/.test(line));
        if (isList) {
          return (
            <ul key={i} style={{ margin: i ? "8px 0 0" : 0, paddingLeft: 18 }}>
              {lines.map((line, j) => <li key={j}>{boldify(line.replace(/^\s*[-*]\s+/, ""))}</li>)}
            </ul>
          );
        }
        return <p key={i} style={{ margin: i ? "8px 0 0" : 0 }}>{boldify(chunk)}</p>;
      })}
    </div>
  );
}
function TableBlock({ block }) {
  const columns = Array.isArray(block.columns) ? block.columns : [];
  const rows = Array.isArray(block.rows) ? block.rows : [];
  if (!columns.length) return null;
  return (
    <div className="panel" style={{ overflow: "hidden" }}>
      {(block.title || block.caption) && (
        <div className="col" style={{ padding: "12px 14px 8px", gap: 3 }}>
          {block.title && <span style={{ fontSize: 12.5, fontWeight: 700 }}>{block.title}</span>}
          {block.caption && <span style={{ fontSize: 11.5, color: "var(--text-3)" }}>{block.caption}</span>}
        </div>
      )}
      <div style={{ overflowX: "auto" }}>
        <table className="dtable">
          <thead><tr>{columns.map(c => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {columns.map((c, j) => {
                  const value = Array.isArray(row) ? row[j] : row && row[c];
                  const numeric = typeof value === "number" || /^-?\$?[\d,]+(\.\d+)?%?$/.test(String(value || "").trim());
                  return <td key={c} className={numeric ? "num" : ""}>{tableCellValue(c, value)}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
function MetricsBlock({ block }) {
  const items = Array.isArray(block.items) ? block.items : [];
  if (!items.length) return null;
  return (
    <div className="panel" style={{ padding: 16, display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(120px,1fr))", gap: 14 }}>
      {items.map((item, i) => (
        <div className="col" key={i} style={{ gap: 2, minWidth: 0 }}>
          <span className="stat-num" style={{ fontSize: 18, overflowWrap: "anywhere" }}>{answerText(item.value)}</span>
          <span className="kicker">{answerText(item.label)}</span>
        </div>
      ))}
    </div>
  );
}
function RankedListBlock({ block }) {
  const items = Array.isArray(block.items) ? block.items : [];
  if (!items.length) return null;
  return (
    <div className="panel" style={{ overflow: "hidden" }}>
      {block.title && <div className="kicker" style={{ padding: "12px 14px 6px" }}>{block.title}</div>}
      <div className="col" style={{ gap: 0 }}>
        {items.map((item, i) => (
          <div className="row" key={i} style={{ padding: "10px 14px", gap: 12, borderTop: i ? "1px solid var(--border)" : "none", alignItems: "flex-start" }}>
            <span className="mono" style={{ color: "var(--text-3)", width: 22, flex: "none", fontSize: 11 }}>{String(i + 1).padStart(2, "0")}</span>
            <div className="col" style={{ gap: 3, minWidth: 0, flex: 1 }}>
              <span style={{ fontSize: 13, fontWeight: 700, overflowWrap: "anywhere" }}>{answerText(item.label)}</span>
              {item.description && <span style={{ fontSize: 12, color: "var(--text-2)", lineHeight: 1.35 }}>{answerText(item.description)}</span>}
            </div>
            {item.value && <span className="mono" style={{ color: "var(--primary)", fontSize: 12, fontWeight: 700, textAlign: "right" }}>{answerText(item.value)}</span>}
          </div>
        ))}
      </div>
    </div>
  );
}
function AnswerBlock({ block }) {
  if (!block || typeof block !== "object") return null;
  if (block.type === "markdown") return <MarkdownBlock content={block.content} />;
  if (block.type === "table") return <TableBlock block={block} />;
  if (block.type === "metrics") return <MetricsBlock block={block} />;
  if (block.type === "ranked_list") return <RankedListBlock block={block} />;
  return null;
}
function FollowupChips({ followups, onFollowup }) {
  if (!followups || !followups.length) return null;
  return (
    <div className="row gap8" style={{ flexWrap: "wrap", paddingTop: 2 }}>
      {followups.map(text => (
        <span key={text} className="sugg" onClick={() => onFollowup && onFollowup(text)}>{text}</span>
      ))}
    </div>
  );
}
function AnswerCard({ answer, onFollowup }) {
  const blocks = (answer.blocks || []).filter((block, i) => {
    return !(i === 0 && block.type === "markdown" && String(block.content || "").trim() === String(answer.summary || "").trim());
  });
  return (
    <div className="col" style={{ gap: 13 }}>
      {answer.title && <div style={{ fontSize: 14, fontWeight: 800 }}>{answer.title}</div>}
      {answer.summary && <p className="prose" style={{ margin: 0 }}>{boldify(answer.summary)}</p>}
      {blocks.map((block, i) => <AnswerBlock key={i} block={block} />)}
      <FollowupChips followups={answer.followups} onFollowup={onFollowup} />
    </div>
  );
}
function AnswerRenderer({ answer, fallbackText, onFollowup }) {
  return <DatasetResponseView answer={answer} fallbackText={fallbackText} onFollowup={onFollowup} />;
}
function DatasetResponseView({ response, answer, fallbackText, onFollowup }) {
  let source = answer;
  if (response && typeof response === "object") {
    source = response.answer && typeof response.answer === "object" ? response.answer : response;
  }
  return <AnswerCard answer={normalizeAnswerForRender(source, fallbackText)} onFollowup={onFollowup} />;
}

/* ---------- chat pieces ---------- */
function UserMsg({ text }) {
  return (
    <div className="msg" style={{ justifyContent: "flex-end" }}>
      <div className="msg-body" style={{ display: "flex", justifyContent: "flex-end" }}>
        <span className="bubble-user">{text}</span>
      </div>
      <div className="msg-av user">RC</div>
    </div>
  );
}
function AiMsg({ ds, msg, onFollowup, onSaveInsight }) {
  // Save insight only appears for completed answers to a real question against
  // an active dataset; saving is manual (this is not automatic history).
  const canSave = !!(onSaveInsight && msg.question && (msg.text || msg.answer) && ds && ds.dataset_id);
  return (
    <div className="msg">
      <div className="msg-av ai"><Icon name="sparkle" size={15} /></div>
      <div className="msg-body col" style={{ gap: 13 }}>
        <div className="msg-name">Alumni AI</div>
        <DatasetResponseView response={msg.response_payload} answer={msg.answer} fallbackText={msg.text} onFollowup={onFollowup} />
        {(msg.op || canSave) && (
          <div className="row gap8" style={{ color: "var(--text-3)", fontSize: 11.5, alignItems: "center" }}>
            {msg.op && (
              <React.Fragment>
                <span>Safe operation</span><span>·</span>
                <span className="chip chip-primary chip-mono"><Icon name="bolt" size={11} /> {msg.op}</span>
              </React.Fragment>
            )}
            {canSave && (
              <button
                className="btn btn-ghost"
                style={{ marginLeft: "auto", flex: "none" }}
                disabled={!!msg.insightSaving || !!msg.insightSaved}
                onClick={onSaveInsight}
                title={msg.insightSaved ? "Saved to your insights" : "Save this answer as a reusable insight"}
              >
                <Icon name="bookmark" size={13} />
                {msg.insightSaved ? " Saved ✓" : msg.insightSaving ? " Saving…" : " Save insight"}
              </button>
            )}
          </div>
        )}
        {msg.insightError && (
          <div style={{ color: "var(--warn)", fontSize: 12 }}>Save failed: {msg.insightError}</div>
        )}
      </div>
    </div>
  );
}
function Thinking() {
  return (
    <div className="msg">
      <div className="msg-av ai"><Icon name="sparkle" size={15} /></div>
      <div className="msg-body" style={{ paddingTop: 6 }}>
        <div className="row gap6">
          {[0, 1, 2].map(i => <span key={i} className="think-dot" style={{ animationDelay: i * 0.16 + "s" }} />)}
        </div>
      </div>
    </div>
  );
}

/* ---------- data panel + rail ---------- */
function DatasetPreview({ preview = { columns: [], rows: [], loading: false, error: "" } }) {
  const previewColumns = preview.columns || [];
  const previewRows = preview.rows || [];
  const columns = previewColumns.length
    ? previewColumns
    : (previewRows[0] ? Object.keys(previewRows[0]) : []);
  const rows = previewRows.slice(0, 10);

  if (preview.loading) {
    return <div style={{ padding: "10px 0 4px", color: "var(--text-3)", fontSize: 12.5 }}>Loading preview...</div>;
  }
  if (preview.error) {
    return <div style={{ padding: "10px 0 4px", color: "var(--warn)", fontSize: 12.5 }}>{preview.error}</div>;
  }
  if (!rows.length) {
    return <div style={{ padding: "10px 0 4px", color: "var(--text-3)", fontSize: 12.5 }}>No preview rows available.</div>;
  }

  return (
    <div className="panel" style={{ overflow: "auto", maxHeight: 220 }}>
      <table className="dtable">
        <thead>
          <tr>{columns.map(c => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {columns.map(c => (
                <td key={c} className={typeof r[c] === "number" ? "num" : ""}>{previewVal(r[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function DataPanel({ ds, preview = { columns: [], rows: [], loading: false, error: "" } }) {
  return (
    <div className="col" style={{ width: 360, flex: "none", borderLeft: "1px solid var(--border)", background: "var(--surface)" }}>
      <div className="row" style={{ padding: "15px 18px", gap: 10, borderBottom: "1px solid var(--border)" }}>
        <Icon name="file" size={16} style={{ color: "var(--primary)" }} />
        <span className="mono" style={{ fontSize: 12.5, fontWeight: 600, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{ds.name}</span>
        <button className="btn-icon" style={{ width: 28, height: 28 }}><Icon name="download" size={14} /></button>
      </div>
      <div className="col" style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
        <div className="row" style={{ padding: "14px 18px", gap: 18 }}>
          {[["Rows", ds.rows_n.toLocaleString()], ["Columns", ds.cols_n], ["Missing", ds.totalMissing]].map(([k, v]) => (
            <div className="col" key={k} style={{ gap: 2 }}>
              <span className="stat-num" style={{ fontSize: 18 }}>{v}</span>
              <span className="kicker">{k}</span>
            </div>
          ))}
        </div>
        <div className="divider" style={{ margin: "0 18px" }} />
        <div className="row" style={{ padding: "14px 18px 8px", justifyContent: "space-between" }}>
          <span className="kicker">Columns</span><span className="kicker">{ds.cols_n}</span>
        </div>
        <div className="col" style={{ padding: "0 12px 12px", gap: 1 }}>
          {ds.columns.map(c => (
            <div className="row" key={c} style={{ padding: "7px 8px", gap: 8, borderRadius: 8 }}>
              <span style={{ fontSize: 12.5, fontWeight: 500, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c}</span>
              {ds.meta[c].missing > 0 && <span title={ds.meta[c].missing + " missing"} style={{ fontSize: 10, color: "var(--warn)", fontWeight: 600 }}>{ds.meta[c].missing}⚠</span>}
              <TypePill t={ds.meta[c].type} />
            </div>
          ))}
        </div>
        <div className="divider" style={{ margin: "0 18px" }} />
        <div className="row" style={{ padding: "14px 18px 8px", justifyContent: "space-between" }}>
          <span className="kicker">Preview</span><span className="kicker">{preview.loading ? "loading" : `${Math.min(preview.rows.length, 10)} rows`}</span>
        </div>
        <div style={{ padding: "0 12px 16px" }}>
          <DatasetPreview preview={preview} />
        </div>
      </div>
    </div>
  );
}
function Rail({ ds, view, onNavigate, onNewAnalysis }) {
  // Conversations, Datasets, and Saved insights are live views; History stays a placeholder.
  const nav = [["chat", "Conversations", "chat"], ["database", "Datasets", "datasets"], ["bookmark", "Saved insights", "insights"], ["history", "History", null]];
  return (
    <div className="col" style={{ width: 212, flex: "none", borderRight: "1px solid var(--border)", background: "var(--surface-2)", padding: 14, gap: 6 }}>
      <button className="btn btn-primary" style={{ width: "100%", marginBottom: 8 }} onClick={onNewAnalysis}><Icon name="plus" size={15} /> New analysis</button>
      {nav.map(([ic, label, target]) => (
        <div
          key={label}
          className={"rail-item" + (target && view === target ? " active" : "")}
          style={target ? { cursor: "pointer" } : { cursor: "default" }}
          onClick={() => target && onNavigate && onNavigate(target)}
        >
          <Icon name={ic} size={16} cls="rail-ico" /> {label}
        </div>
      ))}
      <div className="kicker" style={{ padding: "16px 11px 8px" }}>Active dataset</div>
      {ds ? (
        <div className="rail-item active" style={{ fontSize: 12.5, cursor: "pointer" }} title={ds.name}
          onClick={() => onNavigate && onNavigate("datasets")}>
          <Icon name="file" size={15} cls="rail-ico" />
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{ds.name}</span>
        </div>
      ) : (
        <div className="rail-item" style={{ fontSize: 12.5, color: "var(--text-3)", cursor: "pointer" }}
          onClick={() => onNavigate && onNavigate("datasets")}>
          <Icon name="file" size={15} cls="rail-ico" />
          <span>None selected</span>
        </div>
      )}
      <div style={{ flex: 1 }} />
      <div className="row gap10" style={{ padding: "8px 6px", borderTop: "1px solid var(--border)" }}>
        <div style={{ width: 32, height: 32, borderRadius: "50%", background: "linear-gradient(135deg,#8E6FF0,#4B2FB0)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 700, flex: "none" }}>RC</div>
        <div className="col" style={{ gap: 1, lineHeight: 1.2 }}>
          <span style={{ fontSize: 12.5, fontWeight: 600 }}>Riley Chen</span>
          <span style={{ fontSize: 11, color: "var(--text-3)" }}>Alumni Relations</span>
        </div>
      </div>
    </div>
  );
}

/* ---------- dataset library ---------- */
function DatasetLibrary({ datasets, activeDatasetId, error, onSelect, onRename, onDelete, onUpload }) {
  const inputRef = useRef(null);
  return (
    <div className="col" style={{ flex: "0 0 auto", padding: "26px 34px", gap: 14, maxWidth: 860, width: "100%" }}>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
        <div className="col" style={{ gap: 3 }}>
          <span style={{ fontSize: 16, fontWeight: 800 }}>Datasets</span>
          <span className="kicker">{datasets.length} saved</span>
        </div>
        <button className="btn btn-primary" onClick={() => inputRef.current && inputRef.current.click()}>
          <Icon name="upload" size={15} /> Upload dataset
        </button>
        <input ref={inputRef} type="file" accept=".csv,.xlsx" style={{ display: "none" }}
          onChange={e => { const file = e.target.files[0]; e.target.value = ""; if (file) onUpload(file); }} />
      </div>
      {error && (
        <div className="row gap8" style={{ color: "var(--warn)", fontSize: 12.5, fontWeight: 500 }}>
          <Icon name="bolt" size={14} />{error}
        </div>
      )}
      {!datasets.length ? (
        <div className="panel col" style={{ padding: "34px 24px", alignItems: "center", gap: 8 }}>
          <Icon name="database" size={22} style={{ color: "var(--text-3)" }} />
          <span style={{ fontSize: 13.5, fontWeight: 700 }}>No datasets yet</span>
          <span style={{ fontSize: 12.5, color: "var(--text-3)" }}>Upload a .csv or .xlsx file to get started.</span>
        </div>
      ) : (
        datasets.map(d => {
          const active = d.dataset_id === activeDatasetId;
          return (
            <div
              key={d.dataset_id}
              className="panel row"
              data-dataset-id={d.dataset_id}
              style={{
                padding: "13px 16px", gap: 12, alignItems: "center", cursor: "pointer",
                borderColor: active ? "var(--primary)" : undefined,
                background: active ? "var(--primary-weak)" : undefined,
              }}
              onClick={() => onSelect(d)}
            >
              <Icon name="file" size={16} style={{ color: "var(--primary)", flex: "none" }} />
              <div className="col" style={{ flex: 1, minWidth: 0, gap: 3 }}>
                <div className="row gap8" style={{ alignItems: "center", minWidth: 0 }}>
                  <span style={{ fontSize: 13, fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={d.display_name || d.original_filename}>
                    {d.display_name || d.original_filename || "Untitled dataset"}
                  </span>
                  {active && <span className="chip chip-primary" style={{ flex: "none" }}>Active</span>}
                  {d.status === "missing" && (
                    <span className="chip" style={{ flex: "none", color: "var(--warn)", borderColor: "var(--warn)" }}>File missing</span>
                  )}
                </div>
                <span style={{ fontSize: 12, color: "var(--text-3)" }}>
                  {d.row_count != null ? d.row_count.toLocaleString() : "—"} rows · {d.column_count != null ? d.column_count : "—"} columns
                  {d.uploaded_at ? ` · uploaded ${formatUploadDate(d.uploaded_at)}` : ""}
                </span>
              </div>
              <button className="btn btn-ghost" style={{ flex: "none" }} onClick={e => { e.stopPropagation(); onRename(d); }}>Rename</button>
              <button className="btn btn-ghost" style={{ flex: "none", color: "var(--warn)" }} onClick={e => { e.stopPropagation(); onDelete(d); }}>Delete</button>
            </div>
          );
        })
      )}
    </div>
  );
}

/* ---------- saved insights ---------- */
function insightPreviewText(answer) {
  const text = String(answer || "").replace(/\s+/g, " ").trim();
  return text.length > 140 ? text.slice(0, 140).trim() + "…" : text;
}
function InsightDetail({ insight, datasetAvailable, isActiveDataset, onBack, onRename, onDelete, onUseDataset }) {
  const [showFullResponse, setShowFullResponse] = useState(false);
  const hasFullResponse = !!(insight.response_payload && typeof insight.response_payload === "object");
  useEffect(() => {
    setShowFullResponse(false);
  }, [insight.insight_id]);

  return (
    <div className="col" style={{ flex: "0 0 auto", padding: "26px 34px", gap: 14, maxWidth: 860, width: "100%" }}>
      <div className="row" style={{ alignItems: "center", gap: 10 }}>
        <button className="btn btn-ghost" style={{ flex: "none" }} onClick={onBack}>← All insights</button>
        <div style={{ flex: 1 }} />
        <button className="btn btn-ghost" style={{ flex: "none" }} onClick={() => onRename(insight)}>Rename</button>
        <button className="btn btn-ghost" style={{ flex: "none", color: "var(--warn)" }} onClick={() => onDelete(insight)}>Delete</button>
      </div>
      <div className="panel col" style={{ padding: "20px 22px", gap: 14 }}>
        <div className="col" style={{ gap: 6 }}>
          <span style={{ fontSize: 17, fontWeight: 800 }}>{insight.title}</span>
          <div className="row gap8" style={{ alignItems: "center", flexWrap: "wrap" }}>
            <span className="chip" style={{ flex: "none" }}>
              <Icon name="file" size={12} /> {insight.dataset_name_snapshot}
            </span>
            {!datasetAvailable && (
              <span className="chip" style={{ flex: "none", color: "var(--warn)", borderColor: "var(--warn)" }}>Dataset deleted</span>
            )}
            {insight.created_at && <span style={{ fontSize: 12, color: "var(--text-3)" }}>Saved {formatUploadDate(insight.created_at)}</span>}
            {insight.updated_at && insight.updated_at !== insight.created_at && (
              <span style={{ fontSize: 12, color: "var(--text-3)" }}>· Edited {formatUploadDate(insight.updated_at)}</span>
            )}
          </div>
          {insight.tags.length > 0 && (
            <div className="row gap6" style={{ flexWrap: "wrap" }}>
              {insight.tags.map(tag => <span key={tag} className="chip" style={{ flex: "none", fontSize: 11 }}>{tag}</span>)}
            </div>
          )}
        </div>
        <div className="divider" />
        <div className="col" style={{ gap: 4 }}>
          <span className="kicker">Question</span>
          <span style={{ fontSize: 13.5 }}>{insight.question}</span>
        </div>
        <div className="col" style={{ gap: 4 }}>
          <span className="kicker">Saved answer</span>
          <div style={{ fontSize: 13, lineHeight: 1.55, whiteSpace: "pre-wrap" }}>{insight.answer}</div>
          {hasFullResponse && (
            <button
              className="btn btn-primary"
              data-action="open-full-response"
              style={{ flex: "none", alignSelf: "flex-start", marginTop: 8 }}
              onClick={() => setShowFullResponse(open => !open)}
            >
              <Icon name="sparkle" size={14} /> {showFullResponse ? "Hide full response" : "Open full response"}
            </button>
          )}
        </div>
        {showFullResponse && hasFullResponse && (
          <div className="col" data-insight-full-response="true" style={{ gap: 12, paddingTop: 2 }}>
            <div className="divider" />
            <span className="kicker">Full response</span>
            <DatasetResponseView response={insight.response_payload} fallbackText={insight.answer} />
          </div>
        )}
        <div className="row">
          <button
            className="btn btn-primary"
            style={{ flex: "none" }}
            disabled={!datasetAvailable || isActiveDataset}
            title={datasetAvailable ? "" : "The dataset for this insight has been deleted."}
            onClick={() => onUseDataset(insight)}
          >
            <Icon name="database" size={14} /> {isActiveDataset ? "Dataset is active" : "Open dataset"}
          </button>
        </div>
      </div>
    </div>
  );
}
function InsightsLibrary({ insights, loading, error, activeDatasetId, datasets, selectedInsightId, onSelectInsight, onRename, onDelete, onUseDataset }) {
  const [filter, setFilter] = useState("all"); // "all" | "current"
  const datasetIds = new Set((datasets || []).map(d => d.dataset_id));
  const datasetAvailable = (insight) => insight.dataset_status !== "deleted" && datasetIds.has(insight.dataset_id);

  const selected = insights.find(i => i.insight_id === selectedInsightId);
  if (selected) {
    return (
      <InsightDetail
        insight={selected}
        datasetAvailable={datasetAvailable(selected)}
        isActiveDataset={selected.dataset_id === activeDatasetId}
        onBack={() => onSelectInsight(null)}
        onRename={onRename}
        onDelete={onDelete}
        onUseDataset={onUseDataset}
      />
    );
  }

  const visible = filter === "current" && activeDatasetId
    ? insights.filter(i => i.dataset_id === activeDatasetId)
    : insights;

  return (
    <div className="col" style={{ flex: "0 0 auto", padding: "26px 34px", gap: 14, maxWidth: 860, width: "100%" }}>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
        <div className="col" style={{ gap: 3 }}>
          <span style={{ fontSize: 16, fontWeight: 800 }}>Saved insights</span>
          <span className="kicker">{visible.length} {filter === "current" ? "for this dataset" : "saved"}</span>
        </div>
        <div className="row gap6">
          <button className={"btn " + (filter === "all" ? "btn-primary" : "btn-ghost")} onClick={() => setFilter("all")}>All insights</button>
          <button
            className={"btn " + (filter === "current" ? "btn-primary" : "btn-ghost")}
            disabled={!activeDatasetId}
            title={activeDatasetId ? "" : "Select a dataset to filter by it."}
            onClick={() => setFilter("current")}
          >
            Current dataset only
          </button>
        </div>
      </div>
      {error && (
        <div className="row gap8" style={{ color: "var(--warn)", fontSize: 12.5, fontWeight: 500 }}>
          <Icon name="bolt" size={14} />{error}
        </div>
      )}
      {loading ? (
        <div style={{ color: "var(--text-3)", fontSize: 12.5 }}>Loading saved insights…</div>
      ) : !visible.length ? (
        <div className="panel col" style={{ padding: "34px 24px", alignItems: "center", gap: 8 }}>
          <Icon name="bookmark" size={22} style={{ color: "var(--text-3)" }} />
          <span style={{ fontSize: 13.5, fontWeight: 700 }}>{filter === "current" ? "No insights for this dataset yet" : "No saved insights yet"}</span>
          <span style={{ fontSize: 12.5, color: "var(--text-3)" }}>Ask a question, then press “Save insight” under a useful answer.</span>
        </div>
      ) : (
        visible.map(insight => (
          <div
            key={insight.insight_id}
            className="panel row"
            data-insight-id={insight.insight_id}
            style={{ padding: "13px 16px", gap: 12, alignItems: "flex-start", cursor: "pointer" }}
            onClick={() => onSelectInsight(insight.insight_id)}
          >
            <Icon name="bookmark" size={16} style={{ color: "var(--primary)", flex: "none", marginTop: 2 }} />
            <div className="col" style={{ flex: 1, minWidth: 0, gap: 3 }}>
              <div className="row gap8" style={{ alignItems: "center", minWidth: 0 }}>
                <span style={{ fontSize: 13, fontWeight: 700, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={insight.title}>{insight.title}</span>
                <span className="chip" style={{ flex: "none", fontSize: 11 }}>{insight.dataset_name_snapshot}</span>
                {!datasetAvailable(insight) && (
                  <span className="chip" style={{ flex: "none", fontSize: 11, color: "var(--warn)", borderColor: "var(--warn)" }}>Dataset deleted</span>
                )}
              </div>
              <span style={{ fontSize: 12.5, color: "var(--text-2)", fontStyle: "italic", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{insight.question}</span>
              <span style={{ fontSize: 12, color: "var(--text-3)" }}>{insightPreviewText(insight.answer)}</span>
              {insight.created_at && <span style={{ fontSize: 11, color: "var(--text-3)" }}>Saved {formatUploadDate(insight.created_at)}</span>}
            </div>
            <button className="btn btn-ghost" style={{ flex: "none" }} onClick={e => { e.stopPropagation(); onSelectInsight(insight.insight_id); }}>Open</button>
            <button className="btn btn-ghost" style={{ flex: "none" }} onClick={e => { e.stopPropagation(); onRename(insight); }}>Rename</button>
            <button className="btn btn-ghost" style={{ flex: "none", color: "var(--warn)" }} onClick={e => { e.stopPropagation(); onDelete(insight); }}>Delete</button>
          </div>
        ))
      )}
    </div>
  );
}

/* ---------- composer ---------- */
function Composer({ onSend, busy, suggestions, showSugg }) {
  const [val, setVal] = useState("");
  const [focus, setFocus] = useState(false);
  const send = () => { const t = val.trim(); if (!t || busy) return; setVal(""); onSend(t); };
  return (
    <div style={{ maxWidth: 760, margin: "0 auto", width: "100%" }}>
      {showSugg && (
        <div className="row gap8" style={{ marginBottom: 10, flexWrap: "wrap", justifyContent: "center" }}>
          {suggestions.map(s => <span key={s} className="sugg" onClick={() => !busy && onSend(s)}>{s}</span>)}
        </div>
      )}
      <div className={"composer" + (focus ? " focus" : "")}>
        <Icon name="sparkle" size={18} style={{ color: "var(--primary)", flex: "none" }} />
        <input value={val} onChange={e => setVal(e.target.value)} onKeyDown={e => e.key === "Enter" && send()}
          onFocus={() => setFocus(true)} onBlur={() => setFocus(false)}
          placeholder="Ask anything about your alumni data…"
          style={{ flex: 1, border: "none", outline: "none", background: "transparent", font: "inherit", fontSize: 14, color: "var(--text)" }} />
        <button className="btn-icon" onClick={send} disabled={busy}
          style={{ background: val.trim() ? "var(--primary)" : "var(--surface-2)", borderColor: "transparent", color: val.trim() ? "var(--on-primary)" : "var(--text-3)", width: 32, height: 32 }}>
          <Icon name="send" size={16} />
        </button>
      </div>
    </div>
  );
}

/* ---------- upload view ---------- */
function UploadView({ onLoad, loadError, theme, onToggle }) {
  const [drag, setDrag] = useState(false);
  const [err, setErr] = useState("");
  const inputRef = useRef(null);
  const apiMode = !!(window.Alumni && window.Alumni.isApi && window.Alumni.isApi());
  const allowedPattern = apiMode ? /\.(csv|xlsx)$/i : /\.(csv|tsv|txt)$/i;
  const acceptedTypes = apiMode ? ".csv,.xlsx" : ".csv,.tsv,.txt";
  const supportedLabel = apiMode ? ".csv or .xlsx" : ".csv";
  const handleFile = (file) => {
    if (!file) return;
    if (!allowedPattern.test(file.name)) { setErr(`Please upload a ${supportedLabel} file.`); return; }
    setErr("");
    onLoad(file);
  };
  const canLoadSample = !apiMode && window.SAMPLE_CSV && window.SAMPLE_NAME;
  const loadSample = () => canLoadSample && handleFile(new File([window.SAMPLE_CSV], window.SAMPLE_NAME, { type: "text/csv" }));
  return (
    <div className="screen col" data-theme={theme} style={{ width: "100%", minHeight: "100vh" }}>
      <div className="row" style={{ height: 56, padding: "0 22px", flex: "none", gap: 14 }}>
        <Brand />
        <div style={{ flex: 1 }} />
        <button className="btn-icon" onClick={onToggle} title="Toggle theme"><Icon name={theme === "light" ? "moon" : "sun"} size={16} /></button>
        <div style={{ width: 32, height: 32, borderRadius: "50%", background: "linear-gradient(135deg,#8E6FF0,#4B2FB0)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 700 }}>RC</div>
      </div>
      <div className="col" style={{ flex: 1, alignItems: "center", justifyContent: "center", padding: "0 24px" }}>
        <div className="col" style={{ alignItems: "center", maxWidth: 600, width: "100%", gap: 22 }}>
          <span className="chip chip-primary"><Icon name="sparkle" size={13} /> Internal · Alumni Relations</span>
          <div className="col" style={{ alignItems: "center", gap: 14, width: "100%" }}>
            <h1 style={{ margin: 0, width: "100%", fontSize: 40, fontWeight: 800, letterSpacing: "-0.03em", textAlign: "center", lineHeight: 1.12 }}>Talk to your alumni data</h1>
            <p style={{ margin: 0, fontSize: 16, color: "var(--text-2)", textAlign: "center", maxWidth: 440, lineHeight: 1.5 }}>Drop a spreadsheet and ask questions in plain English. Answers are computed with safe backend operations.</p>
          </div>
          <div className={"dropzone" + (drag ? " drag" : "")} style={{ width: "100%", padding: "40px 24px", cursor: "pointer" }}
            onClick={() => inputRef.current.click()}
            onDragOver={e => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
            onDrop={e => { e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0]); }}>
            <div className="dz-ico"><Icon name="upload" size={24} /></div>
            <div style={{ fontSize: 15.5, fontWeight: 700, marginBottom: 5 }}>{drag ? "Drop to upload" : `Drop a ${supportedLabel} file`}</div>
            <div style={{ fontSize: 13, color: "var(--text-2)", marginBottom: 18 }}>or click to browse</div>
            <button className="btn btn-primary" onClick={e => { e.stopPropagation(); inputRef.current.click(); }}><Icon name="file" size={15} /> Choose file</button>
            <input ref={inputRef} type="file" accept={acceptedTypes} style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])} />
            <div className="row gap8" style={{ marginTop: 18, color: "var(--text-3)", fontSize: 11.5 }}>
              <span className="mono">{supportedLabel}</span>
              {canLoadSample && <React.Fragment><span>·</span>
                <span style={{ color: "var(--primary)", fontWeight: 600 }} onClick={e => { e.stopPropagation(); loadSample(); }}>try the sample dataset</span></React.Fragment>}
            </div>
          </div>
          {(err || loadError) && <div className="row gap8" style={{ color: "var(--warn)", fontSize: 12.5, fontWeight: 500, textAlign: "center" }}><Icon name="bolt" size={14} />{err || loadError}</div>}
        </div>
      </div>
    </div>
  );
}

/* ---------- workspace view ---------- */
function Workspace({ ds, preview, theme, onToggle, view, onNavigate, onNewAnalysis, datasets, datasetsError, onSelectDataset, onRenameDataset, onDeleteDataset, onUpload, insights, insightsLoading, insightsError, selectedInsightId, onSelectInsight, onSaveInsight, onRenameInsight, onDeleteInsight, onUseInsightDataset }) {
  const [messages, setMessages] = useState(() => [{ role: "ai", kind: "help", op: null,
    text: `Loaded **${ds.name}** — ${ds.rows_n.toLocaleString()} rows across ${ds.cols_n} columns. Ask me anything, or try one of the suggestions below.` }]);
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef(null);
  const sugg = window.suggestedQuestions(ds);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
    try { window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }); } catch (e) {}
  }, [messages, busy]);

  const send = useCallback((q) => {
    const text = String(q || "").trim();
    if (!text || busy) return;
    if (window.Alumni.isApi && window.Alumni.isApi() && !(ds && ds.dataset_id)) {
      setMessages(m => [...m, { role: "user", text }, { role: "ai", kind: "structured", op: null,
        text: "No active dataset is selected. Open Datasets in the sidebar and select one before asking questions." }]);
      return;
    }
    setMessages(m => [...m, { role: "user", text }]);
    setBusy(true);
    Promise.all([
      window.Alumni.ask(ds, text).catch(e => ({ op: null, kind: "structured", text: "Something went wrong: " + (e.message || e) })),
      new Promise(r => setTimeout(r, 420)),
    ]).then(([ans]) => {
      // Keep the originating question on the AI message so the user can
      // manually save the answer as an insight (this is not automatic history).
      setMessages(m => [...m, { role: "ai", question: text, ...ans }]);
      setBusy(false);
    });
  }, [ds, busy]);

  const patchMessage = (index, patch) =>
    setMessages(m => m.map((msg, i) => (i === index ? { ...msg, ...patch } : msg)));

  const saveInsightFor = (index) => {
    const msg = messages[index];
    if (!msg || msg.insightSaving || msg.insightSaved || !onSaveInsight) return;
    const helpers = (window.Alumni && window.Alumni.helpers) || {};
    const defaultTitle = helpers.defaultInsightTitle ? helpers.defaultInsightTitle(msg.question) : msg.question;
    const title = window.prompt("Save insight as:", defaultTitle);
    if (title == null) return;
    const answerText = helpers.insightTextFromAnswer ? helpers.insightTextFromAnswer(msg.answer, msg.text) : (msg.text || "");
    patchMessage(index, { insightSaving: true, insightError: "" });
    onSaveInsight({
      dataset_id: ds.dataset_id,
      question: msg.question,
      answer: answerText,
      title: title.trim() || defaultTitle,
      response_payload: msg.response_payload || null,
    })
      .then(() => patchMessage(index, { insightSaving: false, insightSaved: true }))
      .catch(e => patchMessage(index, { insightSaving: false, insightError: e.message || String(e) }));
  };

  const onlyGreeting = messages.length === 1;

  return (
    <div className="screen col" data-theme={theme} data-screen-label="Workspace" style={{ width: "100%", minHeight: "100vh" }}>
      <div className="row" style={{ height: 56, padding: "0 18px", borderBottom: "1px solid var(--border)", background: "var(--surface)", flex: "none", gap: 14 }}>
        <Brand />
        <div style={{ flex: 1, display: "flex", justifyContent: "center" }}>
          <div className="chip" style={{ padding: "6px 12px", gap: 9, background: "var(--surface-2)" }}>
            <Icon name="database" size={14} style={{ color: "var(--primary)" }} />
            <span className="mono" style={{ fontSize: 12, color: "var(--text)", fontWeight: 600 }}>{ds.name}</span>
            <span style={{ color: "var(--text-3)", fontSize: 11 }}>{ds.rows_n.toLocaleString()} rows</span>
          </div>
        </div>
        <button className="btn-icon" onClick={onToggle} title="Toggle theme"><Icon name={theme === "light" ? "moon" : "sun"} size={16} /></button>
        <button className="btn btn-ghost"><Icon name="download" size={15} /> Export</button>
      </div>
      <div className="row" style={{ flex: "1 0 auto", alignItems: "stretch" }}>
        <Rail ds={ds} view={view} onNavigate={onNavigate} onNewAnalysis={onNewAnalysis} />
        <div className="col" style={{ flex: 1, minWidth: 0, minHeight: 0 }}>
          {view === "datasets" ? (
            <DatasetLibrary
              datasets={datasets}
              activeDatasetId={ds && ds.dataset_id}
              error={datasetsError}
              onSelect={onSelectDataset}
              onRename={onRenameDataset}
              onDelete={onDeleteDataset}
              onUpload={onUpload}
            />
          ) : view === "insights" ? (
            <InsightsLibrary
              insights={insights}
              loading={insightsLoading}
              error={insightsError}
              activeDatasetId={ds && ds.dataset_id}
              datasets={datasets}
              selectedInsightId={selectedInsightId}
              onSelectInsight={onSelectInsight}
              onRename={onRenameInsight}
              onDelete={onDeleteInsight}
              onUseDataset={onUseInsightDataset}
            />
          ) : (
            <React.Fragment>
              <div ref={scrollRef} className="col" style={{ flex: "0 0 auto", padding: "26px 34px", gap: 28, overflowY: "visible" }}>
                {messages.map((m, i) => m.role === "user" ? <UserMsg key={i} text={m.text} /> : (
                  <AiMsg key={i} ds={ds} msg={m} onFollowup={send}
                    onSaveInsight={window.Alumni.isApi && window.Alumni.isApi() ? () => saveInsightFor(i) : null} />
                ))}
                {busy && <Thinking />}
              </div>
              <div style={{ padding: "0 34px 22px" }}>
                <Composer onSend={send} busy={busy} suggestions={sugg} showSugg={onlyGreeting} />
              </div>
            </React.Fragment>
          )}
        </div>
        <DataPanel ds={ds} preview={preview} />
      </div>
    </div>
  );
}

/* ---------- root ---------- */
const ACTIVE_DATASET_KEY = "alumniActiveDatasetId";

function App() {
  const apiMode = !!(window.Alumni && window.Alumni.isApi && window.Alumni.isApi());
  const [ds, setDs] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [view, setView] = useState("chat");
  const [chatSeq, setChatSeq] = useState(0);
  const [booting, setBooting] = useState(apiMode);
  const [datasetsError, setDatasetsError] = useState("");
  const [preview, setPreview] = useState({ columns: [], rows: [], loading: false, error: "" });
  const [theme, setTheme] = useState(() => (typeof localStorage !== "undefined" && localStorage.getItem("alumniTheme")) || "light");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [insights, setInsights] = useState([]);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [insightsError, setInsightsError] = useState("");
  const [selectedInsightId, setSelectedInsightId] = useState(null);
  const loadSeq = useRef(0);
  useEffect(() => { try { localStorage.setItem("alumniTheme", theme); } catch (e) {} }, [theme]);
  const toggle = () => setTheme(t => t === "light" ? "dark" : "light");

  const rememberActive = (datasetId) => {
    try {
      if (datasetId) localStorage.setItem(ACTIVE_DATASET_KEY, datasetId);
      else localStorage.removeItem(ACTIVE_DATASET_KEY);
    } catch (e) {}
  };

  const loadPreviewFor = async (datasetId, seq) => {
    setPreview({ columns: [], rows: [], loading: true, error: "" });
    try {
      const p = await window.Alumni.preview(datasetId);
      if (seq !== loadSeq.current) return;
      setPreview({ columns: p.columns || [], rows: p.rows || [], loading: false, error: "" });
      setDs(current => current && current.dataset_id === datasetId ? mergeDatasetPreview(current, p) : current);
    } catch (e) {
      if (seq !== loadSeq.current) return;
      const message = e.message || "Could not load preview.";
      setPreview({ columns: [], rows: [], loading: false, error: `Preview failed: ${message}` });
    }
  };

  const selectDataset = (entry) => {
    if (!entry || !entry.dataset_id) return;
    const seq = ++loadSeq.current;
    setDs(dsFromDatasetEntry(entry));
    rememberActive(entry.dataset_id);
    if (entry.status === "missing") {
      setPreview({ columns: [], rows: [], loading: false, error: "Preview unavailable: the uploaded file for this dataset is missing from storage." });
      return;
    }
    loadPreviewFor(entry.dataset_id, seq);
  };

  // Initial load: fetch saved datasets, restore the previously active one
  // (or the newest), and load its preview.
  useEffect(() => {
    if (!apiMode) return;
    let cancelled = false;
    window.Alumni.datasets()
      .then(list => {
        if (cancelled) return;
        setDatasets(list);
        let storedId = null;
        try { storedId = localStorage.getItem(ACTIVE_DATASET_KEY); } catch (e) {}
        const initial = list.find(d => d.dataset_id === storedId) || list[0] || null;
        if (initial) selectDataset(initial);
        setBooting(false);
      })
      .catch(e => {
        if (cancelled) return;
        setDatasetsError(`Could not load datasets: ${e.message || e}`);
        setBooting(false);
      });
    return () => { cancelled = true; };
  }, []);

  const refreshDatasets = () => {
    if (!apiMode) return;
    window.Alumni.datasets().then(setDatasets).catch(e => setDatasetsError(`Could not load datasets: ${e.message || e}`));
  };

  // Saved insights load when the view opens (and stay fresh after saves);
  // they are never written automatically — only the Save insight button saves.
  useEffect(() => {
    if (!apiMode || view !== "insights") return;
    let cancelled = false;
    setInsightsLoading(true);
    window.Alumni.insights()
      .then(list => { if (!cancelled) { setInsights(list); setInsightsError(""); } })
      .catch(e => { if (!cancelled) setInsightsError(`Could not load saved insights: ${e.message || e}`); })
      .finally(() => { if (!cancelled) setInsightsLoading(false); });
    return () => { cancelled = true; };
  }, [view, apiMode]);

  const saveInsight = async (payload) => {
    const created = await window.Alumni.saveInsight(payload);
    setInsights(list => [created, ...list.filter(i => i.insight_id !== created.insight_id)]);
    return created;
  };

  const renameInsight = async (insight) => {
    const next = window.prompt("Rename insight", insight.title);
    if (next == null) return;
    if (!next.trim()) {
      setInsightsError("Insight title cannot be empty.");
      return;
    }
    try {
      const updated = await window.Alumni.renameInsight(insight.insight_id, next.trim());
      setInsightsError("");
      setInsights(list => list.map(i => (i.insight_id === insight.insight_id ? { ...i, ...updated } : i)));
    } catch (e) {
      setInsightsError(`Rename failed: ${e.message || e}`);
    }
  };

  const deleteInsight = async (insight) => {
    if (!window.confirm(`Delete insight "${insight.title}"? This cannot be undone.`)) return;
    try {
      await window.Alumni.deleteInsight(insight.insight_id);
    } catch (e) {
      // Keep the insight in the UI when the delete fails.
      setInsightsError(`Delete failed: ${e.message || e}`);
      return;
    }
    setInsightsError("");
    setInsights(list => list.filter(i => i.insight_id !== insight.insight_id));
    setSelectedInsightId(current => (current === insight.insight_id ? null : current));
  };

  const useInsightDataset = (insight) => {
    const entry = datasets.find(d => d.dataset_id === insight.dataset_id);
    if (!entry) {
      setInsightsError("The dataset for this insight no longer exists.");
      return;
    }
    setInsightsError("");
    selectDataset(entry);
  };

  const load = async (file) => {
    const seq = loadSeq.current + 1;
    loadSeq.current = seq;
    setError(""); setPreview({ columns: [], rows: [], loading: false, error: "" }); setLoading(true);

    let uploaded;
    try {
      uploaded = await window.Alumni.load(file);
    } catch (e) {
      if (seq === loadSeq.current) {
        setError(`Upload failed: ${e.message || "Could not load that file."}`);
        setDatasetsError(`Upload failed: ${e.message || "Could not load that file."}`);
        setLoading(false);
      }
      return;
    }

    if (seq !== loadSeq.current) return;
    setDs(uploaded);
    setLoading(false);
    setView("chat");
    setDatasetsError("");
    rememberActive(uploaded.dataset_id);
    refreshDatasets();

    if (apiMode && !uploaded.dataset_id) {
      setPreview({ columns: [], rows: [], loading: false, error: "Preview failed: upload response did not include dataset_id." });
      return;
    }

    if (apiMode) {
      await loadPreviewFor(uploaded.dataset_id, seq);
    } else {
      setPreview({ columns: uploaded.columns || [], rows: (uploaded.rows || []).slice(0, 10), loading: false, error: "" });
    }
  };

  const renameDataset = async (entry) => {
    const current = entry.display_name || entry.original_filename || "";
    const next = window.prompt("Rename dataset", current);
    if (next == null) return;
    if (!next.trim()) {
      setDatasetsError("Dataset name cannot be empty.");
      return;
    }
    try {
      const updated = await window.Alumni.renameDataset(entry.dataset_id, next.trim());
      setDatasetsError("");
      setDatasets(list => list.map(d => d.dataset_id === entry.dataset_id ? { ...d, ...updated } : d));
      setDs(current => current && current.dataset_id === entry.dataset_id
        ? { ...current, name: updated.display_name }
        : current);
    } catch (e) {
      setDatasetsError(`Rename failed: ${e.message || e}`);
    }
  };

  const deleteDataset = async (entry) => {
    const label = entry.display_name || entry.original_filename || entry.dataset_id;
    if (!window.confirm(`Delete dataset "${label}"? This cannot be undone.`)) return;
    try {
      await window.Alumni.deleteDataset(entry.dataset_id);
    } catch (e) {
      setDatasetsError(`Delete failed: ${e.message || e}`);
      return;
    }
    setDatasetsError("");
    const remaining = datasets.filter(d => d.dataset_id !== entry.dataset_id);
    setDatasets(remaining);
    if (ds && ds.dataset_id === entry.dataset_id) {
      if (remaining.length) {
        selectDataset(remaining[0]);
      } else {
        loadSeq.current += 1;
        setDs(null);
        setPreview({ columns: [], rows: [], loading: false, error: "" });
        rememberActive(null);
      }
    }
  };

  if (loading || booting) return (
    <div className="screen col" data-theme={theme} style={{ width: "100%", minHeight: "100vh", alignItems: "center", justifyContent: "center", gap: 18 }}>
      <div className="brand-mark" style={{ width: 44, height: 44, animation: "pulse 1.1s ease-in-out infinite" }} />
      <div className="col" style={{ alignItems: "center", gap: 5 }}>
        <span style={{ fontWeight: 700, fontSize: 15 }}>{loading ? "Profiling your spreadsheet…" : "Loading your datasets…"}</span>
        <span style={{ fontSize: 12.5, color: "var(--text-3)" }}>{loading ? "Reading columns, inferring types, scanning for gaps" : "Restoring your saved dataset library"}</span>
      </div>
    </div>
  );

  return ds
    ? <Workspace
        key={`${ds.dataset_id || ds.name}:${chatSeq}`}
        ds={ds}
        preview={preview}
        theme={theme}
        onToggle={toggle}
        view={view}
        onNavigate={setView}
        onNewAnalysis={() => { setChatSeq(s => s + 1); setView("chat"); }}
        datasets={datasets}
        datasetsError={datasetsError}
        onSelectDataset={selectDataset}
        onRenameDataset={renameDataset}
        onDeleteDataset={deleteDataset}
        onUpload={load}
        insights={insights}
        insightsLoading={insightsLoading}
        insightsError={insightsError}
        selectedInsightId={selectedInsightId}
        onSelectInsight={setSelectedInsightId}
        onSaveInsight={saveInsight}
        onRenameInsight={renameInsight}
        onDeleteInsight={deleteInsight}
        onUseInsightDataset={useInsightDataset}
      />
    : <UploadView onLoad={load} loadError={error || datasetsError} theme={theme} onToggle={toggle} />;
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
