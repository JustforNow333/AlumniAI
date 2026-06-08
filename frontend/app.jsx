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
function cellVal(ds, col, raw) {
  const m = ds.meta[col];
  if (raw === "" || raw == null) return <span style={{ color: "var(--text-3)" }}>—</span>;
  if (m.type === "num") return <span className="mono">{window.fmtNum(window.toNum(raw), m)}</span>;
  return raw;
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
    name: preview.filename || ds.name,
    columns,
    meta,
    rows,
    rows_n: preview.row_count != null ? preview.row_count : ds.rows_n,
    cols_n: preview.column_count != null ? preview.column_count : (columns.length || ds.cols_n),
    totalMissing: preview.missing_count != null ? preview.missing_count : sumMissing(missingValues) || ds.totalMissing,
  };
}

/* ---------- result renderers ---------- */
function GroupResult({ ds, result }) {
  const { rows, groupCol, valueCol, agg, currency } = result;
  const max = Math.max(...rows.map(r => Math.abs(r.value))) || 1;
  const label = agg === "count" ? "count" : `${agg} · ${valueCol}`;
  const fmt = v => agg === "count" ? v.toLocaleString() : window.fmtNum(v, { currency });
  return (
    <div className="panel" style={{ overflow: "hidden" }}>
      <table className="dtable">
        <thead><tr><th>{groupCol}</th><th style={{ textAlign: "right" }}>{label}</th></tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td style={{ fontWeight: 600 }}>{r.key}</td>
              <td className="num" style={{ position: "relative", color: "var(--primary)", fontWeight: 700,
                background: `linear-gradient(90deg, var(--primary-weak) ${(Math.abs(r.value)/max*100).toFixed(1)}%, transparent ${(Math.abs(r.value)/max*100).toFixed(1)}%)` }}>
                {fmt(r.value)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function TopResult({ ds, result }) {
  const { rows, showCols, sortCol } = result;
  return (
    <div className="panel" style={{ overflow: "hidden" }}>
      <table className="dtable">
        <thead><tr>{showCols.map(c =>
          <th key={c} style={{ textAlign: ds.meta[c].type === "num" ? "right" : "left" }}>{c}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {showCols.map(c => {
                const isSort = c === sortCol, isNum = ds.meta[c].type === "num";
                if (ds.meta[c].type === "text" && /industry|chapter|degree/i.test(c) && c !== showCols[0])
                  return <td key={c}><span className="chip" style={{ padding: "2px 8px" }}>{r[c] || "—"}</span></td>;
                return <td key={c} className={isNum ? "num" : ""} style={isSort ? { color: "var(--primary)", fontWeight: 700 } : (c === showCols[0] ? { fontWeight: 600 } : { color: "var(--text-2)" })}>
                  {isNum ? window.fmtNum(window.toNum(r[c]), ds.meta[c]) : (r[c] || "—")}
                </td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function CorrResult({ result }) {
  const { r, n } = result;
  if (r == null) return null;
  const pos = ((r + 1) / 2) * 100;
  return (
    <div className="panel" style={{ padding: 18, display: "flex", gap: 22, alignItems: "center" }}>
      <div className="col" style={{ gap: 2, flex: "none" }}>
        <span className="stat-num" style={{ fontSize: 34, letterSpacing: "-0.03em", lineHeight: 1, color: r >= 0 ? "var(--pos)" : "var(--primary)" }}>{r.toFixed(2)}</span>
        <span className="kicker">Pearson r</span>
      </div>
      <div className="vdivider" />
      <div className="col" style={{ gap: 9, flex: 1 }}>
        <div style={{ position: "relative", height: 8, borderRadius: 5, background: "var(--surface-3)" }}>
          <div style={{ position: "absolute", left: "50%", top: -3, bottom: -3, width: 1, background: "var(--border-2)" }} />
          <div style={{ position: "absolute", left: `calc(${pos}% - 6px)`, top: -2, width: 12, height: 12, borderRadius: "50%", background: r >= 0 ? "var(--pos)" : "var(--primary)", boxShadow: "0 1px 3px rgba(0,0,0,.3)" }} />
        </div>
        <div className="row" style={{ justifyContent: "space-between", fontSize: 10.5, color: "var(--text-3)" }}>
          <span>−1.0</span><span>0</span><span>+1.0</span>
        </div>
        <span style={{ fontSize: 11.5, color: "var(--text-3)" }}>{n.toLocaleString()} alumni with both values</span>
      </div>
    </div>
  );
}
function ColResult({ ds, result }) {
  const s = result;
  if (s.kind === "numeric") {
    const c = ds.meta[s.col];
    const stats = [["Mean", s.mean], ["Median", s.median], ["Min", s.min], ["Max", s.max], ["Sum", s.sum], ["Count", s.count]];
    return (
      <div className="panel" style={{ padding: 16, display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14 }}>
        {stats.map(([k, v], i) => (
          <div className="col" key={k} style={{ gap: 2 }}>
            <span className="stat-num" style={{ fontSize: 19 }}>{k === "Count" ? v.toLocaleString() : window.fmtNum(v, c)}</span>
            <span className="kicker">{k}</span>
          </div>
        ))}
      </div>
    );
  }
  if (s.kind === "date") {
    const f = d => d ? new Date(d).toISOString().slice(0, 10) : "—";
    return (
      <div className="panel" style={{ padding: 16, display: "flex", gap: 28 }}>
        {[["Earliest", f(s.earliest)], ["Latest", f(s.latest)], ["Records", s.count.toLocaleString()]].map(([k, v]) => (
          <div className="col" key={k} style={{ gap: 2 }}><span className="stat-num" style={{ fontSize: 18 }}>{v}</span><span className="kicker">{k}</span></div>
        ))}
      </div>
    );
  }
  const max = s.top[0][1];
  return (
    <div className="panel" style={{ overflow: "hidden" }}>
      <table className="dtable">
        <thead><tr><th>{s.col}</th><th style={{ textAlign: "right" }}>count</th></tr></thead>
        <tbody>{s.top.map(([k, v], i) => (
          <tr key={i}><td style={{ fontWeight: 600 }}>{k}</td>
            <td className="num" style={{ color: "var(--primary)", fontWeight: 700, background: `linear-gradient(90deg,var(--primary-weak) ${v/max*100}%, transparent ${v/max*100}%)` }}>{v}</td></tr>
        ))}</tbody>
      </table>
    </div>
  );
}
function MissingResult({ result }) {
  if (result.total === 0) return null;
  const max = result.per[0].n;
  return (
    <div className="panel" style={{ overflow: "hidden" }}>
      <table className="dtable">
        <thead><tr><th>Column</th><th style={{ textAlign: "right" }}>missing</th></tr></thead>
        <tbody>{result.per.map((x, i) => (
          <tr key={i}><td style={{ fontWeight: 600 }}>{x.col}</td>
            <td className="num" style={{ color: "var(--warn)", fontWeight: 700, background: `linear-gradient(90deg,color-mix(in srgb,var(--warn) 14%,transparent) ${x.n/max*100}%, transparent ${x.n/max*100}%)` }}>{x.n}</td></tr>
        ))}</tbody>
      </table>
    </div>
  );
}
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
  return <AnswerCard answer={normalizeAnswerForRender(answer, fallbackText)} onFollowup={onFollowup} />;
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
function AiMsg({ ds, msg, onFollowup }) {
  return (
    <div className="msg">
      <div className="msg-av ai"><Icon name="sparkle" size={15} /></div>
      <div className="msg-body col" style={{ gap: 13 }}>
        <div className="msg-name">Alumni AI</div>
        <AnswerRenderer answer={msg.answer} fallbackText={msg.text} onFollowup={onFollowup} />
        {msg.op && (
          <div className="row gap8" style={{ color: "var(--text-3)", fontSize: 11.5, alignItems: "center" }}>
            <span>Safe operation</span><span>·</span>
            <span className="chip chip-primary chip-mono"><Icon name="bolt" size={11} /> {msg.op}</span>
          </div>
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
function Rail({ ds, onReset }) {
  const nav = [["chat", "Conversations", true], ["database", "Datasets", false], ["bookmark", "Saved insights", false], ["history", "History", false]];
  return (
    <div className="col" style={{ width: 212, flex: "none", borderRight: "1px solid var(--border)", background: "var(--surface-2)", padding: 14, gap: 6 }}>
      <button className="btn btn-primary" style={{ width: "100%", marginBottom: 8 }} onClick={onReset}><Icon name="plus" size={15} /> New analysis</button>
      {nav.map(([ic, label, act]) => <div key={label} className={"rail-item" + (act ? " active" : "")}><Icon name={ic} size={16} cls="rail-ico" /> {label}</div>)}
      <div className="kicker" style={{ padding: "16px 11px 8px" }}>Active dataset</div>
      <div className="rail-item active" style={{ fontSize: 12.5 }}>
        <Icon name="file" size={15} cls="rail-ico" />
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{ds.name}</span>
      </div>
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
function Workspace({ ds, preview, theme, onToggle, onReset }) {
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
    setMessages(m => [...m, { role: "user", text }]);
    setBusy(true);
    Promise.all([
      window.Alumni.ask(ds, text).catch(e => ({ op: null, kind: "structured", text: "Something went wrong: " + (e.message || e) })),
      new Promise(r => setTimeout(r, 420)),
    ]).then(([ans]) => {
      setMessages(m => [...m, { role: "ai", ...ans }]);
      setBusy(false);
    });
  }, [ds, busy]);

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
        <Rail ds={ds} onReset={onReset} />
        <div className="col" style={{ flex: 1, minWidth: 0, minHeight: 0 }}>
          <div ref={scrollRef} className="col" style={{ flex: "0 0 auto", padding: "26px 34px", gap: 28, overflowY: "visible" }}>
            {messages.map((m, i) => m.role === "user" ? <UserMsg key={i} text={m.text} /> : <AiMsg key={i} ds={ds} msg={m} onFollowup={send} />)}
            {busy && <Thinking />}
          </div>
          <div style={{ padding: "0 34px 22px" }}>
            <Composer onSend={send} busy={busy} suggestions={sugg} showSugg={onlyGreeting} />
          </div>
        </div>
        <DataPanel ds={ds} preview={preview} />
      </div>
    </div>
  );
}

/* ---------- root ---------- */
function App() {
  const [ds, setDs] = useState(null);
  const [preview, setPreview] = useState({ columns: [], rows: [], loading: false, error: "" });
  const [theme, setTheme] = useState(() => (typeof localStorage !== "undefined" && localStorage.getItem("alumniTheme")) || "light");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const loadSeq = useRef(0);
  useEffect(() => { try { localStorage.setItem("alumniTheme", theme); } catch (e) {} }, [theme]);
  const toggle = () => setTheme(t => t === "light" ? "dark" : "light");

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
        setLoading(false);
      }
      return;
    }

    if (seq !== loadSeq.current) return;
    setDs(uploaded);
    setLoading(false);

    if (window.Alumni.isApi && window.Alumni.isApi() && !uploaded.dataset_id) {
      setPreview({ columns: [], rows: [], loading: false, error: "Preview failed: upload response did not include dataset_id." });
      return;
    }

    setPreview({ columns: [], rows: [], loading: true, error: "" });
    try {
      const p = await window.Alumni.preview(uploaded.dataset_id || uploaded);
      if (seq !== loadSeq.current) return;
      setPreview({ columns: p.columns || [], rows: p.rows || [], loading: false, error: "" });
      setDs(current => current ? mergeDatasetPreview(current, p) : current);
    } catch (e) {
      if (seq !== loadSeq.current) return;
      const message = e.message || "Could not load preview.";
      console.error("Preview fetch failed after successful upload:", message);
      setPreview({ columns: [], rows: [], loading: false, error: `Preview failed: ${message}` });
    }
  };

  if (loading) return (
    <div className="screen col" data-theme={theme} style={{ width: "100%", minHeight: "100vh", alignItems: "center", justifyContent: "center", gap: 18 }}>
      <div className="brand-mark" style={{ width: 44, height: 44, animation: "pulse 1.1s ease-in-out infinite" }} />
      <div className="col" style={{ alignItems: "center", gap: 5 }}>
        <span style={{ fontWeight: 700, fontSize: 15 }}>Profiling your spreadsheet…</span>
        <span style={{ fontSize: 12.5, color: "var(--text-3)" }}>Reading columns, inferring types, scanning for gaps</span>
      </div>
    </div>
  );

  return ds
    ? <Workspace ds={ds} preview={preview} theme={theme} onToggle={toggle} onReset={() => { loadSeq.current += 1; setDs(null); setPreview({ columns: [], rows: [], loading: false, error: "" }); }} />
    : <UploadView onLoad={load} loadError={error} theme={theme} onToggle={toggle} />;
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
