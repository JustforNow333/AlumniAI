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
function ResultBlock({ ds, msg }) {
  switch (msg.kind) {
    case "group": return <GroupResult ds={ds} result={msg.result} />;
    case "top": return <TopResult ds={ds} result={msg.result} />;
    case "correlation": return <CorrResult result={msg.result} />;
    case "colsummary": return <ColResult ds={ds} result={msg.result} />;
    case "missing": return <MissingResult result={msg.result} />;
    default: return null;
  }
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
function AiMsg({ ds, msg }) {
  return (
    <div className="msg">
      <div className="msg-av ai"><Icon name="sparkle" size={15} /></div>
      <div className="msg-body col" style={{ gap: 13 }}>
        <div className="msg-name">Alumni AI</div>
        <p className="prose" style={{ margin: 0 }}>{boldify(msg.text)}</p>
        <ResultBlock ds={ds} msg={msg} />
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
function DataPanel({ ds }) {
  return (
    <div className="col" style={{ width: 360, flex: "none", borderLeft: "1px solid var(--border)", background: "var(--surface)" }}>
      <div className="row" style={{ padding: "15px 18px", gap: 10, borderBottom: "1px solid var(--border)" }}>
        <Icon name="file" size={16} style={{ color: "var(--primary)" }} />
        <span className="mono" style={{ fontSize: 12.5, fontWeight: 600, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{ds.name}</span>
        <button className="btn-icon" style={{ width: 28, height: 28 }}><Icon name="download" size={14} /></button>
      </div>
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
      <div className="col" style={{ padding: "0 12px 12px", gap: 1, overflowY: "auto" }}>
        {ds.columns.map(c => (
          <div className="row" key={c} style={{ padding: "7px 8px", gap: 8, borderRadius: 8 }}>
            <span style={{ fontSize: 12.5, fontWeight: 500, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c}</span>
            {ds.meta[c].missing > 0 && <span title={ds.meta[c].missing + " missing"} style={{ fontSize: 10, color: "var(--warn)", fontWeight: 600 }}>{ds.meta[c].missing}⚠</span>}
            <TypePill t={ds.meta[c].type} />
          </div>
        ))}
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
  const loadSample = () => handleFile(new File([window.SAMPLE_CSV], window.SAMPLE_NAME, { type: "text/csv" }));
  return (
    <div className="screen col" data-theme={theme} style={{ width: "100%", height: "100%" }}>
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
              <span className="mono">{supportedLabel}</span><span>·</span>
              <span style={{ color: "var(--primary)", fontWeight: 600 }} onClick={e => { e.stopPropagation(); loadSample(); }}>try the sample dataset</span>
            </div>
          </div>
          {(err || loadError) && <div className="row gap8" style={{ color: "var(--warn)", fontSize: 12.5, fontWeight: 500, textAlign: "center" }}><Icon name="bolt" size={14} />{err || loadError}</div>}
        </div>
      </div>
    </div>
  );
}

/* ---------- workspace view ---------- */
function Workspace({ ds, theme, onToggle, onReset }) {
  const [messages, setMessages] = useState(() => [{ role: "ai", kind: "help", op: null,
    text: `Loaded **${ds.name}** — ${ds.rows_n.toLocaleString()} rows across ${ds.cols_n} columns. Ask me anything, or try one of the suggestions below.` }]);
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef(null);
  const sugg = window.suggestedQuestions(ds);

  useEffect(() => {
    const el = scrollRef.current; if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  const send = useCallback((q) => {
    setMessages(m => [...m, { role: "user", text: q }]);
    setBusy(true);
    Promise.all([
      window.Alumni.ask(ds, q).catch(e => ({ op: null, kind: "help", text: "Something went wrong: " + (e.message || e) })),
      new Promise(r => setTimeout(r, 420)),
    ]).then(([ans]) => {
      setMessages(m => [...m, { role: "ai", ...ans }]);
      setBusy(false);
    });
  }, [ds]);

  const onlyGreeting = messages.length === 1;

  return (
    <div className="screen col" data-theme={theme} data-screen-label="Workspace" style={{ width: "100%", height: "100%" }}>
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
      <div className="row" style={{ flex: 1, minHeight: 0 }}>
        <Rail ds={ds} onReset={onReset} />
        <div className="col" style={{ flex: 1, minWidth: 0 }}>
          <div ref={scrollRef} className="col" style={{ flex: 1, padding: "26px 34px", gap: 28, overflowY: "auto", minHeight: 0 }}>
            {messages.map((m, i) => m.role === "user" ? <UserMsg key={i} text={m.text} /> : <AiMsg key={i} ds={ds} msg={m} />)}
            {busy && <Thinking />}
          </div>
          <div style={{ padding: "0 34px 22px" }}>
            <Composer onSend={send} busy={busy} suggestions={sugg} showSugg={onlyGreeting} />
          </div>
        </div>
        <DataPanel ds={ds} />
      </div>
    </div>
  );
}

/* ---------- root ---------- */
function App() {
  const [ds, setDs] = useState(null);
  const [theme, setTheme] = useState(() => (typeof localStorage !== "undefined" && localStorage.getItem("alumniTheme")) || "light");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => { try { localStorage.setItem("alumniTheme", theme); } catch (e) {} }, [theme]);
  const toggle = () => setTheme(t => t === "light" ? "dark" : "light");

  const load = (file) => {
    setError(""); setLoading(true);
    window.Alumni.load(file)
      .then(d => { setDs(d); setLoading(false); })
      .catch(e => { setError(e.message || "Could not load that file."); setLoading(false); });
  };

  if (loading) return (
    <div className="screen col" data-theme={theme} style={{ width: "100%", height: "100%", alignItems: "center", justifyContent: "center", gap: 18 }}>
      <div className="brand-mark" style={{ width: 44, height: 44, animation: "pulse 1.1s ease-in-out infinite" }} />
      <div className="col" style={{ alignItems: "center", gap: 5 }}>
        <span style={{ fontWeight: 700, fontSize: 15 }}>Profiling your spreadsheet…</span>
        <span style={{ fontSize: 12.5, color: "var(--text-3)" }}>Reading columns, inferring types, scanning for gaps</span>
      </div>
    </div>
  );

  return ds
    ? <Workspace ds={ds} theme={theme} onToggle={toggle} onReset={() => setDs(null)} />
    : <UploadView onLoad={load} loadError={error} theme={theme} onToggle={toggle} />;
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
