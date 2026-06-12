/* engine.jsx — client-side analysis engine for Alumni AI prototype.
   Mirrors the Flask backend: CSV parse → profile → safe intent ops → narration.
   No eval, no codegen — a fixed set of pandas-like operations. Exposed on window. */

/* ----------------------------- CSV parse ----------------------------- */
function parseCSV(text) {
  text = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const rows = [];
  let row = [], field = "", inQ = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQ) {
      if (c === '"') { if (text[i+1] === '"') { field += '"'; i++; } else inQ = false; }
      else field += c;
    } else {
      if (c === '"') inQ = true;
      else if (c === ",") { row.push(field); field = ""; }
      else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
      else field += c;
    }
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  const nonEmpty = rows.filter(r => r.some(c => String(c).trim() !== ""));
  if (!nonEmpty.length) throw new Error("Spreadsheet is empty or has no readable columns.");
  let header = nonEmpty[0].map(h => String(h).trim());
  // de-dup blank/duplicate column names (backend parity)
  const counts = {};
  header = header.map(h => {
    let base = h || "Unnamed";
    counts[base] = (counts[base] || 0) + 1;
    return counts[base] === 1 ? base : `${base}_${counts[base]}`;
  });
  const dataRows = nonEmpty.slice(1).map(r => {
    const o = {};
    header.forEach((h, i) => { o[h] = (r[i] ?? "").trim(); });
    return o;
  });
  return { columns: header, rows: dataRows };
}

/* ----------------------------- profiling ----------------------------- */
const numRe = /^-?\$?\s*-?[\d,]*\.?\d+%?$/;
function toNum(v) {
  if (v == null) return NaN;
  const s = String(v).trim().replace(/[$,%\s]/g, "");
  if (s === "") return NaN;
  const n = Number(s);
  return Number.isFinite(n) ? n : NaN;
}
function looksDate(v) { return v && !isNaN(Date.parse(v)) && /[-/:]/.test(v) && /\d{2,4}/.test(v); }

function profile(parsed) {
  const { columns, rows } = parsed;
  const meta = {};
  for (const col of columns) {
    const vals = rows.map(r => r[col]);
    const nonEmpty = vals.filter(v => v !== "" && v != null);
    const missing = vals.length - nonEmpty.length;
    let type = "text", currency = false, isYear = false;
    const nameL = col.toLowerCase();
    const numHits = nonEmpty.filter(v => numRe.test(String(v).trim())).length;
    const dateHits = nonEmpty.filter(looksDate).length;
    if (nonEmpty.length && numHits / nonEmpty.length >= 0.85) {
      type = "num";
      currency = nonEmpty.some(v => String(v).includes("$"));
      isYear = /year/.test(nameL) && nonEmpty.every(v => { const n = toNum(v); return n >= 1900 && n <= 2100; });
    } else if ((/(date|time)/.test(nameL) || (nonEmpty.length && dateHits / nonEmpty.length >= 0.8))) {
      type = "date";
    }
    meta[col] = { name: col, type, currency, year: isYear, missing, nonEmpty: nonEmpty.length };
  }
  const totalMissing = Object.values(meta).reduce((a, m) => a + m.missing, 0);
  return { columns, rows, meta, rows_n: rows.length, cols_n: columns.length, totalMissing };
}

/* ----------------------------- formatting ----------------------------- */
function fmtNum(v, col) {
  if (v == null || isNaN(v)) return "—";
  if (col && col.year) return String(Math.round(v));
  const cur = col && col.currency;
  const abs = Math.abs(v);
  if (cur) {
    if (abs >= 1e6) return "$" + (v/1e6).toFixed(1) + "M";
    if (abs >= 1e5) return "$" + (v/1e3).toFixed(1) + "K";
    return "$" + Math.round(v).toLocaleString();
  }
  if (Number.isInteger(v)) return v.toLocaleString();
  return (Math.round(v*100)/100).toLocaleString();
}

/* ----------------------------- column matching ----------------------------- */
function norm(s){ return String(s).toLowerCase().replace(/[^a-z0-9]+/g," ").trim(); }
function mentioned(question, columns) {
  const q = " " + norm(question) + " ";
  const hits = [];
  for (const c of columns) {
    const nc = norm(c);
    if (!nc) continue;
    const pos = q.indexOf(" " + nc + " ");
    if (pos >= 0) hits.push([pos, -nc.length, c]);
  }
  hits.sort((a,b)=> a[0]-b[0] || a[1]-b[1]);
  return hits.map(h => h[2]);
}
function colAfter(question, kw, columns) {
  const low = question.toLowerCase();
  const idx = low.lastIndexOf(" " + kw + " ");
  if (idx < 0) return null;
  const tail = question.slice(idx + kw.length + 2);
  const m = mentioned(tail, columns);
  return m[0] || null;
}

/* ----------------------------- operations ----------------------------- */
function groupBy(ds, groupCol, valueCol, op) {
  const g = {};
  for (const r of ds.rows) {
    const key = r[groupCol] === "" || r[groupCol] == null ? "Missing" : r[groupCol];
    if (!g[key]) g[key] = [];
    if (op === "count") g[key].push(1);
    else { const n = toNum(r[valueCol]); if (!isNaN(n)) g[key].push(n); }
  }
  let rows = Object.entries(g).map(([k, arr]) => {
    let v;
    if (op === "count") v = arr.length;
    else if (op === "sum") v = arr.reduce((a,b)=>a+b,0);
    else if (op === "mean") v = arr.reduce((a,b)=>a+b,0)/(arr.length||1);
    else if (op === "min") v = Math.min(...arr);
    else if (op === "max") v = Math.max(...arr);
    return { key: k, value: v };
  });
  const asc = op === "min";
  rows.sort((a,b)=> asc ? a.value-b.value : b.value-a.value);
  return rows.slice(0, 12);
}
function topRows(ds, sortCol, asc, limit) {
  const num = ds.meta[sortCol].type === "num";
  const rows = [...ds.rows].sort((a,b)=>{
    let x = a[sortCol], y = b[sortCol];
    if (num) { x = toNum(x); y = toNum(y); x = isNaN(x)?-Infinity:x; y = isNaN(y)?-Infinity:y; return asc ? x-y : y-x; }
    x = String(x).toLowerCase(); y = String(y).toLowerCase();
    return asc ? (x<y?-1:x>y?1:0) : (x>y?-1:x<y?1:0);
  });
  return rows.slice(0, limit);
}
function corr(ds, c1, c2) {
  const xs = [], ys = [];
  for (const r of ds.rows) { const a = toNum(r[c1]), b = toNum(r[c2]); if (!isNaN(a)&&!isNaN(b)){ xs.push(a); ys.push(b);} }
  const n = xs.length;
  if (n < 2) return { r: null, n };
  const mx = xs.reduce((a,b)=>a+b,0)/n, my = ys.reduce((a,b)=>a+b,0)/n;
  let sxy=0,sx=0,sy=0;
  for (let i=0;i<n;i++){ const dx=xs[i]-mx, dy=ys[i]-my; sxy+=dx*dy; sx+=dx*dx; sy+=dy*dy; }
  const d = Math.sqrt(sx*sy);
  return { r: d===0?null:sxy/d, n };
}
function colSummary(ds, col) {
  const m = ds.meta[col];
  const vals = ds.rows.map(r=>r[col]).filter(v=>v!==""&&v!=null);
  if (m.type === "num") {
    const nums = vals.map(toNum).filter(v=>!isNaN(v)).sort((a,b)=>a-b);
    const n = nums.length, sum = nums.reduce((a,b)=>a+b,0);
    const mean = sum/(n||1);
    const median = n? (n%2? nums[(n-1)/2] : (nums[n/2-1]+nums[n/2])/2) : 0;
    const sd = Math.sqrt(nums.reduce((a,b)=>a+(b-mean)**2,0)/(n||1));
    return { kind:"numeric", col, count:n, mean, median, min:nums[0], max:nums[n-1], sum, sd };
  }
  if (m.type === "date") {
    const ds2 = vals.map(v=>new Date(v)).filter(d=>!isNaN(d)).sort((a,b)=>a-b);
    return { kind:"date", col, earliest:ds2[0], latest:ds2[ds2.length-1], count:ds2.length };
  }
  const vc = {};
  vals.forEach(v=>{ vc[v]=(vc[v]||0)+1; });
  const top = Object.entries(vc).sort((a,b)=>b[1]-a[1]).slice(0,8);
  return { kind:"categorical", col, unique:Object.keys(vc).length, top };
}

/* ----------------------------- intent + narration ----------------------------- */
function detectAgg(q){
  if (/\b(average|mean|avg)\b/.test(q)) return "mean";
  if (/\b(total|sum)\b/.test(q)) return "sum";
  if (/\b(how many|count|number of)\b/.test(q)) return "count";
  if (/\b(maximum|max|highest|largest|biggest)\b/.test(q)) return "max_hint";
  if (/\b(minimum|min|lowest|smallest)\b/.test(q)) return "min_hint";
  return null;
}
const aggLabel = { sum:"total", mean:"average", count:"number of records", min:"minimum", max:"maximum" };

function ask(ds, question) {
  const q = question.toLowerCase();
  const cols = ds.columns;
  const ment = mentioned(question, cols);
  const numMent = ment.filter(c => ds.meta[c].type === "num");
  const nonNum = ment.filter(c => ds.meta[c].type !== "num");

  // missing / nulls
  if (/\b(missing|null|blank|empty|incomplete)\b/.test(q)) {
    const per = cols.map(c => ({ col:c, n: ds.meta[c].missing })).filter(x=>x.n>0).sort((a,b)=>b.n-a.n);
    const worst = per[0];
    const text = ds.totalMissing === 0
      ? "This dataset is complete — no missing values in any column."
      : `There are ${ds.totalMissing.toLocaleString()} missing values across ${per.length} column${per.length>1?"s":""}.${worst?` ${worst.col} has the most (${worst.n}).`:""}`;
    return { op:"summarize · missing", kind:"missing", text, result:{ per, total: ds.totalMissing } };
  }

  // correlation
  if (/\b(correlation|correlate|relationship|related|relate|vs\.?|versus)\b/.test(q) && numMent.length >= 2) {
    const [c1,c2] = numMent;
    const { r, n } = corr(ds, c1, c2);
    let strength = "no", dir = "";
    if (r != null) {
      const a = Math.abs(r);
      strength = a>=0.7?"strong":a>=0.4?"moderate":a>=0.2?"weak":"little to no";
      dir = r>=0?"positive":"negative";
    }
    const text = r==null
      ? `Not enough overlapping numeric data to correlate ${c1} and ${c2}.`
      : `${c1} and ${c2} show a ${strength}${dir?` ${dir}`:""} correlation (r = ${r.toFixed(2)}) across ${n.toLocaleString()} alumni.`;
    return { op:"correlation", kind:"correlation", text, result:{ c1, c2, r, n } };
  }

  // column summary (explicit)
  if (/\b(summary|summarize|summarise|describe|distribution|profile of|stats? (on|for|about))\b/.test(q) && ment.length) {
    const s = colSummary(ds, ment[0]);
    return { op:"summarize_column", kind:"colsummary", text: narrateCol(s, ds), result:s };
  }

  // grouped aggregate
  const aggRaw = detectAgg(q);
  if (aggRaw) {
    let agg = aggRaw;
    const hasBy = /\bby\b/.test(q);
    if (aggRaw === "max_hint") agg = hasBy ? "max" : (numMent.length ? "sum" : null);
    if (aggRaw === "min_hint") agg = "min";
    // ranked total language ("highest total giving by class")
    if (agg) {
      let groupCol = colAfter(question, "by", cols) || nonNum[0] || ment[ment.length-1] || null;
      let valueCol = numMent.find(c => c !== groupCol) || ment.find(c => c !== groupCol) || null;
      if (agg === "count" && !valueCol) valueCol = groupCol;
      if (groupCol && valueCol && (hasBy || agg==="count" || agg==="max" || agg==="min" || agg==="sum" || agg==="mean")) {
        const rows = groupBy(ds, groupCol, valueCol, agg);
        if (rows.length) {
          const vcol = ds.meta[valueCol];
          const text = narrateGroup(rows, groupCol, valueCol, agg, vcol);
          return { op:`group_by · ${agg}`, kind:"group", text,
            result:{ rows, groupCol, valueCol, agg, currency: agg!=="count" && vcol.currency } };
        }
      }
    }
  }

  // top rows
  if (/\b(top|highest|largest|biggest|lowest|smallest|bottom|most|least|rank)\b/.test(q) && ment.length) {
    const asc = /\b(lowest|smallest|bottom|least)\b/.test(q);
    const sortCol = numMent[0] || ment[0];
    const limit = (q.match(/\btop\s+(\d{1,2})\b/) ? parseInt(RegExp.$1) : 5);
    const rows = topRows(ds, sortCol, asc, Math.min(limit, 10));
    const showCols = pickDisplayCols(ds, sortCol);
    const text = narrateTop(rows, sortCol, asc, ds, showCols);
    return { op:`top_rows · ${asc?"asc":"desc"}`, kind:"top", text, result:{ rows, sortCol, asc, showCols } };
  }

  // fallback
  return {
    op:null, kind:"help",
    text:`I can answer that with the columns I found. Try a grouped total (e.g. “total ${guessNumCol(ds)} by ${guessCatCol(ds)}”), a ranking (“top alumni by ${guessNumCol(ds)}”), a correlation between two numeric columns, or “what's missing?”.`,
    result:null
  };
}

/* ----------------------------- narration helpers ----------------------------- */
function narrateGroup(rows, groupCol, valueCol, agg, vcol) {
  const top = rows[0];
  const fmt = v => agg==="count" ? v.toLocaleString() : fmtNum(v, vcol);
  const label = aggLabel[agg];
  const lead = agg==="count"
    ? `**${top.key}** has the most records (${top.value.toLocaleString()})`
    : `**${top.key}** leads with **${fmt(top.value)}** in ${label} ${valueCol}`;
  const rest = rows.slice(1,3).map(r=>`${r.key} (${fmt(r.value)})`).join(" and ");
  return `${lead}${rest?`, followed by ${rest}`:""}. Grouped ${valueCol} by ${groupCol} across the dataset.`;
}
function narrateTop(rows, sortCol, asc, ds, showCols) {
  if (!rows.length) return "No rows to rank.";
  const nameCol = showCols[0];
  const r0 = rows[0];
  const v = ds.meta[sortCol].type==="num" ? fmtNum(toNum(r0[sortCol]), ds.meta[sortCol]) : r0[sortCol];
  return `**${r0[nameCol]}** has the ${asc?"lowest":"highest"} ${sortCol} at **${v}**. Showing the ${asc?"bottom":"top"} ${rows.length} by ${sortCol}.`;
}
function narrateCol(s, ds) {
  if (s.kind==="numeric") {
    const c = ds.meta[s.col];
    return `**${s.col}** ranges from ${fmtNum(s.min,c)} to ${fmtNum(s.max,c)}, averaging **${fmtNum(s.mean,c)}** across ${s.count.toLocaleString()} values (median ${fmtNum(s.median,c)}).`;
  }
  if (s.kind==="date") {
    const f = d => d? d.toISOString().slice(0,10) : "—";
    return `**${s.col}** spans ${f(s.earliest)} to ${f(s.latest)} across ${s.count.toLocaleString()} dated records.`;
  }
  return `**${s.col}** has **${s.unique.toLocaleString()}** unique values. The most common is ${s.top[0][0]} (${s.top[0][1]}).`;
}
function pickDisplayCols(ds, sortCol) {
  const text = ds.columns.filter(c => ds.meta[c].type==="text");
  const nameCol = text.find(c=>/name/i.test(c)) || text[0] || ds.columns[0];
  const extra = ds.columns.filter(c => c!==nameCol && c!==sortCol).slice(0,2);
  return [nameCol, ...extra, sortCol].filter((v,i,a)=>a.indexOf(v)===i);
}
function guessNumCol(ds){ return ds.columns.find(c=>ds.meta[c].type==="num") || "a value"; }
function guessCatCol(ds){ return ds.columns.find(c=>ds.meta[c].type==="text") || "a category"; }

function suggestedQuestions(ds) {
  const num = ds.columns.filter(c=>ds.meta[c].type==="num");
  const cat = ds.columns.filter(c=>ds.meta[c].type==="text" && ds.meta[c].name.toLowerCase()!=="email" && !/name/i.test(c));
  const out = [];
  const giving = num.find(c=>/giv|donat|amount|revenue|total|spend|sales/i.test(c)) || num[0];
  if (giving && cat[0]) out.push(`Total ${giving} by ${cat[0]}`);
  if (giving) out.push(`Top alumni by ${giving}`);
  if (num.length>=2) out.push(`Relationship between ${num[0]} and ${num[1]}`);
  if (cat[1]) out.push(`Average ${giving||num[0]} by ${cat[1]}`);
  out.push("What's missing in this data?");
  return out.slice(0,4);
}

Object.assign(window, { parseCSV, profile, ask, fmtNum, toNum, suggestedQuestions, colSummary });
