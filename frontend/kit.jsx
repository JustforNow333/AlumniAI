/* kit.jsx — shared building blocks + mock data for Alumni AI mockups.
   Exports to window so every screen file can use them. */

/* ----------------------------- icons ----------------------------- */
const ICON = {
  send:    "M7 12h11M13 6l6 6-6 6",
  upload:  "M12 16V5M7 10l5-5 5 5M5 19h14",
  file:    "M13 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9zM13 3v6h6",
  table:   "M4 5h16v14H4zM4 10h16M4 15h16M10 5v14",
  chat:    "M5 5h14v10H9l-4 4z",
  sparkle: "M12 4l1.6 4.6L18 10l-4.4 1.4L12 16l-1.6-4.6L6 10l4.4-1.4z",
  search:  "M11 11m-6 0a6 6 0 1 0 12 0a6 6 0 1 0-12 0M20 20l-3.5-3.5",
  plus:    "M12 5v14M5 12h14",
  chevron: "M9 6l6 6-6 6",
  chevronD:"M6 9l6 6 6-6",
  grid:    "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
  history: "M4 12a8 8 0 1 0 8-8 8 8 0 0 0-7 4M4 4v4h4M12 8v4l3 2",
  settings:"M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6M5 12l-2 1 2 4 2-.6M19 12l2 1-2 4-2-.6M12 5V3M12 21v-2",
  download:"M12 4v10M8 11l4 4 4-4M5 19h14",
  filter:  "M4 5h16l-6 8v5l-4 2v-7z",
  check:   "M5 12l4 4 10-10",
  close:   "M6 6l12 12M18 6L6 18",
  more:    "M6 12h.01M12 12h.01M18 12h.01",
  bolt:    "M13 3L5 13h6l-1 8 8-10h-6z",
  bookmark:"M7 4h10v16l-5-4-5 4z",
  sun:     "M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10M12 2v2M12 20v2M4 12H2M22 12h-2M5.6 5.6L7 7M17 17l1.4 1.4M5.6 18.4L7 17M17 7l1.4-1.4",
  moon:    "M20 14.5A8 8 0 0 1 9.5 4 7 7 0 1 0 20 14.5z",
  database:"M12 4c4 0 7 1.3 7 3s-3 3-7 3-7-1.3-7-3 3-3 7-3M5 7v5c0 1.7 3 3 7 3s7-1.3 7-3V7M5 12v5c0 1.7 3 3 7 3s7-1.3 7-3v-5",
};
function Icon({ name, size = 18, style, cls }) {
  return (
    <svg className={"ico " + (cls||"")} width={size} height={size} viewBox="0 0 24 24" style={style}>
      <path d={ICON[name] || ICON.file} />
    </svg>
  );
}

/* ----------------------------- brand ----------------------------- */
function Brand({ small }) {
  return (
    <div className="brand">
      <div className="brand-mark" />
      {!small && <div className="brand-name">Alumni<span> AI</span></div>}
    </div>
  );
}

/* ------------------------- mock dataset --------------------------- */
const DATASET = {
  name: "alumni_master_2026.xlsx",
  rows: 4812,
  cols: 11,
  missing: 137,
  columns: [
    { name: "Name",            type: "txt" },
    { name: "Grad Year",       type: "num" },
    { name: "Degree",          type: "txt" },
    { name: "Industry",        type: "txt" },
    { name: "Company",         type: "txt" },
    { name: "City",            type: "txt" },
    { name: "Engagement Score",type: "num" },
    { name: "Lifetime Giving", type: "num" },
    { name: "Last Contact",    type: "date" },
    { name: "Email",           type: "txt" },
    { name: "Chapter",         type: "txt" },
  ],
  preview: [
    ["Priya Nair",      2014, "B.S. CS",   "Technology", "Stripe",      "San Jose",  82, "$4,200", "2026-04-18"],
    ["Marcus Webb",     2009, "MBA",       "Finance",    "Citadel",     "New York",  91, "$18,500","2026-05-02"],
    ["Elena Sokolova",  2018, "B.A. Econ", "Consulting", "McKinsey",    "Chicago",   67, "$900",   "2025-11-30"],
    ["Daniel Osei",     2011, "B.S. ME",   "Manufacturing","Tesla",     "Austin",    74, "$3,100", "2026-03-12"],
    ["Aisha Rahman",    2016, "M.S. Bio",  "Healthcare", "Genentech",   "Boston",    88, "$6,750", "2026-04-29"],
    ["Tom Becker",      2007, "B.A. Hist", "Education",  "Khan Academy","Seattle",   59, "$450",   "2025-09-08"],
    ["Lucia Romano",    2019, "B.S. CS",   "Technology", "Figma",       "San Jose",  79, "$1,250", "2026-05-10"],
    ["Jordan Pierce",   2013, "MBA",       "Finance",    "Goldman",     "New York",  85, "$12,300","2026-02-21"],
  ],
};
const PREVIEW_COLS = ["Name","Grad Year","Degree","Industry","Company","City","Engmt","Giving","Last Contact"];

/* answer content reused across answer-card variants */
const GIVING_BY_YEAR = [
  { k: "2009", v: 486200, n: "$486.2K" },
  { k: "2011", v: 412800, n: "$412.8K" },
  { k: "2013", v: 388500, n: "$388.5K" },
  { k: "2007", v: 351900, n: "$351.9K" },
  { k: "2014", v: 298400, n: "$298.4K" },
  { k: "2016", v: 244100, n: "$244.1K" },
];

/* ----------------------------- pieces ----------------------------- */
function TypePill({ t }) {
  const map = { num:["type-num","number"], txt:["type-txt","text"], date:["type-date","date"] };
  const [cls,label] = map[t] || map.txt;
  return <span className={"type-pill "+cls}>{label}</span>;
}

function PreviewTable({ rows = DATASET.preview, dense }) {
  return (
    <table className="dtable">
      <thead>
        <tr>{PREVIEW_COLS.map((c,i)=>(
          <th key={c} style={{textAlign: i>=6 ? "right":"left"}}>{c}</th>
        ))}</tr>
      </thead>
      <tbody>
        {rows.map((r,i)=>(
          <tr key={i}>
            <td style={{fontWeight:600}}>{r[0]}</td>
            <td className="num muted">{r[1]}</td>
            <td className="muted">{r[2]}</td>
            <td><span className="chip" style={{padding:"2px 8px"}}>{r[3]}</span></td>
            <td className="muted">{r[4]}</td>
            <td className="muted">{r[5]}</td>
            <td className="num">{r[6]}</td>
            <td className="num" style={{color:"var(--primary)",fontWeight:600}}>{r[7]}</td>
            <td className="num muted">{r[8]}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function MiniBars({ data = GIVING_BY_YEAR, labelW = 120 }) {
  const max = Math.max(...data.map(d=>d.v));
  return (
    <div className="minibars">
      {data.map((d,i)=>(
        <div className="mbar" key={d.k} style={{gridTemplateColumns:`${labelW}px 1fr auto`}}>
          <span className="lab">Class of {d.k}</span>
          <span className="track"><span className="fill" style={{width:(d.v/max*100)+"%", opacity: 1-i*0.07}} /></span>
          <span className="val">{d.n}</span>
        </div>
      ))}
    </div>
  );
}

function Composer({ value, focus }) {
  return (
    <div className={"composer"+(focus?" focus":"")}>
      <Icon name="sparkle" size={18} style={{color:"var(--primary)",flex:"none"}} />
      {value
        ? <span style={{flex:1,fontSize:14,color:"var(--text)",fontWeight:500}}>{value}</span>
        : <span className="ph">Ask anything about your alumni data…</span>}
      <button className="btn-icon" style={{background:"var(--primary)",borderColor:"transparent",color:"var(--on-primary)",width:32,height:32}}>
        <Icon name="send" size={16} />
      </button>
    </div>
  );
}

function OpBadge({ op = "group_by · sum", cols }) {
  return (
    <span className="chip chip-primary chip-mono" title="Safe pandas operation used to compute this answer">
      <Icon name="bolt" size={12} style={{flex:"none"}} /> {op}
    </span>
  );
}

Object.assign(window, {
  Icon, ICON, Brand, DATASET, PREVIEW_COLS, GIVING_BY_YEAR,
  TypePill, PreviewTable, MiniBars, Composer, OpBadge,
});
