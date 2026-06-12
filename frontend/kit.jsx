/* kit.jsx — shared building blocks for Alumni AI screens. */

const ICON = {
  bolt: "M13 3L5 13h6l-1 8 8-10h-6z",
  bookmark: "M7 4h10v16l-5-4-5 4z",
  chat: "M5 5h14v10H9l-4 4z",
  database: "M12 4c4 0 7 1.3 7 3s-3 3-7 3-7-1.3-7-3 3-3 7-3M5 7v5c0 1.7 3 3 7 3s7-1.3 7-3V7M5 12v5c0 1.7 3 3 7 3s7-1.3 7-3v-5",
  download: "M12 4v10M8 11l4 4 4-4M5 19h14",
  file: "M13 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9zM13 3v6h6",
  history: "M4 12a8 8 0 1 0 8-8 8 8 0 0 0-7 4M4 4v4h4M12 8v4l3 2",
  moon: "M20 14.5A8 8 0 0 1 9.5 4 7 7 0 1 0 20 14.5z",
  plus: "M12 5v14M5 12h14",
  send: "M7 12h11M13 6l6 6-6 6",
  sparkle: "M12 4l1.6 4.6L18 10l-4.4 1.4L12 16l-1.6-4.6L6 10l4.4-1.4z",
  sun: "M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10M12 2v2M12 20v2M4 12H2M22 12h-2M5.6 5.6L7 7M17 17l1.4 1.4M5.6 18.4L7 17M17 7l1.4-1.4",
  upload: "M12 16V5M7 10l5-5 5 5M5 19h14",
};

function Icon({ name, size = 18, style, cls }) {
  return (
    <svg className={"ico " + (cls || "")} width={size} height={size} viewBox="0 0 24 24" style={style}>
      <path d={ICON[name] || ICON.file} />
    </svg>
  );
}

function Brand({ small }) {
  return (
    <div className="brand">
      <div className="brand-mark" />
      {!small && <div className="brand-name">Alumni<span> AI</span></div>}
    </div>
  );
}

function TypePill({ t }) {
  const map = {
    num: ["type-num", "number"],
    text: ["type-txt", "text"],
    txt: ["type-txt", "text"],
    date: ["type-date", "date"],
  };
  const [cls, label] = map[t] || map.text;
  return <span className={"type-pill " + cls}>{label}</span>;
}

Object.assign(window, {
  Icon,
  ICON,
  Brand,
  TypePill,
});
