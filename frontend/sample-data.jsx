/* sample-data.jsx — a realistic alumni roster, generated deterministically.
   Produces a CSV string (with $ currency + a few blanks) so the prototype's
   parser/profiler/intent layer all exercise real paths. */
(function () {
  const first = ["Priya","Marcus","Elena","Daniel","Aisha","Tom","Lucia","Jordan","Nina","Omar","Sofia","Liam","Grace","Andre","Maya","Ethan","Carmen","Noah","Ruth","Hassan","Iris","Kevin","Tara","Diego","Hana","Paul","Yuki","Sam","Leah","Victor","Anya","Ben","Mei","Carlos","Zoe","Raj","Clara","Owen","Fatima","Luke","Nadia","Eric","Sara","Gabe"];
  const last = ["Nair","Webb","Sokolova","Osei","Rahman","Becker","Romano","Pierce","Park","Haddad","Reyes","Murphy","Liu","Santos","Cohen","Brooks","Diaz","Klein","Mensah","Ali","Novak","Tran","Singh","Moreno","Sato","Walsh","Tanaka","Cole","Friedman","Petrov","Volkov","Hughes","Chen","Vega","Adams","Patel","Fischer","Doyle","Khan","Barnes","Aziz","Lund","Beck","Ortiz"];
  const degrees = ["B.S. CS","MBA","B.A. Econ","B.S. ME","M.S. Bio","B.A. History","B.S. EE","M.S. Stats","B.A. English","B.S. Physics","M.Eng","B.S. Finance"];
  const ind = {
    Technology:["Stripe","Figma","Datadog","Nvidia","Airbnb"],
    Finance:["Citadel","Goldman","BlackRock","Two Sigma","Capital One"],
    Consulting:["McKinsey","BCG","Bain","Deloitte"],
    Healthcare:["Genentech","Pfizer","Kaiser","Moderna"],
    Education:["Khan Academy","Coursera","Stanford","NYU"],
    Manufacturing:["Tesla","Boeing","3M","Ford"],
    Media:["Netflix","Spotify","NYT","Conde Nast"],
    Nonprofit:["Red Cross","UNICEF","Gates Fdn"],
  };
  const indNames = Object.keys(ind);
  const cities = ["San Jose","New York","Chicago","Austin","Boston","Seattle","Los Angeles","Denver","Atlanta","Miami"];
  const chapters = ["Bay Area","NYC Metro","New England","Pacific NW","Midwest","SoCal","Southeast"];

  // mulberry32 deterministic PRNG
  let seed = 20260603;
  function rnd(){ seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed>>>15, 1|seed); t = t + Math.imul(t ^ t>>>7, 61|t) ^ t; return ((t ^ t>>>14) >>> 0) / 4294967296; }
  const pick = a => a[Math.floor(rnd()*a.length)];

  const N = 44;
  const header = ["Name","Grad Year","Degree","Industry","Company","City","Engagement Score","Lifetime Giving","Last Contact","Email","Chapter"];
  const rows = [header.join(",")];
  for (let i = 0; i < N; i++) {
    const fn = first[i % first.length], ln = last[(i*3+7) % last.length];
    const year = 2005 + Math.floor(rnd()*17);                 // 2005–2021
    const industry = pick(indNames);
    const company = pick(ind[industry]);
    const eng = 42 + Math.floor(rnd()*54);                    // 42–95
    // older + more engaged → more giving, with noise
    const age = 2022 - year;
    const giving = Math.max(0, Math.round((age*180 + eng*60 + (rnd()-0.3)*4000) / 50) * 50);
    const m = 1 + Math.floor(rnd()*12), d = 1 + Math.floor(rnd()*27);
    const cy = rnd() < 0.25 ? 2025 : 2026;
    const date = `${cy}-${String(m).padStart(2,"0")}-${String(d).padStart(2,"0")}`;
    const email = `${fn.toLowerCase()}.${ln.toLowerCase()}@alumni.edu`;
    // sprinkle a few blanks to exercise missing-value profiling
    const engCell = (i === 9 || i === 23) ? "" : String(eng);
    const cityCell = (i === 14) ? "" : pick(cities);
    rows.push([
      `${fn} ${ln}`, year, pick(degrees), industry, company, cityCell,
      engCell, `$${giving}`, date, email, pick(chapters)
    ].join(","));
  }
  window.SAMPLE_CSV = rows.join("\n");
  window.SAMPLE_NAME = "alumni_master_2026.csv";
})();
