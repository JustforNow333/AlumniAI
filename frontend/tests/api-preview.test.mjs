import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

function loadApi({ config, fetchImpl }) {
  const context = {
    window: { ALUMNI_CONFIG: config },
    fetch: fetchImpl,
    FormData: class FormData {
      append(key, value) {
        this[key] = value;
      }
    },
    FileReader: class FileReader {},
  };
  vm.createContext(context);
  vm.runInContext(readFileSync("frontend/api.jsx", "utf8"), context);
  return context.window.Alumni;
}

function loadApiWithWindow({ config, fetchImpl, windowExtras = {} }) {
  const context = {
    window: { ALUMNI_CONFIG: config, ...windowExtras },
    fetch: fetchImpl,
    FormData: class FormData {
      append(key, value) {
        this[key] = value;
      }
    },
    FileReader: class FileReader {},
  };
  vm.createContext(context);
  vm.runInContext(readFileSync("frontend/api.jsx", "utf8"), context);
  return context.window.Alumni;
}

function plain(value) {
  return JSON.parse(JSON.stringify(value));
}

test("load parses upload JSON dataset_id before preview uses the exact backend route", async () => {
  const calls = [];
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (url.endsWith("/api/upload")) {
        return {
          ok: true,
          status: 201,
          json: async () => ({
            dataset_id: "uploaded-id",
            filename: "alumni.csv",
            summary: {
              rows: 1,
              columns: 1,
              column_names: ["Name"],
              column_types: { Name: "object" },
              missing_values: { Name: 0 },
              preview: [{ Name: "Alice" }],
            },
          }),
        };
      }
      return {
        ok: true,
        json: async () => ({
          dataset_id: "uploaded-id",
          filename: "alumni.csv",
          row_count: 1,
          column_count: 1,
          missing_count: 0,
          data_types: { Name: "object" },
          missing_values: { Name: 0 },
          column_names: ["Name"],
          preview: [{ Name: "Alice" }],
        }),
      };
    },
  });

  const ds = await Alumni.load({ name: "alumni.csv" });
  const preview = await Alumni.preview(ds.dataset_id);

  assert.equal(ds.dataset_id, "uploaded-id");
  assert.equal(calls[0].url, "/api/upload");
  assert.equal(calls[1].url, "/api/datasets/uploaded-id/preview");
  assert.deepEqual(plain(preview), {
    dataset_id: "uploaded-id",
    filename: "alumni.csv",
    columns: ["Name"],
    rows: [{ Name: "Alice" }],
    row_count: 1,
    column_count: 1,
    missing_count: 0,
    data_types: { Name: "object" },
    missing_values: { Name: 0 },
  });
});

test("load accepts dataset_id from upload metadata", async () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "http://localhost:5000" },
    fetchImpl: async () => ({
      ok: true,
      status: 201,
      json: async () => ({
        metadata: { dataset_id: "metadata-id" },
        filename: "alumni.csv",
        summary: {
          rows: 0,
          columns: 0,
          column_names: [],
          column_types: {},
          missing_values: {},
          preview: [],
        },
      }),
    }),
  });

  const ds = await Alumni.load({ name: "alumni.csv" });

  assert.equal(ds.dataset_id, "metadata-id");
});

test("load rejects when upload succeeds without dataset_id", async () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "http://localhost:5000" },
    fetchImpl: async () => ({
      ok: true,
      status: 201,
      json: async () => ({ filename: "alumni.csv", summary: {} }),
    }),
  });

  await assert.rejects(
    () => Alumni.load({ name: "alumni.csv" }),
    /did not include dataset_id/
  );
});

test("preview fetches by dataset_id and adapts the existing backend response shape", async () => {
  const calls = [];
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "http://localhost:5000/" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return {
        ok: true,
        json: async () => ({
          column_names: ["Name", "Score"],
          preview: [
            { Name: "Alice", Score: 12 },
            { Name: null, Score: null },
          ],
        }),
      };
    },
  });

  const preview = await Alumni.preview({ dataset_id: "abc 123" });

  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "http://localhost:5000/api/datasets/abc%20123/preview");
  assert.equal(calls[0].options, undefined);
  assert.deepEqual(plain(preview), {
    dataset_id: "abc 123",
    columns: ["Name", "Score"],
    rows: [
      { Name: "Alice", Score: 12 },
      { Name: null, Score: null },
    ],
  });
});

test("preview also accepts the preferred columns/rows response shape", async () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "http://localhost:5000" },
    fetchImpl: async () => ({
      ok: true,
      json: async () => ({
        dataset_id: "server-id",
        columns: ["A"],
        rows: [{ A: "value" }],
        row_count: 3,
        column_count: 1,
      }),
    }),
  });

  const preview = await Alumni.preview("client-id");

  assert.deepEqual(plain(preview), {
    dataset_id: "server-id",
    columns: ["A"],
    rows: [{ A: "value" }],
    row_count: 3,
    column_count: 1,
  });
});

test("preview rejects with the backend error message when the request fails", async () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "http://localhost:5000" },
    fetchImpl: async () => ({
      ok: false,
      status: 404,
      json: async () => ({ error: "Dataset not found." }),
    }),
  });

  await assert.rejects(
    () => Alumni.preview("missing-id"),
    /Dataset not found\./
  );
});

test("preview falls back to local dataset rows outside API mode", async () => {
  const Alumni = loadApi({
    config: { useApi: false, apiBase: "" },
    fetchImpl: async () => {
      throw new Error("fetch should not be called");
    },
  });
  const rows = Array.from({ length: 12 }, (_, i) => ({ Row: i + 1 }));

  const preview = await Alumni.preview({ columns: ["Row"], rows });

  assert.deepEqual(plain(preview), {
    columns: ["Row"],
    rows: rows.slice(0, 10),
  });
});

test("ask posts the active dataset_id and keeps structured answer blocks", async () => {
  const calls = [];
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return {
        ok: true,
        json: async () => ({
          dataset_id: "dataset-1",
          answer: {
            title: "Top Customers",
            summary: "Echo has the highest revenue.",
            blocks: [
              {
                type: "table",
                title: "Top rows",
                columns: ["Customer", "Revenue"],
                rows: [["Echo", "500"]],
                caption: "Sorted by Revenue.",
              },
              {
                type: "metrics",
                items: [{ label: "Rows analyzed", value: "6" }],
              },
              {
                type: "ranked_list",
                title: "Recommended records",
                items: [{ label: "Echo", value: "500", description: "Highest revenue." }],
              },
            ],
            followups: ["Show this by category"],
          },
          answer_text: "Echo has the highest revenue.",
          operation: { type: "top_rows", ascending: false },
          result: {},
        }),
      };
    },
  });

  const msg = await Alumni.ask({ dataset_id: "dataset-1" }, "Top rows by revenue");

  assert.equal(calls[0].url, "/api/ask");
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    dataset_id: "dataset-1",
    question: "Top rows by revenue",
  });
  assert.equal(msg.kind, "structured");
  assert.equal(msg.text, "Echo has the highest revenue.");
  assert.equal(msg.op, "top_rows · desc");
  assert.deepEqual(plain(msg.answer.followups), ["Show this by category"]);
  assert.equal(msg.answer.blocks[0].type, "table");
  assert.equal(msg.answer.blocks[1].type, "metrics");
  assert.equal(msg.answer.blocks[2].type, "ranked_list");
  assert.equal(msg.response_payload.question, "Top rows by revenue");
  assert.deepEqual(plain(msg.response_payload.answer), plain(msg.answer));
  assert.deepEqual(plain(msg.response_payload.operation), { type: "top_rows", ascending: false });
  assert.deepEqual(plain(msg.response_payload.answer.blocks[0].rows), [["Echo", "500"]]);
});

test("ask safely adapts legacy plain-text backend answers", async () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async () => ({
      ok: true,
      json: async () => ({
        answer: "Plain answer from backend.",
        operation: null,
        result: null,
      }),
    }),
  });

  const msg = await Alumni.ask({ dataset_id: "dataset-1" }, "Question");

  assert.equal(msg.kind, "structured");
  assert.equal(msg.answer.summary, "Plain answer from backend.");
  assert.deepEqual(plain(msg.answer.blocks), [
    { type: "markdown", content: "Plain answer from backend." },
  ]);
});

test("ask sanitizes alumni people results for visible columns and total match metrics", async () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async () => ({
      ok: true,
      json: async () => ({
        answer: {
          title: "Tech alumni",
          summary: "Found matching alumni.",
          blocks: [
            {
              type: "metrics",
              items: [
                { label: "Rows shown", value: "100" },
                { label: "Display limit", value: "100" },
              ],
            },
            {
              type: "table",
              title: "Rows",
              columns: [
                "Nickname",
                "First Name",
                "LastName",
                "Occupation",
                "Employer",
                "Match Reason",
                "confidence",
                "internal_reason",
                "classification",
                "LinkedinURL",
              ],
              rows: [
                [
                  "Ada",
                  "Ada",
                  "Lovelace",
                  "Software Engineer",
                  "Local Bakery",
                  "Matched OCCUPATION",
                  "0.99",
                  "internal",
                  "technical_title",
                  "linkedin.com/in/ada",
                ],
                [
                  "Grace",
                  "Grace",
                  "Hopper",
                  "CEO",
                  "Google",
                  "Matched EMPLOYER",
                  "0.95",
                  "internal",
                  "known_tech_company",
                  "",
                ],
              ],
            },
          ],
          followups: [],
        },
        operation: { type: "contains_any" },
        result: {
          intent: "people_filter",
          entity: "alumni",
          answer_label: "Alumni matching criteria",
          total_matches: 74,
          displayed_count: 2,
          display_limit: 100,
          uncertain_count: 8,
          visible_columns: ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL"],
        },
      }),
    }),
  });

  const msg = await Alumni.ask({ dataset_id: "dataset-1" }, "Show tech alumni");
  const metrics = msg.answer.blocks.find(block => block.type === "metrics");
  const table = msg.answer.blocks.find(block => block.type === "table");

  assert.deepEqual(plain(metrics.items), [
    { label: "Alumni matching criteria", value: "74" },
    { label: "Showing", value: "2" },
    { label: "Uncertain not counted", value: "8" },
  ]);
  assert.deepEqual(plain(table.columns), ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL"]);
  assert.deepEqual(plain(table.rows), [
    ["Ada", "Lovelace", "Software Engineer", "Local Bakery", "linkedin.com/in/ada"],
    ["Grace", "Hopper", "CEO", "Google", ""],
  ]);
});

test("frontend helpers resolve alumni columns and LinkedIn links", () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async () => {
      throw new Error("fetch should not be called");
    },
  });

  assert.equal(Alumni._test.canonicalDisplayColumn("last_name"), "Last Name");
  assert.equal(Alumni._test.canonicalDisplayColumn("LastName"), "Last Name");
  assert.equal(Alumni._test.canonicalDisplayColumn("LinkedinURL"), "LinkedIn URL");
  assert.equal(Alumni._test.isLinkedInColumn("linkedin_url"), true);
  assert.equal(Alumni._test.linkedInHref("linkedin.com/in/ada"), "https://linkedin.com/in/ada");
  assert.equal(Alumni._test.linkedInHref("https://linkedin.com/in/ada"), "https://linkedin.com/in/ada");
  assert.equal(Alumni._test.linkedInHref(""), "");
  assert.equal(Alumni._test.isDebugColumn("classification_reason"), true);
  assert.equal(Alumni._test.isDebugColumn("Match Reason"), true);
  assert.equal(Alumni._test.isDebugColumn("internal_reason"), true);
  assert.equal(Alumni._test.isDebugColumn("classification"), true);
});

test("ask adapts alumni tech query without showing an analysis-plan error", async () => {
  const calls = [];
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return {
        ok: true,
        json: async () => ({
          answer: {
            title: "Analysis Result",
            summary: "Alumni matching criteria: 2",
            blocks: [
              {
                type: "metrics",
                items: [{ label: "Display limit", value: "100" }],
              },
              {
                type: "table",
                columns: ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL"],
                rows: [
                  ["Ada", "Lovelace", "Software Engineer", "Local Bakery", "linkedin.com/in/ada"],
                  ["Grace", "Hopper", "Founder", "FanAmp", ""],
                ],
              },
            ],
            followups: [],
          },
          operation: { type: "contains_any" },
          result: {
            intent: "people_filter",
            entity: "alumni",
            answer_label: "Alumni matching criteria",
            total_matches: 2,
            displayed_count: 2,
            display_limit: 100,
            uncertain_count: 0,
            visible_columns: ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL"],
          },
        }),
      };
    },
  });

  const query = "How many alumni are working in tech either as software engineers or as other roles in a tech company?";
  const msg = await Alumni.ask({ dataset_id: "dataset-1" }, query);
  const body = JSON.parse(calls[0].options.body);
  const rendered = JSON.stringify(msg.answer);
  const metrics = msg.answer.blocks.find(block => block.type === "metrics");

  assert.equal(body.question, query);
  assert.equal(rendered.includes("Analysis Plan Error"), false);
  assert.equal(rendered.includes("could not create a valid analysis plan"), false);
  assert.deepEqual(plain(metrics.items), [
    { label: "Alumni matching criteria", value: "2" },
  ]);
});

test("summary fetches by dataset_id", async () => {
  const calls = [];
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url) => {
      calls.push(url);
      return {
        ok: true,
        json: async () => ({ rows: 6, columns: 5 }),
      };
    },
  });

  const summary = await Alumni.summary({ dataset_id: "summary id" });

  assert.equal(calls[0], "/api/datasets/summary%20id/summary");
  assert.deepEqual(plain(summary), { rows: 6, columns: 5 });
});

test("local ask fallback is only used outside API mode and is normalized", async () => {
  const Alumni = loadApiWithWindow({
    config: { useApi: false, apiBase: "" },
    fetchImpl: async () => {
      throw new Error("fetch should not be called");
    },
    windowExtras: {
      ask: () => ({ op: null, kind: "help", text: "Local demo answer." }),
    },
  });

  const msg = await Alumni.ask({ dataset_id: "local" }, "Question");

  assert.equal(msg.kind, "structured");
  assert.equal(msg.answer.summary, "Local demo answer.");
});

test("ask renders non-tech industry people results generically with clean stats", async () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async () => ({
      ok: true,
      json: async () => ({
        answer: {
          title: "Consulting alumni",
          summary: "Found consulting alumni.",
          blocks: [
            {
              type: "metrics",
              items: [
                { label: "Total keyword hits", value: "44" },
                { label: "Display limit", value: "100" },
              ],
            },
            {
              type: "table",
              columns: [
                "Nickname",
                "First Name",
                "LastName",
                "Occupation",
                "Employer",
                "Match Reason",
                "internal_reason",
                "LinkedinURL",
              ],
              rows: [
                ["Pat", "Pat", "Partner", "Partner", "McKinsey", "Matched EMPLOYER", "internal", "linkedin.com/in/pat"],
                ["Sam", "Sam", "Strategy", "Strategy Consultant", "Family Business", "Matched OCCUPATION", "internal", ""],
              ],
            },
          ],
          followups: [],
        },
        operation: { type: "contains_any" },
        result: {
          intent: "people_filter",
          entity: "alumni",
          filter_type: "industry",
          industry: "consulting",
          answer_label: "Alumni matching criteria",
          criteria_label: "working in consulting",
          total_dataset_rows: 300,
          total_keyword_hits: 44,
          total_matches: 12,
          displayed_count: 2,
          display_limit: 100,
          uncertain_count: 3,
          visible_columns: ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL"],
        },
      }),
    }),
  });

  const msg = await Alumni.ask({ dataset_id: "dataset-1" }, "Which alumni work in consulting?");
  const metrics = msg.answer.blocks.find(block => block.type === "metrics");
  const table = msg.answer.blocks.find(block => block.type === "table");
  const rendered = JSON.stringify(msg.answer);

  // Main stat is answer_label + total_matches; uncertain is separate; limits/hits are not the answer.
  assert.deepEqual(plain(metrics.items), [
    { label: "Alumni matching criteria", value: "12" },
    { label: "Showing", value: "2" },
    { label: "Uncertain not counted", value: "3" },
  ]);
  assert.equal(rendered.includes("Display limit"), false);
  assert.equal(rendered.includes("Total keyword hits"), false);
  assert.equal(rendered.includes("Analysis Plan Error"), false);

  // visible_columns drive the table; LinkedIn URL is last; debug columns are hidden.
  assert.deepEqual(plain(table.columns), ["First Name", "Last Name", "Occupation", "Employer", "LinkedIn URL"]);
  assert.equal(table.columns[table.columns.length - 1], "LinkedIn URL");
  assert.equal(rendered.includes("Match Reason"), false);
  assert.equal(rendered.includes("internal_reason"), false);
  assert.equal(rendered.includes("Nickname"), false);
  assert.deepEqual(plain(table.rows), [
    ["Pat", "Partner", "Partner", "McKinsey", "linkedin.com/in/pat"],
    ["Sam", "Strategy", "Strategy Consultant", "Family Business", ""],
  ]);
});

test("history() fetches the history list and normalizes stored response payloads", async () => {
  const calls = [];
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return {
        ok: true,
        json: async () => ({
          history: [
            {
              history_id: "history-1",
              dataset_id: "dataset-1",
              dataset_filename: "alumni.csv",
              title: "Consulting alumni",
              question: "Who works in consulting?",
              answer_text: "12 alumni match consulting.",
              status: "success",
              created_at: "2026-06-13T10:00:00",
              response_payload: {
                answer: {
                  title: "Consulting alumni",
                  summary: "12 alumni match consulting.",
                  blocks: [
                    { type: "metrics", items: [{ label: "Alumni matching criteria", value: "12" }] },
                    {
                      type: "table",
                      columns: ["First Name", "LastName", "Occupation", "Employer", "internal_reason"],
                      rows: [["Ada", "Lovelace", "Consultant", "McKinsey", "debug"]],
                    },
                  ],
                  followups: [],
                },
                result: {
                  intent: "people_filter",
                  entity: "alumni",
                  total_matches: 12,
                  visible_columns: ["First Name", "Last Name", "Occupation", "Employer"],
                },
              },
            },
          ],
        }),
      };
    },
  });

  const history = await Alumni.history();
  const table = history[0].response_payload.answer.blocks.find(block => block.type === "table");

  assert.equal(calls[0].url, "/api/history");
  assert.equal(history[0].history_id, "history-1");
  assert.equal(history[0].title, "Consulting alumni");
  assert.deepEqual(plain(table.columns), ["First Name", "Last Name", "Occupation", "Employer"]);
  assert.deepEqual(plain(table.rows), [["Ada", "Lovelace", "Consultant", "McKinsey"]]);
});

test("history() rejects with the backend error message when the list fails", async () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async () => ({
      ok: false,
      status: 500,
      json: async () => ({ error: "History registry is invalid JSON." }),
    }),
  });

  await assert.rejects(
    () => Alumni.history(),
    /History registry is invalid JSON\./
  );
});

test("history item helpers use the expected backend routes", async () => {
  const calls = [];
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "http://localhost:5000" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return {
        ok: true,
        json: async () => ({
          history_id: "history 1",
          dataset_id: "dataset-1",
          dataset_filename: "alumni.csv",
          question: "Question?",
          answer_text: "Answer.",
          response_payload: { answer: { summary: "Answer.", blocks: [], followups: [] } },
        }),
      };
    },
  });

  await Alumni.historyItem("history 1");
  await Alumni.createHistoryItem({ dataset_id: "dataset-1", question: "Question?", answer_text: "Answer." });
  await Alumni.deleteHistoryItem("history 1");
  await Alumni.clearHistory();

  assert.equal(calls[0].url, "http://localhost:5000/api/history/history%201");
  assert.equal(calls[1].url, "http://localhost:5000/api/history");
  assert.equal(calls[1].options.method, "POST");
  assert.deepEqual(JSON.parse(calls[1].options.body), { dataset_id: "dataset-1", question: "Question?", answer_text: "Answer." });
  assert.equal(calls[2].url, "http://localhost:5000/api/history/history%201");
  assert.equal(calls[2].options.method, "DELETE");
  assert.equal(calls[3].url, "http://localhost:5000/api/history");
  assert.equal(calls[3].options.method, "DELETE");
});

test("history mutation helpers surface backend validation errors", async () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => ({
      ok: false,
      status: options && options.method === "DELETE" ? 404 : 400,
      json: async () => ({ error: options && options.method === "DELETE" ? "History item not found." : "question must not be empty." }),
    }),
  });

  await assert.rejects(
    () => Alumni.createHistoryItem({ dataset_id: "dataset-1", question: "", answer_text: "Answer." }),
    /question must not be empty/
  );
  await assert.rejects(
    () => Alumni.deleteHistoryItem("missing-history"),
    /History item not found/
  );
  await assert.rejects(
    () => Alumni.clearHistory(),
    /History item not found/
  );
});

test("saveHistoryAsInsight sends the history snapshot to the saved insight API", async () => {
  const calls = [];
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return {
        ok: true,
        json: async () => ({
          insight_id: "insight-1",
          dataset_id: "dataset-1",
          dataset_name_snapshot: "alumni.csv",
          title: "Consulting alumni",
          question: "Who works in consulting?",
          answer: "12 alumni match consulting.",
          response_payload: { answer: { summary: "12 alumni match consulting.", blocks: [], followups: [] } },
        }),
      };
    },
  });

  const responsePayload = {
    answer: {
      title: "Consulting alumni",
      summary: "12 alumni match consulting.",
      blocks: [{ type: "metrics", items: [{ label: "Alumni matching criteria", value: "12" }] }],
      followups: [],
    },
    result: { intent: "people_filter", entity: "alumni", total_matches: 12 },
  };
  await Alumni.saveHistoryAsInsight({
    history_id: "history-1",
    dataset_id: "dataset-1",
    dataset_filename: "alumni.csv",
    title: "Consulting alumni",
    question: "Who works in consulting?",
    answer_text: "12 alumni match consulting.",
    response_payload: responsePayload,
  });

  assert.equal(calls[0].url, "/api/insights");
  assert.equal(calls[0].options.method, "POST");
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    dataset_id: "dataset-1",
    title: "Consulting alumni",
    question: "Who works in consulting?",
    answer: "12 alumni match consulting.",
    response_payload: {
      ...responsePayload,
      answer_text: "12 alumni match consulting.",
      operation: null,
    },
  });
});

test("saveHistoryAsInsight rejects unusable history items before fetch", async () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async () => {
      throw new Error("fetch should not be called");
    },
  });

  await assert.rejects(
    () => Alumni.saveHistoryAsInsight({ question: "Question?", answer_text: "Answer." }),
    /Cannot save this history item/
  );
});

test("history helpers are inert outside API mode", async () => {
  const Alumni = loadApi({
    config: { useApi: false, apiBase: "" },
    fetchImpl: async () => {
      throw new Error("fetch should not be called");
    },
  });

  assert.deepEqual(plain(await Alumni.history()), []);
  await assert.rejects(() => Alumni.historyItem("id"), /History requires API mode/);
  await assert.rejects(() => Alumni.createHistoryItem({}), /History requires API mode/);
  await assert.rejects(() => Alumni.deleteHistoryItem("id"), /History requires API mode/);
  await assert.rejects(() => Alumni.clearHistory(), /History requires API mode/);
});

test("normalizeHistoryEntry tolerates missing fields and malformed payloads", () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async () => {
      throw new Error("fetch should not be called");
    },
  });

  assert.equal(Alumni._test.normalizeHistoryEntry(null), null);
  assert.equal(Alumni._test.normalizeHistoryEntry({ question: "No id" }), null);

  const item = Alumni._test.normalizeHistoryEntry({
    id: "legacy-id",
    question: "  What changed? ",
    answer: "Legacy answer",
    response_payload: ["bad"],
    dataset_status: "unexpected",
  });

  assert.deepEqual(plain(item), {
    id: "legacy-id",
    history_id: "legacy-id",
    dataset_id: "",
    dataset_filename: "Unknown dataset",
    dataset_status: "ready",
    title: "What changed",
    question: "What changed?",
    answer_text: "Legacy answer",
    answer: "Legacy answer",
    response_payload: null,
    status: "success",
    created_at: "",
    updated_at: "",
    metadata: {},
  });
});

test("datasets() fetches the library list and normalizes entries", async () => {
  const calls = [];
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return {
        ok: true,
        json: async () => ({
          count: 2,
          datasets: [
            {
              dataset_id: "new-id",
              display_name: "Renamed Alumni",
              original_filename: "alumni.xlsx",
              stored_filename: "new-id_alumni.xlsx",
              uploaded_at: "2026-06-11T10:00:00",
              row_count: 332,
              column_count: 12,
              columns: ["First Name", "Last Name"],
              file_type: "xlsx",
              status: "ready",
            },
            {
              dataset_id: "old-id",
              display_name: "",
              original_filename: "older.csv",
              uploaded_at: "2026-06-10T09:00:00",
              row_count: 5,
              column_count: 2,
              columns: ["A", "B"],
              file_type: "csv",
              status: "missing",
            },
          ],
        }),
      };
    },
  });

  const list = await Alumni.datasets();

  assert.equal(calls[0].url, "/api/datasets");
  assert.equal(list.length, 2);
  assert.equal(list[0].dataset_id, "new-id");
  assert.equal(list[0].display_name, "Renamed Alumni");
  assert.equal(list[0].status, "ready");
  // display_name falls back to original filename; missing status survives.
  assert.equal(list[1].display_name, "older.csv");
  assert.equal(list[1].status, "missing");
});

test("datasets() rejects with the backend error message when the list fails", async () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async () => ({
      ok: false,
      status: 500,
      json: async () => ({ error: "Dataset registry is invalid JSON." }),
    }),
  });

  await assert.rejects(() => Alumni.datasets(), /Dataset registry is invalid JSON\./);
});

test("datasets() resolves empty without fetch outside API mode", async () => {
  const Alumni = loadApi({
    config: { useApi: false, apiBase: "" },
    fetchImpl: async () => {
      throw new Error("fetch should not be called");
    },
  });

  assert.deepEqual(plain(await Alumni.datasets()), []);
});

test("renameDataset PATCHes the dataset and returns updated metadata", async () => {
  const calls = [];
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return {
        ok: true,
        json: async () => ({
          dataset_id: "abc 123",
          display_name: "Class of 2026",
          original_filename: "alumni.csv",
          status: "ready",
        }),
      };
    },
  });

  const updated = await Alumni.renameDataset("abc 123", "Class of 2026");

  assert.equal(calls[0].url, "/api/datasets/abc%20123");
  assert.equal(calls[0].options.method, "PATCH");
  assert.deepEqual(JSON.parse(calls[0].options.body), { display_name: "Class of 2026" });
  assert.equal(updated.display_name, "Class of 2026");
});

test("renameDataset rejects with backend validation errors", async () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async () => ({
      ok: false,
      status: 400,
      json: async () => ({ error: "display_name must not be empty." }),
    }),
  });

  await assert.rejects(() => Alumni.renameDataset("abc", "  "), /display_name must not be empty\./);
});

test("deleteDataset DELETEs the dataset and surfaces clean 404 errors", async () => {
  const calls = [];
  let respondNotFound = false;
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (respondNotFound) {
        return { ok: false, status: 404, json: async () => ({ error: "Dataset not found." }) };
      }
      return { ok: true, json: async () => ({ deleted: true, dataset_id: "gone-id" }) };
    },
  });

  const result = await Alumni.deleteDataset("gone-id");
  assert.equal(calls[0].url, "/api/datasets/gone-id");
  assert.equal(calls[0].options.method, "DELETE");
  assert.deepEqual(plain(result), { deleted: true, dataset_id: "gone-id" });

  respondNotFound = true;
  await assert.rejects(() => Alumni.deleteDataset("gone-id"), /Dataset not found\./);
});

test("normalizeDatasetEntry tolerates missing metadata fields", () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async () => {
      throw new Error("fetch should not be called");
    },
  });
  const normalize = Alumni._test.normalizeDatasetEntry;

  assert.equal(normalize(null), null);
  assert.equal(normalize({}), null);
  const minimal = normalize({ dataset_id: "x" });
  assert.equal(minimal.display_name, "Untitled dataset");
  assert.equal(minimal.status, "ready");
  assert.deepEqual(plain(minimal.columns), []);
  assert.equal(minimal.row_count, null);
});

/* ---- saved insights (manual snapshots, not history) ---- */

test("insights GETs the list, supports dataset filtering, and normalizes entries", async () => {
  const calls = [];
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return {
        ok: true,
        json: async () => ({
          insights: [
            {
              insight_id: "ins-1",
              dataset_id: "ds-1",
              dataset_name_snapshot: "alumni.csv",
              dataset_status: "ready",
              title: "Tech alumni",
              question: "Which alumni work in tech?",
              answer: "2 alumni work in tech.",
              created_at: "2026-06-12T10:00:00",
              updated_at: "2026-06-12T10:00:00",
              tags: ["tech"],
            },
            { insight_id: "" },
          ],
          count: 2,
        }),
      };
    },
  });

  const all = await Alumni.insights();
  assert.equal(calls[0].url, "/api/insights");
  assert.equal(all.length, 1);
  assert.equal(all[0].insight_id, "ins-1");
  assert.equal(all[0].dataset_status, "ready");
  assert.equal(all[0].answer, "2 alumni work in tech.");

  await Alumni.insights("ds 1");
  assert.equal(calls[1].url, "/api/insights?dataset_id=ds%201");
});

test("saveInsight POSTs dataset_id, question, answer, and title", async () => {
  const calls = [];
  const responsePayload = {
    answer: {
      summary: "2 alumni work in tech.",
      blocks: [
        { type: "metrics", items: [{ label: "Alumni matching criteria", value: "2" }] },
        { type: "table", columns: ["First Name", "Employer"], rows: [["Ada", "Google"]] },
      ],
      followups: [],
    },
    result: { intent: "people_filter", entity: "alumni", total_matches: 2 },
  };
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return {
        ok: true,
        status: 201,
        json: async () => ({
          insight_id: "ins-9",
          dataset_id: "ds-1",
          dataset_name_snapshot: "alumni.csv",
          title: "Tech alumni",
          question: "Which alumni work in tech?",
          answer: "2 alumni work in tech.",
          response_payload: responsePayload,
        }),
      };
    },
  });

  const created = await Alumni.saveInsight({
    dataset_id: "ds-1",
    title: "Tech alumni",
    question: "Which alumni work in tech?",
    answer: "2 alumni work in tech.",
    response_payload: responsePayload,
  });

  assert.equal(calls[0].url, "/api/insights");
  assert.equal(calls[0].options.method, "POST");
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    dataset_id: "ds-1",
    question: "Which alumni work in tech?",
    answer: "2 alumni work in tech.",
    title: "Tech alumni",
    response_payload: responsePayload,
  });
  assert.equal(created.insight_id, "ins-9");
  assert.deepEqual(plain(created.response_payload.answer.blocks[1].rows), [["Ada", "Google"]]);
});

test("saveInsight fills a default title from the question when title is missing", async () => {
  const calls = [];
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return { ok: true, status: 201, json: async () => ({ insight_id: "ins-2" }) };
    },
  });

  await Alumni.saveInsight({ dataset_id: "ds-1", question: "Which alumni work in consulting?", answer: "5 direct matches." });
  assert.equal(JSON.parse(calls[0].options.body).title, "Which alumni work in consulting");
});

test("saveInsight refuses to save without dataset, question, or answer", async () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async () => {
      throw new Error("fetch should not be called");
    },
  });

  await assert.rejects(() => Alumni.saveInsight({ question: "Q", answer: "A" }), /active dataset/);
  await assert.rejects(() => Alumni.saveInsight({ dataset_id: "ds", answer: "A" }), /original question/);
  await assert.rejects(() => Alumni.saveInsight({ dataset_id: "ds", question: "Q", answer: " " }), /completed answer/);
});

test("renameInsight PATCHes the title and surfaces errors", async () => {
  const calls = [];
  let fail = false;
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (fail) return { ok: false, status: 400, json: async () => ({ error: "title must not be empty." }) };
      return { ok: true, json: async () => ({ insight_id: "ins 1", title: "Renamed" }) };
    },
  });

  const updated = await Alumni.renameInsight("ins 1", "Renamed");
  assert.equal(calls[0].url, "/api/insights/ins%201");
  assert.equal(calls[0].options.method, "PATCH");
  assert.deepEqual(JSON.parse(calls[0].options.body), { title: "Renamed" });
  assert.equal(updated.title, "Renamed");

  fail = true;
  await assert.rejects(() => Alumni.renameInsight("ins 1", "  "), /title must not be empty\./);
});

test("deleteInsight DELETEs and surfaces clean 404 errors", async () => {
  const calls = [];
  let respondNotFound = false;
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (respondNotFound) {
        return { ok: false, status: 404, json: async () => ({ error: "Saved insight not found." }) };
      }
      return { ok: true, json: async () => ({ deleted: true, insight_id: "ins-1" }) };
    },
  });

  const result = await Alumni.deleteInsight("ins-1");
  assert.equal(calls[0].url, "/api/insights/ins-1");
  assert.equal(calls[0].options.method, "DELETE");
  assert.deepEqual(plain(result), { deleted: true, insight_id: "ins-1" });

  respondNotFound = true;
  await assert.rejects(() => Alumni.deleteInsight("ins-1"), /Saved insight not found\./);
});

test("insight fetches a single saved insight with full question and answer", async () => {
  const calls = [];
  const responsePayload = {
    answer: {
      title: "Tech alumni",
      summary: "Full saved answer text.",
      blocks: [
        {
          type: "table",
          columns: ["First Name", "Last Name", "Employer"],
          rows: [["Ada", "Lovelace", "Google"]],
        },
      ],
      followups: [],
    },
    result: { intent: "people_filter", entity: "alumni", total_matches: 1 },
  };
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return {
        ok: true,
        json: async () => ({
          insight_id: "ins-1",
          question: "Which alumni work in tech?",
          answer: "Full saved answer text.",
          response_payload: responsePayload,
          dataset_status: "deleted",
        }),
      };
    },
  });

  const insight = await Alumni.insight("ins-1");
  assert.equal(calls[0].url, "/api/insights/ins-1");
  assert.equal(insight.question, "Which alumni work in tech?");
  assert.equal(insight.answer, "Full saved answer text.");
  assert.equal(insight.dataset_status, "deleted");
  const table = insight.response_payload.answer.blocks.find(block => block.type === "table");
  assert.deepEqual(plain(table.columns), ["First Name", "Last Name", "Employer"]);
  assert.deepEqual(plain(table.rows), [["Ada", "Lovelace", "Google"]]);
});

test("insights are API-mode only; demo mode resolves empty and rejects writes", async () => {
  const Alumni = loadApi({
    config: { useApi: false },
    fetchImpl: async () => {
      throw new Error("fetch should not be called in demo mode");
    },
  });

  assert.deepEqual(plain(await Alumni.insights()), []);
  await assert.rejects(() => Alumni.saveInsight({ dataset_id: "x", question: "q", answer: "a" }), /API mode/);
  await assert.rejects(() => Alumni.renameInsight("x", "t"), /API mode/);
  await assert.rejects(() => Alumni.deleteInsight("x"), /API mode/);
});

test("normalizeInsightEntry tolerates missing fields and bad dataset_status", () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async () => {
      throw new Error("fetch should not be called");
    },
  });
  const normalize = Alumni._test.normalizeInsightEntry;

  assert.equal(normalize(null), null);
  assert.equal(normalize({}), null);
  const minimal = normalize({ insight_id: "x", question: "Which alumni work in law?" });
  assert.equal(minimal.dataset_name_snapshot, "Unknown dataset");
  assert.equal(minimal.dataset_status, "ready");
  assert.equal(minimal.title, "Which alumni work in law");
  assert.equal(minimal.response_payload, null);
  assert.deepEqual(plain(minimal.tags), []);
  assert.deepEqual(plain(minimal.metadata), {});
});

test("defaultInsightTitle shortens long questions and strips punctuation", () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async () => {
      throw new Error("fetch should not be called");
    },
  });
  const title = Alumni._test.defaultInsightTitle;

  assert.equal(title("Which alumni work in tech?"), "Which alumni work in tech");
  assert.equal(title("   "), "Saved insight");
  const long = title("Which alumni are working in consulting, advisory, or professional services firms across all graduating classes?");
  assert.ok(long.length <= 82);
  assert.ok(long.endsWith("…"));
});

test("insightTextFromAnswer flattens summary, markdown, and metrics into snapshot text", () => {
  const Alumni = loadApi({
    config: { useApi: true, apiBase: "" },
    fetchImpl: async () => {
      throw new Error("fetch should not be called");
    },
  });
  const flatten = Alumni._test.insightTextFromAnswer;

  const text = flatten(
    {
      summary: "5 direct consulting matches.",
      blocks: [
        { type: "metrics", items: [{ label: "Alumni matching criteria", value: "5" }] },
        { type: "table", columns: ["First Name", "Employer"], rows: [["A", "EY"], ["B", "KPMG"]], caption: "Searched columns: Occupation, Employer" },
        { type: "markdown", content: "Assumptions: consulting taxonomy used." },
      ],
    },
    "fallback"
  );
  assert.ok(text.startsWith("5 direct consulting matches."));
  assert.ok(text.includes("Alumni matching criteria: 5"));
  assert.ok(text.includes("Table: 2 rows (First Name, Employer)"));
  assert.ok(text.includes("Assumptions: consulting taxonomy used."));

  assert.equal(flatten(null, "plain fallback"), "plain fallback");
});
