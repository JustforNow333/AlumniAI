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
