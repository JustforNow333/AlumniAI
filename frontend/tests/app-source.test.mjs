import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync("frontend/app.jsx", "utf8");

test("saved insight full response uses the shared dataset response renderer", () => {
  assert.match(source, /function DatasetResponseView/);
  assert.match(source, /<DatasetResponseView response=\{msg\.response_payload\}/);
  assert.match(source, /<DatasetResponseView response=\{insight\.response_payload\}/);
});

test("saved insight detail places an Open full response toggle under Saved answer", () => {
  const savedAnswerIndex = source.indexOf('<span className="kicker">Saved answer</span>');
  const buttonIndex = source.indexOf('data-action="open-full-response"');
  const fullResponseIndex = source.indexOf('data-insight-full-response="true"');

  assert.notEqual(savedAnswerIndex, -1);
  assert.notEqual(buttonIndex, -1);
  assert.notEqual(fullResponseIndex, -1);
  assert.ok(savedAnswerIndex < buttonIndex);
  assert.ok(buttonIndex < fullResponseIndex);
  assert.match(source, /setShowFullResponse\(open => !open\)/);
  assert.match(source, /Open full response/);
});

test("history page is a live sidebar view with empty and item states", () => {
  assert.match(source, /\["history", "History", "history"\]/);
  assert.match(source, /function HistoryLibrary/);
  assert.match(source, />History<\/span>/);
  assert.match(source, /Recent dataset analyses/);
  assert.match(source, /No history yet/);
  assert.match(source, /Ask a question about a dataset and it will appear here\./);
  assert.match(source, /data-history-id=\{item\.history_id\}/);
});

test("history detail opens the stored response without an ask rerun", () => {
  const detailSource = source.slice(
    source.indexOf("function HistoryDetail"),
    source.indexOf("function HistoryLibrary")
  );
  assert.match(source, /function HistoryDetail/);
  assert.match(source, /data-history-full-response="true"/);
  assert.match(source, /<DatasetResponseView response=\{item\.response_payload\} fallbackText=\{item\.answer_text\} \/>/);
  assert.doesNotMatch(detailSource, /window\.Alumni\.ask/);
});

test("history delete and save-as-insight actions are wired", () => {
  assert.match(source, /onDeleteHistory=\{deleteHistory\}/);
  assert.match(source, /onSaveHistoryAsInsight=\{saveHistoryAsInsight\}/);
  assert.match(source, /window\.Alumni\.deleteHistoryItem\(item\.history_id\)/);
  assert.match(source, /window\.Alumni\.clearHistory\(\)/);
  assert.match(source, /window\.Alumni\.saveHistoryAsInsight\(item\)/);
  assert.match(source, /Save as insight/);
});
