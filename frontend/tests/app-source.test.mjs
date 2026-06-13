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
