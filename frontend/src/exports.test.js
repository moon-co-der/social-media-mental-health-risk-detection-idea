import test from 'node:test';
import assert from 'node:assert/strict';
import Papa from 'papaparse';
import { rowPredictionsCsv, runMetadataJson, summaryReport, summaryReportJson, userSummaryCsv } from './exports.js';
import { E2_CLASSES, EMOTION_LABELS } from './analytics.js';

test('row CSV exports all E2 and emotion values, quotes Unicode, and excludes private fields', () => {
  const probabilities = Object.fromEntries(E2_CLASSES.map((label, i) => [label, i / 10]));
  const emotions = Object.fromEntries(EMOTION_LABELS.map((label, i) => [label, i / 100]));
  const value = rowPredictionsCsv([{ row_number: 1, user_id: 'user_日本語,"x"', timestamp: '2025-01-01T00:00:00Z',
    predicted_class: 'supportive', e2_class_probabilities: probabilities, e2_score_uncalibrated: 0.2,
    goemotions_scores: emotions, raw_text: 'private post', original_username: 'name', note: 'comma, quote " and 日本語' }]);
  const parsed = Papa.parse(value, { header: true }).data[0];
  assert.equal(parsed.user_id, 'user_日本語,"x"');
  assert.equal(parsed.e2_probability_supportive, '0.4');
  assert.equal(parsed.goemotions_joy, '0.17');
  assert.equal(value.includes('private post'), false);
  assert.equal(value.includes('original_username'), false);
  assert.equal(value.includes('日本語'), true);
  assert.equal(value.includes('comma'), false);
  assert.equal(Object.keys(parsed).length, 9 + E2_CLASSES.length + EMOTION_LABELS.length);
  assert.equal(Object.hasOwn(parsed, 'raw_text'), false);
});

test('user summary CSV serializes allowlisted fields and emotion aggregates', () => {
  const parsed = Papa.parse(userSummaryCsv([{ user_id: 'user_abc', anchor: '2025-01-01', eligible: true, recent_posts: 5, baseline_posts: 15,
    recent_emotion_means: { joy: 0.4 }, baseline_emotion_means: { joy: 0.2 }, emotion_differences_recent_minus_baseline: { joy: 0.2 }, raw_text: 'private post', author: 'private author' }]), { header: true }).data[0];
  assert.equal(parsed.recent_joy, '0.4');
  assert.equal(parsed.difference_joy, '0.2');
  assert.equal(parsed.recent_window, '[0, 90) days before anchor');
  assert.equal(Object.hasOwn(parsed, 'raw_text'), false);
  assert.equal(Object.hasOwn(parsed, 'author'), false);
  assert.equal(userSummaryCsv([]).startsWith('user_id,anchor'), true);
});

test('summary report uses actual API values and has no raw text or original identifiers', () => {
  const report = summaryReport({ run: { completed_at: '2025-01-01', warnings: ['uncalibrated'], raw_text: 'no' },
    summary: { rows_received: 2, nonempty_text_rows: 1, empty_text_rows: 1, unique_users: 0, temporal_eligible_users: 0, parsed_timestamp_min: null, parsed_timestamp_max: null },
    predictions: [{ row_number: 1, predicted_class: 'attempt', e2_score_uncalibrated: 0.4, e2_class_probabilities: { attempt: 0.8, behavior: 0.1 }, goemotions_scores: { joy: 0.5 }, raw_text: 'private' }], users: [] });
  assert.equal(report.uploaded_rows, 2);
  assert.equal(report.processed_rows, 1);
  assert.equal(report.e2_predicted_class_distribution[0].count, 1);
  assert.equal(report.goemotions_mean_sigmoid_scores.find(x => x.label === 'joy').mean, 0.5);
  assert.equal(JSON.stringify(report).includes('private'), false);
  assert.equal(JSON.parse(summaryReportJson({ predictions: [] })).processed_rows, 0);
});

test('metadata export is valid JSON with only run and summary fields', () => {
  const value = runMetadataJson({ completed_at: '2025-01-01T00:00:00Z', raw_text: 'private post' }, { rows_received: 4, raw_user: 'private user' });
  assert.deepEqual(JSON.parse(value), { run: { completed_at: '2025-01-01T00:00:00Z' }, summary: { rows_received: 4 } });
  assert.equal(value.includes('private'), false);
});
