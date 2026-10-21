import test from 'node:test';
import assert from 'node:assert/strict';
import { E2_CLASSES, adaptiveDistribution, describeValues, deviationSummary, e2Summary, emotionMeans, featureAssociations, filteredPredictions, groupScoreSeries, histogram, inspectUploadedRows, perUserScoreMeans, temporalTimelinePoints, userTemporalComparison } from './analytics.js';

test('E2 class distribution includes zero classes and correct percentages', () => {
  const summary = e2Summary([{ predicted_class: 'attempt', e2_score_uncalibrated: 0.2 }, { predicted_class: 'attempt', e2_score_uncalibrated: 0.6 }, { predicted_class: 'supportive', e2_score_uncalibrated: 0.4 }]);
  assert.deepEqual(summary.classes.map(x => x.label), E2_CLASSES);
  assert.equal(summary.classes.find(x => x.label === 'ideation').count, 0);
  assert.ok(Math.abs(summary.classes.find(x => x.label === 'attempt').percentage - 200 / 3) < 1e-10);
  assert.ok(Math.abs(summary.score.mean - 0.4) < 1e-12);
  assert.equal(summary.score.median, 0.4);
});

test('empty analysis and histograms do not fabricate zero statistics', () => {
  const summary = e2Summary([]);
  assert.equal(summary.classes.every(x => x.count === 0 && x.percentage === null), true);
  assert.deepEqual(summary.score, { count: 0, mean: null, median: null, min: null, max: null });
  assert.equal(histogram([]).every(x => x.count === 0), true);
});

test('emotion means sort descending and break ties alphabetically', () => {
  const rows = emotionMeans([{ goemotions_scores: { joy: 0.8, anger: 0.8, admiration: 0.1 } }]);
  assert.deepEqual(rows.slice(0, 2).map(x => x.label), ['anger', 'joy']);
  assert.equal(rows.find(x => x.label === 'neutral').mean, null);
});

test('temporal comparison uses provided eligibility and suppresses ineligible differences', () => {
  const user = { eligible: false, recent_posts: 3, baseline_posts: 16, recent_e2_mean: 0.3, baseline_e2_mean: 0.2, e2_mean_difference_recent_minus_baseline: 0.1, recent_emotion_means: { joy: 0.4 }, baseline_emotion_means: { joy: 0.2 }, emotion_differences_recent_minus_baseline: { joy: 0.2 } };
  const comparison = userTemporalComparison(user);
  assert.equal(comparison.recentPosts, 3);
  assert.equal(comparison.emotions.find(x => x.label === 'joy').recent, 0.4);
  assert.equal(comparison.emotions.find(x => x.label === 'joy').difference, null);
  assert.equal(userTemporalComparison({ ...user, eligible: true }).emotions.find(x => x.label === 'joy').difference, 0.2);
});

test('timeline measure and selected user filter use actual per-post fields', () => {
  const predictions = [
    { user_id: 'u1', timestamp: '2025-01-02T00:00:00Z', e2_score_uncalibrated: 0.3, goemotions_scores: { joy: 0.6 } },
    { user_id: 'u2', timestamp: '2025-01-02T00:00:00Z', e2_score_uncalibrated: 0.9, goemotions_scores: { joy: 0.9 } },
  ];
  assert.equal(temporalTimelinePoints(predictions, 'u1', '2025-01-02T00:00:00Z', 'joy')[0].score, 0.6);
  assert.equal(temporalTimelinePoints(predictions, 'u1', '2025-01-02T00:00:00Z', 'e2_score_uncalibrated')[0].score, 0.3);
  assert.equal(filteredPredictions(predictions, { userId: 'u1' }).length, 1);
});

test('date-filtered exports exclude posts without dates and honor inclusive dates', () => {
  const rows = [{ timestamp: '2025-01-01T12:00:00Z' }, { timestamp: '2025-01-02T00:00:00Z' }, { timestamp: null }];
  assert.deepEqual(filteredPredictions(rows, { start: '2025-01-01', end: '2025-01-01' }), [rows[0]]);
});

test('adaptive distribution resolves narrow scores and supports a 0–2 E2 scale', () => {
  const narrow = adaptiveDistribution([0.0339, 0.0375, 0.0443]);
  assert.ok(narrow.domain[1] - narrow.domain[0] < 0.05);
  assert.equal(narrow.bins.reduce((sum, bin) => sum + bin.count, 0), 3);
  const full = adaptiveDistribution([0.8, 1.4], true);
  assert.deepEqual(full.domain, [0, 2]);
  assert.equal(describeValues([0.01, 0.01]).nearConstant, true);
});

test('deviation flags use transparent 2 SD or Tukey fences', () => {
  const result = deviationSummary([0.1, 0.11, 0.09, 0.8], 0.1);
  assert.equal(result.rows.length, 4);
  assert.equal(result.rows[3].outlier, true);
});

test('upload inspection reports text issues and strips original user identifiers', () => {
  const result = inspectUploadedRows([
    { body: 'Hello there, friend!', who: 'raw-name', when: '2025-01-01T00:00:00Z' },
    { body: 'hello THERE friend', who: 'raw-name', when: '2025-01-08T00:00:00Z' },
    { body: '  ', who: '', when: '' },
  ], { textColumn: 'body', userColumn: 'who', timestampColumn: 'when' });
  assert.equal(result.quality.empty_text_rows, 1);
  assert.equal(result.quality.normalized_near_duplicate_text_rows, 1);
  assert.equal(result.quality.mapped_columns.timestamp, true);
  assert.equal(JSON.stringify(result.features).includes('raw-name'), false);
  assert.equal(result.features[1].posting_gap_days, 7);
});

test('weekly series, per-user means, and feature correlations use available cohort rows', () => {
  const predictions = [
    { row_number: 1, user_id: 'u1', timestamp: '2025-01-06T00:00:00Z', e2_score_uncalibrated: 0.2 },
    { row_number: 2, user_id: 'u1', timestamp: '2025-01-07T00:00:00Z', e2_score_uncalibrated: 0.4 },
    { row_number: 3, user_id: 'u2', timestamp: '2025-01-15T00:00:00Z', e2_score_uncalibrated: 0.8 },
  ];
  assert.equal(groupScoreSeries(predictions, 'e2_score_uncalibrated', 'week').length, 2);
  assert.equal(perUserScoreMeans(predictions, 'e2_score_uncalibrated')[0].n, 1);
  const associations = featureAssociations([{ row_number: 1, word_count: 1 }, { row_number: 2, word_count: 2 }, { row_number: 3, word_count: 3 }], predictions, 'e2_score_uncalibrated');
  assert.equal(associations.find(row => row.label === 'Post length (words)').n, 3);
});
