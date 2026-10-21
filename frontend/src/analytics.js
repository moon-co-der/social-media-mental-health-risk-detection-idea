export const E2_CLASSES = ['attempt', 'behavior', 'ideation', 'indicator', 'supportive', 'uninformative'];
export const EMOTION_LABELS = ['admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral'];

export function e2Summary(predictions = []) {
  const total = predictions.length;
  const classes = E2_CLASSES.map(label => {
    const count = predictions.filter(p => p.predicted_class === label).length;
    return { label, count, percentage: total ? count / total * 100 : null };
  });
  const scores = predictions.map(p => p.e2_score_uncalibrated).filter(Number.isFinite).sort((a, b) => a - b);
  const mean = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
  const median = scores.length ? (scores[Math.floor((scores.length - 1) / 2)] + scores[Math.ceil((scores.length - 1) / 2)]) / 2 : null;
  return { total, classes, score: { count: scores.length, mean, median, min: scores[0] ?? null, max: scores.at(-1) ?? null } };
}

export function histogram(values = [], bins = 10, min = 0, max = 1) {
  const counts = Array.from({ length: bins }, () => 0);
  const valid = values.filter(Number.isFinite);
  for (const value of valid) {
    const index = Math.min(bins - 1, Math.max(0, Math.floor((value - min) / (max - min) * bins)));
    counts[index] += 1;
  }
  return counts.map((count, i) => ({ from: min + (max - min) * i / bins, to: min + (max - min) * (i + 1) / bins, count }));
}

export function emotionMeans(predictions = []) {
  return EMOTION_LABELS.map(label => {
    const values = predictions.map(p => p.goemotions_scores?.[label]).filter(Number.isFinite);
    return { label, mean: values.length ? values.reduce((sum, n) => sum + n, 0) / values.length : null, count: values.length };
  }).sort((a, b) => (b.mean ?? -Infinity) - (a.mean ?? -Infinity) || a.label.localeCompare(b.label));
}

export function userTemporalComparison(user) {
  if (!user) return null;
  const eligible = Boolean(user.eligible) && user.recent_posts > 0 && user.baseline_posts > 0;
  return {
    eligible: Boolean(user.eligible), recentPosts: user.recent_posts ?? 0, baselinePosts: user.baseline_posts ?? 0,
    e2: { recent: user.recent_e2_mean ?? null, baseline: user.baseline_e2_mean ?? null,
      difference: eligible ? user.e2_mean_difference_recent_minus_baseline ?? null : null },
    emotions: EMOTION_LABELS.map(label => ({ label,
      recent: user.recent_emotion_means?.[label] ?? null,
      baseline: user.baseline_emotion_means?.[label] ?? null,
      difference: eligible ? user.emotion_differences_recent_minus_baseline?.[label] ?? null : null,
    })),
  };
}

export function filteredPredictions(predictions, { userId = '', start = '', end = '' } = {}) {
  const from = start ? Date.parse(start) : -Infinity;
  const to = end ? Date.parse(`${end}T23:59:59.999Z`) : Infinity;
  const hasDateFilter = Boolean(start || end);
  return predictions.filter(p => (!userId || p.user_id === userId) && (!hasDateFilter || (p.timestamp && Date.parse(p.timestamp) >= from && Date.parse(p.timestamp) <= to)));
}

export function temporalTimelinePoints(predictions = [], userId, anchorValue, measure) {
  if (!userId || !anchorValue) return [];
  const anchor = Date.parse(anchorValue);
  return predictions.filter(p => p.user_id === userId && p.timestamp).map(p => {
    const score = scoreValue(p, measure);
    return { ageDays: (anchor - Date.parse(p.timestamp)) / 86400000, date: p.timestamp.slice(0, 10), score };
  }).filter(p => Number.isFinite(p.score)).sort((a, b) => b.ageDays - a.ageDays);
}

export function quantile(values, q) {
  if (!values.length) return null;
  const position = (values.length - 1) * q;
  const lower = Math.floor(position), upper = Math.ceil(position);
  return values[lower] + (values[upper] - values[lower]) * (position - lower);
}

export function describeValues(values = []) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return { n: 0, mean: null, median: null, sd: null, min: null, max: null, q1: null, q3: null, iqr: null, range: null, nearConstant: false };
  const mean = sorted.reduce((sum, value) => sum + value, 0) / sorted.length;
  const sd = Math.sqrt(sorted.reduce((sum, value) => sum + (value - mean) ** 2, 0) / sorted.length);
  const q1 = quantile(sorted, 0.25), median = quantile(sorted, 0.5), q3 = quantile(sorted, 0.75), range = sorted.at(-1) - sorted[0];
  return { n: sorted.length, mean, median, sd, min: sorted[0], max: sorted.at(-1), q1, q3, iqr: q3 - q1, range, nearConstant: sd < 0.01 || range < 0.05 };
}

export function adaptiveDistribution(values = [], fullScale = false, binCount = 12) {
  const valid = values.filter(Number.isFinite);
  const stats = describeValues(valid);
  if (!valid.length) return { stats, domain: null, bins: [], density: [], points: [] };
  const scaleMaximum = Math.max(1, Math.ceil(stats.max));
  const pad = Math.max(stats.range * 0.12, 0.005);
  const domain = fullScale ? [0, scaleMaximum] : [Math.max(0, stats.min - pad), Math.min(scaleMaximum, stats.max + pad)];
  const span = Math.max(Number.EPSILON, domain[1] - domain[0]);
  const bins = Array.from({ length: binCount }, (_, i) => ({ from: domain[0] + span * i / binCount, to: domain[0] + span * (i + 1) / binCount, count: 0 }));
  valid.forEach(value => { const ix = Math.min(binCount - 1, Math.max(0, Math.floor((value - domain[0]) / span * binCount))); bins[ix].count += 1; });
  const bandwidth = Math.max(stats.sd ? 1.06 * stats.sd * valid.length ** -0.2 : 0, span / 60);
  const density = Array.from({ length: 81 }, (_, i) => {
    const x = domain[0] + span * i / 80;
    const d = valid.reduce((sum, value) => sum + Math.exp(-0.5 * ((x - value) / bandwidth) ** 2), 0) / (valid.length * bandwidth * Math.sqrt(2 * Math.PI));
    return { x, countScale: d * valid.length * span / binCount };
  });
  const points = valid.map((value, i) => ({ value, jitter: ((i * 37) % 17) - 8, row: i + 1 }));
  return { stats, domain, bins, density, points };
}

export function deviationSummary(values = [], baseline = null) {
  const stats = describeValues(values);
  if (baseline == null || !stats.n) return { baseline, stats, rows: [] };
  const deviations = values.map((score, index) => ({ index, score, deviation: score - baseline })).filter(x => Number.isFinite(x.score));
  const ds = describeValues(deviations.map(row => row.deviation));
  const lowFence = ds.q1 == null ? null : ds.q1 - 1.5 * ds.iqr;
  const highFence = ds.q3 == null ? null : ds.q3 + 1.5 * ds.iqr;
  return { baseline, stats, rows: deviations.map(row => ({ ...row, outlier: Math.abs(row.deviation) > 2 * (ds.sd ?? 0) || row.deviation < lowFence || row.deviation > highFence })) };
}

export function pearson(xValues = [], yValues = []) {
  const pairs = xValues.map((x, i) => [x, yValues[i]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  const n = pairs.length;
  if (n < 2) return { r: null, n };
  const xm = pairs.reduce((s, p) => s + p[0], 0) / n, ym = pairs.reduce((s, p) => s + p[1], 0) / n;
  const numerator = pairs.reduce((s, p) => s + (p[0] - xm) * (p[1] - ym), 0);
  const denominator = Math.sqrt(pairs.reduce((s, p) => s + (p[0] - xm) ** 2, 0) * pairs.reduce((s, p) => s + (p[1] - ym) ** 2, 0));
  return { r: denominator ? numerator / denominator : null, n };
}

export function groupScoreSeries(predictions = [], measure, period = 'week') {
  const groups = new Map();
  predictions.forEach(row => {
    if (!row.timestamp) return;
    const date = new Date(row.timestamp);
    if (!Number.isFinite(date.getTime())) return;
    let key;
    if (period === 'month') key = `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, '0')}`;
    else {
      const monday = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
      monday.setUTCDate(monday.getUTCDate() - ((monday.getUTCDay() + 6) % 7));
      key = monday.toISOString().slice(0, 10);
    }
    const score = scoreValue(row, measure);
    if (!Number.isFinite(score)) return;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(score);
  });
  return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([date, scores]) => {
    const stats = describeValues(scores);
    const scaleMaximum = measure === 'e2_score_uncalibrated' ? 2 : 1;
    return { date, n: stats.n, mean: stats.mean, sd: stats.sd, low: Math.max(0, stats.mean - stats.sd), high: Math.min(scaleMaximum, stats.mean + stats.sd) };
  });
}

export function perUserScoreMeans(predictions = [], measure) {
  const groups = new Map();
  predictions.forEach(row => {
    if (!row.user_id) return;
    const score = scoreValue(row, measure);
    if (!Number.isFinite(score)) return;
    if (!groups.has(row.user_id)) groups.set(row.user_id, []);
    groups.get(row.user_id).push(score);
  });
  return [...groups.entries()].map(([user_id, values]) => ({ user_id, n: values.length, mean: values.reduce((a, b) => a + b, 0) / values.length })).sort((a, b) => a.n - b.n || a.user_id.localeCompare(b.user_id));
}

export function inspectUploadedRows(rows = [], { textColumn, userColumn = '', timestampColumn = '' } = {}) {
  const exact = new Set(), normalized = new Set();
  let empty = 0, short = 0, exactDuplicates = 0, normalizedDuplicates = 0;
  const features = rows.map((row, index) => {
    const text = String(row[textColumn] ?? ''), trimmed = text.trim();
    if (!trimmed) empty += 1;
    else {
      const words = trimmed.split(/\s+/u);
      if (words.length < 5) short += 1;
      if (exact.has(trimmed)) exactDuplicates += 1; else exact.add(trimmed);
      const norm = trimmed.normalize('NFKC').toLocaleLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');
      if (norm && normalized.has(norm)) normalizedDuplicates += 1; else if (norm) normalized.add(norm);
    }
    const stampRaw = timestampColumn ? String(row[timestampColumn] ?? '').trim() : '';
    const timestamp = stampRaw && !/^[+-]?(?:\d+\.?\d*|\.\d+)$/.test(stampRaw) ? Date.parse(stampRaw) : NaN;
    const pronouns = trimmed.match(/\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves)\b/giu) ?? [];
    const negations = /\b(?:no|not|never|none|cannot|can't|won't|don't|didn't|isn't|aren't|wasn't|weren't|couldn't|wouldn't|shouldn't)\b/i.test(trimmed);
    return { row_number: index + 1, char_length: [...trimmed].length, word_count: trimmed ? trimmed.split(/\s+/u).length : 0,
      first_person_pronouns: pronouns.length, negation_present: Number(negations), timestamp: Number.isFinite(timestamp) ? timestamp : null,
      hour_utc: Number.isFinite(timestamp) ? new Date(timestamp).getUTCHours() + new Date(timestamp).getUTCMinutes() / 60 : null,
      raw_user_key: userColumn ? String(row[userColumn] ?? '').trim() : '' };
  });
  const previousByUser = new Map();
  const sorted = [...features].filter(f => f.raw_user_key && f.timestamp != null).sort((a, b) => a.timestamp - b.timestamp);
  for (const feature of sorted) {
    const previous = previousByUser.get(feature.raw_user_key);
    feature.posting_gap_days = previous == null ? null : (feature.timestamp - previous) / 86400000;
    previousByUser.set(feature.raw_user_key, feature.timestamp);
  }
  const safeFeatures = features.map(({ raw_user_key, ...feature }) => feature);
  return {
    quality: { rows: rows.length, empty_text_rows: empty, exact_duplicate_text_rows: exactDuplicates,
      normalized_near_duplicate_text_rows: Math.max(0, normalizedDuplicates - exactDuplicates), very_short_text_rows_under_5_words: short,
      mapped_columns: { text: Boolean(textColumn), user_id: Boolean(userColumn), timestamp: Boolean(timestampColumn) } },
    features: safeFeatures,
  };
}

export function featureAssociations(features = [], predictions = [], measure = 'e2_score_uncalibrated') {
  const byRow = new Map(features.map(feature => [feature.row_number, feature]));
  const scoreByRow = new Map(predictions.map(row => [row.row_number, scoreValue(row, measure)]));
  const specs = [
    ['Post length (characters)', feature => feature.char_length],
    ['Post length (words)', feature => feature.word_count],
    ['First-person pronouns', feature => feature.first_person_pronouns],
    ['Negation present', feature => feature.negation_present],
    ['Time of day (UTC sine)', feature => feature.hour_utc == null ? null : Math.sin(2 * Math.PI * feature.hour_utc / 24)],
    ['Time of day (UTC cosine)', feature => feature.hour_utc == null ? null : Math.cos(2 * Math.PI * feature.hour_utc / 24)],
    ['Posting gap (days)', feature => feature.posting_gap_days],
  ];
  return specs.map(([label, read]) => {
    const rows = [...byRow.entries()].map(([rowNumber, feature]) => [read(feature), scoreByRow.get(rowNumber)]);
    const corr = pearson(rows.map(row => row[0]), rows.map(row => row[1]));
    return { label, ...corr };
  });
}

export function scoreValue(row, measure) {
  return measure === 'e2_score_uncalibrated' ? row?.e2_score_uncalibrated
    : measure.startsWith('e2_probability_') ? row?.e2_class_probabilities?.[measure.slice(15)] : row?.goemotions_scores?.[measure];
}
