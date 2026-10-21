import Papa from 'papaparse';
import { E2_CLASSES, EMOTION_LABELS, e2Summary, emotionMeans, userTemporalComparison } from './analytics.js';

export function userSummaryCsv(users = []) {
  const fields = ['user_id', 'anchor', 'recent_window', 'baseline_window', 'eligible', 'reason', 'recent_posts', 'baseline_posts',
    'recent_e2_mean', 'baseline_e2_mean', 'e2_mean_difference_recent_minus_baseline',
    ...EMOTION_LABELS.flatMap(label => [`recent_${label}`, `baseline_${label}`, `difference_${label}`])];
  const data = users.map(user => {
    const comparison = userTemporalComparison(user);
    return fields.map(key => {
      if (key === 'recent_window') return '[0, 90) days before anchor';
      if (key === 'baseline_window') return '[90, 270) days before anchor';
      if (key.startsWith('recent_') && EMOTION_LABELS.includes(key.slice(7))) return comparison?.emotions.find(e => e.label === key.slice(7))?.recent ?? '';
      if (key.startsWith('baseline_') && EMOTION_LABELS.includes(key.slice(9))) return comparison?.emotions.find(e => e.label === key.slice(9))?.baseline ?? '';
      if (key.startsWith('difference_') && EMOTION_LABELS.includes(key.slice(11))) return comparison?.emotions.find(e => e.label === key.slice(11))?.difference ?? '';
      return user[key] ?? '';
    });
  });
  return Papa.unparse({ fields, data });
}

export function rowPredictionsCsv(predictions = []) {
  const fields = ['row_number', 'user_id', 'timestamp', 'predicted_class', 'e2_score_uncalibrated',
    'e2_token_count', 'goemotions_token_count', 'e2_truncated_at_512', 'goemotions_truncated_at_512',
    ...E2_CLASSES.map(label => `e2_probability_${label}`), ...EMOTION_LABELS.map(label => `goemotions_${label}`)];
  const data = predictions.map(row => fields.map(field => {
    if (field.startsWith('e2_probability_')) return row.e2_class_probabilities?.[field.slice('e2_probability_'.length)] ?? '';
    if (field.startsWith('goemotions_')) return row.goemotions_scores?.[field.slice('goemotions_'.length)] ?? '';
    return row[field] ?? '';
  }));
  return Papa.unparse({ fields, data });
}

export function summaryReport(result) {
  const predictions = result?.predictions ?? [];
  const summary = e2Summary(predictions);
  const emotions = emotionMeans(predictions);
  const users = result?.users ?? [];
  return {
    analysis_timestamp: result?.run?.completed_at ?? null,
    uploaded_rows: result?.summary?.rows_received ?? null,
    processed_rows: result?.summary?.nonempty_text_rows ?? predictions.length,
    excluded_rows: result?.summary?.empty_text_rows ?? null,
    timestamp_range: { min: result?.summary?.parsed_timestamp_min ?? null, max: result?.summary?.parsed_timestamp_max ?? null },
    data_quality: Object.fromEntries(['exact_duplicate_text_rows', 'normalized_near_duplicate_text_rows',
      'very_short_text_rows_under_5_words', 'e2_truncated_rows', 'goemotions_truncated_rows',
      'truncated_by_either_model_rows', 'missing_or_invalid_timestamp_rows', 'numeric_timestamp_rows', 'mapped_columns']
      .filter(key => key in (result?.summary ?? {})).map(key => [key, result.summary[key]])),
    users_analyzed: result?.summary?.unique_users ?? users.length,
    temporally_eligible_users: result?.summary?.temporal_eligible_users ?? users.filter(u => u.eligible).length,
    e2_predicted_class_distribution: summary.classes,
    e2_behavior_plus_attempt_score_uncalibrated: summary.score,
    goemotions_mean_sigmoid_scores: emotions,
    temporal_comparisons: users.map(user => ({ user_id: user.user_id, ...userTemporalComparison(user) })),
    methodology: {
      e2_score: 'P(behavior) + P(attempt); uncalibrated model-derived research measure, not a clinical probability.',
      goemotions: '28 independent sigmoid scores; labels may be present simultaneously and scores do not sum to one.',
      temporal_windows: 'Recent [0, 90) days and baseline [90, 270) days before each user anchor; comparisons shown only when API eligibility is true.',
      limitations: result?.run?.warnings ?? [],
      research_caveat: 'Research prototype only. No clinical validation, causal inference, or generalization to arbitrary uploads has been established.',
    },
  };
}

export function summaryReportJson(result) {
  return JSON.stringify(summaryReport(result), null, 2);
}

export function runMetadataJson(run, summary) {
  const runFields = ['completed_at', 'elapsed_inference_seconds', 'model_loading_seconds', 'e2_model',
    'e2_checkpoint_revision', 'e2_configured_revision', 'emotion_model', 'emotion_revision', 'device',
    'max_length_tokens', 'batch_size', 'temporal', 'warnings'];
  const summaryFields = ['rows_received', 'nonempty_text_rows', 'empty_text_rows', 'duplicate_rows',
    'unique_users', 'missing_user_id_rows', 'missing_or_invalid_timestamp_rows', 'numeric_timestamp_rows',
    'temporal_analysis_available', 'parsed_timestamp_min', 'parsed_timestamp_max',
    'temporal_eligible_users', 'temporal_ineligible_users'];
  return JSON.stringify({
    run: Object.fromEntries(runFields.filter(key => key in (run ?? {})).map(key => [key, run[key]])),
    summary: Object.fromEntries(summaryFields.filter(key => key in (summary ?? {})).map(key => [key, summary[key]])),
  }, null, 2);
}
