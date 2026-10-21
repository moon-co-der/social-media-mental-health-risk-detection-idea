import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import Papa from 'papaparse';
import { rowPredictionsCsv, runMetadataJson, summaryReportJson, userSummaryCsv } from './exports.js';
import { E2_CLASSES, EMOTION_LABELS, adaptiveDistribution, describeValues, deviationSummary, e2Summary, emotionMeans, featureAssociations, filteredPredictions, groupScoreSeries, inspectUploadedRows, perUserScoreMeans, scoreValue, temporalTimelinePoints, userTemporalComparison } from './analytics.js';
import './style.css';

const API = (import.meta.env.VITE_API_URL || (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '')).replace(/\/+$/, '');
const EMOTIONS = EMOTION_LABELS;
const PAGES = ['Home', 'Analyze', 'Results', 'Transparency'];
const SAVED_EMOTION_SUMMARY = [
  { emotion: 'nervousness', mean: -0.0001, median: -0.0021, increase: 115, decrease: 143 },
  { emotion: 'fear', mean: -0.0039, median: -0.0031, increase: 101, decrease: 157 },
  { emotion: 'disappointment', mean: -0.0060, median: -0.0071, increase: 112, decrease: 146 },
  { emotion: 'neutral', mean: 0.0070, median: 0.0030, increase: 137, decrease: 121 },
  { emotion: 'embarrassment', mean: -0.0005, median: -0.0004, increase: 108, decrease: 150 },
  { emotion: 'curiosity', mean: 0.0017, median: -0.0026, increase: 120, decrease: 138 },
];

function saveFile(name, value, type) {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([value], { type }));
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function TimelineChart({ data, measureLabel }) {
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(Math.max(0, data.length - 1));
  const from = Math.min(start, Math.max(0, data.length - 1));
  const to = Math.max(from, Math.min(end, Math.max(0, data.length - 1)));
  const view = data.slice(from, to + 1);
  const width = 900, height = 300, left = 55, right = 20, top = 18, bottom = 42;
  const plotW = width - left - right, plotH = height - top - bottom;
  const minAge = Math.min(...view.map(p => p.ageDays), 0);
  const maxAge = Math.max(...view.map(p => p.ageDays), 1);
  const span = Math.max(1, maxAge - minAge);
  const x = age => left + ((maxAge - Math.max(minAge, Math.min(maxAge, age))) / span) * plotW;
  const y = score => top + (1 - score) * plotH;
  const dateLabel = age => view.reduce((closest, p) => Math.abs(p.ageDays - age) < Math.abs(closest.ageDays - age) ? p : closest, view[0] || { date: '' }).date;
  const zone = (lo, hi) => {
    const a = Math.max(minAge, lo), b = Math.min(maxAge, hi);
    return b > a ? { x: x(b), width: x(a) - x(b) } : null;
  };
  const recent = zone(0, 90), baseline = zone(90, 270);
  return <div>
    <svg className="timeline-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Chronological uncalibrated E2 score timeline">
      {recent && <rect x={recent.x} y={top} width={recent.width} height={plotH} fill="#dcece6"/>}
      {baseline && <rect x={baseline.x} y={top} width={baseline.width} height={plotH} fill="#e8eee9"/>}
      {[0, .5, 1].map(v => <g key={v}><line x1={left} x2={width-right} y1={y(v)} y2={y(v)} stroke="#dfe6e1" strokeDasharray="4 4"/><text x={left-10} y={y(v)+4} textAnchor="end">{v.toFixed(1)}</text></g>)}
      {view.map((p, i) => <circle key={`${p.date}-${i}`} cx={x(p.ageDays)} cy={y(p.score)} r="4" fill="#276c57"><title>{`${p.date} · ${measureLabel}: ${p.score.toFixed(4)} · 1 contributing post`}</title></circle>)}
      {[maxAge, minAge].map((age, i) => <text key={i} x={x(age)} y={height-12} textAnchor={i ? 'end' : 'start'}>{dateLabel(age)}</text>)}
    </svg>
    <div className="timeline-legend"><span><i className="recent-key"/>Recent [0, 90) days</span><span><i className="baseline-key"/>Baseline [90, 270) days</span><span>Unshaded: outside both windows</span></div>
    <div className="range-controls"><label>Older start<input aria-label="Older start of chart range" type="range" min="0" max={Math.max(0, data.length-1)} value={from} onChange={e => { const next=Number(e.target.value); setStart(next); if(next>to)setEnd(next); }}/></label><label>Recent end<input aria-label="Recent end of chart range" type="range" min="0" max={Math.max(0, data.length-1)} value={to} onChange={e => { const next=Number(e.target.value); setEnd(next); if(next<from)setStart(next); }}/></label></div>
    <p className="muted">Each point is one uploaded post. Unobserved dates are left blank; points are not connected. The range controls change only the chart view, not the temporal eligibility windows or summary statistics.</p>
  </div>;
}

function HorizontalBars({ title, rows, valueKey = 'mean', format = value => value.toFixed(4), maxValue, axisLabel = 'Mean score (0–1)', empty = 'No values available for this analysis.' }) {
  const values = rows.map(row => row[valueKey]).filter(Number.isFinite);
  if (!rows.length || !values.length) return <section className="chart-card"><h3>{title}</h3><p className="muted">{empty}</p></section>;
  const max = maxValue ?? Math.max(...values, 0.0001);
  return <section className="chart-card"><h3>{title}</h3><p className="chart-axis-label">{axisLabel}</p><div className="bar-chart" role="list" aria-label={title}>{rows.map(row => {
    const value = row[valueKey];
    const detail = `${row.label}: ${Number.isFinite(value) ? format(value) : 'not available'}${row.count != null ? ` · ${row.count} posts${Number.isFinite(row.percentage) ? ` (${row.percentage.toFixed(1)}%)` : ''}` : ''}`;
    return <div className={`bar-row${row.n != null && row.n < 5 ? ' low-count' : ''}`} role="listitem" tabIndex={0} aria-label={detail} key={row.label} title={detail}><span className="bar-label">{row.label}{row.n != null && row.n < 5 ? ' · few posts' : ''}</span><span className="bar-track"><i style={{ width: `${Number.isFinite(value) ? Math.max(0, value / max * 100) : 0}%` }}/></span><b>{Number.isFinite(value) ? format(value) : '—'}{row.percentage != null ? ` (${row.percentage.toFixed(1)}%)` : ''}</b></div>;
  })}</div></section>;
}

function DistributionPanel({ title, values, rowNumbers, label, fullScale, setFullScale, caption }) {
  const validRows = values.map((value, i) => ({ value, row: rowNumbers?.[i] ?? i + 1 })).filter(item => Number.isFinite(item.value));
  const chart = adaptiveDistribution(validRows.map(item => item.value), fullScale);
  const { stats, domain, bins, density, points } = chart;
  const fullScaleMaximum = Math.max(1, Math.ceil(stats.max ?? 1));
  if (!stats.n) return <section className="chart-card"><h3>{title}</h3><p className="empty-state">No valid values are available for this measure.</p></section>;
  const width = 760, height = 250, left = 38, right = 18, top = 18, bottom = 35;
  const plotW = width - left - right, plotH = height - top - bottom;
  const maximum = Math.max(1, ...bins.map(bin => bin.count), ...density.map(point => point.countScale));
  const x = value => left + (value - domain[0]) / (domain[1] - domain[0]) * plotW;
  const y = count => top + plotH - count / maximum * plotH;
  const line = density.map((point, i) => `${i ? 'L' : 'M'}${x(point.x)},${y(point.countScale)}`).join(' ');
  const markers = [['Mean', stats.mean], ['Median', stats.median], ['−1 SD', stats.mean - stats.sd], ['+1 SD', stats.mean + stats.sd]];
  return <section className="chart-card"><h3>{title}</h3><div className="distribution-controls"><label><input type="checkbox" checked={fullScale} onChange={e => setFullScale(e.target.checked)}/> Full scale 0–{fullScaleMaximum}</label><span>n={stats.n}{stats.n < 30 ? ' · Small sample (n &lt; 30)' : ''}</span></div><p className="chart-axis-label">{label} · {fullScale ? `full 0–${fullScaleMaximum} scale` : 'adaptive range'}</p>
    <svg className="distribution-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title}; ${label}, n ${stats.n}`}>
      {[0, .5, 1].map(t => <g key={t}><line x1={left} x2={width-right} y1={top + plotH * (1-t)} y2={top + plotH * (1-t)} stroke="#dfe6e1"/><text x={left-8} y={top + plotH * (1-t)+4} textAnchor="end">{(maximum*t).toFixed(0)}</text></g>)}<text x={left} y={12}>Post count / expected density</text>
      {bins.map((bin, i) => <g key={i}><rect x={x(bin.from)+1} y={y(bin.count)} width={Math.max(1, plotW / bins.length - 2)} height={Math.max(0, top + plotH - y(bin.count))} fill="#c8dbcf"><title>{bin.from.toFixed(4)}–{bin.to.toFixed(4)}: {bin.count} posts</title></rect></g>)}
      <path d={line} fill="none" stroke="#295f49" strokeWidth="2" aria-label="Kernel density estimate"/>
      {markers.map(([name, value]) => value >= domain[0] && value <= domain[1] ? <g key={name}><line x1={x(value)} x2={x(value)} y1={top} y2={top+plotH} stroke={name === 'Median' ? '#76548a' : '#b26c39'} strokeDasharray="4 3"/><title>{name}: {value.toFixed(4)}</title></g> : null)}
      {points.map((point, i) => <circle key={i} cx={x(point.value)} cy={height-15 + point.jitter/4} r="2.4" fill="#365d79"><title>Upload row {validRows[i]?.row ?? i+1}: {label} {point.value.toFixed(5)}</title></circle>)}
      <text x={left} y={height-2}>{domain[0].toFixed(4)}</text><text x={width-right} y={height-2} textAnchor="end">{domain[1].toFixed(4)}</text>
    </svg>
    <div className="stat-legend">{markers.map(([name, value]) => <span key={name}>{name}: {value.toFixed(4)}</span>)}</div>
    <p className="distribution-summary">n={stats.n} · mean {stats.mean.toFixed(5)} · median {stats.median.toFixed(5)} · SD {stats.sd.toFixed(5)} · min {stats.min.toFixed(5)} · max {stats.max.toFixed(5)} · IQR {stats.iqr.toFixed(5)}</p>
    {stats.nearConstant && <p className="warning">Near-constant output by the project’s review rule (SD &lt; 0.01 or range &lt; 0.05). Variation may be too small to interpret.</p>}
    <p className="muted">{caption} The density curve is a descriptive smoothing. Points show individual posts.</p>
  </section>;
}

function DeviationPlot({ rows, baseline, baselineLabel }) {
  if (baseline == null || !rows.length) return <p className="empty-state">This deviation view needs a baseline and valid scores. Select a user with a recent window to use that option.</p>;
  const width=760,height=Math.max(180,rows.length*17+45),left=110,right=35,top=18,bottom=28;
  const maximum=Math.max(0.005,...rows.map(r=>Math.abs(r.deviation))), x=d=>left+(d+maximum)/(2*maximum)*(width-left-right), zero=x(0);
  return <><p className="muted">Baseline: {baselineLabel} = {baseline.toFixed(5)}. Deviation = post score − baseline. Above baseline is blue; below baseline is orange. Outliers use |deviation| &gt; 2 SD or Tukey 1.5×IQR fences.</p><svg className="deviation-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Deviation from ${baselineLabel}, n ${rows.length}`}><line x1={zero} x2={zero} y1={top} y2={height-bottom} stroke="#52675b" strokeDasharray="4 3"/>{rows.map((row,i)=><g key={row.row_number}><text x={left-8} y={top+i*17+4} textAnchor="end">row {row.row_number}</text><rect x={Math.min(zero,x(row.deviation))} y={top+i*17-4} width={Math.max(1,Math.abs(x(row.deviation)-zero))} height="9" rx="3" fill={row.deviation>=0?'#356b8a':'#bb7e34'}><title>Row {row.row_number}: deviation {row.deviation.toFixed(5)}{row.outlier?' · flagged by the specified outlier rule':''}</title></rect>{row.outlier&&<text x={width-right+4} y={top+i*17+4}>outlier</text>}</g>)}<text x={left} y={height-6}>−{maximum.toFixed(4)}</text><text x={width-right} y={height-6} textAnchor="end">+{maximum.toFixed(4)}</text></svg><p className="muted">n={rows.length}{rows.length<30?' · Small sample (n < 30)':''}. Outlier flags identify statistical distance only; they are not person-level risk labels.</p></>;
}

function featureMean(features, label) {
  const values = features.map(feature => {
    if (label === 'Post length (characters)') return feature.char_length;
    if (label === 'Post length (words)') return feature.word_count;
    if (label === 'First-person pronouns') return feature.first_person_pronouns;
    if (label === 'Negation present') return feature.negation_present;
    if (label === 'Posting gap (days)') return feature.posting_gap_days;
    if (label === 'Time of day (UTC sine)') return feature.hour_utc == null ? null : Math.sin(feature.hour_utc * Math.PI / 12);
    if (label === 'Time of day (UTC cosine)') return feature.hour_utc == null ? null : Math.cos(feature.hour_utc * Math.PI / 12);
    return null;
  }).filter(Number.isFinite);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function DataQualityPanel({ quality, predictions, nearConstant }) {
  if (!quality) return <section className="card"><h2>Data quality</h2><p className="muted">Data quality summary is unavailable.</p></section>;
  const columns=quality.mapped_columns ?? {};
  return <section className="card data-quality"><h2>Data quality</h2><div className="quality-grid"><span>Rows: <b>{quality.rows_received ?? quality.rows}</b></span><span>Empty text: <b>{quality.empty_text_rows ?? 0}</b></span><span>Exact duplicate text: <b>{quality.exact_duplicate_text_rows ?? 0}</b></span><span>Normalized near-duplicates: <b>{quality.normalized_near_duplicate_text_rows ?? 0}</b></span><span>Very short (&lt;5 words): <b>{quality.very_short_text_rows_under_5_words ?? 0}</b></span><span>Missing user IDs: <b>{quality.missing_user_id_rows ?? 'Unavailable'}</b></span><span>Missing/invalid timestamps: <b>{quality.missing_or_invalid_timestamp_rows ?? 'Unavailable'}</b></span><span>At E2 512-token limit: <b>{quality.e2_truncated_rows ?? 'Unavailable'}</b></span><span>At GoEmotions 512-token limit: <b>{quality.goemotions_truncated_rows ?? 'Unavailable'}</b></span></div><p>Mapped columns: text {columns.text?'found':'missing'} · user ID {columns.user_id?'found':'missing'} · timestamp {columns.timestamp?'found':'missing'}.</p><p className="muted">Near-duplicates are repeated text after case, punctuation, spacing, and Unicode normalization. Very short means fewer than 5 whitespace-separated words. Token counts are model-tokenizer counts; a count above 512 indicates truncation.</p>{nearConstant.length>0&&<p className="warning">Near-constant outputs (SD &lt; 0.01 or range &lt; 0.05): {nearConstant.join(', ')}. Variation may be too small to interpret.</p>}{predictions.length<30&&<p className="warning">Small sample: n={predictions.length} (below 30). Descriptive patterns may be unstable.</p>}</section>;
}

function DeviationDrivers({ rows, predictions, features, associations, emotionStats, flaggedRow, setFlaggedRow }) {
  if (!rows.length) return <p className="empty-state">No statistical outliers were flagged under the stated rule.</p>;
  const getFeatureValues = feature => [
    ['Post length (characters)', feature.char_length, 'Post length (characters)'], ['Post length (words)', feature.word_count, 'Post length (words)'],
    ['First-person pronouns', feature.first_person_pronouns, 'First-person pronouns'], ['Negation present', feature.negation_present, 'Negation present'],
    ['UTC hour (sine encoding)', feature.hour_utc == null ? null : Math.sin(feature.hour_utc * Math.PI / 12), 'Time of day (UTC sine)'], ['UTC hour (cosine encoding)', feature.hour_utc == null ? null : Math.cos(feature.hour_utc * Math.PI / 12), 'Time of day (UTC cosine)'],
    ['Posting gap (days)', feature.posting_gap_days, 'Posting gap (days)'],
  ];
  return <div className="driver-list">{rows.map(row=>{
    const prediction=predictions.find(p=>p.row_number===row.row_number), feature=features.find(f=>f.row_number===row.row_number);
    const top=Object.entries(prediction?.goemotions_scores??{}).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).slice(0,3);
    const available=associations.filter(a=>a.r!=null&&a.n>=2);
    return <details key={row.row_number} className="driver-card" open={String(row.row_number)===flaggedRow} onToggle={e=>{if(e.currentTarget.open)setFlaggedRow(String(row.row_number));}}><summary>Row {row.row_number} · deviation {row.deviation>0?'+':''}{row.deviation.toFixed(5)} · What co-occurs with this score?</summary><div className="driver-content"><p>No causal driver can be established. The values below describe this post and associations across the uploaded cohort.</p><h4>Top GoEmotions scores vs cohort mean</h4>{top.length?<ul>{top.map(([emotion,value])=>{const average=emotionStats.find(e=>e.label===emotion)?.mean;return <li key={emotion}>{emotion}: {value.toFixed(4)} vs cohort mean {average?.toFixed(4)??'unavailable'}</li>;})}</ul>:<p>No emotion values are available.</p>}<h4>Text and timing features; cohort correlation with selected score</h4>{!feature?<p>Local feature summary is unavailable.</p>:<ul>{getFeatureValues(feature).map(([label,value,associationLabel],i)=>{const assoc=associations.find(a=>a.label===associationLabel);const cohortMean=featureMean(features,associationLabel);return <li key={`${label}-${i}`}>{label}: {Number.isFinite(value)?value.toFixed(2):'unavailable'} vs cohort mean {Number.isFinite(cohortMean)?cohortMean.toFixed(2):'unavailable'}; correlation r={assoc?.r?.toFixed(3)??'unavailable'}, n={assoc?.n??0}</li>;})}</ul>}<p>{available.length?'Measured features show associations with score variation in this upload; these correlations are descriptive and do not establish cause.':'No clear driver found in the available features.'} Correlations with n &lt; 30 should be treated as especially unstable.</p></div></details>;
  })}</div>;
}

function RankedDots({ rows, label }) {
  const sorted = [...rows].filter(row => Number.isFinite(row.value)).sort((a, b) => a.value - b.value || a.row_number - b.row_number);
  if (!sorted.length) return <p className="empty-state">No post scores are available to rank.</p>;
  const width = 760, height = Math.max(180, sorted.length * 7 + 50), left = 50, right = 150, top = 20, bottom = 30;
  const values = sorted.map(row => row.value), min = Math.min(...values), max = Math.max(...values), span = Math.max(max-min, 0.01);
  const x = value => left + (value-min)/span*(width-left-right), extremes = new Set([...sorted.slice(0,5), ...sorted.slice(-5)].map(row => row.row_number));
  return <div className="rank-plot-wrap"><svg viewBox={`0 0 ${width} ${height}`} className="rank-svg" role="img" aria-label={`Posts ranked by ${label}; n ${sorted.length}`}><line x1={left} x2={width-right} y1={height-bottom} y2={height-bottom} stroke="#9eaaa2"/>{sorted.map((row,i)=><g key={row.row_number}><circle cx={x(row.value)} cy={top+i*7} r="3" fill="#456e58"><title>Upload row {row.row_number}: {label} {row.value.toFixed(5)}</title></circle>{extremes.has(row.row_number)&&<text x={width-right+6} y={top+i*7+4}>row {row.row_number} · {row.value.toFixed(4)}</text>}</g>)}<text x={left} y={height-6}>{min.toFixed(4)}</text><text x={width-right} y={height-6} textAnchor="end">{max.toFixed(4)}</text></svg><p className="muted">Sorted low to high · upload row labels · n={sorted.length}{sorted.length<30?' · Small sample (n < 30)':''}</p></div>;
}

function GroupTrendChart({ rows, title, label }) {
  if (!rows.length) return <p className="empty-state">Valid timestamps are unavailable for this grouped trend.</p>;
  const width=760,height=240,left=52,right=24,top=18,bottom=36, vals=rows.flatMap(r=>[r.low,r.high]), min=Math.min(...vals),max=Math.max(...vals), span=Math.max(max-min,0.01);
  const x=i=>left+(rows.length===1?0:i/(rows.length-1))*(width-left-right), y=v=>top+(max-v)/span*(height-top-bottom);
  const band=[...rows.map((r,i)=>`${x(i)},${y(r.high)}`),...rows.map((r,i)=>`${x(rows.length-1-i)},${y(rows[rows.length-1-i].low)}`)].join(' ');
  return <svg className="trend-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title}; ${label}; ${rows.length} time groups`}><polygon points={band} fill="#bed2c4" opacity=".55"/><polyline points={rows.map((r,i)=>`${x(i)},${y(r.mean)}`).join(' ')} fill="none" stroke="#295f49" strokeWidth="2"/>{rows.map((r,i)=><g key={r.date}><circle cx={x(i)} cy={y(r.mean)} r="4" fill="#295f49"><title>{r.date}: mean {r.mean.toFixed(5)}, SD {r.sd.toFixed(5)}, n={r.n}</title></circle><text x={x(i)} y={height-10} textAnchor="middle">{rows.length<=12||i%Math.ceil(rows.length/12)===0?r.date:''}</text></g>)}</svg>;
}

function ComparisonBars({ rows }) {
  if (!rows.length) return <p className="muted">No eligible recent/baseline comparison is available for this user.</p>;
  return <div className="comparison-chart">{rows.map(row => <div className="comparison-row" key={row.label}><b>{row.label}</b><div><span>Recent {Number.isFinite(row.recent) ? row.recent.toFixed(4) : '—'}</span><i><em style={{ width: `${Math.max(0, (row.recent ?? 0) * 100)}%` }}/></i></div><div><span>Baseline {Number.isFinite(row.baseline) ? row.baseline.toFixed(4) : '—'}</span><i><em className="baseline-fill" style={{ width: `${Math.max(0, (row.baseline ?? 0) * 100)}%` }}/></i></div><small>Δ {Number.isFinite(row.difference) ? row.difference.toFixed(4) : '—'}</small></div>)}</div>;
}

function App() {
  const [page, setPage] = useState(PAGES[0]);
  const [file, setFile] = useState(null);
  const [fields, setFields] = useState([]);
  const [preview, setPreview] = useState([]);
  const [textCol, setTextCol] = useState('');
  const [userCol, setUserCol] = useState('');
  const [timeCol, setTimeCol] = useState('');
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [validating, setValidating] = useState(false);
  const [error, setError] = useState('');
  const [validation, setValidation] = useState(null);
  const [result, setResult] = useState(null);
  const [tab, setTab] = useState('Overview');
  const [selected, setSelected] = useState('');
  const [topEmotionCount, setTopEmotionCount] = useState(5);
  const [timelineMeasure, setTimelineMeasure] = useState('e2_score_uncalibrated');
  const [postUserSearch, setPostUserSearch] = useState('');
  const [postSort, setPostSort] = useState('row_number');
  const [postPage, setPostPage] = useState(0);
  const [exportStart, setExportStart] = useState('');
  const [exportEnd, setExportEnd] = useState('');
  const [dataQuality, setDataQuality] = useState(null);
  const [localFeatures, setLocalFeatures] = useState([]);
  const [distributionMeasure, setDistributionMeasure] = useState('e2_score_uncalibrated');
  const [fullScale, setFullScale] = useState(false);
  const [trendPeriod, setTrendPeriod] = useState('week');
  const [baselineMode, setBaselineMode] = useState('median');
  const [flaggedRow, setFlaggedRow] = useState('');

  const chooseFile = (next) => {
    setError(''); setResult(null); setValidation(null); setDataQuality(null); setLocalFeatures([]);
    if (!next) return;
    if (!next.name.toLowerCase().endsWith('.csv')) { setFile(null); setError('Choose a .csv file.'); return; }
    if (next.size > 2 * 1024 * 1024) { setFile(null); setError('File exceeds the current 2 MiB limit. Choose a smaller CSV; it was not uploaded.'); return; }
    setFile(next);
    Papa.parse(next, { header: true, preview: 8, skipEmptyLines: 'greedy', complete: ({ meta, data }) => {
      const columns = meta.fields || [];
      setFields(columns); setPreview(data);
      const selectedText = columns.find(x => /text|post|content/i.test(x)) || columns[0] || '';
      const selectedUser = columns.find(x => /user|author|account/i.test(x)) || '';
      const selectedTime = columns.find(x => /time|date|created/i.test(x)) || '';
      setTextCol(selectedText); setUserCol(selectedUser); setTimeCol(selectedTime);
    }, error: (e) => setError(e.message) });
  };

  useEffect(() => {
    if (!file || !textCol) return undefined;
    let current = true;
    Papa.parse(file, { header: true, skipEmptyLines: true, complete: ({ data: allRows }) => {
      if (!current) return;
      const checked = inspectUploadedRows(allRows, { textColumn: textCol, userColumn: userCol, timestampColumn: timeCol });
      setDataQuality(checked.quality); setLocalFeatures(checked.features);
    } });
    return () => { current = false; };
  }, [file, textCol, userCol, timeCol]);

  const validateCsv = async () => {
    if (!file || !textCol) { setError('Choose a CSV and map its text column.'); return; }
    if (!API) { setError('Set VITE_API_URL to the deployed Python API address in the frontend hosting settings.'); return; }
    setValidating(true); setError(''); setValidation(null);
    try {
      const form = new FormData(); form.append('file', file); form.append('text_column', textCol);
      form.append('user_column', userCol); form.append('timestamp_column', timeCol);
      const response = await fetch(`${API}/validate`, { method: 'POST', body: form, signal: AbortSignal.timeout(30000) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || `API returned ${response.status}`);
      setValidation(body);
    } catch (e) { setError(`${e.message}. Confirm the Python service is reachable and check the CSV columns and format.`); }
    finally { setValidating(false); }
  };

  const analyze = async () => {
    if (!file || !textCol || !validation) { setError('Validate the CSV with the selected columns before running inference.'); return; }
    if (!API) { setError('Set VITE_API_URL to the deployed Python API address in the frontend hosting settings.'); return; }
    setBusy(true); setError(''); setResult(null);
    try {
      const form = new FormData();
      form.append('file', file); form.append('text_column', textCol);
      form.append('user_column', userCol); form.append('timestamp_column', timeCol);
      form.append('pseudonym_seed', validation.pseudonym_seed);
      const response = await fetch(`${API}/analyze`, { method: 'POST', body: form, signal: AbortSignal.timeout(300000) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || `API returned ${response.status}`);
      setResult(body); setSelected(body.users?.[0]?.user_id || ''); setTab('Overview'); setPostUserSearch(''); setPostSort('row_number'); setPostPage(0); setExportStart(''); setExportEnd(''); setValidation(null); setPage('Results');
    } catch (e) {
      setError(`${e.message}. Confirm the Python service is reachable and its required model checkpoints are configured, then retry.`);
    } finally { setBusy(false); }
  };

  const currentUser = result?.users?.find(u => u.user_id === selected);
  const timeline = useMemo(() => {
    if (!result || !currentUser) return [];
    return temporalTimelinePoints(result.predictions, selected, currentUser.anchor, timelineMeasure);
  }, [result, selected, currentUser, timelineMeasure]);
  const predictions = result?.predictions ?? [];
  const e2Stats = useMemo(() => e2Summary(predictions), [result]);
  const emotionStats = useMemo(() => emotionMeans(predictions), [result]);
  const selectedTemporal = useMemo(() => userTemporalComparison(currentUser), [currentUser]);
  const temporalEmotionRows = selectedTemporal?.eligible ? [...selectedTemporal.emotions].sort((a, b) => Math.max(b.recent ?? 0, b.baseline ?? 0) - Math.max(a.recent ?? 0, a.baseline ?? 0) || a.label.localeCompare(b.label)).slice(0, 10) : [];
  const filteredTableRows = useMemo(() => {
    const rows = predictions.filter(p => !postUserSearch || (p.user_id || '').toLowerCase().includes(postUserSearch.toLowerCase()));
    rows.sort((a, b) => postSort === 'timestamp' ? String(a.timestamp ?? '').localeCompare(String(b.timestamp ?? '')) : postSort === 'score' ? (b.e2_score_uncalibrated ?? -1) - (a.e2_score_uncalibrated ?? -1) : a.row_number - b.row_number);
    return rows;
  }, [result, postUserSearch, postSort]);
  const pageSize = 20;
  const visibleTableRows = filteredTableRows.slice(postPage * pageSize, (postPage + 1) * pageSize);
  const exportFilteredRows = filteredPredictions(predictions, { userId: selected, start: exportStart, end: exportEnd });
  const distributionRows = predictions.map(row => ({ row_number: row.row_number, value: scoreValue(row, distributionMeasure) })).filter(row => Number.isFinite(row.value));
  const distributionLabel = distributionMeasure === 'e2_score_uncalibrated' ? 'E2 behavior + attempt · uncalibrated model score, not a clinical risk probability'
    : distributionMeasure.startsWith('e2_probability_') ? `E2 ${distributionMeasure.slice(15)} probability · uncalibrated model score, not a clinical risk probability`
      : `GoEmotions ${distributionMeasure} independent sigmoid score`;
  const measureName = distributionMeasure === 'e2_score_uncalibrated' ? 'E2 behavior + attempt score'
    : distributionMeasure.startsWith('e2_probability_') ? `E2 ${distributionMeasure.slice(15)} score` : `GoEmotions ${distributionMeasure}`;
  const qualitySummary = { ...(dataQuality ?? {}), ...(result?.summary ?? {}) };
  const nearConstantMeasures = useMemo(() => {
    const candidates = [{ name: 'E2 behavior + attempt', values: predictions.map(p => p.e2_score_uncalibrated) },
      ...E2_CLASSES.map(label => ({ name: `E2 ${label} probability`, values: predictions.map(p => p.e2_class_probabilities?.[label]) })),
      ...EMOTIONS.map(emotion => ({ name: `GoEmotions ${emotion}`, values: predictions.map(p => p.goemotions_scores?.[emotion]) }))];
    return candidates.filter(item => describeValues(item.values).n && describeValues(item.values).nearConstant).map(item => item.name);
  }, [result]);
  const trendRows = useMemo(() => groupScoreSeries(predictions, distributionMeasure, trendPeriod), [result, distributionMeasure, trendPeriod]);
  const userMeanRows = useMemo(() => perUserScoreMeans(predictions, distributionMeasure), [result, distributionMeasure]);
  const deviationScope = baselineMode === 'user_recent' ? predictions.filter(row => selected && row.user_id === selected) : predictions;
  const cohortDescription = describeValues(predictions.map(row => scoreValue(row, distributionMeasure)));
  let deviationBaseline = baselineMode === 'median' ? cohortDescription.median : baselineMode === 'mean' ? cohortDescription.mean : null;
  let baselineLabel = baselineMode === 'median' ? 'cohort median' : 'cohort mean';
  if (baselineMode === 'user_recent') {
    baselineLabel = 'selected user’s [0, 90) day mean';
    deviationBaseline = distributionMeasure === 'e2_score_uncalibrated' ? currentUser?.recent_e2_mean
      : !distributionMeasure.startsWith('e2_probability_') ? currentUser?.recent_emotion_means?.[distributionMeasure] : null;
  }
  const rawDeviations = deviationSummary(deviationScope.map(row => scoreValue(row, distributionMeasure)), deviationBaseline);
  const deviationRows = rawDeviations.rows.map(item => ({ ...item, row_number: deviationScope[item.index]?.row_number })).filter(row => row.row_number != null);
  const outlierRows = deviationRows.filter(row => row.outlier);
  const associations = useMemo(() => featureAssociations(localFeatures, predictions, distributionMeasure), [result, localFeatures, distributionMeasure]);
  const availableDistributionMeasures = [
    ['e2_score_uncalibrated', 'E2 behavior + attempt score'],
    ...E2_CLASSES.map(label => [`e2_probability_${label}`, `E2 class probability · ${label}`]),
    ...EMOTIONS.map(label => [label, `GoEmotions · ${label}`]),
  ];

  const exportUsers = () => saveFile('user-summary.csv', userSummaryCsv(result?.users || []), 'text/csv;charset=utf-8');
  const exportRows = () => saveFile('row-predictions-pseudonymous.csv', rowPredictionsCsv(result?.predictions || []), 'text/csv;charset=utf-8');
  const exportSelectedRows = () => saveFile('selected-user-date-filtered-predictions.csv', rowPredictionsCsv(exportFilteredRows), 'text/csv;charset=utf-8');

  return <div className="shell">
    <header>
      <button type="button" className="brand" onClick={() => setPage(PAGES[0])}><span className="brandmark">T</span><span>Temporal NLP Research<small>Academic demonstration</small></span></button>
      <nav>{PAGES.map(item => <button key={item} className={page === item ? 'active' : ''} onClick={() => setPage(item)}>{item}</button>)}</nav>
    </header>
    <main>
      {page === PAGES[0] && <>
        <section className="hero"><span className="eyebrow">FINAL-YEAR NLP RESEARCH PROJECT</span><h1>Explore how emotional expression shifts over time.</h1><p>A research app for exploring emotion patterns and a classifier signal in dated Reddit posts.</p>
          <h2 className="section-title">How it works</h2><ol className="steps"><li><b>Upload a CSV</b><span>Add posts and, if available, user IDs and dates.</span></li><li><b>Score the language</b><span>The models estimate emotion scores and a risk-related language signal.</span></li><li><b>View temporal trends</b><span>Explore descriptive averages across recent and earlier time windows.</span></li></ol>
          <div className="hero-actions"><button className="primary" onClick={() => setPage('Analyze')}>Analyze a CSV <span>→</span></button></div>
          <p className="caveat">Research prototype. Outputs are not clinical assessments. <button className="text-link" onClick={() => setPage('Transparency')}>Read the transparency notes</button></p>
        </section>
        <section className="card findings"><h2>Patterns in the existing project data</h2><p>Average nervousness and fear scores were slightly lower in the more recent time window; the average neutral score was slightly higher.</p><p className="muted">For example: nervousness −0.0001 · fear −0.0039 · neutral +0.0070 mean change. These are descriptive cohort-level model scores, not clinical changes or individual predictions.</p></section>
        <footer>Existing project cohort: 83,516 posts from 680 users. New uploads have not yet been evaluated. <button className="text-link" onClick={() => setPage('Transparency')}>How to interpret these results</button></footer>
      </>}

      {page === 'Analyze' && <>
        <div className="pagehead"><span className="eyebrow">UPLOAD & CONFIGURE</span><h1>Analyze a small CSV</h1><p>Your upload is processed by the Python inference service. Raw text is not included in returned results.</p></div>
        <section className="card upload">
          <label className={`drop${dragging ? ' dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); chooseFile(e.dataTransfer.files?.[0]); }}><input className="file-input" type="file" accept=".csv,text/csv" aria-label="Choose a CSV file" onChange={e => chooseFile(e.target.files?.[0])} /><span className="uploadicon">↑</span><b>{file ? file.name : 'Choose or drop a CSV file'}</b><span>CSV · current prototype limit 2 MiB / 100 rows (not benchmark-validated)</span></label>
          {fields.length > 0 && <>
            <h2>Map your columns</h2><div className="grid three">
              {[['Post text', textCol, setTextCol, true], ['User ID', userCol, setUserCol, false], ['Timestamp', timeCol, setTimeCol, false]].map(([label, value, setValue, required]) => <label className="field" key={label}>{label}{required && <i> required</i>}<select value={value} onChange={e => { setValue(e.target.value); setValidation(null); }}><option value="">{required ? 'Select column' : 'Not provided'}</option>{fields.map(f => <option key={f}>{f}</option>)}</select></label>)}
            </div>
            <h2>Preview <span className="muted">First {preview.length} rows</span></h2><div className="tablewrap"><table><thead><tr>{fields.map(c => <th key={c}>{c}</th>)}</tr></thead><tbody>{preview.map((row, i) => <tr key={i}>{fields.map(c => <td key={c}>{String(row[c] ?? '').slice(0, 100)}</td>)}</tr>)}</tbody></table></div>
          </>}
          {validation && <section className="validation-summary"><h2>Validation summary</h2><div className="stats">{[['Rows', validation.summary.rows_received], ['Nonempty text', validation.summary.nonempty_text_rows], ['Empty text excluded', validation.summary.empty_text_rows], ['Duplicate rows retained', validation.summary.duplicate_rows], ['Unique user IDs', validation.summary.unique_users], ['Missing user IDs', validation.summary.missing_user_id_rows], ['Temporally eligible users', validation.summary.temporal_eligible_users]].map(([label, value]) => <div className="stat card" key={label}><span>{label}</span><b>{value}</b></div>)}</div><p>Timestamp range: {validation.summary.parsed_timestamp_min || 'Unavailable'} to {validation.summary.parsed_timestamp_max || 'Unavailable'}. Invalid/missing timestamps: {validation.summary.missing_or_invalid_timestamp_rows}; numeric timestamp strings rejected: {validation.summary.numeric_timestamp_rows}.</p>{!validation.summary.temporal_analysis_available && <p>User ID and timestamp columns are required for temporal comparison; post-level inference can still run.</p>}{validation.users.length > 0 && <div className="tablewrap"><table><thead><tr><th>Pseudonymous user</th><th>Recent posts / 5</th><th>Baseline posts / 15</th><th>Eligibility</th><th>Reason</th></tr></thead><tbody>{validation.users.map(u => <tr key={u.user_id}><td>{u.user_id}</td><td>{u.recent_posts}</td><td>{u.baseline_posts}</td><td>{u.eligible ? 'Eligible' : 'Not eligible'}</td><td>{u.reason}</td></tr>)}</tbody></table></div>}</section>}
          <div className="actions"><button className="secondary" disabled={!file || validating || busy || !textCol} onClick={validateCsv}>{validating ? 'Checking CSV…' : 'Validate CSV'}</button><button className="primary" disabled={!file || busy || validating || !textCol || !validation} onClick={analyze}>{busy ? 'Running real model inference…' : 'Run analysis'}</button><span className="muted">Inference requires both trained model artifacts on the API service.</span></div>
          {error && <div className="error"><b>Could not run analysis</b><p>{error}</p></div>}
        </section>
      </>}

      {page === 'Results' && (!result ? <section className="empty card"><h2>No analysis yet</h2><p>Upload a CSV to generate model predictions and temporal summaries.</p><button className="primary" onClick={() => setPage('Analyze')}>Go to Analyze</button></section> : <section>
        <div className="pagehead"><span className="eyebrow">ANALYSIS RESULTS</span><h1>Observed model outputs</h1><p>Run completed {new Date(result.run.completed_at).toLocaleString()} · {result.summary.rows_received} rows</p></div>
        <div className="tabs">{['Overview', 'User Timeline', 'Export Results'].map(x => <button className={tab === x ? 'selected' : ''} onClick={() => setTab(x)} key={x}>{x}</button>)}</div>
        {tab === 'Overview' && <>
          <section className="card export-callout"><div><h2>Export Results</h2><p>Download the full analysis or filter post-level results by the selected pseudonymous user and date range.</p></div><button className="primary" onClick={() => setTab('Export Results')}>Open export options</button></section>
          <div className="stats">{[['Uploaded rows', result.summary.rows_received], ['Rows processed', result.predictions.length], ['Empty-text rows excluded', result.summary.empty_text_rows], ['Users analyzed', result.summary.unique_users], ['Temporally eligible users', result.summary.temporal_eligible_users], ['Mean E2 behavior + attempt score', e2Stats.score.mean == null ? 'Unavailable' : e2Stats.score.mean.toFixed(4)]].map(([label, value]) => <div className="stat card" key={label}><span>{label}</span><b>{value ?? 'Unavailable'}</b></div>)}</div>
          <div className="notice"><b>Interpret with care</b><span>{result.run.warnings.join(' ')}</span></div>
          <DataQualityPanel quality={qualitySummary} predictions={predictions} nearConstant={nearConstantMeasures}/>
          <div className="grid two chart-grid">
            <HorizontalBars title="Predicted E2 class distribution" axisLabel="Predicted posts (bar length shows count)" rows={e2Stats.classes.map(row => ({ label: row.label, count: row.count, percentage: row.percentage, mean: row.count }))} valueKey="mean" format={value => `${value} posts`} empty="No posts were processed. This is a predicted label distribution, not the prevalence of a condition." />
            <section className="chart-card"><h3>E2 class counts and percentages</h3><div className="tablewrap"><table><thead><tr><th>Predicted class</th><th>Posts</th><th>Share</th></tr></thead><tbody>{e2Stats.classes.map(row => <tr key={row.label}><td>{row.label}</td><td>{row.count}</td><td>{row.percentage == null ? '—' : `${row.percentage.toFixed(1)}%`}</td></tr>)}</tbody></table></div><p className="muted">Predicted classes only; not actual condition prevalence.</p></section>
            <section className="chart-card"><h3>Explore score distributions</h3><label className="field">Score measure<select value={distributionMeasure} onChange={e => setDistributionMeasure(e.target.value)}>{availableDistributionMeasures.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><DistributionPanel title={`${measureName} distribution`} values={distributionRows.map(row => row.value)} rowNumbers={distributionRows.map(row => row.row_number)} label={distributionLabel} fullScale={fullScale} setFullScale={setFullScale} caption={distributionLabel.includes('E2') ? 'E2 values are an uncalibrated model score, not a clinical risk probability.' : 'GoEmotions values are independent sigmoid scores and do not sum to 100%.'}/></section>
            <section className="chart-card"><h3>Posts ranked by score</h3><p className="muted">Top and bottom five are labelled by upload row. {distributionLabel.includes('E2') ? 'E2 values are an uncalibrated model score, not a clinical risk probability.' : 'GoEmotions scores are independent sigmoid values.'}</p><RankedDots rows={distributionRows} label={distributionLabel}/></section>
            <section className="chart-card"><h3>Mean score over time</h3><label className="field">Group by<select value={trendPeriod} onChange={e => setTrendPeriod(e.target.value)}><option value="week">Week</option><option value="month">Month</option></select></label>{trendRows.length ? <><p className="muted">Mean with a shaded ±1 SD band · n={predictions.length}{predictions.length < 30 ? ' · Small sample (n &lt; 30)' : ''}. {distributionLabel.includes('E2') ? 'E2 values are an uncalibrated model score, not a clinical risk probability.' : 'GoEmotions scores are independent sigmoid values.'}</p><GroupTrendChart rows={trendRows} title={`${measureName} by ${trendPeriod}`} label={distributionLabel}/></> : <p className="empty-state">Timestamps are needed for a grouped trend. Scores remain available without dates.</p>}</section>
            <div className="emotion-profile-wrap"><label className="field">Emotion profile — show<select value={topEmotionCount} onChange={e => setTopEmotionCount(Number(e.target.value))}><option value={5}>Top 5</option><option value={10}>Top 10</option><option value={28}>All 28</option></select></label><HorizontalBars title="GoEmotions profile: mean sigmoid scores" rows={emotionStats.slice(0, topEmotionCount)} format={value => value.toFixed(4)} maxValue={1} empty="No emotion scores are available."/><p className="muted">n={predictions.length}. Scores are independent sigmoid values; multiple emotions can score highly at once and they do not form a probability distribution.</p></div>
          </div>
          <section className="card"><h2>Mean score by pseudonymous user</h2><p className="muted">Each user mean includes its post count; users with fewer than 5 posts are marked low-count. n={predictions.length}{predictions.length < 30 ? ' · Small sample (n &lt; 30)' : ''}. {distributionLabel.includes('E2') ? 'E2 values are an uncalibrated model score, not a clinical risk probability.' : 'GoEmotions scores are independent sigmoid values.'}</p><HorizontalBars title={`${measureName} by user`} rows={userMeanRows} axisLabel={distributionLabel} format={value => value.toFixed(4)} empty="User IDs are unavailable. Post-level scores remain available."/></section>
          <section className="card deviation-section"><h2>Deviation from baseline</h2><p>Each bar is a post’s score minus the selected baseline. The default baseline is the cohort median. Positive values are above baseline; negative values are below.</p><div className="deviation-controls"><label className="field">Baseline<select value={baselineMode} onChange={e => setBaselineMode(e.target.value)}><option value="median">Cohort median</option><option value="mean">Cohort mean</option><option value="user_recent" disabled={distributionMeasure.startsWith('e2_probability_') || !currentUser}>Selected user’s recent [0, 90) day mean</option></select></label>{baselineMode === 'user_recent' && <label className="field">User<select value={selected} onChange={e => setSelected(e.target.value)}>{result.users.map(user => <option key={user.user_id} value={user.user_id}>{user.user_id}</option>)}</select></label>}</div><p className="muted">{distributionLabel.includes('E2') ? 'E2 values are an uncalibrated model score, not a clinical risk probability.' : 'GoEmotions values are independent sigmoid scores; they do not sum to one.'}</p><DeviationPlot rows={deviationRows} baseline={deviationBaseline} baselineLabel={baselineLabel}/><h3>What co-occurs with flagged deviations?</h3><p className="muted">Feature values and correlations are descriptive associations across this upload; they do not establish cause. Raw post text is not shown.</p><DeviationDrivers rows={outlierRows} predictions={predictions} features={localFeatures} associations={associations} emotionStats={emotionStats} flaggedRow={flaggedRow} setFlaggedRow={setFlaggedRow}/></section>
          <section className="card"><h2>All 28 GoEmotions mean scores</h2><p className="muted">Exact means derived from the returned per-post scores.</p><div className="tablewrap"><table><thead><tr><th>Emotion</th><th>Mean sigmoid score</th><th>Posts with value</th></tr></thead><tbody>{emotionStats.map(row => <tr key={row.label}><td>{row.label}</td><td>{row.mean?.toFixed(4) ?? '—'}</td><td>{row.count}</td></tr>)}</tbody></table></div></section>
          <section className="card post-table-card"><h2>Post predictions <span className="muted">Raw text is not shown</span></h2><p className="muted">E2 score is P(behavior) + P(attempt), an uncalibrated model output.</p><div className="table-filters"><label className="field">Search pseudonymous user ID<input value={postUserSearch} onChange={e => { setPostUserSearch(e.target.value); setPostPage(0); }} placeholder="user_…"/></label><label className="field">Sort by<select value={postSort} onChange={e => setPostSort(e.target.value)}><option value="row_number">Upload row</option><option value="timestamp">Timestamp</option><option value="score">E2 score</option></select></label></div>
            {!filteredTableRows.length ? <div className="empty-state">No posts match these filters.</div> : <><div className="tablewrap"><table><thead><tr><th>Row</th><th>Pseudonymous user</th><th>Timestamp</th><th>E2 score · uncalibrated</th><th>Six E2 probabilities</th><th>Highest GoEmotions score</th></tr></thead><tbody>{visibleTableRows.map(p => { const top = Object.entries(p.goemotions_scores || {}).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]; return <tr key={p.row_number}><td>{p.row_number}</td><td>{p.user_id || 'Unavailable'}</td><td>{p.timestamp || 'Unavailable'}</td><td>{Number.isFinite(p.e2_score_uncalibrated) ? p.e2_score_uncalibrated.toFixed(4) : '—'}</td><td><details><summary>View scores</summary><ul>{E2_CLASSES.map(label => <li key={label}>{label}: {p.e2_class_probabilities?.[label]?.toFixed(4) ?? '—'}</li>)}</ul></details></td><td>{top ? `${top[0]} · ${top[1].toFixed(3)}` : '—'}</td></tr>; })}</tbody></table></div><div className="pagination"><span>{filteredTableRows.length} matching posts</span><button className="secondary" disabled={postPage === 0} onClick={() => setPostPage(n => Math.max(0, n - 1))}>Previous</button><span>Page {postPage + 1} of {Math.max(1, Math.ceil(filteredTableRows.length / pageSize))}</span><button className="secondary" disabled={(postPage + 1) * pageSize >= filteredTableRows.length} onClick={() => setPostPage(n => n + 1)}>Next</button></div></>}
          </section>
        </>}
        {tab === 'User Timeline' && <section className="card"><h2>User timeline and temporal comparison</h2>{result.users.length === 0 ? <p>User IDs and timestamps are needed for temporal analysis. Post-level inference is still available.</p> : <>
          <label className="field">Pseudonymous user<select value={selected} onChange={e => setSelected(e.target.value)}>{result.users.map(u => <option key={u.user_id} value={u.user_id}>{u.user_id}</option>)}</select></label>
          <div className="stats compact"><div className="stat"><span>Recent posts / 5 required</span><b>{currentUser?.recent_posts}</b></div><div className="stat"><span>Baseline posts / 15 required</span><b>{currentUser?.baseline_posts}</b></div><div className="stat"><span>Eligibility</span><b>{currentUser?.eligible ? 'Eligible' : 'Not eligible'}</b></div></div>
          <p>{currentUser?.reason}</p>
          <label className="field measure-select">Timeline measure<select value={timelineMeasure} onChange={e => setTimelineMeasure(e.target.value)}><option value="e2_score_uncalibrated">E2 behavior + attempt score</option>{E2_CLASSES.map(label => <option key={`e2-${label}`} value={`e2_probability_${label}`}>E2 probability · {label}</option>)}{EMOTIONS.map(label => <option key={`emo-${label}`} value={label}>GoEmotions · {label}</option>)}</select></label>
          {!currentUser?.eligible && <p className="warning">This user does not meet both temporal post-count thresholds. The chart is post-level and exploratory; no recent/baseline difference is reported.</p>}
          {!timeline.length ? <p className="empty-state">No dated predictions are available for this measure and user.</p> : timeline.length < 2 ? <p className="empty-state">Only one dated post is available. There are not enough observations to show a trend.</p> : <><div className="chart"><TimelineChart key={`${currentUser.user_id}-${timelineMeasure}`} data={timeline} measureLabel={timelineMeasure === 'e2_score_uncalibrated' ? 'E2 behavior + attempt score (uncalibrated)' : timelineMeasure.replaceAll('_', ' ')}/></div><p className="muted">n={timeline.length}{timeline.length < 30 ? ' · Small sample (n &lt; 30)' : ''}. {timelineMeasure.startsWith('e2_') ? 'E2 values are an uncalibrated model score, not a clinical risk probability. ' : 'GoEmotions scores are independent sigmoid values and do not sum to 100%. '}Points are not connected; gaps mean no uploaded posts. The sliders change the chart view only.</p></>}
          <h3>Recent versus baseline</h3><p className="muted">Recent window: [0, 90) days ({selectedTemporal?.recentPosts ?? 0} posts). Baseline window: [90, 270) days ({selectedTemporal?.baselinePosts ?? 0} posts). Differences are shown only for eligible users with sufficient observations.</p>
          {selectedTemporal?.eligible ? <><ComparisonBars rows={[{ label: 'E2 behavior + attempt', ...selectedTemporal.e2 }, ...temporalEmotionRows]}/><p className="muted">Recent n={selectedTemporal.recentPosts}; baseline n={selectedTemporal.baselinePosts}. Descriptive model-score comparisons only. They do not show that an emotion caused a change in mental health.</p></> : <p className="empty-state">No temporal comparison is reported because this user is ineligible or one/both window means are unavailable.</p>}
          <h3>All emotion comparisons</h3><div className="tablewrap"><table><thead><tr><th>Emotion</th><th>Recent mean</th><th>Baseline mean</th><th>Difference</th></tr></thead><tbody>{selectedTemporal?.emotions.map(row => <tr key={row.label}><td>{row.label}</td><td>{row.recent?.toFixed(4) ?? '—'}</td><td>{row.baseline?.toFixed(4) ?? '—'}</td><td>{selectedTemporal.eligible ? row.difference?.toFixed(4) ?? '—' : 'Not eligible'}</td></tr>)}</tbody></table></div>
        </>}</section>}
        {tab === 'Export Results' && <section className="card"><h2>Export Results</h2><p>Exports omit post text and original identifiers. Row-level files contain pseudonymous IDs and model outputs, so handle them carefully.</p><div className="export-group"><h3>Full analysis</h3><p>Includes all processed posts and all users from this run.</p><div className="actions"><button className="secondary" onClick={exportRows}>All post predictions · CSV</button><button className="secondary" onClick={exportUsers}>All temporal summaries · CSV</button><button className="secondary" onClick={() => saveFile('research-analysis-summary.json', summaryReportJson(result), 'application/json;charset=utf-8')}>Research summary report · JSON</button><button className="secondary" onClick={() => saveFile('run-metadata.json', runMetadataJson(result.run, result.summary), 'application/json;charset=utf-8')}>Run metadata · JSON</button></div></div><div className="export-group"><h3>Current user and date-filtered posts</h3>{result.users.length ? <label className="field">Pseudonymous user<select value={selected} onChange={e => setSelected(e.target.value)}>{result.users.map(u => <option key={u.user_id} value={u.user_id}>{u.user_id}</option>)}</select></label> : <p>User IDs are unavailable; post-level export can still be filtered by date.</p>}<div className="date-filters"><label className="field">From date<input type="date" value={exportStart} onChange={e => setExportStart(e.target.value)}/></label><label className="field">Through date<input type="date" value={exportEnd} onChange={e => setExportEnd(e.target.value)}/></label></div><p>{exportFilteredRows.length} posts match the current selection. Rows without timestamps are excluded when a date filter is set.</p><div className="actions"><button className="secondary" onClick={exportSelectedRows}>Selected user/date posts · CSV</button><button className="secondary" disabled={!currentUser} onClick={() => saveFile('selected-user-temporal-summary.csv', userSummaryCsv(currentUser ? [currentUser] : []), 'text/csv;charset=utf-8')}>Selected user temporal summary · CSV</button></div></div></section>}
      </section>)}

      {page === PAGES[3] && <>
        <div className="pagehead" id="transparency"><span className="eyebrow">TRANSPARENCY</span><h1>What has and hasn’t been validated</h1><p>The project reports historical model results and descriptive patterns in an existing dataset; those metrics have not been independently reproduced against the original test split, and new uploads have not yet been evaluated.</p></div>
        <div className="grid two">
          <section className="card"><h2>Existing project dataset</h2><p>The repository records a prepared cohort of 83,516 posts from 680 users; a notebook audit reports 258 eligible users and 16,639 posts in the temporal cohort. These describe the existing research dataset, not predictions for your upload.</p><p>Saved aggregate summaries list six exploratory emotion associations. Their schemas were inspected; they do not contain row-level predictions or establish causal or clinical effects.</p></section>
          <section className="card"><h2>Held-out evaluation</h2><p>The E2 held-out test used 443 examples and reported accuracy 0.6388, macro-F1 0.5079, and weighted-F1 0.6231. These results have not been independently reproduced against the original test split.</p><p>Current upload evaluation: <b>Not yet evaluated.</b></p></section>
          <section className="card"><h2>What the scores mean</h2><ul><li>E2 scores are uncalibrated classifier outputs, not clinical risk probabilities, diagnoses, or predictions of future behavior.</li><li>GoEmotions scores are independent sigmoid values, not a distribution summing to one.</li><li>Inputs are truncated at 512 tokens, so longer posts may be scored using only part of their text.</li></ul></section>
          <section className="card"><h2>How to read mean change</h2><p>Mean change is the difference between an emotion’s average score in the recent window (0–90 days before a user’s latest dated post) and the earlier window (90–270 days before it). A positive value means the average score was higher in the recent window; a negative value means it was lower.</p><p>These temporal summaries are descriptive. They do not establish causation or a clinical change. The project has no clinical validation, and generalization to arbitrary uploads has not been established.</p></section>
        </div>
        <section className="card"><h2>Limits of these visualizations</h2><ul><li>Small or uneven uploads can make distributions, trends, per-user means, and outlier flags unstable; charts warn when fewer than 30 posts are available.</li><li>Feature correlations and text summaries are exploratory associations. They do not establish what caused a score or explain an individual person.</li><li>Adaptive axes improve visibility of small score differences but can visually magnify them; use the full scale and the reported summary statistics for context.</li></ul></section>
        <section className="card"><h2>Aggregate emotion summary</h2><p className="table-note">How to read this: mean change compares average model scores in recent and earlier windows for the existing project cohort only. Negative values are lower in the recent window; positive values are higher. This table is not generated from your upload.</p><div className="tablewrap"><table><thead><tr><th>Emotion</th><th>Mean change</th><th>Median change</th><th>Users with increase</th><th>Users with decrease</th></tr></thead><tbody>{SAVED_EMOTION_SUMMARY.map(row => <tr key={row.emotion}><td>{row.emotion}</td><td>{row.mean.toFixed(4)}</td><td>{row.median.toFixed(4)}</td><td>{row.increase}</td><td>{row.decrease}</td></tr>)}</tbody></table></div><p>The saved summary contains means, medians, and increase/decrease counts, but no p-values or confidence intervals. The notebook also records statistical tests and adjustments; those historical results have not been recomputed from your upload.</p></section>
        <details className="card researcher-details"><summary>For researchers: reproducibility gaps</summary><ul><li>The reported E2 metrics have not been independently reproduced against the original held-out split.</li><li>The original training data and complete checkpoint provenance are unavailable here.</li><li>The original cohort filtering script and source dataset are absent.</li><li>Exact original text construction is only partially documented; arbitrary user text may differ.</li><li>GoEmotions is revision-pinned, but first model load requires network access or a local cache.</li></ul></details>
        <footer>Current uploaded CSVs are analyzed only when the Python API and both model artifacts are available.</footer>
      </>}
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
