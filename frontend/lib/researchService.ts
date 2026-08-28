import type { ReportData } from './utils';

/**
 * Metadata describing each model variant in the optimization ladder.
 * Keep this list in sync with the JSON files in /public/reports.
 */
export interface ReportMeta {
  /** Inner JSON filename (without extension) used to fetch the report. */
  name: string;
  /** Human-readable variant label. */
  label: string;
  /** Short tag shown as a badge. */
  tag: string;
  /** One-line pitch shown under the name. */
  description: string;
  /** Position in the optimization ladder. */
  step: number;
}

export const REPORT_ORDER: ReportMeta[] = [
  {
    name: 'baseline_model',
    label: 'Baseline 1D-CNN',
    tag: 'FP32',
    description: 'Original 12 K-parameter convolutional classifier trained on raw float32 arrays.',
    step: 0,
  },
  {
    name: 'quantized_int8',
    label: 'INT8 Quantized',
    tag: 'PTQ',
    description: 'Post-training quantization of the baseline to 8-bit integer weights and activations.',
    step: 1,
  },
  {
    name: 'pruned_model_50',
    label: 'Pruned 50%',
    tag: 'Pruning',
    description: 'Magnitude pruning that removes the 50% least-significant weights from the baseline.',
    step: 2,
  },
  {
    name: 'pruned_model_70',
    label: 'Pruned 70%',
    tag: 'Pruning',
    description: 'Aggressive 70% sparse variant for heavily resource-constrained edge targets.',
    step: 3,
  },
  {
    name: 'pruned_quantized',
    label: 'Pruned + Quantized',
    tag: 'Final',
    description: 'The deployed edge model: 70% pruned, then INT8 quantized and flashed to the MCU.',
    step: 4,
  },
];

/** Key performance metrics available on every report. */
export type ReportMetricKey =
  | 'accuracy'
  | 'precision'
  | 'recall'
  | 'f1_score'
  | 'ami_sensitivity'
  | 'ami_positive_predictivity'
  | 'ami_effectiveness'
  | 'specificity';

export interface MetricDescriptor {
  key: ReportMetricKey;
  label: string;
  short: string;
  description: string;
  group: 'Classification' | 'AAMI EC57';
}

export const METRIC_CATALOG: MetricDescriptor[] = [
  {
    key: 'accuracy',
    label: 'Accuracy',
    short: 'Acc',
    description: 'Proportion of all predictions that are correct.',
    group: 'Classification',
  },
  {
    key: 'precision',
    label: 'Precision',
    short: 'Prec',
    description: 'Of the beats flagged as arrhythmia, how many truly are.',
    group: 'Classification',
  },
  {
    key: 'recall',
    label: 'Recall / Sensitivity',
    short: 'Rec',
    description: 'Of the true arrhythmias, how many the model catches.',
    group: 'Classification',
  },
  {
    key: 'specificity',
    label: 'Specificity',
    short: 'Spec',
    description: 'Of the normal beats, how many the model correctly rejects.',
    group: 'Classification',
  },
  {
    key: 'f1_score',
    label: 'F1 Score',
    short: 'F1',
    description: 'Harmonic mean of precision and recall.',
    group: 'Classification',
  },
  {
    key: 'ami_sensitivity',
    label: 'AAMI Sensitivity (Se)',
    short: 'Se',
    description: 'AAMI EC57: TP / (TP + FN) — arrhythmia detection power.',
    group: 'AAMI EC57',
  },
  {
    key: 'ami_positive_predictivity',
    label: 'AAMI +Predictivity (+P)',
    short: '+P',
    description: 'AAMI EC57: TP / (TP + FP) — reliability of a positive alert.',
    group: 'AAMI EC57',
  },
  {
    key: 'ami_effectiveness',
    label: 'AAMI Effectiveness',
    short: 'Eff',
    description: 'AAMI EC57: geometric mean of Se and +P.',
    group: 'AAMI EC57',
  },
];

/**
 * Returns the ordered list of model variants that have real report data
 * available under /public/reports. Report names are the source of truth.
 */
export async function getReportList(): Promise<ReportMeta[]> {
  return REPORT_ORDER;
}

/**
 * Fetches a single benchmark report by its base name from the public
 * directory. If the report does not exist, it returns null rather than
 * fabricating randomized data — the UI handles missing reports gracefully.
 */
export async function getReportByName(reportName: string): Promise<ReportData | null> {
  try {
    const res = await fetch(`/reports/${reportName}.json`, { cache: 'no-store' });
    if (!res.ok) return null;
    return (await res.json()) as ReportData;
  } catch (error) {
    console.error(`Error fetching report ${reportName}:`, error);
    return null;
  }
}

/**
 * Loads every available report in ladder order, skipping any whose JSON
 * file is missing.
 */
export async function loadAllReports(): Promise<ReportData[]> {
  const metas = await getReportList();
  const reports = await Promise.all(metas.map((m) => getReportByName(m.name)));
  return reports.filter((r): r is ReportData => r !== null);
}
