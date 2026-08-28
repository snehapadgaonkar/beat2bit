import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Strongly-typed shape of a benchmark report as emitted by the backend
 * `report_generator.py`. Every field mirrors the JSON written by
 * `src/benchmarking/report_generator.py`.
 */
export interface ReportData {
  metadata: {
    model_name: string;
    timestamp: string;
    report_version: string;
    generator: string;
  };
  dataset_info: {
    dataset: string;
    samples: number;
    features: number;
    classes: string[];
    split: string;
  };
  model_evaluation: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    ami_sensitivity: number;
    ami_positive_predictivity: number;
    ami_effectiveness: number;
    specificity: number;
    confusion_matrix: {
      tn: number;
      fp: number;
      fn: number;
      tp: number;
    };
  };
  complexity_analysis: {
    parameters: {
      total_parameters: number;
      trainable_parameters: number;
      non_trainable_parameters: number;
    };
    memory_size: {
      fp32_mb: number;
      int8_mb: number;
      compression_ratio_fp32_to_int8: number;
    };
    computational_complexity: {
      total_flops: number;
      gflops: number;
      mflops: number;
    };
  };
  latency_benchmarking: {
    batch_sizes: number[];
    latency_stats: {
      [key: string]: {
        mean_latency_ms: number;
        median_latency_ms: number;
        std_latency_ms: number;
        min_latency_ms: number;
        max_latency_ms: number;
        p95_latency_ms: number;
        p99_latency_ms: number;
        n_measurements: number;
      };
    };
    throughput_stats: {
      [key: string]: {
        throughput_samples_per_sec: number;
        latency_per_sample_ms: number;
      };
    };
    summary: {
      optimal_batch_size_for_latency: number;
      optimal_batch_size_for_throughput: number;
      latency_range_ms: {
        min: number;
        max: number;
      };
      throughput_range_samples_per_sec: {
        min: number;
        max: number;
      };
    };
  };
  summary: {
    model_performance: {
      accuracy: number;
      f1_score: number;
      ami_sensitivity: number;
      ami_positive_predictivity: number;
      ami_effectiveness: number;
    };
    model_complexity: {
      total_parameters: number;
      model_size_fp32_mb: number;
      model_size_int8_mb: number;
      compression_ratio: number;
      total_flops: number;
      gflops: number;
    };
    latency_performance: {
      single_sample_latency_ms: number;
      latency_p95_ms: number;
      throughput_samples_per_sec: number;
    };
  };
  recommendations: string[];
}

/* ------------------------------------------------------------------ */
/* Formatting helpers                                                  */
/* ------------------------------------------------------------------ */

/** Format a 0–1 ratio as a percentage string (e.g. 0.87 -> "87.0%"). */
export function pct(value: number, digits = 1): string {
  if (Number.isNaN(value) || value == null) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

/** Format milliseconds, adapting the decimals to the magnitude. */
export function ms(value: number): string {
  if (value == null || Number.isNaN(value)) return '—';
  if (value >= 100) return `${value.toFixed(0)} ms`;
  if (value >= 10) return `${value.toFixed(1)} ms`;
  return `${value.toFixed(2)} ms`;
}

/** Format a byte-count in megabytes with adaptive precision. */
export function mb(value: number): string {
  if (value == null || Number.isNaN(value)) return '—';
  if (value >= 1) return `${value.toFixed(2)} MB`;
  if (value >= 0.1) return `${value.toFixed(2)} MB`;
  return `${(value * 1024).toFixed(0)} KB`;
}

/** Format a large integer with thousands separators. */
export function int(value: number): string {
  if (value == null || Number.isNaN(value)) return '—';
  return value.toLocaleString('en-US');
}

/** Format a possibly-fractional count of parameters. */
export function params(value: number): string {
  if (value == null || Number.isNaN(value)) return '—';
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toFixed(0);
}

/** Format a FLOP count into GFLOPS / MFLOPS / raw. */
export function flops(value: number): string {
  if (value == null || Number.isNaN(value)) return '—';
  const g = value / 1e9;
  // Only use GFLOPs when the value won't round to "0.00" at 2 decimals
  // (e.g. 2.84 MFLOPs = 0.00284 GFLOPs would show as "0.00 GFLOPs" — misleading)
  if (g >= 0.01) return `${g.toFixed(2)} GFLOPs`;
  const m = value / 1e6;
  if (m >= 0.1) return `${m.toFixed(2)} MFLOPs`;
  return `${Math.round(value).toLocaleString('en-US')} FLOPs`;
}

/** Compact latency percentile labels for charts. */
export const LATENCY_STAT_KEYS: Array<{ key: keyof import('./utils').ReportData['latency_benchmarking']['latency_stats'][string]; label: string; color: string }> = [
  { key: 'min_latency_ms', label: 'min', color: '#94a3b8' },
  { key: 'p95_latency_ms', label: 'p50 / mean', color: '#0ea5e9' },
  { key: 'mean_latency_ms', label: 'mean', color: '#e11d48' },
  { key: 'p95_latency_ms', label: 'p95', color: '#f59e0b' },
  { key: 'p99_latency_ms', label: 'p99', color: '#8b5cf6' },
  { key: 'max_latency_ms', label: 'max', color: '#64748b' },
];

export const MODEL_COLORS = {
  baseline_model: '#64748b',
  quantized_int8: '#e11d48',
  pruned_model_50: '#f59e0b',
  pruned_model_70: '#8b5cf6',
  pruned_quantized: '#10b981',
};
