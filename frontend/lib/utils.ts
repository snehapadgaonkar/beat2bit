import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

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