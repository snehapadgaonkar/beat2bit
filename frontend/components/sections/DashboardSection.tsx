'use client';

import React from 'react';
import {
  Activity,
  Timer,
  Layers,
  Cpu,
  Zap,
  Gauge,
  Boxes,
  ArrowUpRight,
  Waves,
} from 'lucide-react';
import type { ReportData } from '@/lib/utils';
import { pct, ms, mb, params, flops, int, MODEL_COLORS } from '@/lib/utils';
import { REPORT_ORDER } from '@/lib/researchService';
import { ECGChart } from '@/components/ECGChart';
import { LatencyDistributionChart, ThroughputChart } from '@/components/charts/LatencyChart';
import { cn } from '@/lib/utils';

interface Props {
  reports: ReportData[];
  loading: boolean;
  selectedName: string;
  onSelectName: (name: string) => void;
}

export function DashboardSection({ reports, loading, selectedName, onSelectName }: Props) {
  const selected = reports.find((r) => r.metadata.model_name === selectedName) ?? reports[0];
  const meta = (name: string) => REPORT_ORDER.find((m) => m.name === name);

  if (!selected) return <div className="text-slate-400">Loading telemetry…</div>;

  const e = selected.model_evaluation;
  const c = selected.complexity_analysis;
  const l = selected.summary.latency_performance;
  const ls = selected.latency_benchmarking;
  const best = REPORT_ORDER.find((m) => m.name === selected.metadata.model_name);

  const throughputData = reports.map((r) => ({
    name: meta(r.metadata.model_name)?.label ?? r.metadata.model_name,
    throughput: r.summary.latency_performance.throughput_samples_per_sec,
  }));

  const kpis = [
    { label: 'Inference Latency', value: ms(l.single_sample_latency_ms), sub: 'mean · batch=1', icon: Timer, color: 'text-sky-500' },
    { label: 'Latency p95', value: ms(l.latency_p95_ms), sub: 'tail latency', icon: Gauge, color: 'text-amber-500' },
    { label: 'Throughput', value: `${int(l.throughput_samples_per_sec)}/s`, sub: 'single-sample', icon: Activity, color: 'text-emerald-500' },
    { label: 'Model Size (INT8)', value: mb(c.memory_size.int8_mb), sub: `FP32 ${mb(c.memory_size.fp32_mb)}`, icon: Boxes, color: 'text-purple-500' },
    { label: 'Parameters', value: params(c.parameters.total_parameters), sub: 'trainable', icon: Layers, color: 'text-rose-500' },
    { label: 'FLOPs', value: flops(c.computational_complexity.total_flops), sub: 'per inference', icon: Zap, color: 'text-orange-500' },
    { label: 'Compression', value: `${c.memory_size.compression_ratio_fp32_to_int8.toFixed(1)}×`, sub: 'FP32 → INT8', icon: Gauge, color: 'text-blue-500' },
    { label: 'Accuracy', value: pct(e.accuracy), sub: 'AAMI split', icon: Cpu, color: 'text-slate-900' },
  ];

  return (
    <div className="space-y-8 pb-16">
      {/* Header + device selector */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wider text-rose-500">Live Telemetry</p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">Edge device dashboard</h1>
          <p className="mt-2 max-w-xl text-slate-600">
            Real-time inference metrics from the edge microcontroller running the selected Beat2Bit model.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {reports.map((r) => {
            const m = meta(r.metadata.model_name);
            if (!m) return null;
            const active = r.metadata.model_name === selected.metadata.model_name;
            return (
              <button
                key={r.metadata.model_name}
                onClick={() => onSelectName(r.metadata.model_name)}
                className={cn(
                  'flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition-all',
                  active ? 'border-rose-200 bg-rose-50 text-rose-700 ring-1 ring-rose-200' : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                )}
              >
                <span className="h-2 w-2 rounded-full" style={{ background: MODEL_COLORS[r.metadata.model_name as keyof typeof MODEL_COLORS] ?? '#e11d48' }} />
                {m.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* KPI grid */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {kpis.map((k, i) => (
          <div key={i} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition-shadow hover:shadow-md">
            <div className="flex items-center gap-2 text-slate-500">
              <k.icon className={cn('h-4 w-4', k.color)} />
              <span className="text-xs font-medium uppercase tracking-wide">{k.label}</span>
            </div>
            <div className="mt-3 text-2xl font-bold text-slate-900">{loading ? '—' : k.value}</div>
            <div className="mt-1 text-xs text-slate-400">{k.sub}</div>
          </div>
        ))}
      </div>

      {/* ECG monitor */}
      <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="flex items-center gap-2 font-semibold text-slate-900">
              <Waves className="h-4 w-4 text-rose-500" /> Live ECG Monitor
            </h2>
            <p className="text-sm text-slate-500">Streaming waveform · {selected.dataset_info.samples.toLocaleString()} sample corpus</p>
          </div>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-600">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" /> Streaming
          </span>
        </div>
        <ECGChart />
      </div>

      {/* Latency + throughput */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <h3 className="mb-1 flex items-center gap-2 font-semibold text-slate-900">
            <Timer className="h-4 w-4 text-sky-500" /> Latency Distribution
          </h3>
          <p className="mb-4 text-sm text-slate-500">{best?.label} · batch=1 · {ls.latency_stats.batch_size_1?.n_measurements ?? '—'} measurements</p>
          <LatencyDistributionChart report={selected} />
        </div>

        <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <h3 className="mb-1 flex items-center gap-2 font-semibold text-slate-900">
            <ArrowUpRight className="h-4 w-4 text-emerald-500" /> Throughput Comparison
          </h3>
          <p className="mb-4 text-sm text-slate-500">Single-sample throughput across the optimization ladder.</p>
          <ThroughputChart data={throughputData} />
        </div>
      </div>

      {/* Batch analysis */}
      <div className="rounded-3xl border border-slate-200 bg-gradient-to-br from-slate-900 to-slate-800 p-6 text-white shadow-sm">
        <h3 className="font-semibold">Batch sizing analysis</h3>
        <p className="mt-1 text-sm text-slate-300">Optimal configuration detected by the backend latency benchmarker.</p>
        <div className="mt-5 grid gap-3 sm:grid-cols-3">
          <BatchStat label="Best batch for latency" value={`${ls.summary.optimal_batch_size_for_latency}`} sub="lowest per-sample latency" />
          <BatchStat label="Best batch for throughput" value={`${ls.summary.optimal_batch_size_for_throughput}`} sub="max samples per second" />
          <BatchStat
            label="Throughput range"
            value={`${int(ls.summary.throughput_range_samples_per_sec.min)} – ${int(ls.summary.throughput_range_samples_per_sec.max)}`}
            sub="samples / second"
          />
        </div>
      </div>
    </div>
  );
}

function BatchStat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
      <p className="text-xs uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-2 text-2xl font-bold">{value}</p>
      <p className="mt-1 text-xs text-slate-400">{sub}</p>
    </div>
  );
}
