'use client';

import React from 'react';
import { Layers, Scissors, Box, Cpu } from 'lucide-react';
import type { ReportData } from '@/lib/utils';
import { params, mb, ms } from '@/lib/utils';

interface Props {
  reports: ReportData[];
}

export function ModelSection({ reports }: Props) {
  const find = (n: string) => reports.find((r) => r.metadata.model_name === n);
  const baseline = find('baseline_model');
  const quantized = find('quantized_int8');
  const pruned70 = find('pruned_model_70');
  const final = find('pruned_quantized');

  const stages = [
    {
      icon: Layers,
      color: 'bg-blue-100 text-blue-600',
      title: '1 · Baseline 1D CNN',
      body: 'A fully-connected 1D convolutional network trained on raw float32 arrays. Spatial convolutions learn morphological features like QRS widening and T-wave inversion directly from the signal.',
      metric: baseline
        ? `${params(baseline.complexity_analysis.parameters.total_parameters)} params · ${mb(baseline.complexity_analysis.memory_size.fp32_mb)} (FP32)`
        : '—',
    },
    {
      icon: Box,
      color: 'bg-sky-100 text-sky-600',
      title: '2 · INT8 Quantization (PTQ)',
      body: 'Weights and activations are converted from 32-bit floats to 8-bit integers via post-training quantization, shrinking the model ~4× and slashing energy — usually with <1% accuracy loss.',
      metric: quantized
        ? `${mb(quantized.complexity_analysis.memory_size.int8_mb)} (INT8) · ${quantized.complexity_analysis.memory_size.compression_ratio_fp32_to_int8.toFixed(1)}× smaller`
        : '—',
    },
    {
      icon: Scissors,
      color: 'bg-purple-100 text-purple-600',
      title: '3 · Magnitude Pruning',
      body: 'Near-zero weights are pruned to introduce sparsity (up to 70%), cutting the multiply-accumulate operations required per beat while preserving the learned features.',
      metric: pruned70
        ? `${params(pruned70.complexity_analysis.parameters.total_parameters)} params · ${Math.round((1 - pruned70.complexity_analysis.parameters.total_parameters / (baseline?.complexity_analysis.parameters.total_parameters ?? 1)) * 100)}% sparsity`
        : '—',
    },
    {
      icon: Cpu,
      color: 'bg-emerald-100 text-emerald-600',
      title: '4 · TFLite Micro Deployment',
      body: 'The optimized model is converted to a C byte array and flashed to the microcontroller along with the TensorFlow Lite for Microcontrollers runtime.',
      metric: final
        ? `${mb(final.complexity_analysis.memory_size.int8_mb)} on-device · ${ms(final.summary.latency_performance.single_sample_latency_ms)} latency`
        : '—',
    },
  ];

  const howMuch = final
    ? {
        size: (1 - final.complexity_analysis.memory_size.int8_mb / (baseline?.complexity_analysis.memory_size.fp32_mb ?? 1)) * 100,
        params: (1 - final.complexity_analysis.parameters.total_parameters / (baseline?.complexity_analysis.parameters.total_parameters ?? 1)) * 100,
        latency: (1 - final.summary.latency_performance.single_sample_latency_ms / (baseline?.summary.latency_performance.single_sample_latency_ms ?? 1)) * 100,
      }
    : { size: 0, params: 0, latency: 0 };

  return (
    <div className="space-y-14 pb-16">
      <div className="mx-auto max-w-3xl text-center">
        <p className="text-sm font-semibold uppercase tracking-wider text-rose-500">From Python to silicon</p>
        <h1 className="mt-4 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">The Beat2Bit pipeline</h1>
        <p className="mt-4 text-lg leading-relaxed text-slate-600">
          Four stages take a full-precision deep network to a deployable, ultra-low-power edge model — with measurable
          gains at every step.
        </p>
      </div>

      {/* Impact stats */}
      <div className="grid gap-3 sm:grid-cols-3">
        <ImpactStat label="Model size reduction" value={`${Math.max(0, howMuch.size).toFixed(0)}%`} sub="vs FP32 baseline" />
        <ImpactStat label="Parameter reduction" value={`${Math.max(0, howMuch.params).toFixed(0)}%`} sub="via magnitude pruning" />
        <ImpactStat label="Latency reduction" value={`${Math.max(0, howMuch.latency).toFixed(0)}%`} sub="faster per inference" />
      </div>

      {/* Timeline */}
      <div className="relative mx-auto max-w-4xl">
        <div className="absolute left-6 top-0 bottom-0 w-px bg-gradient-to-b from-blue-200 via-purple-200 to-emerald-200" />
        <div className="space-y-8">
          {stages.map((s, i) => (
            <div key={i} className="relative flex gap-5 pl-0">
              <div className={`z-10 grid h-12 w-12 shrink-0 place-items-center rounded-2xl shadow-sm ${s.color}`}>
                <s.icon className="h-6 w-6" />
              </div>
              <div className="flex-1 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <h3 className="text-lg font-semibold text-slate-900">{s.title}</h3>
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">{s.metric}</span>
                </div>
                <p className="mt-3 leading-relaxed text-slate-600">{s.body}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ImpactStat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 text-center shadow-sm">
      <p className="text-sm text-slate-500">{label}</p>
      <p className="mt-2 text-3xl font-bold text-emerald-600">{value}</p>
      <p className="mt-1 text-xs text-slate-400">{sub}</p>
    </div>
  );
}
