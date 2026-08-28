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
    <div className="space-y-10 pb-12 sm:space-y-14 sm:pb-16">
      {/* ── Intro ────────────────────────────────────────────────────────────── */}
      <div className="w-full text-center">
        <p className="text-sm font-semibold uppercase tracking-wider text-rose-500">From Python to silicon</p>
        <h1 className="mt-4 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl sm:text-4xl">
          The Beat2Bit pipeline
        </h1>
        <p className="mt-4 text-base leading-relaxed text-slate-600 sm:text-lg">
          Four stages take a full-precision deep network to a deployable, ultra-low-power edge model — with measurable
          gains at every step.
        </p>
      </div>

      {/* ── Impact stats — 1 col → 3 col ────────────────────────────────────── */}
      <div className="grid gap-3 sm:grid-cols-3">
        <ImpactStat label="Model size reduction" value={`${Math.max(0, howMuch.size).toFixed(0)}%`} sub="vs FP32 baseline" />
        <ImpactStat label="Parameter reduction" value={`${Math.max(0, howMuch.params).toFixed(0)}%`} sub="via magnitude pruning" />
        <ImpactStat label="Latency reduction" value={`${Math.max(0, howMuch.latency).toFixed(0)}%`} sub="faster per inference" />
      </div>

      {/* ── Vertical timeline ────────────────────────────────────────────────── */}
      <div className="relative w-full mx-auto">
        {/* Connector line — hidden on very small screens to avoid overlap */}
        <div className="absolute bottom-0 left-5 top-0 hidden w-px bg-gradient-to-b from-blue-200 via-purple-200 to-emerald-200 sm:block sm:left-6" />
        <div className="space-y-5 sm:space-y-8">
          {stages.map((s, i) => (
            <div key={i} className="relative flex gap-3 sm:gap-5">
              <div className={`z-10 grid h-10 w-10 shrink-0 place-items-center rounded-2xl shadow-sm sm:h-12 sm:w-12 ${s.color}`}>
                <s.icon className="h-5 w-5 sm:h-6 sm:w-6" />
              </div>
              <div className="flex-1 rounded-3xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
                <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-start sm:justify-between">
                  <h3 className="text-base font-semibold text-slate-900 sm:text-lg">{s.title}</h3>
                  <span className="w-max rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">{s.metric}</span>
                </div>
                <p className="mt-3 text-sm leading-relaxed text-slate-600 sm:text-base">{s.body}</p>
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
    <div className="rounded-2xl border border-slate-200 bg-white p-4 text-center shadow-sm sm:p-5">
      <p className="text-xs text-slate-500 sm:text-sm">{label}</p>
      <p className="mt-2 text-2xl font-bold text-emerald-600 sm:text-3xl">{value}</p>
      <p className="mt-1 text-[10px] text-slate-400 sm:text-xs">{sub}</p>
    </div>
  );
}
