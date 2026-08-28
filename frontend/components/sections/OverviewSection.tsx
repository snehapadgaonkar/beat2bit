'use client';

import React from 'react';
import {
  ArrowRight,
  WifiOff,
  BatteryCharging,
  ShieldCheck,
  Activity,
  Cpu,
  Zap,
  Waves,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { ReportData } from '@/lib/utils';
import { pct, ms, mb } from '@/lib/utils';
import type { TabId } from '@/components/SiteHeader';

interface Props {
  reports: ReportData[];
  loading: boolean;
  onNavigate: (id: TabId) => void;
}

export function OverviewSection({ reports, loading, onNavigate }: Props) {
  const find = (name: string) => reports.find((r) => r.metadata.model_name === name);
  const baseline = find('baseline_model');
  // INT8 quantized is the AAMI-compliant deployed model (Se=75%, +P=71.2%)
  const final = find('quantized_int8') ?? find('pruned_quantized');

  const accuracy = baseline?.model_evaluation.accuracy ?? 0.9383;
  const finalLatency = final?.summary.latency_performance.single_sample_latency_ms ?? 0.073;
  const finalSize = final?.complexity_analysis.memory_size.int8_mb ?? 0.0338;
  const compression = final?.complexity_analysis.memory_size.compression_ratio_fp32_to_int8 ?? 2.59;

  const heroStats = [
    { label: 'Classification Accuracy', value: pct(accuracy), sub: 'AAMI patient independent', icon: TargetIcon },
    { label: 'Inference Latency', value: ms(finalLatency), sub: 'single sample · MCU', icon: Activity },
    { label: 'Deployed Model Size', value: mb(finalSize), sub: 'INT8 on-device', icon: Cpu },
    { label: 'Compression', value: `${compression.toFixed(1)}×`, sub: 'FP32 → INT8', icon: Zap },
  ];

  return (
    <div className="space-y-16 pb-12 sm:space-y-24 sm:pb-16">
      {/* ── Hero ─────────────────────────────────────────────────────────────── */}
      <section className="relative -mx-4 -mt-8 overflow-hidden bg-slate-900 px-4 py-16 sm:-mx-6 sm:px-6 sm:py-20 lg:-mx-8 lg:px-8 lg:py-28">
        {/* Ambient glow blobs */}
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute -top-24 -right-24 h-64 w-64 rounded-full bg-rose-500/30 blur-3xl sm:h-96 sm:w-96" />
          <div className="absolute bottom-0 left-1/4 h-64 w-64 rounded-full bg-cyan-500/10 blur-3xl sm:h-96 sm:w-96" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_1px_1px,rgba(255,255,255,0.06)_1px,transparent_0)] [background-size:24px_24px]" />
        </div>

        <div className="relative mx-auto max-w-7xl">
          {/* Heading + CTA */}
          <div className="mx-auto max-w-3xl text-center">
            <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-xs font-medium text-rose-200 backdrop-blur">
              <span className="h-1.5 w-1.5 rounded-full bg-rose-400 animate-pulse" />
              Ultra-Low-Power Edge AI · Research Release
            </span>

            <h1 className="mt-6 text-3xl font-extrabold tracking-tight text-white sm:text-5xl lg:text-6xl">
              Clinical-grade ECG detection,
              <span className="block bg-gradient-to-r from-rose-400 via-rose-300 to-orange-300 bg-clip-text text-transparent">
                engineered for the edge.
              </span>
            </h1>

            <p className="mx-auto mt-5 max-w-2xl text-base leading-relaxed text-slate-300 sm:mt-6 sm:text-lg">
              Beat2Bit compresses a state-of-the-art 1D convolutional network into a TinyML footprint that runs
              entirely on a microcontroller — no cloud, no connectivity, no compromise on clinical accuracy.
            </p>

            <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
              <Button
                size="lg"
                onClick={() => onNavigate('research')}
                className="h-12 w-full rounded-full bg-rose-500 px-7 text-base font-semibold text-white shadow-lg shadow-rose-500/30 transition-all hover:bg-rose-400 sm:w-auto"
              >
                Explore the Research <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                onClick={() => onNavigate('dashboard')}
                className="h-12 w-full rounded-full border-white/20 bg-white/5 px-7 text-base font-semibold text-white backdrop-blur hover:bg-white/10 sm:w-auto"
              >
                Live Dashboard
              </Button>
            </div>
          </div>

          {/* Stat tiles — 2 col on mobile → 4 col on lg */}
          <div className="mt-12 grid grid-cols-2 gap-3 sm:mt-16 lg:grid-cols-4">
            {heroStats.map((s, i) => (
              <div
                key={i}
                className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur transition-colors hover:border-white/20 sm:p-5"
              >
                <div className="flex items-center gap-2 text-rose-300">
                  <s.icon className="h-4 w-4 shrink-0" />
                  <span className="text-[10px] uppercase tracking-wider text-slate-300 sm:text-xs">{s.label}</span>
                </div>
                <div className="mt-3 text-2xl font-bold text-white sm:text-3xl">{loading ? '—' : s.value}</div>
                <div className="mt-1 text-[10px] text-slate-400 sm:text-xs">{s.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Value props ───────────────────────────────────────────────────────── */}
      <section className="mx-auto w-full px-4 sm:px-6 lg:px-8">
        <div className="mb-10 max-w-2xl sm:mb-12">
          <p className="text-sm font-semibold uppercase tracking-wider text-rose-500">Why Beat2Bit</p>
          <h2 className="mt-3 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
            A research problem with real clinical stakes
          </h2>
          <p className="mt-4 text-base leading-relaxed text-slate-600 sm:text-lg">
            Continuous cardiac monitoring in remote and low-resource settings demands models that are small,
            fast, private, and battery-efficient — while still meeting medical accuracy standards.
          </p>
        </div>

        {/* Feature cards — 1 col → 3 col */}
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3 w-full">
          {[
            {
              icon: WifiOff,
              title: '100% Offline Inference',
              desc: 'Inference runs entirely on the microcontroller, removing any dependency on cloud connectivity for true anywhere-in-the-world operation.',
              color: 'from-slate-100 to-slate-50 text-slate-700',
            },
            {
              icon: BatteryCharging,
              title: 'Ultra-Low Power',
              desc: 'INT8 quantization and magnitude pruning cut energy to microjoules per inference, extending wearable battery life from days to weeks.',
              color: 'from-emerald-50 to-emerald-50/40 text-emerald-600',
            },
            {
              icon: ShieldCheck,
              title: 'Privacy Preserving',
              desc: 'Biometric data is processed, classified, and discarded on-device — inherently compliant with HIPAA and GDPR.',
              color: 'from-blue-50 to-blue-50/40 text-blue-600',
            },
          ].map((f, i) => (
            <div
              key={i}
              className="group relative rounded-3xl border border-slate-200 bg-white p-6 shadow-sm transition-all hover:-translate-y-1 hover:shadow-xl sm:p-8"
            >
              <div className={`mb-5 grid h-12 w-12 place-items-center rounded-2xl bg-gradient-to-br sm:mb-6 sm:h-14 sm:w-14 ${f.color}`}>
                <f.icon className="h-6 w-6 sm:h-7 sm:w-7" />
              </div>
              <h3 className="text-lg font-semibold text-slate-900 sm:text-xl">{f.title}</h3>
              <p className="mt-3 text-sm leading-relaxed text-slate-600 sm:text-base">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Pathologies ───────────────────────────────────────────────────────── */}
      <section className="mx-auto w-full px-4 sm:px-6 lg:px-8">
        <div className="mb-10 max-w-2xl sm:mb-12">
          <p className="text-sm font-semibold uppercase tracking-wider text-rose-500">Pathologies detected</p>
          <h2 className="mt-3 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
            Trained to catch the beats that matter
          </h2>
        </div>

        {/* 2 col on mobile → 4 col on lg */}
        <div className="grid grid-cols-2 gap-4 sm:gap-5 lg:grid-cols-4 w-full">
          {[
            { name: 'Normal Sinus Rhythm', code: 'N', desc: 'Regular 60–100 bpm P-QRS-T sequence.', color: 'text-slate-600', bg: 'bg-slate-100' },
            { name: 'PVC', code: 'V', desc: 'Premature ventricular contractions — wide, bizarre QRS.', color: 'text-rose-500', bg: 'bg-rose-50' },
            { name: 'APC', code: 'A', desc: 'Atrial premature contractions — early, narrow beats.', color: 'text-orange-500', bg: 'bg-orange-50' },
            { name: 'Atrial Fibrillation', code: 'AF', desc: 'Irregular rhythm with high stroke risk.', color: 'text-purple-600', bg: 'bg-purple-50' },
          ].map((d, i) => (
            <div
              key={i}
              className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm transition-all hover:-translate-y-1 hover:shadow-lg sm:p-6"
            >
              <div className={`grid h-10 w-10 place-items-center rounded-xl sm:h-12 sm:w-12 ${d.bg} ${d.color}`}>
                <Waves className="h-5 w-5 sm:h-6 sm:w-6" />
              </div>
              <h3 className="mt-3 text-sm font-semibold text-slate-900 sm:mt-4 sm:text-base">{d.name}</h3>
              <p className="mt-1.5 text-xs leading-relaxed text-slate-500 sm:text-sm">{d.desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function TargetIcon({ className }: { className?: string }) {
  return <span className={className}>◎</span>;
}
