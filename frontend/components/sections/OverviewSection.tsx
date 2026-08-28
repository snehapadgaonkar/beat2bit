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
  const final = find('pruned_quantized');

  const accuracy = baseline?.model_evaluation.accuracy ?? 0.87;
  const finalLatency = final?.summary.latency_performance.single_sample_latency_ms ?? 0.92;
  const finalSize = final?.complexity_analysis.memory_size.int8_mb ?? 0.004;
  const compression = baseline?.complexity_analysis.memory_size.compression_ratio_fp32_to_int8 ?? 3.92;

  const heroStats = [
    { label: 'Classification Accuracy', value: pct(accuracy), sub: 'AAMI patient independent', icon: TargetIcon },
    { label: 'Inference Latency', value: ms(finalLatency), sub: 'single sample · MCU', icon: Activity },
    { label: 'Deployed Model Size', value: mb(finalSize), sub: 'INT8 on-device', icon: Cpu },
    { label: 'Compression', value: `${compression.toFixed(1)}×`, sub: 'FP32 → INT8', icon: Zap },
  ];

  return (
    <div className="space-y-24 pb-16">
      {/* Hero */}
      <section className="relative -mx-4 -mt-8 overflow-hidden bg-slate-900 px-4 py-20 sm:px-6 lg:px-8 lg:py-28">
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute -top-24 -right-24 h-96 w-96 rounded-full bg-rose-500/30 blur-3xl" />
          <div className="absolute bottom-0 left-1/4 h-96 w-96 rounded-full bg-cyan-500/10 blur-3xl" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_1px_1px,rgba(255,255,255,0.06)_1px,transparent_0)] [background-size:24px_24px]" />
        </div>
        <div className="relative mx-auto max-w-7xl">
          <div className="mx-auto max-w-3xl text-center">
            <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-xs font-medium text-rose-200 backdrop-blur">
              <span className="h-1.5 w-1.5 rounded-full bg-rose-400 animate-pulse" />
              Ultra-Low-Power Edge AI · Research Release
            </span>
            <h1 className="mt-6 text-4xl font-extrabold tracking-tight text-white sm:text-5xl lg:text-6xl">
              Clinical-grade ECG detection,
              <span className="block bg-gradient-to-r from-rose-400 via-rose-300 to-orange-300 bg-clip-text text-transparent">
                engineered for the edge.
              </span>
            </h1>
            <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-slate-300">
              Beat2Bit compresses a state-of-the-art 1D convolutional network into a TinyML footprint that runs
              entirely on a microcontroller — no cloud, no connectivity, no compromise on clinical accuracy.
            </p>
            <div className="mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row">
              <Button
                size="lg"
                onClick={() => onNavigate('research')}
                className="h-12 rounded-full bg-rose-500 px-7 text-base font-semibold text-white shadow-lg shadow-rose-500/30 transition-all hover:bg-rose-400"
              >
                Explore the Research <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                onClick={() => onNavigate('dashboard')}
                className="h-12 rounded-full border-white/20 bg-white/5 px-7 text-base font-semibold text-white backdrop-blur hover:bg-white/10"
              >
                Live Dashboard
              </Button>
            </div>
          </div>

          {/* Stats */}
          <div className="mt-16 grid grid-cols-2 gap-3 lg:grid-cols-4">
            {heroStats.map((s, i) => (
              <div
                key={i}
                className="rounded-2xl border border-white/10 bg-white/5 p-5 backdrop-blur transition-colors hover:border-white/20"
              >
                <div className="flex items-center gap-2 text-rose-300">
                  <s.icon className="h-4 w-4" />
                  <span className="text-xs uppercase tracking-wider text-slate-300">{s.label}</span>
                </div>
                <div className="mt-3 text-3xl font-bold text-white">{loading ? '—' : s.value}</div>
                <div className="mt-1 text-xs text-slate-400">{s.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Methodology / value props */}
      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-12 max-w-2xl">
          <p className="text-sm font-semibold uppercase tracking-wider text-rose-500">Why Beat2Bit</p>
          <h2 className="mt-3 text-3xl font-bold tracking-tight text-slate-900">
            A research problem with real clinical stakes
          </h2>
          <p className="mt-4 text-lg leading-relaxed text-slate-600">
            Continuous cardiac monitoring in remote and low-resource settings demands models that are small,
            fast, private, and battery-efficient — while still meeting medical accuracy standards.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
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
              className="group relative rounded-3xl border border-slate-200 bg-white p-8 shadow-sm transition-all hover:-translate-y-1 hover:shadow-xl"
            >
              <div className={`mb-6 grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br ${f.color}`}>
                <f.icon className="h-7 w-7" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900">{f.title}</h3>
              <p className="mt-3 leading-relaxed text-slate-600">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Clinical capabilities (diseases) */}
      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-12 max-w-2xl">
          <p className="text-sm font-semibold uppercase tracking-wider text-rose-500">Pathologies detected</p>
          <h2 className="mt-3 text-3xl font-bold tracking-tight text-slate-900">Trained to catch the beats that matter</h2>
        </div>
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { name: 'Normal Sinus Rhythm', code: 'N', desc: 'Regular 60–100 bpm P-QRS-T sequence.', color: 'text-slate-600', bg: 'bg-slate-100' },
            { name: 'PVC', code: 'V', desc: 'Premature ventricular contractions — wide, bizarre QRS.', color: 'text-rose-500', bg: 'bg-rose-50' },
            { name: 'APC', code: 'A', desc: 'Atrial premature contractions — early, narrow beats.', color: 'text-orange-500', bg: 'bg-orange-50' },
            { name: 'Atrial Fibrillation', code: 'AF', desc: 'Irregular rhythm with high stroke risk.', color: 'text-purple-600', bg: 'bg-purple-50' },
          ].map((d, i) => (
            <div
              key={i}
              className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm transition-all hover:-translate-y-1 hover:shadow-lg"
            >
              <div className={`grid h-12 w-12 place-items-center rounded-xl ${d.bg} ${d.color}`}>
                <Waves className="h-6 w-6" />
              </div>
              <h3 className="mt-4 font-semibold text-slate-900">{d.name}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-slate-500">{d.desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

// Small helper icon component for the stat tiles.
function TargetIcon({ className }: { className?: string }) {
  return <span className={className}>◎</span>;
}
