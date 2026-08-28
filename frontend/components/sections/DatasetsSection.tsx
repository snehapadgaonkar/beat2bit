'use client';

import React from 'react';
import { Database, BadgeCheck, Layers } from 'lucide-react';
import type { ReportData } from '@/lib/utils';
import { int } from '@/lib/utils';

interface Props {
  reports: ReportData[];
}

export function DatasetsSection({ reports }: Props) {
  const primary = reports.find((r) => r.metadata.model_name === 'baseline_model');

  const datasets = [
    {
      name: 'MIT-BIH Arrhythmia',
      desc: 'The gold-standard benchmark: 48 half-hour ambulatory ECG recordings with 109,000+ beat-by-beat annotations from multiple cardiologists.',
      tags: ['360 Hz', '2 Channels', '109k Beats'],
      color: 'border-blue-100 bg-blue-50/50',
      dot: 'bg-blue-500',
    },
    {
      name: 'PTB-XL Large-Scale',
      desc: 'A large clinical corpus of 21,837 twelve-lead ECGs from 18,885 patients covering a wide range of pathologies and demographics.',
      tags: ['500 Hz', '12-Lead', '21k Records'],
      color: 'border-emerald-100 bg-emerald-50/50',
      dot: 'bg-emerald-500',
    },
    {
      name: 'European ST-T',
      desc: 'Strengthens robustness to ST-segment changes, ischemia, and baseline-wander artifact across lead configurations.',
      tags: ['250 Hz', '2 Leads', '90 Records'],
      color: 'border-purple-100 bg-purple-50/50',
      dot: 'bg-purple-500',
    },
  ];

  return (
    <div className="space-y-14 pb-16">
      <div className="mx-auto max-w-3xl text-center">
        <p className="text-sm font-semibold uppercase tracking-wider text-rose-500">Training corpora</p>
        <h1 className="mt-4 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">Trained on the world&apos;s data</h1>
        <p className="mt-4 text-lg leading-relaxed text-slate-600">
          Beat2Bit ensembles three gold-standard clinical datasets so the model generalizes across demographics,
          hardware, and electrode configurations.
        </p>
      </div>

      {/* Primary dataset spec strip */}
      <div className="mx-auto flex max-w-4xl flex-wrap items-center justify-center gap-x-8 gap-y-2 rounded-2xl border border-slate-200 bg-white px-6 py-4 text-sm text-slate-600 shadow-sm">
        <span className="inline-flex items-center gap-2">
          <BadgeCheck className="h-4 w-4 text-emerald-500" /> {primary?.dataset_info.dataset ?? 'MIT-BIH Arrhythmia'}
        </span>
        <span className="inline-flex items-center gap-2">
          <Database className="h-4 w-4 text-rose-500" /> {int(primary?.dataset_info.samples ?? 2000)} samples
        </span>
        <span className="inline-flex items-center gap-2">
          <Layers className="h-4 w-4 text-sky-500" /> {primary?.dataset_info.features ?? 180} features
        </span>
        <span className="inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1 text-xs font-medium">
          {primary?.dataset_info.split ?? 'Patient-independent AAMI'}
        </span>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {datasets.map((d, i) => (
          <div key={i} className={`relative overflow-hidden rounded-3xl border bg-white p-7 shadow-sm transition-all hover:-translate-y-1 hover:shadow-lg ${d.color.split(' ')[0]}`}>
            <div className={`absolute -top-8 -right-8 h-28 w-28 rounded-full ${d.color.split(' ')[1]} blur-2xl`} />
            <div className="relative">
              <div className="mb-4 grid h-12 w-12 place-items-center rounded-xl bg-white shadow-sm">
                <Database className="h-6 w-6 text-slate-700" />
              </div>
              <h3 className="text-lg font-semibold text-slate-900">{d.name}</h3>
              <p className="mt-3 text-sm leading-relaxed text-slate-600">{d.desc}</p>
              <div className="mt-5 flex items-center gap-2">
                <span className={`h-1.5 w-1.5 rounded-full ${d.dot}`} />
                <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Lead config</span>
              </div>
              <div className="mt-2 flex flex-wrap gap-2">
                {d.tags.map((t, k) => (
                  <span key={k} className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-600 shadow-sm ring-1 ring-slate-200">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
