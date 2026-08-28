'use client';

import type { ReportData } from '@/lib/utils';

/**
 * Visual AAMI / scikit-learn-style 2x2 confusion matrix, rendered as a
 * heat-map grid with the underlying counts from the backend report.
 */
export function ConfusionMatrix({ report }: { report: ReportData }) {
  const { tn, fp, fn, tp } = report.model_evaluation.confusion_matrix;
  const total = tn + fp + fn + tp;

  const cells = [
    { label: 'Actual Normal', heading: 'True Negative', value: tn, col: 'Predict Normal', bg: 'bg-emerald-50 text-emerald-800 ring-emerald-100', bar: 'bg-emerald-500' },
    { label: 'Actual Normal', heading: 'False Positive', value: fp, col: 'Predict Arrhythmia', bg: 'bg-rose-50 text-rose-700 ring-rose-100', bar: 'bg-rose-500' },
    { label: 'Actual Arrhythmia', heading: 'False Negative', value: fn, col: 'Predict Normal', bg: 'bg-amber-50 text-amber-700 ring-amber-100', bar: 'bg-amber-500' },
    { label: 'Actual Arrhythmia', heading: 'True Positive', value: tp, col: 'Predict Arrhythmia', bg: 'bg-slate-900 text-white ring-slate-900', bar: 'bg-emerald-400' },
  ];

  return (
    <div className="w-full">
      <div className="grid grid-cols-2 gap-2">
        {cells.map((c, i) => {
          const sharePct = total > 0 ? (c.value / total) * 100 : 0;
          return (
            <div key={i} className={`relative overflow-hidden rounded-2xl p-4 ring-1 ${c.bg}`}>
              <div className="flex items-center justify-between text-[11px] uppercase tracking-wide opacity-70">
                <span>{c.col}</span>
                <span>{c.label}</span>
              </div>
              <div className="mt-2 flex items-end justify-between">
                <div>
                  <div className="text-3xl font-bold tabular-nums">{c.value}</div>
                  <div className="text-xs font-medium opacity-80">{c.heading}</div>
                </div>
                <div className="text-sm font-semibold tabular-nums opacity-80">{sharePct.toFixed(1)}%</div>
              </div>
              <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-white/40">
                <div className={`h-full rounded-full ${c.bar} transition-all duration-700`} style={{ width: `${sharePct}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
