'use client';

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
} from 'recharts';
import type { ReportData } from '@/lib/utils';

interface StatDef {
  key: keyof ReportData['latency_benchmarking']['latency_stats'][string];
  label: string;
  color: string;
}

// p50 is approximated by the median for a cleaner distribution read-out.
const STATS: StatDef[] = [
  { key: 'min_latency_ms', label: 'min', color: '#94a3b8' },
  { key: 'median_latency_ms', label: 'p50', color: '#38bdf8' },
  { key: 'mean_latency_ms', label: 'mean', color: '#e11d48' },
  { key: 'p95_latency_ms', label: 'p95', color: '#f59e0b' },
  { key: 'p99_latency_ms', label: 'p99', color: '#8b5cf6' },
  { key: 'max_latency_ms', label: 'max', color: '#64748b' },
];

/** Latency distribution for the selected model (single-sample, batch 1). */
export function LatencyDistributionChart({ report }: { report: ReportData }) {
  const stats = report.latency_benchmarking.latency_stats.batch_size_1;
  const data = STATS.map((s) => ({ key: s.label, value: stats[s.key], color: s.color }));

  return (
    <div className="h-[280px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 24, right: 8, left: -16, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eef2f7" />
          <XAxis dataKey="key" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v} ms`} />
          <Tooltip
            formatter={(value) => [`${Number(value).toFixed(2)} ms`, 'Latency']}
            cursor={{ fill: 'rgba(226,232,240,0.4)' }}
            contentStyle={{ borderRadius: 12, border: '1px solid #e2e8f0', boxShadow: '0 8px 24px rgba(15,23,42,0.08)', fontSize: 13 }}
          />
          <Bar dataKey="value" radius={[6, 6, 0, 0]} barSize={38}>
            {data.map((d) => (
              <Cell key={d.key} fill={d.color} />
            ))}
            <LabelList dataKey="value" position="top" formatter={(v) => `${Number(v).toFixed(2)}`} style={{ fill: '#334155', fontSize: 11, fontWeight: 600 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Throughput across model variants (single-sample, batch 1). */
export function ThroughputChart({ data }: { data: Array<{ name: string; throughput: number }> }) {
  return (
    <div className="h-[280px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 24, right: 8, left: -20, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eef2f7" />
          <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} interval={0} angle={-14} height={52} textAnchor="end" />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}`} />
          <Tooltip
            formatter={(value) => [`${Number(value).toLocaleString()} samples/s`, 'Throughput']}
            cursor={{ fill: 'rgba(226,232,240,0.4)' }}
            contentStyle={{ borderRadius: 12, border: '1px solid #e2e8f0', boxShadow: '0 8px 24px rgba(15,23,42,0.08)', fontSize: 13 }}
          />
          <Bar dataKey="throughput" fill="#10b981" radius={[6, 6, 0, 0]} barSize={34}>
            <LabelList dataKey="throughput" position="top" formatter={(v) => `${Math.round(Number(v)).toLocaleString()}`} style={{ fill: '#059669', fontSize: 11, fontWeight: 600 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
