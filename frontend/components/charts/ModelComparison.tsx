'use client';

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LabelList,
} from 'recharts';

export interface ComparisonPoint {
  name: string;
  accuracy: number;
  f1: number;
  latency: number;
  size: number;
  params: number;
}

/** Grouped bar chart comparing accuracy & F1 across every model variant. */
export function AccuracyChart({ data }: { data: ComparisonPoint[] }) {
  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 24, right: 16, left: -16, bottom: 4 }} barGap={4}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eef2f7" />
          <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} interval={0} angle={-14} height={52} textAnchor="end" />
          <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
          <Tooltip
            formatter={(value, name) => [`${Number(value).toFixed(1)}%`, name === 'accuracy' ? 'Accuracy' : 'F1 Score']}
            cursor={{ fill: 'rgba(226,232,240,0.4)' }}
            contentStyle={{ borderRadius: 12, border: '1px solid #e2e8f0', boxShadow: '0 8px 24px rgba(15,23,42,0.08)', fontSize: 13 }}
          />
          <Legend iconType="circle" wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
          <Bar dataKey="accuracy" name="accuracy" fill="#e11d48" radius={[6, 6, 0, 0]} barSize={18}>
            <LabelList dataKey="accuracy" position="top" formatter={(v) => `${Number(v).toFixed(0)}%`} style={{ fill: '#e11d48', fontSize: 11, fontWeight: 600 }} />
          </Bar>
          <Bar dataKey="f1" name="f1" fill="#6366f1" radius={[6, 6, 0, 0]} barSize={18}>
            <LabelList dataKey="f1" position="top" formatter={(v) => `${Number(v).toFixed(0)}%`} style={{ fill: '#6366f1', fontSize: 11, fontWeight: 600 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Grouped bar chart comparing latency & model size (different scales, dual axis). */
export function EfficiencyChart({ data }: { data: ComparisonPoint[] }) {
  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 24, right: 8, left: -16, bottom: 4 }} barGap={4}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eef2f7" />
          <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} interval={0} angle={-14} height={52} textAnchor="end" />
          <YAxis yAxisId="left" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v} ms`} />
          <YAxis yAxisId="right" orientation="right" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v} KB`} />
          <Tooltip
            formatter={(value, name) => (name === 'latency' ? [`${Number(value).toFixed(2)} ms`, 'Latency'] : [`${Number(value).toFixed(0)} KB`, 'Size (KB)'])}
            cursor={{ fill: 'rgba(226,232,240,0.4)' }}
            contentStyle={{ borderRadius: 12, border: '1px solid #e2e8f0', boxShadow: '0 8px 24px rgba(15,23,42,0.08)', fontSize: 13 }}
          />
          <Legend iconType="circle" wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
          <Bar yAxisId="left" dataKey="latency" name="latency" fill="#0ea5e9" radius={[6, 6, 0, 0]} barSize={18}>
            <LabelList dataKey="latency" position="top" formatter={(v) => `${Number(v).toFixed(1)}`} style={{ fill: '#0ea5e9', fontSize: 11, fontWeight: 600 }} />
          </Bar>
          <Bar yAxisId="right" dataKey="size" name="size" fill="#8b5cf6" radius={[6, 6, 0, 0]} barSize={18}>
            <LabelList dataKey="size" position="top" formatter={(v) => `${Number(v).toFixed(0)}`} style={{ fill: '#8b5cf6', fontSize: 11, fontWeight: 600 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
