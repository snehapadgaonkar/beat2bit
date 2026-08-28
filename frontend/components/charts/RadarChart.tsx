'use client';

import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import type { ReportData } from '@/lib/utils';

interface AxisMetric {
  key: string;
  label: string;
  get: (r: ReportData) => number;
}

const AXES: AxisMetric[] = [
  { key: 'accuracy', label: 'Accuracy', get: (r) => r.model_evaluation.accuracy },
  { key: 'precision', label: 'Precision', get: (r) => r.model_evaluation.precision },
  { key: 'recall', label: 'Recall', get: (r) => r.model_evaluation.recall },
  { key: 'specificity', label: 'Specificity', get: (r) => r.model_evaluation.specificity },
  { key: 'f1', label: 'F1 Score', get: (r) => r.model_evaluation.f1_score },
  { key: 'eff', label: 'AAMI Eff.', get: (r) => r.model_evaluation.ami_effectiveness },
];

export function MetricRadar({ report, color = '#e11d48' }: { report: ReportData; color?: string }) {
  const data = AXES.map((a) => ({
    metric: a.label,
    value: Number((a.get(report) * 100).toFixed(1)),
    full: 100,
  }));

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="72%">
          <PolarGrid stroke="#e2e8f0" />
          <PolarAngleAxis dataKey="metric" tick={{ fill: '#64748b', fontSize: 12 }} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#cbd5e1', fontSize: 10 }} />
          <Tooltip
            formatter={(value) => `${value}%`}
            contentStyle={{
              borderRadius: 12,
              border: '1px solid #e2e8f0',
              boxShadow: '0 8px 24px rgba(15,23,42,0.08)',
              fontSize: 13,
            }}
          />
          <Radar dataKey="value" stroke={color} fill={color} fillOpacity={0.22} strokeWidth={2} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
