'use client';

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart } from 'recharts';
import { cn } from '@/lib/utils';

interface ECGPoint {
  t: number;
  voltage: number;
}

const SAMPLE_RATE = 360; // Hz (MIT-BIH standard)
const WINDOW = 640; // visible samples (~1.8s)
const RR = 360; // one beat every 360 samples (60 bpm * ... ≈ 100 bpm)
const TOTAL = 3600; // precomputed signal length

/** Baseline wander + a realistic P-QRS-T morphology for a single beat. */
function beatVoltage(phase: number, abnormal: boolean): number {
  const gauss = (x: number, mu: number, sigma: number, amp: number) =>
    amp * Math.exp(-Math.pow((x - mu) / sigma, 2));

  if (abnormal) {
    // Premature ventricular contraction: wide, bizarre QRS, no P wave.
    return (
      gauss(phase, 0.35, 0.045, 1.35) + // broad R
      gauss(phase, 0.42, 0.055, -0.9) + // deep S
      gauss(phase, 0.72, 0.07, -0.55) // inverted T
    );
  }

  return (
    gauss(phase, 0.14, 0.020, 0.16) + // P wave
    gauss(phase, 0.30, 0.008, -0.14) + // Q
    gauss(phase, 0.36, 0.010, 1.45) + // R
    gauss(phase, 0.40, 0.010, -0.28) + // S
    gauss(phase, 0.68, 0.040, 0.38) // T wave
  );
}

function generateSignal(abnormal: boolean): number[] {
  const out: number[] = [];
  for (let i = 0; i < TOTAL; i++) {
    const phase = (i % RR) / RR;
    const wander = Math.sin((i / SAMPLE_RATE) * 2 * Math.PI * 0.25) * 0.06; // slow 0.25 Hz
    const noise = (Math.random() - 0.5) * 0.035; // measurement noise
    out.push(wander + beatVoltage(phase, abnormal) + noise);
  }
  return out;
}

export function ECGChart() {
  const [abnormal, setAbnormal] = useState(false);
  const [signal, setSignal] = useState<number[]>([]);
  const [offset, setOffset] = useState(0);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    setSignal(generateSignal(abnormal));
    setOffset(0);
  }, [abnormal]);

  // Livestream: advance a sample every frame, wrapping around the buffer.
  useEffect(() => {
    const tick = () => {
      setOffset((o) => (o + 1) % TOTAL);
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, []);

  const data = useMemo<ECGPoint[]>(() => {
    if (signal.length === 0) return [];
    return Array.from({ length: WINDOW }, (_, j) => {
      const idx = (offset + j) % TOTAL;
      return { t: (offset + j) % TOTAL, voltage: signal[idx] };
    });
  }, [signal, offset]);

  return (
    <div className="w-full">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
        <div>
          <h3 className="font-semibold text-slate-900">Live ECG Viewer</h3>
          <p className="text-sm text-slate-500">
            {SAMPLE_RATE} Hz window · MIT-BIH Arrhythmia waveform · p50 {abnormal ? 'PVC' : 'Normal sinus'} beat
          </p>
        </div>
        <div className="bg-slate-100 rounded-full p-1 inline-flex w-max">
          <button
            onClick={() => setAbnormal(false)}
            className={cn(
              'px-4 py-1.5 rounded-full text-sm font-medium transition-all',
              !abnormal ? 'bg-white text-slate-900 shadow-sm ring-1 ring-slate-200' : 'text-slate-500 hover:text-slate-700'
            )}
          >
            Normal (N)
          </button>
          <button
            onClick={() => setAbnormal(true)}
            className={cn(
              'px-4 py-1.5 rounded-full text-sm font-medium transition-all',
              abnormal ? 'bg-rose-500 text-white shadow-sm' : 'text-slate-500 hover:text-rose-600'
            )}
          >
            PVC (V)
          </button>
        </div>
      </div>

      <div className="h-[280px] w-full rounded-2xl border border-slate-200 bg-slate-950 p-2 shadow-inner">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="ecgFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={abnormal ? '#fb7185' : '#22d3ee'} stopOpacity={0.35} />
                <stop offset="100%" stopColor={abnormal ? '#fb7185' : '#22d3ee'} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="2 4" stroke="#1e293b" strokeOpacity={0.6} vertical={false} />
            <XAxis dataKey="t" tick={false} axisLine={false} />
            <YAxis domain={[-1.1, 1.8]} tick={false} axisLine={false} />
            <Tooltip
              labelFormatter={() => 'Live sample'}
              formatter={(value) => [`${Number(value).toFixed(2)} mV`, 'Voltage']}
              contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 10, color: '#e2e8f0', fontSize: 12 }}
            />
            <Area type="monotone" dataKey="voltage" stroke="none" fill="url(#ecgFill)" />
            <Line
              type="monotone"
              dataKey="voltage"
              stroke={abnormal ? '#fb7185' : '#22d3ee'}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 flex items-center justify-between text-xs text-slate-500">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" /> Streaming @ {SAMPLE_RATE} Hz
        </span>
        <span>P · QRS · T morphology</span>
      </div>
    </div>
  );
}
