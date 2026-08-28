'use client';

import React from 'react';
import { HeartPulse, Database, Cpu, ShieldCheck } from 'lucide-react';

export function SiteFooter() {
  return (
    <footer className="border-t border-slate-200/80 bg-white">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 sm:py-10 lg:px-8">
        <div className="grid gap-8 sm:grid-cols-2 md:grid-cols-3">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2">
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-rose-500 to-rose-600">
                <HeartPulse className="h-4 w-4 text-white" />
              </span>
              <span className="font-bold tracking-tight text-slate-900">Beat2Bit</span>
            </div>
            <p className="mt-3 max-w-xs text-sm leading-relaxed text-slate-500">
              Ultra-low-power edge AI for ECG arrhythmia detection. Research-driven TinyML, built for remote medical devices.
            </p>
          </div>

          {/* Research stack */}
          <div className="text-sm text-slate-500">
            <h4 className="mb-3 text-sm font-semibold text-slate-900">Research Stack</h4>
            <ul className="space-y-2">
              <li className="flex items-center gap-2"><Cpu className="h-4 w-4 shrink-0 text-rose-500" /> 1D-CNN + INT8 + Pruning</li>
              <li className="flex items-center gap-2"><ShieldCheck className="h-4 w-4 shrink-0 text-emerald-500" /> AAMI EC57 compliant metrics</li>
              <li className="flex items-center gap-2"><Database className="h-4 w-4 shrink-0 text-rose-500" /> MIT-BIH · PTB-XL · EU ST-T</li>
            </ul>
          </div>

          {/* AAMI */}
          <div className="text-sm text-slate-500 sm:col-span-2 md:col-span-1">
            <h4 className="mb-3 text-sm font-semibold text-slate-900">AAMI Metrics</h4>
            <p>
              All reported figures follow the AAMI EC57 standard for arrhythmia detector evaluation and are measured on a
              patient-independent validation split.
            </p>
            <p className="mt-3 text-xs text-slate-400">© {new Date().getFullYear()} Beat2Bit. Research preview build.</p>
          </div>
        </div>
      </div>
    </footer>
  );
}
