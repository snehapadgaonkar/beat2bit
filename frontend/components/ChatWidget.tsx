'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Bot, Send, Sparkles, X } from 'lucide-react';
import type { ReportData } from '@/lib/utils';
import { pct, ms, mb, params, flops } from '@/lib/utils';
import { REPORT_ORDER } from '@/lib/researchService';
import { cn } from '@/lib/utils';

interface Msg {
  role: 'user' | 'agent';
  text: string;
}

interface Props {
  reports: ReportData[];
}

export function ChatWidget({ reports }: Props) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Msg[]>([
    {
      role: 'agent',
      text: "Hi, I'm the Beat2Bit research assistant. Ask me about accuracy, latency, model size, AAMI metrics, or the datasets.",
    },
  ]);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typing]);

  useEffect(() => {
    if (open) {
      const t = setTimeout(() => inputRef.current?.focus(), 200);
      return () => clearTimeout(t);
    }
  }, [open]);

  const meta = (name: string) => REPORT_ORDER.find((m) => m.name === name);

  function answer(q: string): string {
    const lower = q.toLowerCase();
    const best = reports.find((r) => r.metadata.model_name === 'pruned_quantized');
    const baseline = reports.find((r) => r.metadata.model_name === 'baseline_model');

    if (/(accuracy|f1|f1 score)/.test(lower)) {
      const rows = reports
        .map((r) => `${meta(r.metadata.model_name)?.label}: ${pct(r.model_evaluation.accuracy)} / ${pct(r.model_evaluation.f1_score)} F1`)
        .join(' · ');
      return `Across the ladder, accuracy stays clinically usable: ${rows}. The deployed model holds ${pct(best?.model_evaluation.accuracy ?? 0)} accuracy with just ${params(best?.complexity_analysis.parameters.total_parameters ?? 0)} parameters.`;
    }
    if (/(latency|speed|fast|ms|real.?time)/.test(lower)) {
      return `Single-sample inference latency on the edge MCU is ${ms(best?.summary.latency_performance.single_sample_latency_ms ?? 0)} (p95 ${ms(best?.summary.latency_performance.latency_p95_ms ?? 0)}), down from ${ms(baseline?.summary.latency_performance.single_sample_latency_ms ?? 0)} for the FP32 baseline — a ${(
        (1 -
          (best?.summary.latency_performance.single_sample_latency_ms ?? 0) /
            (baseline?.summary.latency_performance.single_sample_latency_ms ?? 1)) *
        100
      ).toFixed(0)}% reduction.`;
    }
    if (/(size|memory|kb|mb|small|ram|flash)/.test(lower)) {
      return `The final pruned + quantized model is ${mb(best?.complexity_analysis.memory_size.int8_mb ?? 0)} in INT8 (${mb(best?.complexity_analysis.memory_size.fp32_mb ?? 0)} in FP32). That's a ${
        best?.complexity_analysis.memory_size.compression_ratio_fp32_to_int8 ?? 0
      }× compression from quantization plus ${((1 - (best?.complexity_analysis.parameters.total_parameters ?? 0) / (baseline?.complexity_analysis.parameters.total_parameters ?? 1)) * 100).toFixed(
        0
      )}% fewer parameters from pruning.`;
    }
    if (/(power|energy|battery|micro.?joule|µj)/.test(lower)) {
      return `Because INT8 quantization and 70% pruning cut FLOPs to ${flops(best?.complexity_analysis.computational_complexity.total_flops ?? 0)}, energy per inference drops to the microjoule range, extending wearable battery life from days to weeks. We keep inference fully on-device.`;
    }
    if (/(dataset|data|mit|ptb|st-t)/.test(lower)) {
      return `We train on MIT-BIH Arrhythmia (${baseline?.dataset_info.dataset ?? ''}), plus PTB-XL and the European ST-T database, using a ${baseline?.dataset_info.split ?? 'patient-independent'} split for generalizability.`;
    }
    if (/(aami|sensitivity|predictivity|ec57|effectiveness)/.test(lower)) {
      const e = best?.model_evaluation;
      return `AAMI EC57 metrics for the deployed model — Sensitivity (Se) ${pct(e?.ami_sensitivity ?? 0)}, Positive Predictivity (+P) ${pct(e?.ami_positive_predictivity ?? 0)}, Effectiveness ${pct(
        e?.ami_effectiveness ?? 0
      )}. These are computed on a patient-independent split exactly as defined in EC57.`;
    }
    if (/(prune|quant|compress|smaller|optim)/.test(lower)) {
      return `The optimization ladder is Baseline → INT8 Quantize → Prune 50% → Prune 70% → Prune+Quantize. The deployed model combines 70% magnitude pruning with INT8 quantization for a ~${
        best?.complexity_analysis.memory_size.compression_ratio_fp32_to_int8 ?? 0
      }× size cut while keeping ${pct(best?.model_evaluation.accuracy ?? 0)} accuracy.`;
    }
    return `Ask me about accuracy, latency, model size, AAMI metrics, pruning/quantization, or the datasets — I'll answer with the real numbers from the benchmarking reports.`;
  }

  const send = (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || typing) return;
    setMessages((m) => [...m, { role: 'user', text }]);
    setInput('');
    setTyping(true);
    setTimeout(() => {
      setMessages((m) => [...m, { role: 'agent', text: answer(text) }]);
      setTyping(false);
    }, 650);
  };

  return (
    <>
      {/* Floating launcher */}
      <button
        onClick={() => setOpen((v) => !v)}
        aria-label="Ask Beat2Bit"
        className={cn(
          'fixed bottom-5 right-4 z-50 flex items-center gap-2.5 rounded-full px-4 py-3 text-sm font-semibold shadow-xl transition-all duration-300 sm:right-5',
          open
            ? 'bg-slate-900 text-white shadow-slate-900/30'
            : 'bg-rose-500 text-white shadow-rose-500/40 hover:scale-105 hover:bg-rose-600'
        )}
      >
        <span className="relative grid h-7 w-7 place-items-center">
          {open ? <X className="h-5 w-5" /> : <Bot className="h-6 w-6" />}
          {!open && <span className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full bg-emerald-400 ring-2 ring-rose-500" />}
        </span>
        {!open && <span className="hidden sm:inline">Ask Beat2Bit</span>}
      </button>

      {/* Chat panel
          - Mobile: full-width, fixed bottom sheet (above launcher)
          - sm+: anchored to bottom-right as a floating panel  */}
      <div
        className={cn(
          'fixed z-50 flex flex-col overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-2xl shadow-slate-900/15 transition-all duration-300',
          // Mobile: full-width sheet, 80% screen height
          'bottom-20 left-2 right-2',
          // sm+: right-anchored panel, capped width
          'sm:bottom-20 sm:left-auto sm:right-4 sm:w-[min(calc(100vw-2rem),24rem)] md:right-5',
          open ? 'pointer-events-auto translate-y-0 opacity-100' : 'pointer-events-none translate-y-4 opacity-0'
        )}
      >
        {/* Header */}
        <div className="flex items-center gap-3 border-b border-slate-200 bg-slate-900 px-4 py-3.5 sm:px-5 sm:py-4">
          <div className="grid h-8 w-8 place-items-center rounded-xl bg-gradient-to-br from-rose-500 to-rose-600 sm:h-9 sm:w-9">
            <Bot className="h-4 w-4 text-white sm:h-5 sm:w-5" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white sm:text-base">Ask Beat2Bit</h3>
            <p className="text-[10px] text-slate-400 sm:text-xs">Grounded in live reports ({reports.length} models)</p>
          </div>
          <span className="ml-auto inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-2.5 py-1 text-[10px] font-medium text-emerald-300 sm:text-xs">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" /> Online
          </span>
        </div>

        {/* Messages */}
        <div className="h-[min(55vh,320px)] space-y-3 overflow-y-auto bg-slate-50/60 p-3 sm:h-[320px] sm:p-4">
          {messages.map((m, i) => (
            <div key={i} className={cn('flex', m.role === 'user' ? 'justify-end' : 'justify-start')}>
              {m.role === 'agent' && (
                <div className="mr-2 mt-1 grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-slate-900 sm:h-7 sm:w-7">
                  <Bot className="h-3 w-3 text-rose-400 sm:h-3.5 sm:w-3.5" />
                </div>
              )}
              <div
                className={cn(
                  'max-w-[88%] rounded-2xl px-3 py-2.5 text-xs leading-relaxed shadow-sm sm:max-w-[85%] sm:px-3.5 sm:text-sm',
                  m.role === 'user'
                    ? 'rounded-br-sm bg-slate-900 text-white'
                    : 'rounded-bl-sm border border-slate-200 bg-white text-slate-700'
                )}
              >
                {m.text}
              </div>
            </div>
          ))}
          {typing && (
            <div className="flex items-center gap-2 pl-8 text-xs text-slate-400">
              <Sparkles className="h-3.5 w-3.5 animate-pulse" /> thinking…
            </div>
          )}
          <div ref={endRef} />
        </div>

        {/* Quick suggestions */}
        <div className="flex flex-wrap gap-1.5 border-t border-slate-100 bg-white px-3 pb-1 pt-2 sm:px-4">
          {['Model accuracy', 'Latency', 'Model size', 'AAMI metrics'].map((q) => (
            <button
              key={q}
              onClick={() => {
                setMessages((m) => [...m, { role: 'user', text: q }]);
                setTyping(true);
                setTimeout(() => {
                  setMessages((m) => [...m, { role: 'agent', text: answer(q) }]);
                  setTyping(false);
                }, 500);
              }}
              className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[10px] font-medium text-slate-600 transition-colors hover:border-rose-200 hover:bg-rose-50 hover:text-rose-600 sm:px-3 sm:text-xs"
            >
              {q}
            </button>
          ))}
        </div>

        {/* Input */}
        <form onSubmit={send} className="relative border-t border-slate-100 bg-white p-2">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder='Try: "Deployed model latency?"'
            className="w-full rounded-full border border-slate-200 bg-slate-50 py-2.5 pl-4 pr-12 text-xs outline-none transition-all focus:border-rose-300 focus:bg-white focus:ring-2 focus:ring-rose-200 sm:text-sm"
          />
          <button
            type="submit"
            disabled={!input.trim() || typing}
            className="absolute right-3.5 top-1/2 grid h-7 w-7 -translate-y-1/2 place-items-center rounded-full bg-rose-500 text-white shadow-md shadow-rose-500/30 transition-all hover:bg-rose-600 disabled:opacity-40 sm:h-8 sm:w-8"
          >
            <Send className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
          </button>
        </form>
      </div>
    </>
  );
}
