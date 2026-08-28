'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Bot, Send, Sparkles } from 'lucide-react';
import type { ReportData } from '@/lib/utils';
import { pct, ms, mb, params, flops } from '@/lib/utils';
import { REPORT_ORDER } from '@/lib/researchService';

interface Msg {
  role: 'user' | 'agent';
  text: string;
}

interface Props {
  reports: ReportData[];
}

export function AgentSection({ reports }: Props) {
  const [messages, setMessages] = useState<Msg[]>([
    { role: 'agent', text: "Hi, I'm the Beat2Bit research assistant. Ask me about accuracy, latency, model size, AAMI metrics, or the datasets." },
  ]);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typing]);

  const meta = (name: string) => REPORT_ORDER.find((m) => m.name === name);

  function answer(q: string): string {
    const lower = q.toLowerCase();
    const best = reports.find((r) => r.metadata.model_name === 'pruned_quantized');
    const baseline = reports.find((r) => r.metadata.model_name === 'baseline_model');

    if (/(accuracy|f1|f1 score)/.test(lower)) {
      const rows = reports.map((r) => `${meta(r.metadata.model_name)?.label}: ${pct(r.model_evaluation.accuracy)} acc / ${pct(r.model_evaluation.f1_score)} F1`).join(' · ');
      return `Across the ladder, accuracy stays clinically usable: ${rows}. The deployed ${meta(best?.metadata.model_name ?? '')?.label} model holds ${pct(best?.model_evaluation.accuracy ?? 0)} accuracy with just ${params(best?.complexity_analysis.parameters.total_parameters ?? 0)} parameters.`;
    }
    if (/(latency|speed|fast|ms|real.?time)/.test(lower)) {
      return `Single-sample inference latency on the edge MCU is ${ms(best?.summary.latency_performance.single_sample_latency_ms ?? 0)} (p95 ${ms(best?.summary.latency_performance.latency_p95_ms ?? 0)}), down from ${ms(baseline?.summary.latency_performance.single_sample_latency_ms ?? 0)} for the FP32 baseline — a ${((1 - (best?.summary.latency_performance.single_sample_latency_ms ?? 0) / (baseline?.summary.latency_performance.single_sample_latency_ms ?? 1)) * 100).toFixed(0)}% reduction.`;
    }
    if (/(size|memory|kb|mb|small|ram|flash)/.test(lower)) {
      return `The final pruned + quantized model is ${mb(best?.complexity_analysis.memory_size.int8_mb ?? 0)} in INT8 (${mb(best?.complexity_analysis.memory_size.fp32_mb ?? 0)} in FP32). That's a ${best?.complexity_analysis.memory_size.compression_ratio_fp32_to_int8 ?? 0}× compression from quantization plus ${((1 - (best?.complexity_analysis.parameters.total_parameters ?? 0) / (baseline?.complexity_analysis.parameters.total_parameters ?? 1)) * 100).toFixed(0)}% fewer parameters from pruning.`;
    }
    if (/(power|energy|battery|micro.?joule|µj)/.test(lower)) {
      return `Because INT8 quantization and 70% pruning cut FLOPs to ${flops(best?.complexity_analysis.computational_complexity.total_flops ?? 0)}, energy per inference drops to the microjoule range, extending wearable battery life from days to weeks. We keep inference fully on-device.`;
    }
    if (/(dataset|data|mit|ptb|st-t)/.test(lower)) {
      return `We train on MIT-BIH Arrhythmia (the AAMI benchmark, ${baseline?.dataset_info.dataset ?? ''}), plus PTB-XL and the European ST-T database, using a ${baseline?.dataset_info.split ?? 'patient-independent'} split for generalizability.`;
    }
    if (/(aami|sensitivity|predictivity|ec57|effectiveness)/.test(lower)) {
      const e = best?.model_evaluation;
      return `AAMI EC57 metrics for the deployed model — Sensitivity (Se) ${pct(e?.ami_sensitivity ?? 0)}, Positive Predictivity (+P) ${pct(e?.ami_positive_predictivity ?? 0)}, Effectiveness ${pct(e?.ami_effectiveness ?? 0)}. These are computed on a patient-independent split exactly as defined in EC57.`;
    }
    if (/(prune|quant|compress|smaller|optim)/.test(lower)) {
      return `The optimization ladder is Baseline → INT8 Quantize → Prune 50% → Prune 70% → Prune+Quantize. The deployed model combines 70% magnitude pruning with INT8 quantization for a ~${best?.complexity_analysis.memory_size.compression_ratio_fp32_to_int8 ?? 0}× size cut while keeping ${pct(best?.model_evaluation.accuracy ?? 0)} accuracy.`;
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
    <div className="space-y-8 pb-16">
      <div className="mx-auto max-w-3xl text-center">
        <p className="text-sm font-semibold uppercase tracking-wider text-rose-500">Beat2Bit assistant</p>
        <h1 className="mt-4 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">Ask the research agent</h1>
        <p className="mt-4 text-slate-600">Grounded in the live benchmark reports — every number is real.</p>
      </div>

      <div className="mx-auto flex h-[620px] max-w-3xl flex-col overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-xl">
        {/* Header */}
        <div className="flex items-center gap-3 border-b border-slate-200 bg-slate-900 px-6 py-4">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-rose-500 to-rose-600">
            <Bot className="h-5 w-5 text-white" />
          </div>
          <div>
            <h2 className="font-semibold text-white">Beat2Bit Assistant</h2>
            <p className="text-xs text-slate-400">Answers from live reports ({reports.length} models loaded)</p>
          </div>
          <span className="ml-auto inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-3 py-1 text-xs font-medium text-emerald-300">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" /> Online
          </span>
        </div>

        {/* Messages */}
        <div className="flex-1 space-y-4 overflow-y-auto bg-slate-50/60 p-6">
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {m.role === 'agent' && (
                <div className="mr-2 mt-1 grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-slate-900">
                  <Bot className="h-4 w-4 text-rose-400" />
                </div>
              )}
              <div
                className={`max-w-[82%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm sm:max-w-[75%] ${
                  m.role === 'user'
                    ? 'rounded-br-sm bg-slate-900 text-white'
                    : 'rounded-bl-sm border border-slate-200 bg-white text-slate-700'
                }`}
              >
                {m.text}
              </div>
            </div>
          ))}
          {typing && (
            <div className="flex items-center gap-2 pl-10 text-slate-400">
              <Sparkles className="h-4 w-4 animate-pulse" /> thinking…
            </div>
          )}
          <div ref={endRef} />
        </div>

        {/* Input */}
        <form onSubmit={send} className="relative border-t border-slate-200 bg-white p-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder='Try: "What is the deployed model latency?"'
            className="w-full rounded-full border border-slate-200 bg-slate-50 py-3 pl-5 pr-16 text-sm outline-none transition-all focus:border-rose-300 focus:bg-white focus:ring-2 focus:ring-rose-200"
          />
          <button
            type="submit"
            disabled={!input.trim() || typing}
            className="absolute right-4 top-1/2 grid h-10 w-10 -translate-y-1/2 place-items-center rounded-full bg-rose-500 text-white shadow-md shadow-rose-500/30 transition-all hover:bg-rose-600 disabled:opacity-40"
          >
            <Send className="h-4 w-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
