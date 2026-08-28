'use client';

import React, { useState, useEffect } from 'react';
import { SiteHeader, type TabId } from '@/components/SiteHeader';
import { SiteFooter } from '@/components/SiteFooter';
import { OverviewSection } from '@/components/sections/OverviewSection';
import { ResearchSection } from '@/components/sections/ResearchSection';
import { DashboardSection } from '@/components/sections/DashboardSection';
import { ModelSection } from '@/components/sections/ModelSection';
import { DatasetsSection } from '@/components/sections/DatasetsSection';
import { AgentSection } from '@/components/sections/AgentSection';
import { loadAllReports } from '@/lib/researchService';
import type { ReportData } from '@/lib/utils';

export default function Beat2BitWebsite() {
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [reports, setReports] = useState<ReportData[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedName, setSelectedName] = useState('pruned_quantized');

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const data = await loadAllReports();
        if (cancelled) return;
        setReports(data);
        if (data.length > 0) {
          const final = data.find((r) => r.metadata.model_name === 'pruned_quantized');
          setSelectedName(final?.metadata.model_name ?? data[0].metadata.model_name);
        }
      } catch (err) {
        console.error('Failed to load reports:', err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const navigate = (id: TabId) => {
    setActiveTab(id);
    setMobileOpen(false);
  };

  const sectionProps = { reports, loading, selectedName, onSelectName: setSelectedName };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 antialiased selection:bg-rose-100 selection:text-rose-900">
      <SiteHeader
        activeTab={activeTab}
        onNavigate={navigate}
        mobileOpen={mobileOpen}
        onToggleMobile={() => setMobileOpen((v) => !v)}
      />

      <main className="relative">
        <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
          <div className="absolute left-1/2 top-0 h-[420px] w-[700px] -translate-x-1/2 rounded-full bg-rose-200/30 blur-[120px]" />
        </div>
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 sm:py-10 lg:px-8">
          {activeTab === 'overview' && <OverviewSection reports={reports} loading={loading} onNavigate={navigate} />}
          {activeTab === 'research' && <ResearchSection {...sectionProps} />}
          {activeTab === 'dashboard' && <DashboardSection {...sectionProps} />}
          {activeTab === 'model' && <ModelSection reports={reports} />}
          {activeTab === 'datasets' && <DatasetsSection reports={reports} />}
          {activeTab === 'agent' && <AgentSection reports={reports} />}
        </div>
      </main>

      <SiteFooter />
    </div>
  );
}
