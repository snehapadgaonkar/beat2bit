'use client';

import React from 'react';
import {
  HeartPulse,
  Menu,
  X,
  LayoutDashboard,
  FlaskConical,
  Activity,
  Cpu,
  Database,
  Bot,
} from 'lucide-react';
import { cn } from '@/lib/utils';

export type TabId = 'overview' | 'research' | 'dashboard' | 'model' | 'datasets' | 'agent';

export interface NavItem {
  id: TabId;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

export const NAV_ITEMS: NavItem[] = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'research', label: 'Research', icon: FlaskConical },
  { id: 'dashboard', label: 'Dashboard', icon: Activity },
  { id: 'model', label: 'Model', icon: Cpu },
  { id: 'datasets', label: 'Datasets', icon: Database },
  { id: 'agent', label: 'AI Agent', icon: Bot },
];

export function SiteHeader({
  activeTab,
  onNavigate,
  mobileOpen,
  onToggleMobile,
}: {
  activeTab: TabId;
  onNavigate: (id: TabId) => void;
  mobileOpen: boolean;
  onToggleMobile: () => void;
}) {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-200/80 bg-white/85 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Brand */}
        <button
          onClick={() => onNavigate('overview')}
          className="flex items-center gap-2.5 group"
          aria-label="Beat2Bit home"
        >
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-rose-500 to-rose-600 shadow-lg shadow-rose-500/30 transition-transform group-hover:scale-105">
            <HeartPulse className="h-5 w-5 text-white" />
          </span>
          <span className="text-lg font-bold tracking-tight text-slate-900">
            Beat<span className="text-rose-500">2</span>Bit
          </span>
        </button>

        {/* Desktop nav */}
        <nav className="hidden items-center gap-1 lg:flex">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={cn(
                  'flex items-center gap-2 rounded-full px-3.5 py-2 text-sm font-medium transition-all',
                  active
                    ? 'bg-slate-900 text-white shadow-sm'
                    : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                )}
              >
                <Icon className={cn('h-4 w-4', active ? 'text-rose-400' : 'text-slate-400')} />
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="hidden lg:block">
          <button
            onClick={() => onNavigate('research')}
            className="rounded-full bg-rose-500 px-4 py-2 text-sm font-semibold text-white shadow-md shadow-rose-500/30 transition-all hover:bg-rose-600"
          >
            View Research
          </button>
        </div>

        {/* Mobile toggle */}
        <button
          onClick={onToggleMobile}
          className="flex h-10 w-10 items-center justify-center rounded-lg text-slate-600 hover:bg-slate-100 lg:hidden"
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </div>

      {/* Mobile drawer */}
      <div
        className={cn(
          'lg:hidden overflow-hidden border-t border-slate-200/80 bg-white transition-all duration-300',
          mobileOpen ? 'max-h-[480px]' : 'max-h-0 border-t-0'
        )}
      >
        <nav className="space-y-1 px-3 py-4">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={cn(
                  'flex w-full items-center gap-3 rounded-xl px-4 py-3 text-base font-medium transition-colors',
                  active ? 'bg-rose-50 text-rose-700' : 'text-slate-700 hover:bg-slate-50'
                )}
              >
                <Icon className={cn('h-5 w-5', active ? 'text-rose-500' : 'text-slate-400')} />
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
