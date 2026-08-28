import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Beat2Bit · Ultra-Low-Power Edge AI for ECG Arrhythmia Detection',
  description:
    'Research-driven TinyML: compression, quantization and pruning for real-time ECG arrhythmia detection on resource-constrained microcontrollers.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-sans antialiased bg-slate-50 text-slate-900">{children}</body>
    </html>
  );
}
