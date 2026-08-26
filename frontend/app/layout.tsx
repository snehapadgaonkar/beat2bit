import "./globals.css"

export const metadata = {
  title: "Edge AI ECG Arrhythmia Detection",
  description: "Research Pipeline & Interactive Dashboard",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="font-sans antialiased bg-slate-50 text-slate-900">{children}</body>
    </html>
  )
}
