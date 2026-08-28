"use client"

import React, { useState, useRef, useEffect } from 'react'
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Activity, Cpu, LineChart, Database, ShieldAlert, Bot,
  Send, ArrowRight, ShieldCheck, BatteryCharging, WifiOff,
  HeartPulse, Menu, X, CheckCircle2
} from 'lucide-react'
import { ECGChart } from '@/components/ECGChart'
import { getReportList, getReportByName, ReportData } from '@/lib/researchService'

export default function Beat2BitWebsite() {
  const [activeTab, setActiveTab] = useState('research')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const [chatMessages, setChatMessages] = useState([
    { role: 'agent', text: "Hi! I am the Beat2Bit AI assistant. Ask me anything about our ultra-low-power ECG detection model." }
  ])
  const [currentMessage, setCurrentMessage] = useState("")
  const chatEndRef = useRef<HTMLDivElement>(null)

  // Research tab state
  const [reportList, setReportList] = useState<Array<{ name: string; date: string; type: string }>>([])
  const [fullReports, setFullReports] = useState<ReportData[]>([])
  const [selectedReport, setSelectedReport] = useState<ReportData | null>(null)
  const [loading, setLoading] = useState<boolean>(true)

  // Load reports when component mounts or when research tab becomes active
  useEffect(() => {
    async function loadInitialReports() {
      setLoading(true)
      try {
        const reportList = await getReportList()
        setReportList(reportList)
        // Load full report data for all reports
        if (reportList.length > 0) {
          const fullReports = await Promise.all(
            reportList.map(report => getReportByName(report.name))
          )
          const filteredReports = fullReports.filter((report): report is ReportData => report !== null)
          setFullReports(filteredReports)
          // Load the first report by default if available
          if (fullReports.length > 0) {
            setSelectedReport(fullReports[0])
          }
        }
      } catch (error) {
        console.error('Error loading reports:', error)
      } finally {
        setLoading(false)
      }
    }

    loadInitialReports()
  }, [])

  // Reload reports when switching to research tab
  useEffect(() => {
    if (activeTab === 'research') {
      const loadResearchData = async () => {
        setLoading(true)
        try {
          const reportList = await getReportList()
          setReportList(reportList)
          // Load full report data for all reports
          if (reportList.length > 0) {
            const fullReports = await Promise.all(
              reportList.map(report => getReportByName(report.name))
            )
            const filteredReports = fullReports.filter((report): report is ReportData => report !== null)
            setFullReports(filteredReports)
            // Load the first report by default if available
            if (filteredReports.length > 0) {
              setSelectedReport(filteredReports[0])
            }
          }
        } catch (error) {
          console.error('Error loading research data:', error)
        } finally {
          setLoading(false)
        }
      }

      loadResearchData()
    }
  }, [activeTab])

const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault()
    if (!currentMessage.trim()) return

    const userMsg = { role: 'user', text: currentMessage }
    setChatMessages(prev => [...prev, userMsg])
    setCurrentMessage("")

    setTimeout(() => {
      const lower = userMsg.text.toLowerCase()
      let reply = "That's a great question! Beat2Bit is continuously learning. Basically, we use highly compressed 1D CNNs to run real-time ECG analysis right on the edge."
      if (lower.includes('dataset') || lower.includes('data')) {
        reply = "We train on 3 major datasets: MIT-BIH Arrhythmia, PTB-XL, and the European ST-T database to ensure robust generalization."
      } else if (lower.includes('power') || lower.includes('battery') || lower.includes('energy')) {
        reply = "Beat2Bit uses INT8 quantization and magnitude pruning to reduce memory footprint by 4x and minimize inference energy, enabling it to run on microcontrollers for weeks."
      } else if (lower.includes('disease') || lower.includes('detect')) {
        reply = "We primarily detect Arrhythmias, such as Premature Ventricular Contractions (PVC), Atrial Fibrillation (AFib), and Atrial Premature Contractions (APC)."
      }

      setChatMessages(prev => [...prev, { role: 'agent', text: reply }])
    }, 1000)
  }

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [chatMessages])

  const navItems = [
    { id: 'product', label: 'Product', icon: Activity },
    { id: 'datasets', label: 'Datasets', icon: Database },
    { id: 'diseases', label: 'Diseases', icon: ShieldAlert },
    { id: 'dashboard', label: 'Dashboard', icon: LineChart },
    { id: 'model', label: 'Model Pipeline', icon: Cpu },
    { id: 'agent', label: 'AI Agent', icon: Bot },
    { id: 'research', label: 'Research', icon: Bot },
  ]

  const switchTab = (id: string) => {
    setActiveTab(id)
    setMobileMenuOpen(false)
  }

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 selection:bg-rose-100 selection:text-rose-900">
      
      {/* Enterprise Sticky Navigation */}
      <nav className="sticky top-0 z-50 w-full border-b bg-white/80 backdrop-blur-md shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            
            {/* Logo */}
            <div className="flex-shrink-0 flex items-center gap-2 cursor-pointer" onClick={() => switchTab('product')}>
              <div className="bg-rose-500 p-1.5 rounded-lg">
                <HeartPulse className="h-5 w-5 text-white" />
              </div>
              <span className="font-bold text-xl tracking-tight text-slate-900">Beat2Bit</span>
            </div>

            {/* Desktop Nav */}
            <div className="hidden md:flex space-x-1 lg:space-x-2">
              {navItems.map((item) => {
                const Icon = item.icon
                const isActive = activeTab === item.id
                return (
                  <button
                    key={item.id}
                    onClick={() => switchTab(item.id)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                      isActive 
                        ? 'bg-slate-100 text-slate-900 shadow-sm ring-1 ring-slate-200' 
                        : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                    }`}
                  >
                    <Icon className={`w-4 h-4 ${isActive ? 'text-rose-500' : 'text-slate-400'}`} />
                    {item.label}
                  </button>
                )
              })}
            </div>

            {/* Mobile Menu Button */}
            <div className="flex md:hidden items-center">
              <button 
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="text-slate-600 hover:text-slate-900 focus:outline-none p-2"
              >
                {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Nav Dropdown */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t bg-white absolute w-full shadow-lg pb-4">
            <div className="px-2 pt-2 pb-3 space-y-1">
              {navItems.map((item) => {
                const Icon = item.icon
                const isActive = activeTab === item.id
                return (
                  <button
                    key={item.id}
                    onClick={() => switchTab(item.id)}
                    className={`flex items-center gap-3 w-full px-3 py-3 rounded-md text-base font-medium ${
                      isActive ? 'bg-rose-50 text-rose-700' : 'text-slate-700 hover:bg-slate-50'
                    }`}
                  >
                    <Icon className={`w-5 h-5 ${isActive ? 'text-rose-500' : 'text-slate-400'}`} />
                    {item.label}
                  </button>
                )
              })}
            </div>
          </div>
        )}
      </nav>

      {/* Main Content Area */}
      <main className="relative min-h-[calc(100vh-4rem)] overflow-hidden">
        
        {/* Global Background Effects */}
        <div className="absolute inset-0 -z-10 h-full w-full bg-slate-50 bg-[radial-gradient(#e2e8f0_1px,transparent_1px)] [background-size:16px_16px]">
          <div className="absolute left-0 right-0 top-0 -z-10 m-auto h-[310px] w-[310px] rounded-full bg-rose-400 opacity-20 blur-[100px]"></div>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12">
          
          {/* 1. PRODUCT MARKETING TAB */}
          {activeTab === 'product' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-700 space-y-24 pb-20">
              {/* Hero Section */}
              <section className="text-center pt-10 md:pt-20">
                <Badge variant="outline" className="px-4 py-1.5 mb-6 text-sm font-medium bg-white/50 backdrop-blur-sm border-slate-200 text-slate-800 rounded-full shadow-sm">
                  <span className="flex h-2 w-2 rounded-full bg-rose-500 mr-2 animate-pulse"></span>
                  v1.0 is now live on the Edge
                </Badge>
                <h1 className="text-5xl md:text-7xl font-extrabold text-slate-900 tracking-tight max-w-4xl mx-auto leading-[1.1]">
                  Clinical-Grade ECG Detection, <br className="hidden md:block"/>
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-rose-500 to-orange-400">
                    Engineered for the Edge.
                  </span>
                </h1>
                <p className="mt-6 text-lg md:text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
                  Beat2Bit compresses state-of-the-art Deep Learning into a TinyML footprint. No cloud required. No privacy risks. Just real-time, ultra-low-power heart monitoring.
                </p>
                <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center items-center">
                  <Button size="lg" className="bg-slate-900 hover:bg-slate-800 text-white rounded-full px-8 h-14 text-base shadow-lg hover:shadow-xl transition-all" onClick={() => switchTab('model')}>
                    Discover the Model <ArrowRight className="ml-2 w-5 h-5"/>
                  </Button>
                  <Button size="lg" variant="outline" className="bg-white/50 backdrop-blur-sm rounded-full px-8 h-14 text-base border-slate-200 hover:bg-white" onClick={() => switchTab('dashboard')}>
                    View Live Dashboard
                  </Button>
                </div>
              </section>

              {/* Value Props Section */}
              <section className="grid md:grid-cols-3 gap-8 md:gap-12">
                {[
                  {
                    icon: WifiOff,
                    title: "100% Offline Inference",
                    desc: "By running inferences entirely on the microcontroller, Beat2Bit removes reliance on internet connectivity, guaranteeing functionality anywhere in the world.",
                    color: "text-slate-700",
                    bg: "bg-slate-100"
                  },
                  {
                    icon: BatteryCharging,
                    title: "Ultra-Low Power",
                    desc: "Through INT8 quantization and magnitude pruning, our AI uses microjoules of energy per inference, extending wearable battery life from days to weeks.",
                    color: "text-emerald-600",
                    bg: "bg-emerald-100"
                  },
                  {
                    icon: ShieldCheck,
                    title: "Privacy Preserving",
                    desc: "Patient biometric data never leaves the device. Data is processed, classified, and discarded locally, inherently complying with HIPAA and GDPR.",
                    color: "text-blue-600",
                    bg: "bg-blue-100"
                  }
                ].map((feature, idx) => (
                  <div key={idx} className="relative group p-8 bg-white border border-slate-200 rounded-3xl shadow-sm hover:shadow-md transition-all">
                    <div className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-6 ${feature.bg}`}>
                      <feature.icon className={`w-7 h-7 ${feature.color}`} />
                    </div>
                    <h3 className="text-xl font-bold text-slate-900 mb-3">{feature.title}</h3>
                    <p className="text-slate-600 leading-relaxed">{feature.desc}</p>
                  </div>
                ))}
              </section>
            </div>
          )}

          {/* 2. DATASETS TAB */}
          {activeTab === 'datasets' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 max-w-5xl mx-auto py-8">
              <div className="text-center mb-12">
                <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4 tracking-tight">Trained on the World&apos;s Data</h2>
                <p className="text-lg text-slate-600 max-w-2xl mx-auto">To ensure robust generalization across diverse demographics, Beat2Bit leverages three gold-standard clinical datasets.</p>
              </div>

              <div className="grid md:grid-cols-3 gap-6">
                {[
                  {
                    title: "MIT-BIH Arrhythmia",
                    desc: "The gold standard benchmark containing 48 half-hour ambulatory ECG recordings with 109,000+ beat annotations by multiple cardiologists.",
                    tags: ["360 Hz", "2 Channels", "109k Beats"],
                    color: "border-blue-200 bg-blue-50/50"
                  },
                  {
                    title: "PTB-XL Large-Scale",
                    desc: "A massive clinical dataset of 21,837 clinical 12-lead ECGs from 18,885 patients, covering diverse pathologies and demographics.",
                    tags: ["500 Hz", "12 Leads", "21k Records"],
                    color: "border-emerald-200 bg-emerald-50/50"
                  },
                  {
                    title: "European ST-T",
                    desc: "Used specifically to improve the model's robustness against ST-segment changes, ischemia, and varying baseline wander noise.",
                    tags: ["250 Hz", "2 Leads", "90 Records"],
                    color: "border-purple-200 bg-purple-50/50"
                  }
                ].map((ds, idx) => (
                  <div key={idx} className={`p-8 rounded-3xl border shadow-sm hover:shadow-md transition-shadow bg-white ${ds.color.split(' ')[0]} relative overflow-hidden group`}>
                    <div className={`absolute top-0 right-0 w-32 h-32 transform translate-x-8 -translate-y-8 rounded-full ${ds.color.split(' ')[1]} blur-2xl opacity-50 group-hover:opacity-100 transition-opacity`}></div>
                    <Database className="w-8 h-8 text-slate-700 mb-4 relative z-10" />
                    <h3 className="font-bold text-xl text-slate-900 mb-3 relative z-10">{ds.title}</h3>
                    <p className="text-slate-600 mb-6 relative z-10 leading-relaxed">{ds.desc}</p>
                    <div className="flex flex-wrap gap-2 relative z-10">
                      {ds.tags.map(tag => (
                        <Badge key={tag} variant="secondary" className="bg-white border-slate-200 text-slate-700">{tag}</Badge>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 3. DISEASES TAB */}
          {activeTab === 'diseases' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 max-w-4xl mx-auto py-8">
              <div className="mb-10 text-center">
                <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4 tracking-tight">Pathologies Detected</h2>
                <p className="text-lg text-slate-600">Beat2Bit&apos;s neural network performs highly accurate multi-class anomaly detection natively on-device.</p>
              </div>

              <div className="space-y-6">
                {[
                  {
                    title: "Normal Sinus Rhythm (N)",
                    desc: "The baseline electrical activity of a healthy heart. Characterized by a regular rate (60-100 bpm) and a typical P-QRS-T wave sequence.",
                    iconColor: "text-slate-500",
                    bg: "bg-slate-100 border-slate-200"
                  },
                  {
                    title: "Premature Ventricular Contractions (PVC)",
                    desc: "Abnormal heartbeats that begin in the ventricles. These disrupt the regular heart rhythm. While occasional PVCs are common, frequent occurrences may indicate underlying heart disease or risk of dangerous arrhythmias.",
                    iconColor: "text-rose-500",
                    bg: "bg-rose-50 border-rose-100"
                  },
                  {
                    title: "Atrial Premature Contractions (APC)",
                    desc: "Extra heartbeats originating in the atria. They cause the heart to beat prematurely, leading to a compensatory pause. Often benign but can be precursors to atrial fibrillation.",
                    iconColor: "text-orange-500",
                    bg: "bg-orange-50 border-orange-100"
                  },
                  {
                    title: "Atrial Fibrillation (AFib)",
                    desc: "A quivering or irregular heartbeat (arrhythmia) that can lead to blood clots, stroke, heart failure and other heart-related complications. Highly critical to detect accurately.",
                    iconColor: "text-purple-600",
                    bg: "bg-purple-50 border-purple-100"
                  }
                ].map((disease, idx) => (
                  <div key={idx} className="flex flex-col md:flex-row gap-6 p-6 md:p-8 bg-white border border-slate-200 rounded-3xl shadow-sm hover:shadow-md transition-shadow">
                    <div className="md:w-1/5 shrink-0">
                      <div className={`aspect-square rounded-2xl flex items-center justify-center border ${disease.bg}`}>
                        <Activity className={`w-10 h-10 ${disease.iconColor}`} />
                      </div>
                    </div>
                    <div className="md:w-4/5 flex flex-col justify-center">
                      <h3 className="text-xl font-bold text-slate-900 mb-2">{disease.title}</h3>
                      <p className="text-slate-600 leading-relaxed">{disease.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 4. DASHBOARD TAB */}
          {activeTab === 'dashboard' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 py-4">
              <div className="mb-8">
                <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Live Device Telemetry</h2>
                <p className="text-slate-500">Monitoring real-time inference metrics from the edge microcontroller.</p>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                <div className="p-6 bg-slate-900 rounded-2xl text-white shadow-lg">
                  <p className="text-sm font-medium text-slate-400 mb-1">Inference Latency</p>
                  <div className="text-3xl font-bold">12.4 <span className="text-lg text-slate-300 font-normal">ms</span></div>
                  <p className="text-xs text-slate-500 mt-2">per heartbeat window</p>
                </div>
                <div className="p-6 bg-white border border-slate-200 rounded-2xl shadow-sm">
                  <p className="text-sm font-medium text-slate-500 mb-1">Energy Profile</p>
                  <div className="text-3xl font-bold text-emerald-600">15 <span className="text-lg font-normal">µJ</span></div>
                  <p className="text-xs text-slate-500 mt-2">per inference</p>
                </div>
                <div className="p-6 bg-white border border-slate-200 rounded-2xl shadow-sm">
                  <p className="text-sm font-medium text-slate-500 mb-1">Model RAM (SRAM)</p>
                  <div className="text-3xl font-bold text-blue-600">42 <span className="text-lg font-normal">KB</span></div>
                  <p className="text-xs text-slate-500 mt-2">peak memory footprint</p>
                </div>
                <div className="p-6 bg-white border border-slate-200 rounded-2xl shadow-sm">
                  <p className="text-sm font-medium text-slate-500 mb-1">Model Accuracy</p>
                  <div className="text-3xl font-bold text-slate-900">97.8<span className="text-lg font-normal">%</span></div>
                  <p className="text-xs text-slate-500 mt-2">F1 Score (AAMI split)</p>
                </div>
              </div>
              
              <div className="bg-white p-2 rounded-3xl shadow-sm border border-slate-200">
                <ECGChart />
              </div>
            </div>
          )}

          {/* 5. MODEL OVERVIEW TAB */}
          {activeTab === 'model' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 max-w-4xl mx-auto py-8">
              <div className="text-center mb-12">
                <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4 tracking-tight">The Beat2Bit Architecture</h2>
                <p className="text-lg text-slate-600">From Python to Silicon: How we shrink deep learning to fit on a microcontroller.</p>
              </div>

              <div className="relative border-l-2 border-slate-200 ml-4 md:ml-8 space-y-12 pb-8">
                
                <div className="relative pl-8 md:pl-12">
                  <div className="absolute -left-4 top-1 h-8 w-8 rounded-full bg-blue-100 border-4 border-white flex items-center justify-center">
                    <div className="h-3 w-3 bg-blue-500 rounded-full"></div>
                  </div>
                  <h3 className="text-xl font-bold text-slate-900 mb-2">1. Baseline 1D CNN</h3>
                  <p className="text-slate-600 leading-relaxed mb-4">We begin with a highly parameterized 1D Convolutional Neural Network trained on raw floating-point 32 (FP32) arrays. The model uses spatial convolutions to automatically extract morphological features like QRS complex widening or inverted T-waves.</p>
                  <div className="inline-flex items-center text-sm font-mono bg-slate-100 text-slate-700 px-3 py-1 rounded-md">Size: ~250 KB (FP32)</div>
                </div>

                <div className="relative pl-8 md:pl-12">
                  <div className="absolute -left-4 top-1 h-8 w-8 rounded-full bg-emerald-100 border-4 border-white flex items-center justify-center">
                    <div className="h-3 w-3 bg-emerald-500 rounded-full"></div>
                  </div>
                  <h3 className="text-xl font-bold text-slate-900 mb-2">2. INT8 Quantization</h3>
                  <p className="text-slate-600 leading-relaxed mb-4">Floating-point operations are extremely power-hungry. We convert weights and activations from 32-bit floats to 8-bit integers via Post-Training Quantization (PTQ). This reduces model size by ~4x and massively drops energy consumption, usually with less than a 1% drop in accuracy.</p>
                  <div className="inline-flex items-center text-sm font-mono bg-slate-100 text-slate-700 px-3 py-1 rounded-md">Size: ~65 KB (INT8)</div>
                </div>

                <div className="relative pl-8 md:pl-12">
                  <div className="absolute -left-4 top-1 h-8 w-8 rounded-full bg-purple-100 border-4 border-white flex items-center justify-center">
                    <div className="h-3 w-3 bg-purple-500 rounded-full"></div>
                  </div>
                  <h3 className="text-xl font-bold text-slate-900 mb-2">3. Magnitude Pruning</h3>
                  <p className="text-slate-600 leading-relaxed mb-4">We identify neural network weights near zero and aggressively prune them out. By achieving up to 60-70% sparsity, we significantly reduce the computational MAC (Multiply-Accumulate) operations required per beat.</p>
                  <div className="inline-flex items-center text-sm font-mono bg-slate-100 text-slate-700 px-3 py-1 rounded-md">Sparsity: 65%</div>
                </div>

                <div className="relative pl-8 md:pl-12">
                  <div className="absolute -left-4 top-1 h-8 w-8 rounded-full bg-rose-100 border-4 border-white flex items-center justify-center">
                    <CheckCircle2 className="h-5 w-5 text-rose-500" />
                  </div>
                  <h3 className="text-xl font-bold text-slate-900 mb-2">4. TFLite Micro Deployment</h3>
                  <p className="text-slate-600 leading-relaxed">The final optimized model is converted into a C byte array and flashed directly onto the microcontroller&apos;s ROM alongside the TensorFlow Lite for Microcontrollers inference engine.</p>
                </div>

              </div>
            </div>
          )}

          {/* 6. AI AGENT TAB */}
          {activeTab === 'agent' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 max-w-3xl mx-auto py-4">
              <div className="bg-white rounded-3xl border border-slate-200 shadow-xl overflow-hidden flex flex-col h-[700px]">
                
                {/* Chat Header */}
                <div className="bg-slate-900 text-white p-6 flex items-center gap-4">
                  <div className="bg-blue-500/20 p-2 rounded-xl">
                    <Bot className="w-8 h-8 text-blue-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold">Beat2Bit Assistant</h2>
                    <p className="text-sm text-slate-400">Ask me about the architecture, datasets, or edge metrics.</p>
                  </div>
                </div>
                
                {/* Chat Messages */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-slate-50/50">
                  {chatMessages.map((msg, idx) => (
                    <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      {msg.role === 'agent' && (
                        <div className="w-8 h-8 rounded-full bg-slate-900 flex items-center justify-center mr-3 shrink-0 mt-1">
                          <Bot className="w-4 h-4 text-white" />
                        </div>
                      )}
                      <div className={`max-w-[80%] md:max-w-[70%] rounded-2xl p-4 shadow-sm ${
                        msg.role === 'user' 
                          ? 'bg-slate-900 text-white rounded-tr-sm' 
                          : 'bg-white border border-slate-200 text-slate-800 rounded-tl-sm'
                      }`}>
                        <p className="text-sm md:text-base leading-relaxed">{msg.text}</p>
                      </div>
                    </div>
                  ))}
                  <div ref={chatEndRef} />
                </div>
                
                {/* Chat Input */}
                <div className="p-4 bg-white border-t border-slate-200">
                  <form onSubmit={handleSendMessage} className="flex gap-3 max-w-4xl mx-auto relative">
                    <input
                      type="text"
                      value={currentMessage}
                      onChange={(e) => setCurrentMessage(e.target.value)}
                      placeholder="E.g., How much power does the model consume?"
                      className="flex-1 pl-6 pr-14 py-4 bg-slate-50 border border-slate-200 rounded-full text-sm md:text-base focus:outline-none focus:ring-2 focus:ring-rose-500 focus:bg-white transition-all shadow-inner"
                    />
                    <Button 
                      type="submit" 
                      size="icon" 
                      className="absolute right-2 top-2 bottom-2 h-auto w-10 rounded-full bg-rose-500 hover:bg-rose-600 shadow-md"
                    >
                      <Send className="w-4 h-4 text-white" />
                    </Button>
                  </form>
                </div>

              </div>
            </div>
          )}

          {/* 7. RESEARCH TAB */}
          {activeTab === 'research' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 max-w-4xl mx-auto py-8">
              <div className="mb-10 text-center">
                <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4 tracking-tight">Research Details</h2>
                <p className="text-lg text-slate-600">Comprehensive analysis of model performance, optimization trade-offs, and validation results.</p>
              </div>

              {/* Research Content */}
              <div className="space-y-8">
                {/* Model Optimization Trade-offs */}
                <div className="p-6 bg-white border border-slate-200 rounded-3xl shadow-sm hover:shadow-md transition-shadow">
                  <h3 className="text-xl font-bold text-slate-900 mb-4">Model Optimization Trade-offs</h3>
                  <p className="text-slate-600 mb-4">
                    Analysis of how pruning and quantization affect model accuracy, size, and latency.
                  </p>
                  {!loading && fullReports.length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-slate-200">
                        <thead className="bg-slate-50">
                          <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                              Model
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                              Accuracy
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                              Latency (ms)
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                              Size (KB)
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                              FLOPs (M)
                            </th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-200">
                          {fullReports.map((report, index) => (
                            <tr key={index} className="bg-white hover:bg-slate-50">
                              <td className="px-6 py-4 text-sm font-medium text-slate-900">
                                {report.metadata?.model_name?.replace(/_/g, ' ').toUpperCase() || 'Unknown'}
                              </td>
                              <td className="px-6 py-4 text-sm text-slate-600">
                                {(report.summary?.model_performance?.accuracy || 0) * 100}.toFixed(1)%
                              </td>
                              <td className="px-6 py-4 text-sm text-slate-600">
                                {(report.latency_benchmarking?.latency_stats?.batch_size_1?.mean_latency_ms || 0).toFixed(1)}
                              </td>
                              <td className="px-6 py-4 text-sm text-slate-600">
                                {(report.complexity_analysis?.memory_size?.fp32_mb || 0) * 1024}.toFixed(0)
                              </td>
                              <td className="px-6 py-4 text-sm text-slate-600">
                                {(report.complexity_analysis?.computational_complexity?.total_flops || 0) / 1000000}.toFixed(1)
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="text-slate-500">Loading report data...</p>
                  )}
                </div>

                {/* Validation Results */}
                <div className="p-6 bg-white border border-slate-200 rounded-3xl shadow-sm hover:shadow-md transition-shadow">
                  <h3 className="text-xl font-bold text-slate-900 mb-4">Validation Results</h3>
                  <p className="text-slate-600 mb-4">
                    AAMI EC57 compliant metrics and statistical significance testing.
                  </p>
                  {!loading && selectedReport ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-slate-500">Accuracy</p>
                          <p className="font-medium">{selectedReport.model_evaluation.accuracy.toFixed(2) + '%'}</p>
                        </div>
                        <div>
                          <p className="text-slate-500">F1 Score</p>
                          <p className="font-medium">{selectedReport.model_evaluation.f1_score.toFixed(2) + '%'}</p>
                        </div>
                        <div>
                          <p className="text-slate-500">AMI Sensitivity</p>
                          <p className="font-medium">{selectedReport.model_evaluation.ami_sensitivity.toFixed(2) + '%'}</p>
                        </div>
                        <div>
                          <p className="text-slate-500">AMI +P</p>
                          <p className="font-medium">{selectedReport.model_evaluation.ami_positive_predictivity.toFixed(2) + '%'}</p>
                        </div>
                        <div>
                          <p className="text-slate-500">AMI Effectiveness</p>
                          <p className="font-medium">{selectedReport.model_evaluation.ami_effectiveness.toFixed(2) + '%'}</p>
                        </div>
                        <div>
                          <p className="text-slate-500">Latency (ms)</p>
                          <p className="font-medium">{selectedReport.latency_benchmarking.latency_stats.batch_size_1.mean_latency_ms.toFixed(2)}</p>
                        </div>
                        <div>
                          <p className="text-slate-500">Model Size (KB)</p>
                          <p className="font-medium">{(selectedReport.complexity_analysis.memory_size.fp32_mb * 1024).toFixed(0)}</p>
                        </div>
                      </div>
                      <div className="mt-4 p-4 bg-slate-50 rounded-md">
                        <p className="text-slate-500">Confusion Matrix</p>
                        <div className="mt-2">
                          <table className="text-sm">
                            <thead>
                              <tr>
                                <th></th>
                                <th className="text-left">Predicted Negative</th>
                                <th className="text-left">Predicted Positive</th>
                              </tr>
                            </thead>
                            <tbody>
                              <tr>
                                <th className="text-left">Actual Negative</th>
                                <td>{selectedReport.model_evaluation.confusion_matrix.tn}</td>
                                <td>{selectedReport.model_evaluation.confusion_matrix.fp}</td>
                              </tr>
                              <tr>
                                <th className="text-left">Actual Positive</th>
                                <td>{selectedReport.model_evaluation.confusion_matrix.fn}</td>
                                <td>{selectedReport.model_evaluation.confusion_matrix.tp}</td>
                              </tr>
                            </tbody>
                          </table>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-slate-500">Loading validation data...</p>
                  )}
                </div>

                {/* Downloadable Reports */}
                <div className="p-6 bg-white border border-slate-200 rounded-3xl shadow-sm hover:shadow-md transition-shadow">
                  <h3 className="text-xl font-bold text-slate-900 mb-4">Downloadable Reports</h3>
                  <p className="text-slate-600 mb-4">
                    Access detailed benchmarking reports and research documentation.
                  </p>
                  {!loading && reportList.length > 0 ? (
                    <div className="space-y-3">
                      {reportList.map((report, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-slate-50 rounded-md">
                          <div className="flex items-center gap-3">
                            <div className="bg-slate-200 p-2 rounded-md">
                              <Activity className="w-4 h-4 text-slate-600" />
                            </div>
                            <div>
                              <p className="text-sm font-medium text-slate-900">{report.name.replace(/_/g, ' ').toUpperCase()}</p>
                              <p className="text-xs text-slate-500">{report.type} • {report.date}</p>
                            </div>
                          </div>
                          <Button
                            variant="outline"
                            size="sm"
                            className="text-slate-600 hover:text-slate-900 border-slate-200"
                            onClick={() => {
                              // Simulate download - in a real app, this would trigger a download
                              alert(`Downloading report for ${report.name}`);
                            }}
                          >
                            Download
                          </Button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-slate-500">Loading report list...</p>
                  )}
                </div>
              </div>
            </div>
          )}

        </div>
      </main>
    </div>
  )
}