"use client"
import React, { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot } from 'recharts'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

// Simulate an ECG signal with R-peaks
const generateECGData = (isAbnormal: boolean) => {
  const data = []
  for (let i = 0; i < 300; i++) {
    let voltage = Math.sin(i * 0.1) * 0.1 // baseline wander
    
    // R-peak 1
    if (i > 40 && i < 60) {
      voltage += Math.exp(-Math.pow(i - 50, 2) / 10) * 1.5
    }
    
    // R-peak 2 (Abnormal if true)
    if (i > 140 && i < 170) {
      if (isAbnormal) {
        // PVC shape
        voltage -= Math.exp(-Math.pow(i - 150, 2) / 30) * 0.8
        voltage += Math.exp(-Math.pow(i - 160, 2) / 20) * 1.2
      } else {
        // Normal shape
        voltage += Math.exp(-Math.pow(i - 150, 2) / 10) * 1.5
      }
    }

    // R-peak 3
    if (i > 240 && i < 260) {
      voltage += Math.exp(-Math.pow(i - 250, 2) / 10) * 1.5
    }

    data.push({ time: i, voltage: voltage + (Math.random() * 0.05) }) // add noise
  }
  return data
}

export function ECGChart() {
  const [isAbnormal, setIsAbnormal] = useState(false)
  const [data, setData] = useState<any[]>([])

  useEffect(() => {
    setData(generateECGData(isAbnormal))
  }, [isAbnormal])

  return (
    <Card className="w-full shadow-sm">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Interactive ECG Viewer</CardTitle>
          <CardDescription>Simulated MIT-BIH Window (360 Hz)</CardDescription>
        </div>
        <div className="flex gap-2">
          <Badge 
            variant={!isAbnormal ? "default" : "outline"} 
            className="cursor-pointer px-3 py-1"
            onClick={() => setIsAbnormal(false)}
          >
            Normal (N)
          </Badge>
          <Badge 
            variant={isAbnormal ? "destructive" : "outline"} 
            className="cursor-pointer px-3 py-1"
            onClick={() => setIsAbnormal(true)}
          >
            Abnormal (V)
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="time" tick={false} axisLine={false} />
              <YAxis domain={[-1.5, 2]} tick={false} axisLine={false} />
              <Tooltip 
                contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                labelFormatter={() => "Sample"}
              />
              <Line 
                type="monotone" 
                dataKey="voltage" 
                stroke={isAbnormal ? "#ef4444" : "#0f172a"} 
                strokeWidth={2}
                dot={false}
                isAnimationActive={true}
              />
              {/* R-Peak Annotations */}
              <ReferenceDot x={50} y={1.5} r={4} fill="red" stroke="none" />
              <ReferenceDot x={150} y={isAbnormal ? 1.2 : 1.5} r={4} fill={isAbnormal ? "blue" : "red"} stroke="none" />
              <ReferenceDot x={250} y={1.5} r={4} fill="red" stroke="none" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  )
}