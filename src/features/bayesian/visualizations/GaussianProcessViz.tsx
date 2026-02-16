import { useState, useMemo, useCallback, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'
import { Toggle } from '../../../components/ui/Toggle'
import { Select } from '../../../components/ui/Select'
import { Button } from '../../../components/ui/Button'
import { GaussianProcess, makeSineData, makeGapData, makeLinearTrendData } from '../../../lib/algorithms/bayesian/gaussianProcess'
import { createKernel } from '../../../lib/algorithms/bayesian/kernels'
import type { KernelType } from '../../../lib/algorithms/bayesian/kernels'

// ── Colors ──────────────────────────────────────────────────────────
const COLORS = {
  mean: '#34D399',
  band1: 'rgba(99, 102, 241, 0.18)',
  band2: 'rgba(99, 102, 241, 0.08)',
  sample: '#6366F1',
  data: '#F4F4F5',
  prior: '#A1A1AA',
  gridLine: 'rgba(255,255,255,0.06)',
  kernelCurve: '#FBBF24',
}

const KERNEL_OPTIONS = [
  { value: 'rbf', label: 'RBF (Smooth)' },
  { value: 'matern', label: 'Matérn 3/2' },
  { value: 'periodic', label: 'Periodic' },
  { value: 'linear', label: 'Linear' },
]

const PRESET_OPTIONS = [
  { value: 'none', label: 'Click to add' },
  { value: 'sine', label: 'Sine wave' },
  { value: 'gap', label: 'Gap in middle' },
  { value: 'linear', label: 'Linear trend' },
]

const X_DOMAIN: [number, number] = [0, 10]
const Y_DOMAIN: [number, number] = [-3, 3]
const MAX_POINTS = 80

// ── Main GP panel ───────────────────────────────────────────────────
function GPMainPanel({
  gp,
  dataX,
  dataY,
  nSamples,
  showBands,
  showSamples,
  width,
  height,
  onAddPoint,
}: {
  gp: GaussianProcess
  dataX: number[]
  dataY: number[]
  nSamples: number
  showBands: boolean
  showSamples: boolean
  width: number
  height: number
  onAddPoint: (x: number, y: number) => void
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const pad = { top: 20, right: 20, bottom: 35, left: 45 }
  const w = width - pad.left - pad.right
  const h = height - pad.top - pad.bottom

  const xScale = (v: number) => pad.left + ((v - X_DOMAIN[0]) / (X_DOMAIN[1] - X_DOMAIN[0])) * w
  const yScale = (v: number) => pad.top + h - ((v - Y_DOMAIN[0]) / (Y_DOMAIN[1] - Y_DOMAIN[0])) * h
  const xInv = (px: number) => X_DOMAIN[0] + ((px - pad.left) / w) * (X_DOMAIN[1] - X_DOMAIN[0])
  const yInv = (py: number) => Y_DOMAIN[0] + ((pad.top + h - py) / h) * (Y_DOMAIN[1] - Y_DOMAIN[0])

  // Prediction range
  const xRange = useMemo(() => {
    const pts: number[] = []
    for (let i = 0; i <= 120; i++) pts.push(X_DOMAIN[0] + (X_DOMAIN[1] - X_DOMAIN[0]) * (i / 120))
    return pts
  }, [])

  const prediction = useMemo(
    () => gp.predict(xRange),
    [gp, xRange, dataX, dataY]
  )

  // ±1σ and ±2σ bands
  const band1Path = useMemo(() => {
    const upper = xRange.map((x, i) => `${i === 0 ? 'M' : 'L'}${xScale(x)},${yScale(prediction.mean[i] + Math.sqrt(prediction.variance[i]))}`)
    const lower = [...xRange].reverse().map((x, i) => `L${xScale(x)},${yScale(prediction.mean[xRange.length - 1 - i] - Math.sqrt(prediction.variance[xRange.length - 1 - i]))}`)
    return upper.join(' ') + ' ' + lower.join(' ') + ' Z'
  }, [prediction, xRange])

  const band2Path = useMemo(() => {
    const upper = xRange.map((x, i) => `${i === 0 ? 'M' : 'L'}${xScale(x)},${yScale(prediction.mean[i] + 2 * Math.sqrt(prediction.variance[i]))}`)
    const lower = [...xRange].reverse().map((x, i) => `L${xScale(x)},${yScale(prediction.mean[xRange.length - 1 - i] - 2 * Math.sqrt(prediction.variance[xRange.length - 1 - i]))}`)
    return upper.join(' ') + ' ' + lower.join(' ') + ' Z'
  }, [prediction, xRange])

  // Mean line
  const meanPath = useMemo(
    () => xRange.map((x, i) => `${i === 0 ? 'M' : 'L'}${xScale(x)},${yScale(prediction.mean[i])}`).join(' '),
    [prediction, xRange]
  )

  // Function samples
  const samples = useMemo(() => {
    if (nSamples === 0) return []
    if (dataX.length === 0) return gp.samplePrior(xRange, nSamples, 42)
    return gp.samplePosterior(xRange, nSamples, 42)
  }, [gp, xRange, nSamples, dataX, dataY])

  const handleClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const rect = svgRef.current?.getBoundingClientRect()
      if (!rect) return
      const px = e.clientX - rect.left
      const py = e.clientY - rect.top
      const x = xInv(px)
      const y = yInv(py)
      if (x >= X_DOMAIN[0] && x <= X_DOMAIN[1] && y >= Y_DOMAIN[0] && y <= Y_DOMAIN[1]) {
        onAddPoint(x, y)
      }
    },
    [onAddPoint]
  )

  return (
    <svg ref={svgRef} width={width} height={height} className="block cursor-crosshair" onClick={handleClick}>
      {/* Grid */}
      {[0, 2, 4, 6, 8, 10].map((v) => (
        <line key={`gx-${v}`} x1={xScale(v)} y1={pad.top} x2={xScale(v)} y2={pad.top + h} stroke={COLORS.gridLine} />
      ))}
      {[-2, -1, 0, 1, 2].map((v) => (
        <line key={`gy-${v}`} x1={pad.left} y1={yScale(v)} x2={pad.left + w} y2={yScale(v)} stroke={COLORS.gridLine} />
      ))}

      {/* ±2σ band */}
      {showBands && <path d={band2Path} fill={COLORS.band2} />}
      {/* ±1σ band */}
      {showBands && <path d={band1Path} fill={COLORS.band1} />}

      {/* Function samples */}
      {showSamples && samples.map((sample, si) => {
        const path = xRange.map((x, i) => {
          const yVal = Math.max(Y_DOMAIN[0] - 1, Math.min(Y_DOMAIN[1] + 1, sample[i]))
          return `${i === 0 ? 'M' : 'L'}${xScale(x)},${yScale(yVal)}`
        }).join(' ')
        return (
          <path
            key={si}
            d={path}
            fill="none"
            stroke={dataX.length === 0 ? COLORS.prior : COLORS.sample}
            strokeWidth={1.2}
            strokeOpacity={dataX.length === 0 ? 0.3 : 0.25}
          />
        )
      })}

      {/* Mean line */}
      <path d={meanPath} fill="none" stroke={COLORS.mean} strokeWidth={2.5} />

      {/* Data points */}
      {dataX.map((x, i) => (
        <motion.circle
          key={i}
          cx={xScale(x)}
          cy={yScale(dataY[i])}
          r={4.5}
          fill={COLORS.data}
          fillOpacity={0.9}
          stroke={COLORS.data}
          strokeWidth={1}
          strokeOpacity={0.3}
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
        />
      ))}

      {/* Axes */}
      {[0, 2, 4, 6, 8, 10].map((v) => (
        <text key={`tx-${v}`} x={xScale(v)} y={pad.top + h + 16} textAnchor="middle" className="text-[9px] fill-text-tertiary font-mono">{v}</text>
      ))}
      {[-2, -1, 0, 1, 2].map((v) => (
        <text key={`ty-${v}`} x={pad.left - 8} y={yScale(v) + 3} textAnchor="end" className="text-[9px] fill-text-tertiary font-mono">{v}</text>
      ))}
      <text x={pad.left + w / 2} y={height - 3} textAnchor="middle" className="text-[10px] fill-text-tertiary font-mono">x</text>
      <text x={10} y={pad.top + h / 2} textAnchor="middle" transform={`rotate(-90, 10, ${pad.top + h / 2})`} className="text-[10px] fill-text-tertiary font-mono">f(x)</text>

      {/* Click hint */}
      {dataX.length === 0 && (
        <text x={pad.left + w / 2} y={pad.top + h / 2 - 10} textAnchor="middle" className="text-[11px] fill-text-tertiary">
          Click to add data points
        </text>
      )}
    </svg>
  )
}

// ── Kernel explorer panel ───────────────────────────────────────────
function KernelPanel({
  kernelType,
  lengthScale,
  signalVariance,
  width,
  height,
}: {
  kernelType: KernelType
  lengthScale: number
  signalVariance: number
  width: number
  height: number
}) {
  const kernel = useMemo(
    () => createKernel(kernelType, lengthScale, signalVariance),
    [kernelType, lengthScale, signalVariance]
  )

  const pad = { top: 20, right: 15, bottom: 30, left: 40 }
  const w = width - pad.left - pad.right
  const h = height - pad.top - pad.bottom

  // Plot k(0, d) for d from 0 to 5
  const dMax = 5
  const nPts = 80
  const pts = useMemo(() => {
    const result: { d: number; k: number }[] = []
    let maxK = 0
    for (let i = 0; i <= nPts; i++) {
      const d = (i / nPts) * dMax
      const k = kernel.compute(5, 5 + d) // Use x=5 as reference
      if (Math.abs(k) > maxK) maxK = Math.abs(k)
      result.push({ d, k })
    }
    return { pts: result, maxK: Math.max(maxK, 0.01) }
  }, [kernel])

  const xScale = (d: number) => pad.left + (d / dMax) * w
  const yScale = (k: number) => pad.top + h / 2 - (k / (pts.maxK * 1.2)) * (h / 2)

  const linePath = pts.pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(p.d)},${yScale(p.k)}`).join(' ')

  return (
    <svg width={width} height={height} className="block">
      {/* Zero line */}
      <line x1={pad.left} y1={yScale(0)} x2={pad.left + w} y2={yScale(0)} stroke={COLORS.gridLine} />

      {/* Kernel curve */}
      <path d={linePath} fill="none" stroke={COLORS.kernelCurve} strokeWidth={2} />

      {/* Labels */}
      <text x={pad.left + w / 2} y={12} textAnchor="middle" className="text-[10px] fill-text-secondary font-medium">{kernel.name} Kernel</text>
      <text x={pad.left + w / 2} y={height - 4} textAnchor="middle" className="text-[9px] fill-text-tertiary font-mono">|x - x'|</text>
      <text x={8} y={pad.top + h / 2} textAnchor="middle" transform={`rotate(-90, 8, ${pad.top + h / 2})`} className="text-[9px] fill-text-tertiary font-mono">k(x, x')</text>

      {/* Ticks */}
      {[0, 1, 2, 3, 4, 5].map((v) => (
        <text key={v} x={xScale(v)} y={pad.top + h + 14} textAnchor="middle" className="text-[8px] fill-text-tertiary font-mono">{v}</text>
      ))}
    </svg>
  )
}

// ── Kernel comparison strip ─────────────────────────────────────────
function KernelComparisonStrip({
  dataX,
  dataY,
  lengthScale,
  signalVariance,
  noiseVariance,
  width,
}: {
  dataX: number[]
  dataY: number[]
  lengthScale: number
  signalVariance: number
  noiseVariance: number
  width: number
}) {
  const kernelTypes: KernelType[] = ['rbf', 'matern', 'periodic', 'linear']
  const panelW = Math.floor((width - 36) / 4)
  const panelH = 140

  return (
    <div className="flex gap-2 overflow-x-auto">
      {kernelTypes.map((kt) => {
        const kernel = createKernel(kt, lengthScale, signalVariance)
        const gp = new GaussianProcess(kernel, noiseVariance)
        gp.fit(dataX, dataY)

        const xRange: number[] = []
        for (let i = 0; i <= 60; i++) xRange.push(X_DOMAIN[0] + (X_DOMAIN[1] - X_DOMAIN[0]) * (i / 60))
        const pred = gp.predict(xRange)

        const pad = { top: 16, right: 8, bottom: 8, left: 8 }
        const w = panelW - pad.left - pad.right
        const h = panelH - pad.top - pad.bottom

        const xs = (v: number) => pad.left + ((v - X_DOMAIN[0]) / (X_DOMAIN[1] - X_DOMAIN[0])) * w
        const ys = (v: number) => pad.top + h - ((v - Y_DOMAIN[0]) / (Y_DOMAIN[1] - Y_DOMAIN[0])) * h

        const bandPath =
          xRange.map((x, i) => `${i === 0 ? 'M' : 'L'}${xs(x)},${ys(pred.mean[i] + Math.sqrt(pred.variance[i]))}`)
            .join(' ') + ' ' +
          [...xRange].reverse().map((x, i) => `L${xs(x)},${ys(pred.mean[xRange.length - 1 - i] - Math.sqrt(pred.variance[xRange.length - 1 - i]))}`)
            .join(' ') + ' Z'

        const meanPath = xRange.map((x, i) => `${i === 0 ? 'M' : 'L'}${xs(x)},${ys(pred.mean[i])}`).join(' ')

        return (
          <div key={kt} className="bg-obsidian-surface rounded-lg p-1 flex-shrink-0">
            <svg width={panelW} height={panelH}>
              <text x={panelW / 2} y={11} textAnchor="middle" className="text-[9px] fill-text-tertiary font-mono">{kernel.name}</text>
              <path d={bandPath} fill={COLORS.band1} />
              <path d={meanPath} fill="none" stroke={COLORS.mean} strokeWidth={1.5} />
              {dataX.map((x, i) => (
                <circle key={i} cx={xs(x)} cy={ys(dataY[i])} r={2} fill={COLORS.data} fillOpacity={0.7} />
              ))}
            </svg>
          </div>
        )
      })}
    </div>
  )
}

// ── Main Component ──────────────────────────────────────────────────
export function GaussianProcessViz() {
  const [dataX, setDataX] = useState<number[]>([])
  const [dataY, setDataY] = useState<number[]>([])
  const [kernelType, setKernelType] = useState<KernelType>('rbf')
  const [lengthScale, setLengthScale] = useState(1.0)
  const [signalVariance, setSignalVariance] = useState(1.0)
  const [noiseVariance, setNoiseVariance] = useState(0.1)
  const [nSamples, setNSamples] = useState(5)
  const [showBands, setShowBands] = useState(true)
  const [showSamples, setShowSamples] = useState(true)
  const [preset, setPreset] = useState('none')
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(900)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) setContainerWidth(entry.contentRect.width)
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  // Build GP
  const gp = useMemo(() => {
    const kernel = createKernel(kernelType, lengthScale, signalVariance)
    const g = new GaussianProcess(kernel, noiseVariance)
    g.fit(dataX, dataY)
    return g
  }, [kernelType, lengthScale, signalVariance, noiseVariance, dataX, dataY])

  const handleAddPoint = useCallback((x: number, y: number) => {
    if (dataX.length >= MAX_POINTS) return
    setDataX((prev) => [...prev, x])
    setDataY((prev) => [...prev, y])
    setPreset('none')
  }, [dataX.length])

  const handleClear = useCallback(() => {
    setDataX([])
    setDataY([])
    setPreset('none')
  }, [])

  const handlePreset = useCallback((value: string) => {
    setPreset(value)
    if (value === 'none') return
    let data: { x: number[]; y: number[] }
    switch (value) {
      case 'sine':
        data = makeSineData(15, 0.2, Date.now())
        break
      case 'gap':
        data = makeGapData(20, 3, 7, Date.now())
        break
      case 'linear':
        data = makeLinearTrendData(12, 0.3, Date.now())
        break
      default:
        return
    }
    setDataX(data.x)
    setDataY(data.y)
  }, [])

  const isCompact = containerWidth < 700
  const mainWidth = isCompact ? containerWidth - 32 : containerWidth - 32
  const mainHeight = isCompact ? 300 : 400
  const kernelPanelW = isCompact ? containerWidth - 32 : 260
  const kernelPanelH = 160

  return (
    <div className="space-y-4" ref={containerRef}>
      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          <Select label="Kernel" value={kernelType} options={KERNEL_OPTIONS} onChange={(v) => setKernelType(v as KernelType)} className="w-36" />
          <Select label="Preset Data" value={preset} options={PRESET_OPTIONS} onChange={handlePreset} className="w-36" />

          {kernelType !== 'linear' && (
            <Slider label="Length scale ℓ" value={lengthScale} min={0.2} max={5} step={0.1} onChange={setLengthScale} formatValue={(v) => v.toFixed(1)} className="w-32" />
          )}
          <Slider label="Signal σ²" value={signalVariance} min={0.1} max={3} step={0.1} onChange={setSignalVariance} formatValue={(v) => v.toFixed(1)} className="w-28" />
          <Slider label="Noise σₙ²" value={noiseVariance} min={0} max={1} step={0.01} onChange={setNoiseVariance} formatValue={(v) => v.toFixed(2)} className="w-28" />
          <Slider label="Samples" value={nSamples} min={0} max={20} step={1} onChange={setNSamples} formatValue={(v) => String(v)} className="w-24" />

          <div className="h-6 w-px bg-obsidian-border" />

          <Toggle label="Bands" checked={showBands} onChange={setShowBands} />
          <Toggle label="Samples" checked={showSamples} onChange={setShowSamples} />

          <div className="h-6 w-px bg-obsidian-border" />

          <Button variant="secondary" size="sm" onClick={handleClear}>Clear</Button>
        </div>
      </GlassCard>

      {/* Main GP panel */}
      <GlassCard className="p-3 overflow-hidden">
        <GPMainPanel
          gp={gp}
          dataX={dataX}
          dataY={dataY}
          nSamples={nSamples}
          showBands={showBands}
          showSamples={showSamples}
          width={mainWidth}
          height={mainHeight}
          onAddPoint={handleAddPoint}
        />
      </GlassCard>

      {/* Kernel explorer + legend row */}
      <div className={`grid gap-4 ${isCompact ? 'grid-cols-1' : 'grid-cols-[260px_1fr]'}`}>
        <GlassCard className="p-2">
          <KernelPanel
            kernelType={kernelType}
            lengthScale={lengthScale}
            signalVariance={signalVariance}
            width={kernelPanelW}
            height={kernelPanelH}
          />
        </GlassCard>

        <GlassCard className="p-4 space-y-3">
          <p className="text-[10px] font-mono uppercase tracking-wider text-text-tertiary">Legend</p>
          <div className="flex flex-wrap gap-4">
            <div className="flex items-center gap-1.5">
              <div className="w-5 h-0.5 rounded" style={{ backgroundColor: COLORS.mean }} />
              <span className="text-[10px] text-text-tertiary">Mean prediction</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-4 h-3 rounded" style={{ backgroundColor: COLORS.band1 }} />
              <span className="text-[10px] text-text-tertiary">±1σ band</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-4 h-3 rounded" style={{ backgroundColor: COLORS.band2 }} />
              <span className="text-[10px] text-text-tertiary">±2σ band</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-5 h-0.5 rounded" style={{ backgroundColor: COLORS.sample, opacity: 0.4 }} />
              <span className="text-[10px] text-text-tertiary">Function samples</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS.data, opacity: 0.8 }} />
              <span className="text-[10px] text-text-tertiary">Data ({dataX.length} / {MAX_POINTS})</span>
            </div>
          </div>
          <p className="text-[11px] text-text-tertiary leading-relaxed">
            {kernelType === 'rbf' && 'The RBF kernel produces infinitely smooth functions. Short length scales give wiggly fits; long ones give smooth underfitting.'}
            {kernelType === 'matern' && 'The Matérn 3/2 kernel produces once-differentiable functions. Rougher than RBF, often more realistic for real-world data.'}
            {kernelType === 'periodic' && 'The periodic kernel assumes the function repeats. The band stays tight everywhere it expects a repeat of the pattern.'}
            {kernelType === 'linear' && 'The linear kernel reduces to Bayesian linear regression. Function samples are straight lines, the band forms a wedge.'}
          </p>
        </GlassCard>
      </div>

      {/* Kernel comparison strip */}
      {dataX.length > 0 && (
        <div>
          <p className="text-[10px] font-mono uppercase tracking-wider text-text-tertiary mb-2 px-1">Same Data, Four Kernels</p>
          <KernelComparisonStrip
            dataX={dataX}
            dataY={dataY}
            lengthScale={lengthScale}
            signalVariance={signalVariance}
            noiseVariance={noiseVariance}
            width={containerWidth - 16}
          />
        </div>
      )}
    </div>
  )
}
