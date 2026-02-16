import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { motion, AnimatePresence } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Button } from '../../../components/ui/Button'
import { Slider } from '../../../components/ui/Slider'
import { Toggle } from '../../../components/ui/Toggle'
import {
  makeTiltedCloud,
  makeCircularCloud,
  makeThreeClusters,
  fitPCA,
  projectPCA,
  reconstructPCA,
  getProjectionLines,
  varianceAtAngle,
  type PCAResult,
} from '../../../lib/algorithms/basics/pca'

// ── Constants ─────────────────────────────────────────────────────────
const N_POINTS = 60

type DatasetKey = 'tilted' | 'circular' | 'clusters'

const DATASET_LABELS: Record<DatasetKey, string> = {
  tilted: 'Tilted Cloud',
  circular: 'Circular Cloud',
  clusters: 'Three Clusters',
}

const COLORS = {
  pc1: '#6366F1',
  pc2: '#F472B6',
  point: '#E4E4E7',
  projected: '#4ADE80',
  reconstructed: '#FBBF24',
  projLine: 'rgba(99, 102, 241, 0.3)',
}

// Step descriptions for the guided animation
const STEP_DESCRIPTIONS = [
  'The raw data cloud. These two features are correlated.',
  'PC1 (indigo) aligns with the direction of maximum variance.',
  'PC2 (pink) is perpendicular to PC1, capturing the remaining variance.',
  'Project onto PC1: each point drops to the principal axis.',
  'Reconstruct from 1 component. Close to the originals, with some lost detail.',
]

// ── Main Scatter Panel ───────────────────────────────────────────────
function PCAScatterPanel({
  innerWidth,
  innerHeight,
  dataX,
  dataY,
  pca,
  step,
  showOrigAxes,
  showProjLines,
  nComponents,
}: {
  innerWidth: number
  innerHeight: number
  dataX: number[]
  dataY: number[]
  pca: PCAResult
  step: number
  showOrigAxes: boolean
  showProjLines: boolean
  nComponents: 1 | 2
}) {
  // Compute projections and reconstructions
  const projLines = useMemo(
    () => getProjectionLines(dataX, dataY, pca),
    [dataX, dataY, pca]
  )

  const projected = useMemo(
    () => projectPCA(dataX, dataY, pca, nComponents),
    [dataX, dataY, pca, nComponents]
  )

  const reconstructed = useMemo(
    () => reconstructPCA(projected, pca),
    [projected, pca]
  )

  // Scales
  const allX = [...dataX, ...reconstructed.x]
  const allY = [...dataY, ...reconstructed.y]
  const xExtent = d3.extent(allX) as [number, number]
  const yExtent = d3.extent(allY) as [number, number]
  const pad = Math.max(xExtent[1] - xExtent[0], yExtent[1] - yExtent[0]) * 0.2
  const cx = (xExtent[0] + xExtent[1]) / 2
  const cy = (yExtent[0] + yExtent[1]) / 2
  const halfRange = Math.max(xExtent[1] - xExtent[0], yExtent[1] - yExtent[0]) / 2 + pad

  const xScale = d3.scaleLinear().domain([cx - halfRange, cx + halfRange]).range([0, innerWidth])
  const yScale = d3.scaleLinear().domain([cy - halfRange, cy + halfRange]).range([innerHeight, 0])

  const [mx, my] = pca.mean
  const pc1Dir = pca.components[0].direction
  const pc2Dir = pca.components[1].direction

  // PC axis line length
  const axisLen = halfRange * 1.5

  return (
    <>
      {/* Original axes */}
      {showOrigAxes && (
        <>
          <line x1={xScale(cx - halfRange)} y1={yScale(0)} x2={xScale(cx + halfRange)} y2={yScale(0)}
            stroke="rgba(255,255,255,0.08)" strokeWidth={0.5} />
          <line x1={xScale(0)} y1={yScale(cy - halfRange)} x2={xScale(0)} y2={yScale(cy + halfRange)}
            stroke="rgba(255,255,255,0.08)" strokeWidth={0.5} />
        </>
      )}

      {/* Projection lines (step 3+) */}
      {step >= 3 && showProjLines && (
        <AnimatePresence>
          {projLines.map((pl, i) => (
            <motion.line
              key={`pline-${i}`}
              x1={xScale(pl.fromX)}
              y1={yScale(pl.fromY)}
              x2={xScale(pl.toX)}
              y2={yScale(pl.toY)}
              stroke={COLORS.projLine}
              strokeWidth={0.8}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: i * 0.01, duration: 0.3 }}
            />
          ))}
        </AnimatePresence>
      )}

      {/* PC1 axis (step 1+) */}
      {step >= 1 && (
        <motion.line
          x1={xScale(mx - pc1Dir[0] * axisLen)}
          y1={yScale(my - pc1Dir[1] * axisLen)}
          x2={xScale(mx + pc1Dir[0] * axisLen)}
          y2={yScale(my + pc1Dir[1] * axisLen)}
          stroke={COLORS.pc1}
          strokeWidth={2}
          strokeOpacity={0.8}
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.5 }}
        />
      )}

      {/* PC2 axis (step 2+) */}
      {step >= 2 && (
        <motion.line
          x1={xScale(mx - pc2Dir[0] * axisLen)}
          y1={yScale(my - pc2Dir[1] * axisLen)}
          x2={xScale(mx + pc2Dir[0] * axisLen)}
          y2={yScale(my + pc2Dir[1] * axisLen)}
          stroke={COLORS.pc2}
          strokeWidth={2}
          strokeOpacity={0.6}
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.5 }}
        />
      )}

      {/* Reconstructed points (step 4) */}
      {step >= 4 && nComponents === 1 && (
        <>
          {reconstructed.x.map((rx, i) => (
            <motion.line
              key={`rline-${i}`}
              x1={xScale(dataX[i])}
              y1={yScale(dataY[i])}
              x2={xScale(rx)}
              y2={yScale(reconstructed.y[i])}
              stroke={COLORS.reconstructed}
              strokeWidth={0.6}
              strokeOpacity={0.4}
              strokeDasharray="2 2"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3, duration: 0.3 }}
            />
          ))}
          {reconstructed.x.map((rx, i) => (
            <motion.circle
              key={`rpt-${i}`}
              cx={xScale(rx)}
              cy={yScale(reconstructed.y[i])}
              r={3}
              fill={COLORS.reconstructed}
              fillOpacity={0.7}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.01, duration: 0.3 }}
            />
          ))}
        </>
      )}

      {/* Projected points on PC1 (step 3+) */}
      {step >= 3 && (
        <>
          {projLines.map((pl, i) => (
            <motion.circle
              key={`proj-${i}`}
              cx={xScale(pl.toX)}
              cy={yScale(pl.toY)}
              r={3}
              fill={COLORS.projected}
              fillOpacity={0.8}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: i * 0.01 + 0.2, duration: 0.3 }}
            />
          ))}
        </>
      )}

      {/* Data points */}
      {dataX.map((xi, i) => (
        <circle
          key={`pt-${i}`}
          cx={xScale(xi)}
          cy={yScale(dataY[i])}
          r={3.5}
          fill={step >= 4 && nComponents === 1 ? 'rgba(228,228,231,0.3)' : COLORS.point}
          fillOpacity={step >= 4 && nComponents === 1 ? 0.5 : 0.8}
          stroke="rgba(255,255,255,0.15)"
          strokeWidth={0.5}
        />
      ))}

      {/* PC labels */}
      {step >= 1 && (
        <text
          x={xScale(mx + pc1Dir[0] * axisLen * 0.9)}
          y={yScale(my + pc1Dir[1] * axisLen * 0.9) - 8}
          className="text-[10px] font-mono font-medium"
          fill={COLORS.pc1}
          textAnchor="middle"
        >
          PC1 ({(pca.components[0].varianceExplained * 100).toFixed(0)}%)
        </text>
      )}
      {step >= 2 && (
        <text
          x={xScale(mx + pc2Dir[0] * axisLen * 0.9)}
          y={yScale(my + pc2Dir[1] * axisLen * 0.9) - 8}
          className="text-[10px] font-mono font-medium"
          fill={COLORS.pc2}
          textAnchor="middle"
        >
          PC2 ({(pca.components[1].varianceExplained * 100).toFixed(0)}%)
        </text>
      )}
    </>
  )
}

// ── Variance Explained Bar Chart ─────────────────────────────────────
function VarianceBarChart({ pca }: { pca: PCAResult }) {
  const bars = pca.components.map((c, i) => ({
    label: `PC${i + 1}`,
    value: c.varianceExplained * 100,
    color: i === 0 ? COLORS.pc1 : COLORS.pc2,
  }))

  return (
    <div className="flex items-end gap-4 h-24">
      {bars.map((bar) => (
        <div key={bar.label} className="flex flex-col items-center gap-1 flex-1">
          <p className="text-[10px] font-mono font-bold" style={{ color: bar.color }}>
            {bar.value.toFixed(1)}%
          </p>
          <div className="w-full relative" style={{ height: 60 }}>
            <motion.div
              className="absolute bottom-0 w-full rounded-t"
              style={{ backgroundColor: bar.color }}
              initial={{ height: 0 }}
              animate={{ height: `${bar.value}%` }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
            />
          </div>
          <p className="text-[9px] font-mono text-text-tertiary">{bar.label}</p>
        </div>
      ))}
    </div>
  )
}

// ── Rotation Demo Panel ──────────────────────────────────────────────
function RotationDemoPanel({
  innerWidth,
  innerHeight,
  dataX,
  dataY,
  pca,
  rotationAngle,
}: {
  innerWidth: number
  innerHeight: number
  dataX: number[]
  dataY: number[]
  pca: PCAResult
  rotationAngle: number
}) {
  // Scales
  const xExtent = d3.extent(dataX) as [number, number]
  const yExtent = d3.extent(dataY) as [number, number]
  const pad = Math.max(xExtent[1] - xExtent[0], yExtent[1] - yExtent[0]) * 0.2
  const cx = (xExtent[0] + xExtent[1]) / 2
  const cy = (yExtent[0] + yExtent[1]) / 2
  const halfRange = Math.max(xExtent[1] - xExtent[0], yExtent[1] - yExtent[0]) / 2 + pad

  const xScale = d3.scaleLinear().domain([cx - halfRange, cx + halfRange]).range([0, innerWidth])
  const yScale = d3.scaleLinear().domain([cy - halfRange, cy + halfRange]).range([innerHeight, 0])

  const [mx, my] = pca.mean
  const axisLen = halfRange * 1.3

  const dx = Math.cos(rotationAngle)
  const dy = Math.sin(rotationAngle)

  // Variance at this angle
  const currentVar = varianceAtAngle(dataX, dataY, rotationAngle)
  const maxVar = pca.components[0].eigenvalue
  const varRatio = Math.min(currentVar / maxVar, 1)

  // Check if near PCA solution
  const pcAngle = Math.atan2(pca.components[0].direction[1], pca.components[0].direction[0])
  // Handle the fact that PC direction can point either way
  const isNearOptimal = Math.min(
    Math.abs(rotationAngle - pcAngle) % Math.PI,
    Math.abs(rotationAngle - pcAngle + Math.PI) % Math.PI
  ) < 0.15

  return (
    <>
      {/* Data points */}
      {dataX.map((xi, i) => (
        <circle
          key={`rpt-${i}`}
          cx={xScale(xi)}
          cy={yScale(dataY[i])}
          r={2.5}
          fill={COLORS.point}
          fillOpacity={0.5}
        />
      ))}

      {/* Rotatable axis */}
      <line
        x1={xScale(mx - dx * axisLen)}
        y1={yScale(my - dy * axisLen)}
        x2={xScale(mx + dx * axisLen)}
        y2={yScale(my + dy * axisLen)}
        stroke={isNearOptimal ? '#4ADE80' : '#818CF8'}
        strokeWidth={2.5}
        strokeOpacity={0.8}
      />

      {/* Variance meter */}
      <rect x={4} y={innerHeight - 24} width={innerWidth - 8} height={18} rx={4} fill="rgba(0,0,0,0.4)" />
      <rect
        x={6}
        y={innerHeight - 22}
        width={(innerWidth - 12) * varRatio}
        height={14}
        rx={3}
        fill={isNearOptimal ? '#4ADE80' : '#818CF8'}
        fillOpacity={0.6}
      />
      <text x={innerWidth / 2} y={innerHeight - 12} textAnchor="middle" className="text-[9px] font-mono font-medium" fill="white">
        Variance: {(varRatio * 100).toFixed(0)}%
      </text>

      {/* Optimal indicator */}
      {isNearOptimal && (
        <motion.circle
          cx={innerWidth / 2}
          cy={innerHeight / 2}
          r={20}
          fill="none"
          stroke="#4ADE80"
          strokeWidth={2}
          strokeOpacity={0.5}
          animate={{ r: [20, 30, 20], strokeOpacity: [0.5, 0.2, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      )}
    </>
  )
}

// ── Main Component ────────────────────────────────────────────────────
export function PCAViz() {
  const [dataset, setDataset] = useState<DatasetKey>('tilted')
  const [step, setStep] = useState(0)
  const [showOrigAxes, setShowOrigAxes] = useState(true)
  const [showProjLines, setShowProjLines] = useState(true)
  const [nComponents, setNComponents] = useState<1 | 2>(1)
  const [rotationAngle, setRotationAngle] = useState(0.3)

  const data = useMemo(() => {
    switch (dataset) {
      case 'tilted':
        return makeTiltedCloud(N_POINTS, Math.PI / 5, 3, 0.6, 42)
      case 'circular':
        return makeCircularCloud(N_POINTS, 2, 42)
      case 'clusters':
        return makeThreeClusters(N_POINTS, 42)
    }
  }, [dataset])

  const pca = useMemo(() => fitPCA(data.x, data.y), [data])

  const handleStepForward = useCallback(() => {
    setStep((prev) => Math.min(prev + 1, 4))
  }, [])

  const handleStepBack = useCallback(() => {
    setStep((prev) => Math.max(prev - 1, 0))
  }, [])

  const handleReset = useCallback(() => {
    setStep(0)
  }, [])

  return (
    <div className="space-y-4">
      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          {/* Step controls */}
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="sm" onClick={handleReset}>
              Reset
            </Button>
            <Button variant="ghost" size="sm" onClick={handleStepBack} disabled={step === 0}>
              ◀
            </Button>
            <Button variant="ghost" size="sm" onClick={handleStepForward} disabled={step === 4}>
              ▶
            </Button>
            <span className="text-xs font-mono text-text-tertiary ml-1">
              Step {step}/4
            </span>
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          {/* Dataset selector */}
          <div className="flex items-center gap-1">
            {(['tilted', 'circular', 'clusters'] as DatasetKey[]).map((dk) => (
              <Button
                key={dk}
                variant="secondary"
                size="sm"
                active={dataset === dk}
                onClick={() => {
                  setDataset(dk)
                  setStep(0)
                }}
              >
                {DATASET_LABELS[dk]}
              </Button>
            ))}
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          <div className="flex items-center gap-1">
            <Button variant="secondary" size="sm" active={nComponents === 1} onClick={() => setNComponents(1)}>
              1 PC
            </Button>
            <Button variant="secondary" size="sm" active={nComponents === 2} onClick={() => setNComponents(2)}>
              2 PCs
            </Button>
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          <Toggle label="Axes" checked={showOrigAxes} onChange={setShowOrigAxes} />
          <Toggle label="Proj Lines" checked={showProjLines} onChange={setShowProjLines} />
        </div>
      </GlassCard>

      {/* Step description */}
      <div className="px-1">
        <p className="text-sm text-text-secondary">
          {STEP_DESCRIPTIONS[step]}
        </p>
      </div>

      {/* Main panel + variance bar */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        <GlassCard className="lg:col-span-3 overflow-hidden">
          <SVGContainer
            aspectRatio={4 / 3}
            minHeight={300}
            maxHeight={450}
            padding={{ top: 15, right: 20, bottom: 20, left: 20 }}
          >
            {({ innerWidth, innerHeight }) => (
              <PCAScatterPanel
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                dataX={data.x}
                dataY={data.y}
                pca={pca}
                step={step}
                showOrigAxes={showOrigAxes}
                showProjLines={showProjLines}
                nComponents={nComponents}
              />
            )}
          </SVGContainer>
        </GlassCard>

        <GlassCard className="p-4">
          <h4 className="text-xs font-mono uppercase tracking-wider text-text-secondary mb-3">
            Variance Explained
          </h4>
          <VarianceBarChart pca={pca} />
          <p className="text-[9px] text-text-tertiary mt-3">
            {pca.components[0].varianceExplained > 0.8
              ? `PC1 alone captures ${(pca.components[0].varianceExplained * 100).toFixed(0)}% of the variation. Dropping PC2 loses very little.`
              : pca.components[0].varianceExplained > 0.6
                ? `Variance is split more evenly. Both components carry significant information.`
                : `Variance is nearly equal in both directions. PCA cannot compress this data much.`}
          </p>
        </GlassCard>
      </div>

      {/* Rotation demo */}
      <div>
        <p className="text-xs font-medium text-text-secondary mb-2">
          Rotation Demo
        </p>
        <p className="text-[10px] text-text-tertiary mb-3">
          Drag the slider to rotate the axis. PCA finds the angle that maximizes the variance meter.
          Can you find the maximum?
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            <GlassCard className="overflow-hidden">
              <SVGContainer
                aspectRatio={16 / 9}
                minHeight={180}
                maxHeight={260}
                padding={{ top: 10, right: 16, bottom: 30, left: 16 }}
              >
                {({ innerWidth, innerHeight }) => (
                  <RotationDemoPanel
                    innerWidth={innerWidth}
                    innerHeight={innerHeight}
                    dataX={data.x}
                    dataY={data.y}
                    pca={pca}
                    rotationAngle={rotationAngle}
                  />
                )}
              </SVGContainer>
            </GlassCard>
          </div>
          <GlassCard className="p-4 flex flex-col justify-center">
            <Slider
              label="Rotation"
              value={rotationAngle}
              min={-Math.PI / 2}
              max={Math.PI / 2}
              step={0.01}
              onChange={setRotationAngle}
              formatValue={(v) => `${((v * 180) / Math.PI).toFixed(0)}°`}
            />
            <p className="text-[9px] text-text-tertiary mt-3">
              The green pulse means you found the PCA direction. Any other angle captures less variance.
            </p>
          </GlassCard>
        </div>
      </div>
    </div>
  )
}
