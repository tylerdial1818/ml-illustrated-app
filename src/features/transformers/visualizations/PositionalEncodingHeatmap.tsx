import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'
import { Toggle } from '../../../components/ui/Toggle'

// ── NLP Colors ──────────────────────────────────────────────────────────
const COLORS = {
  position: '#38BDF8',
  tokenHighlight: '#818CF8',
}

const CURVE_COLORS = ['#6366F1', '#F472B6', '#34D399', '#FBBF24', '#38BDF8', '#A78BFA']

// ── Token colors for the Before/After panel ─────────────────────────────
const TOKEN_COLORS = ['#6366F1', '#F472B6', '#34D399', '#FBBF24', '#38BDF8']

// ── Positional Encoding Computation ─────────────────────────────────────
function computePE(pos: number, dim: number, dModel: number): number {
  const i = Math.floor(dim / 2)
  const angle = pos / Math.pow(10000, (2 * i) / dModel)
  return dim % 2 === 0 ? Math.sin(angle) : Math.cos(angle)
}

function computePEMatrix(seqLen: number, dModel: number): number[][] {
  const matrix: number[][] = []
  for (let pos = 0; pos < seqLen; pos++) {
    const row: number[] = []
    for (let dim = 0; dim < dModel; dim++) {
      row.push(computePE(pos, dim, dModel))
    }
    matrix.push(row)
  }
  return matrix
}

// ── Sample sentence tokens ──────────────────────────────────────────────
const SENTENCE_TOKENS = ['the', 'dog', 'chased', 'the', 'cat']

function getTokenAtPos(pos: number): string {
  if (pos < SENTENCE_TOKENS.length) return SENTENCE_TOKENS[pos]
  return `pos_${pos}`
}

// ── Before/After Token Positions ─────────────────────────────────────────
// Fixed 2D positions for each unique token (semantic embedding space)
const TOKEN_BASE_POSITIONS: Record<string, { x: number; y: number }> = {
  the: { x: 0.3, y: 0.5 },
  dog: { x: 0.7, y: 0.3 },
  chased: { x: 0.5, y: 0.7 },
  cat: { x: 0.7, y: 0.7 },
}

// ── Before/After Panel ──────────────────────────────────────────────────

function BeforeAfterPanel({
  innerWidth,
  innerHeight,
  dModel,
  showPE,
}: {
  innerWidth: number
  innerHeight: number
  dModel: number
  showPE: boolean
}) {
  const pad = 30
  const plotW = innerWidth - pad * 2
  const plotH = innerHeight - 40

  const xScale = d3.scaleLinear().domain([0, 1]).range([pad, pad + plotW])
  const yScale = d3.scaleLinear().domain([0, 1]).range([30, 30 + plotH])

  // Compute token positions with and without PE
  const tokenPositions = SENTENCE_TOKENS.map((token, pos) => {
    const base = TOKEN_BASE_POSITIONS[token] || { x: 0.5, y: 0.5 }

    // PE offset uses first two dimensions, scaled down for visibility
    const peOffsetX = computePE(pos, 0, dModel) * 0.12
    const peOffsetY = computePE(pos, 1, dModel) * 0.12

    return {
      token,
      pos,
      baseX: base.x,
      baseY: base.y,
      peX: base.x + peOffsetX,
      peY: base.y + peOffsetY,
      color: TOKEN_COLORS[pos % TOKEN_COLORS.length],
    }
  })

  // Check which tokens are duplicated
  const tokenCounts: Record<string, number> = {}
  for (const t of SENTENCE_TOKENS) {
    tokenCounts[t] = (tokenCounts[t] || 0) + 1
  }

  return (
    <g>
      {/* Title */}
      <text
        x={innerWidth / 2}
        y={14}
        textAnchor="middle"
        fontSize={10}
        fontWeight={500}
        className="fill-text-tertiary uppercase tracking-wider"
      >
        {showPE ? 'With Positional Encoding' : 'Without Positional Encoding'}
      </text>

      {/* Faint grid */}
      {[0.25, 0.5, 0.75].map((v) => (
        <g key={`grid-${v}`}>
          <line
            x1={xScale(v)} y1={yScale(0)} x2={xScale(v)} y2={yScale(1)}
            stroke="rgba(255,255,255,0.04)" strokeWidth={0.5}
          />
          <line
            x1={xScale(0)} y1={yScale(v)} x2={xScale(1)} y2={yScale(v)}
            stroke="rgba(255,255,255,0.04)" strokeWidth={0.5}
          />
        </g>
      ))}

      {/* PE offset arrows (only when showing PE) */}
      {showPE &&
        tokenPositions.map((tp) => (
          <motion.line
            key={`arrow-${tp.pos}`}
            x1={xScale(tp.baseX)}
            y1={yScale(tp.baseY)}
            stroke={COLORS.position}
            strokeWidth={1}
            strokeOpacity={0.4}
            strokeDasharray="3,2"
            initial={{ x2: xScale(tp.baseX), y2: yScale(tp.baseY) }}
            animate={{ x2: xScale(tp.peX), y2: yScale(tp.peY) }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
          />
        ))}

      {/* Token dots */}
      {tokenPositions.map((tp) => {
        const targetX = showPE ? tp.peX : tp.baseX
        const targetY = showPE ? tp.peY : tp.baseY
        const isDuplicate = tokenCounts[tp.token] > 1

        return (
          <g key={`token-${tp.pos}`}>
            {/* Dot */}
            <motion.circle
              r={isDuplicate ? 7 : 5}
              fill={tp.color}
              fillOpacity={0.85}
              stroke={isDuplicate ? '#fff' : tp.color}
              strokeWidth={isDuplicate ? 1.5 : 0.5}
              animate={{ cx: xScale(targetX), cy: yScale(targetY) }}
              transition={{ duration: 0.6, ease: 'easeOut' }}
            />

            {/* Label */}
            <motion.text
              textAnchor="middle"
              fontSize={9}
              fontFamily="'JetBrains Mono', monospace"
              fill="#E4E4E7"
              animate={{
                x: xScale(targetX),
                y: yScale(targetY) - (isDuplicate ? 11 : 9),
              }}
              transition={{ duration: 0.6, ease: 'easeOut' }}
            >
              {tp.token}
              <tspan fontSize={7} fill="#71717A">
                {` [${tp.pos}]`}
              </tspan>
            </motion.text>

            {/* Overlap indicator when NOT using PE */}
            {!showPE && isDuplicate && tp.pos > 0 && tp.token === SENTENCE_TOKENS[0] && (
              <motion.text
                textAnchor="middle"
                fontSize={7}
                fill="#F87171"
                initial={{ opacity: 0 }}
                animate={{
                  opacity: 1,
                  x: xScale(targetX),
                  y: yScale(targetY) + 16,
                }}
                transition={{ delay: 0.3, duration: 0.3 }}
              >
                overlapping!
              </motion.text>
            )}
          </g>
        )
      })}

      {/* Explanatory note */}
      <text
        x={innerWidth / 2}
        y={innerHeight - 4}
        textAnchor="middle"
        fontSize={8}
        fill="#71717A"
      >
        {showPE
          ? 'Same words at different positions now separate'
          : '"the" at positions 0 and 3 map to the same point'}
      </text>
    </g>
  )
}

// ── Main Component ──────────────────────────────────────────────────────
export function PositionalEncodingHeatmap() {
  const [seqLen, setSeqLen] = useState(10)
  const [dModel, setDModel] = useState(16)
  const [hoveredPos, setHoveredPos] = useState<number | null>(null)
  const [scrubberPos, setScrubberPos] = useState(3)
  const [showPE, setShowPE] = useState(false)

  const peMatrix = useMemo(() => computePEMatrix(seqLen, dModel), [seqLen, dModel])

  // Color scale: diverging blue -> dark -> amber
  const colorScale = useMemo(
    () =>
      d3
        .scaleLinear<string>()
        .domain([-1, 0, 1])
        .range(['#3B82F6', '#1E1E22', '#FBBF24'])
        .clamp(true),
    []
  )

  // Frequency curves: pick 6 evenly spaced even dimensions
  const curveDims = useMemo(() => {
    const dims: number[] = []
    const step = Math.max(2, Math.floor(dModel / 6) * 2)
    for (let d = 0; d < dModel && dims.length < 6; d += step) {
      dims.push(d % 2 === 0 ? d : d - 1)
    }
    return [...new Set(dims)].slice(0, 6)
  }, [dModel])

  const handleCellHover = useCallback((pos: number | null) => {
    setHoveredPos(pos)
  }, [])

  return (
    <div className="space-y-6">
      {/* Token pills row */}
      <div className="flex flex-wrap gap-2 items-center justify-center">
        {Array.from({ length: seqLen }, (_, pos) => {
          const token = getTokenAtPos(pos)
          const isHighlighted = hoveredPos === pos
          return (
            <motion.div
              key={pos}
              className="px-3 py-1.5 rounded-full text-xs font-mono border transition-colors duration-150"
              style={{
                backgroundColor: isHighlighted
                  ? `${COLORS.position}20`
                  : 'rgba(255,255,255,0.04)',
                borderColor: isHighlighted ? COLORS.position : 'rgba(255,255,255,0.08)',
                color: isHighlighted ? COLORS.position : 'rgba(255,255,255,0.6)',
              }}
              animate={{
                scale: isHighlighted ? 1.1 : 1,
              }}
              transition={{ duration: 0.15 }}
            >
              {token}
              <span className="ml-1.5 text-[9px] opacity-50">{pos}</span>
            </motion.div>
          )
        })}
      </div>

      {/* Dual panel layout: Heatmap + Before/After */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left panel: Heatmap */}
        <GlassCard className="p-4">
          <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-3">
            Sinusoidal Heatmap
          </p>
          <SVGContainer
            aspectRatio={1.2}
            minHeight={280}
            maxHeight={500}
            padding={{ top: 30, right: 10, bottom: 10, left: 40 }}
          >
            {({ innerWidth, innerHeight }) => {
              const cellW = Math.max(2, innerWidth / dModel)
              const cellH = Math.max(4, innerHeight / seqLen)

              return (
                <>
                  {/* Dimension labels on top */}
                  {Array.from({ length: dModel }, (_, d) => {
                    const labelStep = dModel <= 16 ? 2 : dModel <= 32 ? 4 : 8
                    if (d % labelStep !== 0) return null
                    return (
                      <text
                        key={`dim-${d}`}
                        x={d * cellW + cellW / 2}
                        y={-6}
                        textAnchor="middle"
                        className="fill-text-tertiary"
                        fontSize={8}
                        fontFamily="monospace"
                      >
                        {d}
                      </text>
                    )
                  })}

                  {/* Heatmap cells */}
                  {peMatrix.map((row, pos) =>
                    row.map((val, dim) => (
                      <motion.rect
                        key={`${pos}-${dim}`}
                        x={dim * cellW}
                        y={pos * cellH}
                        width={cellW - 0.5}
                        height={cellH - 0.5}
                        rx={1}
                        fill={colorScale(val)}
                        opacity={hoveredPos !== null && hoveredPos !== pos ? 0.3 : 1}
                        onMouseEnter={() => handleCellHover(pos)}
                        onMouseLeave={() => handleCellHover(null)}
                        style={{ cursor: 'pointer' }}
                        transition={{ duration: 0.1 }}
                      />
                    ))
                  )}

                  {/* Position labels on left */}
                  {Array.from({ length: seqLen }, (_, pos) => (
                    <text
                      key={`pos-${pos}`}
                      x={-6}
                      y={pos * cellH + cellH / 2 + 3}
                      textAnchor="end"
                      fontSize={9}
                      fontFamily="monospace"
                      fill={hoveredPos === pos ? COLORS.position : 'rgba(255,255,255,0.4)'}
                      fontWeight={hoveredPos === pos ? 600 : 400}
                    >
                      {pos}
                    </text>
                  ))}

                  {/* Hover row highlight */}
                  {hoveredPos !== null && (
                    <motion.rect
                      x={0}
                      y={hoveredPos * cellH}
                      width={dModel * cellW}
                      height={cellH}
                      fill="none"
                      stroke={COLORS.position}
                      strokeWidth={1.5}
                      rx={2}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.15 }}
                    />
                  )}
                </>
              )
            }}
          </SVGContainer>

          {/* Color legend */}
          <div className="flex items-center gap-2 mt-2 justify-center">
            <span className="text-[9px] text-text-tertiary">-1</span>
            <div
              className="h-2 rounded-full"
              style={{
                width: 100,
                background: 'linear-gradient(to right, #3B82F6, #1E1E22, #FBBF24)',
              }}
            />
            <span className="text-[9px] text-text-tertiary">+1</span>
          </div>
        </GlassCard>

        {/* Right panel: Before/After Embedding Space */}
        <GlassCard className="p-4">
          <div className="flex items-center justify-between mb-3">
            <p className="text-[10px] uppercase tracking-wider text-text-tertiary">
              Embedding Space
            </p>
            <Toggle
              label="Add PE"
              checked={showPE}
              onChange={setShowPE}
            />
          </div>
          <SVGContainer
            aspectRatio={1.2}
            minHeight={280}
            maxHeight={500}
            padding={{ top: 10, right: 20, bottom: 10, left: 20 }}
          >
            {({ innerWidth, innerHeight }) => (
              <BeforeAfterPanel
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                dModel={dModel}
                showPE={showPE}
              />
            )}
          </SVGContainer>
        </GlassCard>
      </div>

      {/* Frequency sub-viz */}
      <GlassCard className="p-4">
        <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-3">
          Frequency Curves by Dimension
        </p>
        <SVGContainer
          aspectRatio={16 / 6}
          minHeight={180}
          maxHeight={300}
          padding={{ top: 20, right: 20, bottom: 30, left: 40 }}
        >
          {({ innerWidth, innerHeight }) => {
            const xScale = d3
              .scaleLinear()
              .domain([0, seqLen - 1])
              .range([0, innerWidth])
            const yScale = d3
              .scaleLinear()
              .domain([-1.1, 1.1])
              .range([innerHeight, 0])

            const curveResolution = 200
            const curveData = curveDims.map((dim) => {
              const points: { x: number; y: number }[] = []
              for (let i = 0; i <= curveResolution; i++) {
                const pos = (i / curveResolution) * (seqLen - 1)
                const val = computePE(pos, dim, dModel)
                points.push({ x: pos, y: val })
              }
              return points
            })

            const yTicks = [-1, -0.5, 0, 0.5, 1]

            return (
              <>
                {/* Grid lines */}
                {yTicks.map((tick) => (
                  <line
                    key={`grid-${tick}`}
                    x1={0}
                    x2={innerWidth}
                    y1={yScale(tick)}
                    y2={yScale(tick)}
                    stroke="rgba(255,255,255,0.06)"
                    strokeWidth={tick === 0 ? 1 : 0.5}
                  />
                ))}

                {/* Y-axis labels */}
                {yTicks.map((tick) => (
                  <text
                    key={`ylabel-${tick}`}
                    x={-8}
                    y={yScale(tick) + 3}
                    textAnchor="end"
                    fontSize={8}
                    fontFamily="monospace"
                    className="fill-text-tertiary"
                  >
                    {tick.toFixed(1)}
                  </text>
                ))}

                {/* X-axis labels */}
                {Array.from(
                  { length: Math.min(seqLen, 10) },
                  (_, i) => Math.round((i / Math.min(seqLen - 1, 9)) * (seqLen - 1))
                )
                  .filter((v, i, arr) => arr.indexOf(v) === i)
                  .map((pos) => (
                    <text
                      key={`xlabel-${pos}`}
                      x={xScale(pos)}
                      y={innerHeight + 18}
                      textAnchor="middle"
                      fontSize={8}
                      fontFamily="monospace"
                      className="fill-text-tertiary"
                    >
                      {pos}
                    </text>
                  ))}

                {/* Axis label */}
                <text
                  x={innerWidth / 2}
                  y={innerHeight + 28}
                  textAnchor="middle"
                  fontSize={9}
                  className="fill-text-tertiary"
                >
                  Position
                </text>

                {/* Curves */}
                {curveData.map((points, ci) => {
                  const linePath = points
                    .map((p, i) =>
                      i === 0
                        ? `M ${xScale(p.x)} ${yScale(p.y)}`
                        : `L ${xScale(p.x)} ${yScale(p.y)}`
                    )
                    .join(' ')

                  return (
                    <motion.path
                      key={`curve-${ci}`}
                      d={linePath}
                      fill="none"
                      stroke={CURVE_COLORS[ci % CURVE_COLORS.length]}
                      strokeWidth={1.5}
                      strokeLinecap="round"
                      initial={{ pathLength: 0, opacity: 0 }}
                      animate={{ pathLength: 1, opacity: 0.85 }}
                      transition={{ duration: 1, delay: ci * 0.15 }}
                    />
                  )
                })}

                {/* Scrubber vertical line */}
                <motion.line
                  x1={xScale(scrubberPos)}
                  x2={xScale(scrubberPos)}
                  y1={0}
                  y2={innerHeight}
                  stroke="rgba(255,255,255,0.3)"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                  animate={{ x1: xScale(scrubberPos), x2: xScale(scrubberPos) }}
                  transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                />

                {/* Dots on curves at scrubber position */}
                {curveDims.map((dim, ci) => {
                  const val = computePE(scrubberPos, dim, dModel)
                  return (
                    <motion.circle
                      key={`dot-${ci}`}
                      r={4}
                      fill={CURVE_COLORS[ci % CURVE_COLORS.length]}
                      stroke="#0F0F11"
                      strokeWidth={1.5}
                      animate={{
                        cx: xScale(scrubberPos),
                        cy: yScale(val),
                      }}
                      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                    />
                  )
                })}

                {/* Curve legend */}
                {curveDims.map((dim, ci) => (
                  <g key={`legend-${ci}`} transform={`translate(${innerWidth - 55}, ${12 + ci * 14})`}>
                    <line
                      x1={0}
                      x2={12}
                      y1={0}
                      y2={0}
                      stroke={CURVE_COLORS[ci % CURVE_COLORS.length]}
                      strokeWidth={2}
                    />
                    <text
                      x={16}
                      y={3}
                      fontSize={8}
                      fontFamily="monospace"
                      fill={CURVE_COLORS[ci % CURVE_COLORS.length]}
                    >
                      d={dim}
                    </text>
                  </g>
                ))}
              </>
            )
          }}
        </SVGContainer>
      </GlassCard>

      {/* Explanatory note */}
      <div className="text-center px-4">
        <p className="text-xs text-text-tertiary leading-relaxed max-w-2xl mx-auto">
          Low dimensions change slowly (broad position awareness). High dimensions change fast
          (fine-grained position). Together they create a unique fingerprint for each position.
        </p>
      </div>

      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap gap-6 items-end">
          <Slider
            label="Sequence Length"
            value={seqLen}
            min={5}
            max={30}
            step={1}
            onChange={setSeqLen}
            className="w-48"
          />
          <Slider
            label="Embedding Dim (d_model)"
            value={dModel}
            min={8}
            max={64}
            step={8}
            onChange={setDModel}
            className="w-48"
          />
          <Slider
            label="Position Scrubber"
            value={scrubberPos}
            min={0}
            max={seqLen - 1}
            step={1}
            onChange={setScrubberPos}
            formatValue={(v) => `pos ${v}`}
            className="w-48"
          />
        </div>
      </GlassCard>
    </div>
  )
}
