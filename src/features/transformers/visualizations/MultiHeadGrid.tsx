import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { motion, AnimatePresence } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Button } from '../../../components/ui/Button'
import { Toggle } from '../../../components/ui/Toggle'
import { TransportControls } from '../../../components/viz/TransportControls'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'
import { adaptHeadCount } from '../../../lib/algorithms/transformers/multiHeadAttention'

// ── NLP Colors ────────────────────────────────────────────────────────
const NLP = {
  query: '#F472B6',
  key: '#34D399',
  value: '#FBBF24',
  position: '#38BDF8',
  token: '#818CF8',
  attentionLow: 'rgba(99, 102, 241, 0.05)',
  attentionHigh: '#6366F1',
}

// ── Tokens ────────────────────────────────────────────────────────────
const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat']
const N = TOKENS.length

// ── Head Metadata ─────────────────────────────────────────────────────
const HEAD_META = [
  { name: 'Head 0', specialty: 'Positional', color: NLP.position, description: 'Attends to nearby tokens' },
  { name: 'Head 1', specialty: 'Syntactic', color: NLP.query, description: 'Subject-verb, det-noun links' },
  { name: 'Head 2', specialty: 'Semantic', color: NLP.key, description: 'Related concepts' },
  { name: 'Head 3', specialty: 'Global', color: NLP.value, description: 'Broad context gathering' },
]

// ── Attention Weight Matrices (each row sums to 1.0) ──────────────────
// Head 0: Positional -- diagonal-heavy, nearby tokens
const HEAD_WEIGHTS: number[][][] = [
  [
    [0.40, 0.25, 0.15, 0.10, 0.05, 0.05], // The
    [0.20, 0.35, 0.25, 0.10, 0.05, 0.05], // cat
    [0.10, 0.20, 0.35, 0.20, 0.10, 0.05], // sat
    [0.05, 0.10, 0.20, 0.35, 0.20, 0.10], // on
    [0.05, 0.05, 0.10, 0.20, 0.35, 0.25], // the
    [0.05, 0.05, 0.05, 0.10, 0.25, 0.50], // mat
  ],
  // Head 1: Syntactic -- determiner-noun, subject-verb links
  [
    [0.10, 0.50, 0.15, 0.05, 0.10, 0.10], // The -> cat
    [0.10, 0.10, 0.50, 0.10, 0.05, 0.15], // cat -> sat
    [0.15, 0.30, 0.10, 0.20, 0.05, 0.20], // sat -> cat
    [0.05, 0.05, 0.15, 0.10, 0.20, 0.45], // on -> mat
    [0.10, 0.05, 0.05, 0.15, 0.10, 0.55], // the -> mat
    [0.10, 0.15, 0.25, 0.30, 0.10, 0.10], // mat -> on/sat
  ],
  // Head 2: Semantic -- related concepts
  [
    [0.20, 0.15, 0.15, 0.10, 0.20, 0.20], // The
    [0.10, 0.10, 0.30, 0.10, 0.05, 0.35], // cat -> sat, mat
    [0.05, 0.35, 0.10, 0.25, 0.05, 0.20], // sat -> cat, on
    [0.10, 0.10, 0.30, 0.10, 0.10, 0.30], // on -> sat, mat
    [0.30, 0.10, 0.10, 0.10, 0.15, 0.25], // the -> The, mat
    [0.10, 0.30, 0.25, 0.15, 0.10, 0.10], // mat -> cat, sat
  ],
  // Head 3: Global -- more uniform distribution
  [
    [0.18, 0.17, 0.17, 0.16, 0.16, 0.16], // The
    [0.16, 0.18, 0.17, 0.17, 0.16, 0.16], // cat
    [0.16, 0.17, 0.18, 0.16, 0.17, 0.16], // sat
    [0.17, 0.16, 0.16, 0.18, 0.16, 0.17], // on
    [0.16, 0.17, 0.16, 0.17, 0.18, 0.16], // the
    [0.16, 0.16, 0.17, 0.16, 0.17, 0.18], // mat
  ],
]

// ── Head metadata for variable head counts ──────────────────────────
const HEAD_COLORS = [NLP.position, NLP.query, NLP.key, NLP.value, '#F472B6', '#38BDF8', '#34D399', '#FBBF24']
const HEAD_SPECIALTIES = [
  'Positional', 'Syntactic', 'Semantic', 'Global',
  'Positional+', 'Syntactic+', 'Semantic+', 'Global+',
]
const HEAD_DESCRIPTIONS = [
  'Attends to nearby tokens', 'Subject-verb, det-noun links', 'Related concepts', 'Broad context gathering',
  'Focused proximity', 'Strong syntax links', 'Sharp semantics', 'Wide gathering',
]

function getHeadMeta(numHeads: number) {
  return Array.from({ length: numHeads }, (_, i) => ({
    name: `Head ${i}`,
    specialty: HEAD_SPECIALTIES[i % HEAD_SPECIALTIES.length],
    color: HEAD_COLORS[i % HEAD_COLORS.length],
    description: HEAD_DESCRIPTIONS[i % HEAD_DESCRIPTIONS.length],
  }))
}

// ── Average the attention heads ───────────────────────────────────────
function averageHeads(weights: number[][][]): number[][] {
  const avg: number[][] = []
  for (let r = 0; r < N; r++) {
    avg[r] = []
    for (let c = 0; c < N; c++) {
      let sum = 0
      for (let h = 0; h < weights.length; h++) {
        sum += weights[h][r][c]
      }
      avg[r][c] = sum / weights.length
    }
  }
  return avg
}

// ── Single Heatmap Head Component ─────────────────────────────────────
function AttentionHeatmap({
  weights,
  headIdx,
  meta,
  cellSize,
  selectedToken,
  dimmed,
  expanded,
  onClick,
}: {
  weights: number[][]
  headIdx: number
  meta: { name: string; specialty: string; color: string; description: string }
  cellSize: number
  selectedToken: number | null
  dimmed: boolean
  expanded: boolean
  onClick: () => void
}) {
  const gridSize = cellSize * N
  const colorScale = d3
    .scaleLinear<string>()
    .domain([0, 0.5])
    .range([NLP.attentionLow, meta.color])
    .clamp(true)

  return (
    <motion.div
      className="flex flex-col items-center cursor-pointer"
      animate={{
        opacity: dimmed ? 0.3 : 1,
        scale: expanded ? 1.05 : 1,
      }}
      transition={{ duration: 0.3 }}
      onClick={onClick}
    >
      {/* Head label */}
      <div className="flex items-center gap-1.5 mb-2">
        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: meta.color }} />
        <span className="text-[10px] font-medium text-text-secondary">{meta.specialty}</span>
      </div>

      {/* Heatmap */}
      <div
        className="rounded-lg overflow-hidden"
        style={{
          border: expanded ? `2px solid ${meta.color}` : '1px solid rgba(255,255,255,0.08)',
          padding: 1,
        }}
      >
        <svg
          width={gridSize + 2}
          height={gridSize + 2}
          viewBox={`-1 -1 ${gridSize + 2} ${gridSize + 2}`}
        >
          {weights.map((row, r) =>
            row.map((w, c) => {
              const isHighlighted =
                selectedToken !== null && r === selectedToken
              const isFaded =
                selectedToken !== null && r !== selectedToken

              return (
                <motion.rect
                  key={`${headIdx}-${r}-${c}`}
                  x={c * cellSize}
                  y={r * cellSize}
                  width={cellSize - 1}
                  height={cellSize - 1}
                  rx={2}
                  fill={colorScale(w)}
                  animate={{
                    opacity: isFaded ? 0.2 : 1,
                    strokeWidth: isHighlighted ? 1 : 0,
                  }}
                  stroke={isHighlighted ? 'rgba(255,255,255,0.3)' : 'transparent'}
                  transition={{ duration: 0.2 }}
                />
              )
            })
          )}

          {/* Row/col labels */}
          {TOKENS.map((token, i) => (
            <g key={`labels-${headIdx}-${i}`}>
              {/* Only show if expanded or if there's room */}
              {expanded && (
                <>
                  <text
                    x={i * cellSize + cellSize / 2}
                    y={-3}
                    textAnchor="middle"
                    className="text-[6px] fill-text-tertiary font-mono"
                  >
                    {token}
                  </text>
                  <text
                    x={-3}
                    y={i * cellSize + cellSize / 2 + 2}
                    textAnchor="end"
                    className="text-[6px] fill-text-tertiary font-mono"
                  >
                    {token}
                  </text>
                </>
              )}
            </g>
          ))}
        </svg>
      </div>

      {/* Description */}
      <span className="text-[8px] text-text-tertiary mt-1.5 text-center max-w-[120px]">
        {meta.description}
      </span>
    </motion.div>
  )
}

// ── Arc Diagram for a selected token across heads ─────────────────────
function TokenArcs({
  headIdx,
  weights,
  selectedToken,
  width,
  color,
}: {
  headIdx: number
  weights: number[][]
  selectedToken: number
  width: number
  color: string
}) {
  const tokenSpacing = width / (N + 1)
  const row = weights[selectedToken]

  const arcColorScale = d3
    .scaleLinear<string>()
    .domain([0, 0.5])
    .range(['rgba(255,255,255,0.05)', color])
    .clamp(true)

  const arcWidthScale = d3
    .scaleLinear()
    .domain([0, 0.5])
    .range([0.5, 3])
    .clamp(true)

  return (
    <svg width={width} height={50} viewBox={`0 0 ${width} 50`}>
      {row.map((w, targetIdx) => {
        if (targetIdx === selectedToken) return null
        const srcX = (selectedToken + 1) * tokenSpacing
        const tgtX = (targetIdx + 1) * tokenSpacing
        const midX = (srcX + tgtX) / 2
        const dist = Math.abs(targetIdx - selectedToken)
        const arcH = 8 + dist * 8
        const arcY = 48 - arcH

        return (
          <motion.path
            key={`arc-${headIdx}-${selectedToken}-${targetIdx}`}
            d={`M ${srcX} 48 Q ${midX} ${arcY} ${tgtX} 48`}
            fill="none"
            stroke={arcColorScale(w)}
            strokeWidth={arcWidthScale(w)}
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.8 }}
            transition={{ duration: 0.4, delay: targetIdx * 0.05 }}
          />
        )
      })}
    </svg>
  )
}

// ── Concatenation Animation (step-based) ─────────────────────────────
// Steps: 0=head outputs, 1=concat bracket, 2=concatenated, 3=W_O, 4=final output
function ConcatenationAnimation({
  selectedToken,
  step,
  numHeads,
}: {
  selectedToken: number
  step: number
  numHeads: number
}) {
  const meta = getHeadMeta(numHeads)
  const vecWidth = Math.max(24, 40 - numHeads * 2)
  const vecHeight = 16
  const totalConcatW = meta.length * vecWidth + (meta.length - 1) * 4
  const projectedW = vecWidth * 1.5

  return (
    <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border p-4">
      <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-3">
        Output Combination for &quot;{TOKENS[selectedToken]}&quot;
      </p>

      <div className="flex items-center justify-center gap-4 overflow-x-auto py-2">
        {/* Head output vectors */}
        <div className="flex items-center gap-1">
          {meta.map((m, h) => (
            <motion.div
              key={`vec-${h}`}
              className="rounded"
              style={{
                width: vecWidth,
                height: vecHeight,
                background: `linear-gradient(90deg, ${m.color}33, ${m.color}88)`,
                border: `1px solid ${m.color}66`,
              }}
              initial={{ opacity: 0 }}
              animate={{ opacity: step >= 0 ? 1 : 0 }}
              transition={{ duration: 0.3, delay: h * 0.08 }}
            >
              <span className="text-[7px] text-white/70 flex items-center justify-center h-full font-mono">
                h{h}
              </span>
            </motion.div>
          ))}
        </div>

        {/* Concat bracket */}
        {step >= 1 && (
          <motion.div
            className="flex flex-col items-center"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
          >
            <span className="text-[9px] text-text-tertiary font-mono">concat</span>
            <svg width={12} height={20} viewBox="0 0 12 20">
              <path d="M6 0 L6 20" stroke="rgba(255,255,255,0.3)" strokeWidth={1.5} />
              <path d="M2 5 L6 0 L10 5" stroke="rgba(255,255,255,0.3)" strokeWidth={1} fill="none" />
              <path d="M2 15 L6 20 L10 15" stroke="rgba(255,255,255,0.3)" strokeWidth={1} fill="none" />
            </svg>
          </motion.div>
        )}

        {/* Concatenated vector */}
        {step >= 2 && (
          <motion.div
            className="rounded border border-obsidian-border"
            style={{
              height: vecHeight,
              background: `linear-gradient(90deg, ${NLP.position}33, ${NLP.query}33, ${NLP.key}33, ${NLP.value}33)`,
            }}
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: totalConcatW, opacity: 1 }}
            transition={{ duration: 0.4 }}
          >
            <span className="text-[7px] text-white/60 flex items-center justify-center h-full font-mono whitespace-nowrap">
              h x {numHeads}
            </span>
          </motion.div>
        )}

        {/* W_O projection */}
        {step >= 3 && (
          <motion.div className="flex items-center gap-2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>
            <svg width={30} height={20} viewBox="0 0 30 20">
              <path d="M0 10 L25 10" stroke="rgba(255,255,255,0.4)" strokeWidth={1.5} />
              <path d="M20 5 L25 10 L20 15" stroke="rgba(255,255,255,0.4)" strokeWidth={1.5} fill="none" />
            </svg>
            <div className="rounded-md border px-3 py-1" style={{ borderColor: NLP.token + '66', backgroundColor: NLP.token + '15' }}>
              <span className="text-[9px] font-mono" style={{ color: NLP.token }}>W_O</span>
            </div>
          </motion.div>
        )}

        {/* Final output */}
        {step >= 4 && (
          <motion.div className="flex items-center gap-2" initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.3 }}>
            <svg width={30} height={20} viewBox="0 0 30 20">
              <path d="M0 10 L25 10" stroke="rgba(255,255,255,0.4)" strokeWidth={1.5} />
              <path d="M20 5 L25 10 L20 15" stroke="rgba(255,255,255,0.4)" strokeWidth={1.5} fill="none" />
            </svg>
            <div
              className="rounded"
              style={{
                width: projectedW,
                height: vecHeight,
                background: `linear-gradient(90deg, ${NLP.token}44, ${NLP.token}88)`,
                border: `1px solid ${NLP.token}66`,
              }}
            >
              <span className="text-[7px] text-white/70 flex items-center justify-center h-full font-mono">
                d_model
              </span>
            </div>
          </motion.div>
        )}
      </div>

      <p className="text-[8px] text-text-tertiary mt-2 text-center">
        {numHeads} head outputs are concatenated then linearly projected back to original dimension
      </p>
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────
export function MultiHeadGrid() {
  const [selectedToken, setSelectedToken] = useState<number | null>(null)
  const [expandedHead, setExpandedHead] = useState<number | null>(null)
  const [showAverage, setShowAverage] = useState(false)
  const [showCombination, setShowCombination] = useState(false)
  const [numHeads, setNumHeads] = useState(4)

  // Adapt head weights to current head count
  const currentWeights = useMemo(
    () => adaptHeadCount(HEAD_WEIGHTS, numHeads),
    [numHeads]
  )
  const currentMeta = useMemo(() => getHeadMeta(numHeads), [numHeads])

  const avgWeights = useMemo(() => averageHeads(currentWeights), [currentWeights])

  // Concatenation transport (5 steps: heads → concat bracket → merged → W_O → output)
  const concatSnapshots = useMemo(() => [0, 1, 2, 3, 4], [])
  const concatPlayer = useAlgorithmPlayer({ snapshots: concatSnapshots, baseFps: 1 })

  const handleHeadClick = useCallback(
    (headIdx: number) => {
      setExpandedHead((prev) => (prev === headIdx ? null : headIdx))
    },
    []
  )

  const handleTokenClick = useCallback((tokenIdx: number) => {
    setSelectedToken((prev) => (prev === tokenIdx ? null : tokenIdx))
  }, [])

  // Adaptive cell size based on head count
  const cellSize = numHeads <= 4 ? 22 : 16
  // Grid columns: h<=2 use cols-2, h<=4 use cols-4, h=8 use cols-4
  const gridCols = numHeads <= 2 ? 'grid-cols-2' : 'grid-cols-2 md:grid-cols-4'
  const arcCols = numHeads <= 2 ? 'grid-cols-2' : 'grid-cols-2 md:grid-cols-4'

  return (
    <div className="space-y-4">
      {/* Token pills */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-center gap-3 mb-1">
          <span className="text-[10px] uppercase tracking-wider text-text-tertiary mr-2">
            Tokens:
          </span>
          {TOKENS.map((token, i) => (
            <motion.button
              key={i}
              className="px-3 py-1 rounded-full text-xs font-mono transition-colors"
              style={{
                backgroundColor:
                  selectedToken === i
                    ? NLP.token + '30'
                    : 'rgba(255,255,255,0.06)',
                border: `1px solid ${
                  selectedToken === i ? NLP.token + '80' : 'rgba(255,255,255,0.1)'
                }`,
                color: selectedToken === i ? NLP.token : '#A1A1AA',
              }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleTokenClick(i)}
            >
              {token}
            </motion.button>
          ))}
        </div>
        <p className="text-[8px] text-text-tertiary">
          Click a token to highlight its attention row across all heads
        </p>
      </GlassCard>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        {/* Number of heads selector */}
        <div>
          <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-1">Heads (h)</p>
          <div className="flex gap-1">
            {[1, 2, 4, 8].map((h) => (
              <button
                key={h}
                onClick={() => { setNumHeads(h); setExpandedHead(null) }}
                className={`px-2 py-1 rounded text-xs font-mono transition-all ${
                  numHeads === h
                    ? 'bg-accent/20 text-accent border border-accent/40'
                    : 'text-text-tertiary hover:text-text-secondary border border-transparent'
                }`}
              >
                {h}
              </button>
            ))}
          </div>
        </div>

        <div className="h-5 w-px bg-obsidian-border" />

        <Toggle
          label="Show averaged attention"
          checked={showAverage}
          onChange={setShowAverage}
        />

        <div className="h-5 w-px bg-obsidian-border" />

        <span className="text-[10px] text-text-tertiary">
          d_k = d_model / {numHeads}
        </span>
      </div>

      {/* Attention heatmap grid */}
      <AnimatePresence mode="wait">
        {showAverage ? (
          <motion.div
            key="averaged"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
          >
            <GlassCard className="p-6">
              <div className="flex flex-col items-center">
                <p className="text-xs font-medium text-text-secondary mb-3">
                  Averaged Attention ({numHeads} heads combined)
                </p>

                {/* Token labels across top */}
                <div className="flex items-center mb-1" style={{ paddingLeft: 36 }}>
                  {TOKENS.map((token, i) => (
                    <div
                      key={i}
                      className="text-[8px] font-mono text-text-tertiary text-center"
                      style={{ width: cellSize }}
                    >
                      {token}
                    </div>
                  ))}
                </div>

                {/* Heatmap with row labels */}
                <SVGContainer
                  aspectRatio={1}
                  minHeight={180}
                  maxHeight={280}
                  padding={{ top: 10, right: 20, bottom: 20, left: 40 }}
                >
                  {({ innerWidth, innerHeight }) => {
                    const cs = Math.min(innerWidth / N, innerHeight / N, 30)
                    const avgColorScale = d3
                      .scaleLinear<string>()
                      .domain([0, 0.3])
                      .range([NLP.attentionLow, NLP.attentionHigh])
                      .clamp(true)

                    return (
                      <>
                        {avgWeights.map((row, r) =>
                          row.map((w, c) => (
                            <motion.rect
                              key={`avg-${r}-${c}`}
                              x={c * cs}
                              y={r * cs}
                              width={cs - 1}
                              height={cs - 1}
                              rx={2}
                              fill={avgColorScale(w)}
                              animate={{
                                opacity:
                                  selectedToken !== null && r !== selectedToken
                                    ? 0.2
                                    : 1,
                              }}
                              transition={{ duration: 0.2 }}
                            />
                          ))
                        )}

                        {/* Weight labels */}
                        {avgWeights.map((row, r) =>
                          row.map((w, c) => {
                            if (cs < 20) return null
                            return (
                              <text
                                key={`avg-lbl-${r}-${c}`}
                                x={c * cs + cs / 2}
                                y={r * cs + cs / 2 + 3}
                                textAnchor="middle"
                                className="text-[7px] font-mono fill-white/60"
                              >
                                {w.toFixed(2)}
                              </text>
                            )
                          })
                        )}

                        {/* Row labels */}
                        {TOKENS.map((token, i) => (
                          <text
                            key={`avg-row-${i}`}
                            x={-4}
                            y={i * cs + cs / 2 + 3}
                            textAnchor="end"
                            className="text-[8px] font-mono fill-text-tertiary"
                          >
                            {token}
                          </text>
                        ))}
                      </>
                    )
                  }}
                </SVGContainer>
              </div>
            </GlassCard>
          </motion.div>
        ) : (
          <motion.div
            key={`individual-${numHeads}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
          >
            <GlassCard className="p-6">
              {/* Arc diagrams when a token is selected */}
              <AnimatePresence>
                {selectedToken !== null && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="overflow-hidden mb-3"
                  >
                    <p className="text-[10px] text-text-secondary mb-2 text-center">
                      Attention arcs from &quot;{TOKENS[selectedToken]}&quot; across heads
                    </p>
                    <div className={`grid ${arcCols} gap-2`}>
                      {currentMeta.map((meta, h) => (
                        <div key={`arc-${h}`} className="flex flex-col items-center">
                          <span
                            className="text-[8px] font-mono mb-0.5"
                            style={{ color: meta.color }}
                          >
                            {meta.specialty}
                          </span>
                          <TokenArcs
                            headIdx={h}
                            weights={currentWeights[h]}
                            selectedToken={selectedToken}
                            width={160}
                            color={meta.color}
                          />
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Heatmap grid */}
              <div className={`grid ${gridCols} gap-6 justify-items-center`}>
                {currentMeta.map((meta, h) => (
                  <AttentionHeatmap
                    key={`${numHeads}-${h}`}
                    weights={currentWeights[h]}
                    headIdx={h}
                    meta={meta}
                    cellSize={cellSize}
                    selectedToken={selectedToken}
                    dimmed={expandedHead !== null && expandedHead !== h}
                    expanded={expandedHead === h}
                    onClick={() => handleHeadClick(h)}
                  />
                ))}
              </div>

              <p className="text-[8px] text-text-tertiary mt-4 text-center">
                Click a head to expand it. Each head learns different attention patterns.
              </p>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Concatenation section */}
      <div className="flex items-center gap-3">
        <Button
          variant="secondary"
          size="sm"
          onClick={() => {
            setShowCombination((v) => !v)
            concatPlayer.reset()
          }}
          active={showCombination}
        >
          {showCombination ? 'Hide Combination' : 'Show Combination'}
        </Button>
        <span className="text-[10px] text-text-tertiary">
          See how multi-head outputs are merged
        </span>
      </div>

      <AnimatePresence>
        {showCombination && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden space-y-3"
          >
            <ConcatenationAnimation
              selectedToken={selectedToken ?? 1}
              step={concatPlayer.currentSnapshot}
              numHeads={numHeads}
            />
            <TransportControls
              isPlaying={concatPlayer.isPlaying}
              isAtStart={concatPlayer.isAtStart}
              isAtEnd={concatPlayer.isAtEnd}
              currentStep={concatPlayer.currentStep}
              totalSteps={concatPlayer.totalSteps}
              speed={concatPlayer.speed}
              onPlay={concatPlayer.play}
              onPause={concatPlayer.pause}
              onTogglePlay={concatPlayer.togglePlay}
              onStepForward={concatPlayer.stepForward}
              onStepBack={concatPlayer.stepBack}
              onReset={concatPlayer.reset}
              onSetSpeed={concatPlayer.setSpeed}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
