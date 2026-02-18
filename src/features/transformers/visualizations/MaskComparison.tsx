import { useState, useCallback, useMemo } from 'react'
import * as d3 from 'd3'
import { motion, AnimatePresence } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Button } from '../../../components/ui/Button'
import { Select } from '../../../components/ui/Select'
import { Slider } from '../../../components/ui/Slider'

// ── NLP Colors ────────────────────────────────────────────────────────
const NLP = {
  query: '#F472B6',
  key: '#34D399',
  value: '#FBBF24',
  position: '#38BDF8',
  token: '#818CF8',
  attentionLow: 'rgba(99, 102, 241, 0.05)',
  attentionHigh: '#6366F1',
  cross: '#E879F9',
}

// ── Tokens ────────────────────────────────────────────────────────────
const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat']
const N = TOKENS.length

// ── Full bidirectional attention weights (encoder, rows sum to 1.0) ───
const ENCODER_WEIGHTS: number[][] = [
  [0.15, 0.25, 0.20, 0.10, 0.15, 0.15], // The
  [0.15, 0.10, 0.30, 0.10, 0.10, 0.25], // cat
  [0.10, 0.35, 0.10, 0.15, 0.10, 0.20], // sat
  [0.10, 0.10, 0.20, 0.10, 0.15, 0.35], // on
  [0.20, 0.10, 0.10, 0.20, 0.10, 0.30], // the
  [0.10, 0.20, 0.25, 0.20, 0.15, 0.10], // mat
]

// ── Pre-softmax logits for the decoder (before masking) ───────────────
// We'll apply causal mask and re-normalize
const DECODER_LOGITS: number[][] = [
  [2.0, 0.5, 0.8, 0.3, 0.4, 0.6],
  [1.2, 1.8, 0.6, 0.4, 0.5, 0.7],
  [0.8, 1.5, 1.6, 0.5, 0.3, 0.9],
  [0.6, 0.8, 1.2, 1.9, 0.7, 0.5],
  [1.0, 0.6, 0.7, 1.0, 1.7, 0.8],
  [0.5, 0.9, 1.1, 0.8, 1.0, 2.1],
]

// Apply a custom mask (boolean[][]) and softmax to get attention weights
function computeMaskedWeights(logits: number[][], mask: boolean[][]): number[][] {
  const size = logits.length
  const weights: number[][] = []
  for (let r = 0; r < size; r++) {
    const row: number[] = []
    let sumExp = 0
    const exps: number[] = []
    for (let c = 0; c < size; c++) {
      if (mask[r][c]) {
        const e = Math.exp(logits[r][c])
        exps.push(e)
        sumExp += e
      } else {
        exps.push(0)
      }
    }
    for (let c = 0; c < size; c++) {
      row.push(mask[r][c] && sumExp > 0 ? exps[c] / sumExp : 0)
    }
    weights.push(row)
  }
  return weights
}

// Create a causal (lower-triangular) mask
function createCausalMask(size: number): boolean[][] {
  return Array.from({ length: size }, (_, r) =>
    Array.from({ length: size }, (_, c) => c <= r)
  )
}

// Create a full (all-true) mask
function createFullMask(size: number): boolean[][] {
  return Array.from({ length: size }, () =>
    Array.from({ length: size }, () => true)
  )
}

const DECODER_WEIGHTS = computeMaskedWeights(DECODER_LOGITS, createCausalMask(N))

// ── Cross-attention data (encoder-decoder models) ─────────────────────
const FRENCH_TOKENS = ['Le', 'chat', 'est', 'assis', 'sur', 'le', 'tapis']
const ENGLISH_TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat']

// Cross-attention weights: each English token attends to French tokens
const CROSS_ATTENTION_WEIGHTS: number[][] = [
  [0.55, 0.10, 0.05, 0.05, 0.05, 0.15, 0.05], // The → Le
  [0.05, 0.60, 0.05, 0.10, 0.05, 0.05, 0.10], // cat → chat
  [0.03, 0.07, 0.35, 0.40, 0.05, 0.05, 0.05], // sat → est+assis
  [0.05, 0.05, 0.05, 0.05, 0.65, 0.10, 0.05], // on → sur
  [0.15, 0.05, 0.05, 0.05, 0.05, 0.55, 0.10], // the → le
  [0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.65], // mat → tapis
]

// ── Generation tokens for the walkthrough ─────────────────────────────
const GEN_TOKENS = ['[START]', 'The', 'cat', 'sat', 'on', 'the', 'mat']

// Generate attention weights for each generation step
function computeGenStepWeights(step: number): number[][] {
  const size = step + 1
  const weights: number[][] = []
  for (let r = 0; r < size; r++) {
    const row: number[] = []
    let sumExp = 0
    const exps: number[] = []
    for (let c = 0; c < size; c++) {
      if (c <= r) {
        // Create plausible logits
        const selfBias = c === r ? 1.5 : 0
        const proxBias = 1.0 / (1 + Math.abs(r - c))
        const logit = selfBias + proxBias + Math.sin((r + 1) * (c + 1) * 0.7) * 0.5
        const e = Math.exp(logit)
        exps.push(e)
        sumExp += e
      } else {
        exps.push(0)
      }
    }
    for (let c = 0; c < size; c++) {
      row.push(c <= r ? exps[c] / sumExp : 0)
    }
    weights.push(row)
  }
  return weights
}

// ── Model type options ────────────────────────────────────────────────
const MODEL_OPTIONS = [
  { value: 'both', label: 'Both (side by side)' },
  { value: 'encoder', label: 'Encoder only (BERT)' },
  { value: 'decoder', label: 'Decoder only (GPT)' },
  { value: 'enc-dec', label: 'Encoder-Decoder (T5)' },
]

// ── Attention Heatmap Panel ───────────────────────────────────────────
function AttentionPanel({
  title,
  description,
  weights,
  tokens,
  maskApplied,
  innerWidth,
  innerHeight,
  panelType,
}: {
  title: string
  description: string
  weights: number[][]
  tokens: string[]
  maskApplied: boolean
  innerWidth: number
  innerHeight: number
  panelType: 'encoder' | 'decoder'
}) {
  const size = tokens.length
  const labelPad = 35
  const availW = innerWidth - labelPad
  const availH = innerHeight - labelPad - 30
  const cellSize = Math.min(availW / size, availH / size, 36)
  const gridSize = cellSize * size
  const startX = labelPad + (availW - gridSize) / 2
  const startY = 28

  const colorScale = d3
    .scaleLinear<string>()
    .domain([0, 0.5])
    .range([NLP.attentionLow, panelType === 'encoder' ? NLP.key : NLP.query])
    .clamp(true)

  return (
    <>
      {/* Title */}
      <text
        x={innerWidth / 2}
        y={12}
        textAnchor="middle"
        className="text-[11px] font-medium"
        fill="#E4E4E7"
      >
        {title}
      </text>

      {/* Column labels (top) */}
      {tokens.map((token, c) => (
        <text
          key={`col-${c}`}
          x={startX + c * cellSize + cellSize / 2}
          y={startY - 4}
          textAnchor="middle"
          className="text-[8px] font-mono fill-text-tertiary"
        >
          {token}
        </text>
      ))}

      {/* Row labels (left) */}
      {tokens.map((token, r) => (
        <text
          key={`row-${r}`}
          x={startX - 4}
          y={startY + r * cellSize + cellSize / 2 + 3}
          textAnchor="end"
          className="text-[8px] font-mono fill-text-tertiary"
        >
          {token}
        </text>
      ))}

      {/* Heatmap cells */}
      {weights.map((row, r) =>
        row.map((w, c) => {
          const isMasked = panelType === 'decoder' && maskApplied && c > r

          return (
            <g key={`cell-${r}-${c}`}>
              {/* Cell background */}
              <motion.rect
                x={startX + c * cellSize}
                y={startY + r * cellSize}
                width={cellSize - 1}
                height={cellSize - 1}
                rx={3}
                fill={isMasked ? 'rgba(255, 255, 255, 0.02)' : colorScale(w)}
                animate={{
                  fill: isMasked ? 'rgba(255, 255, 255, 0.02)' : colorScale(w),
                }}
                transition={{ duration: 0.4 }}
              />

              {/* Masked X pattern */}
              {isMasked && (
                <motion.g
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 0.4 }}
                  transition={{ duration: 0.3, delay: (c - r) * 0.05 }}
                >
                  <line
                    x1={startX + c * cellSize + 3}
                    y1={startY + r * cellSize + 3}
                    x2={startX + (c + 1) * cellSize - 4}
                    y2={startY + (r + 1) * cellSize - 4}
                    stroke="#F87171"
                    strokeWidth={1}
                    strokeOpacity={0.5}
                  />
                  <line
                    x1={startX + (c + 1) * cellSize - 4}
                    y1={startY + r * cellSize + 3}
                    x2={startX + c * cellSize + 3}
                    y2={startY + (r + 1) * cellSize - 4}
                    stroke="#F87171"
                    strokeWidth={1}
                    strokeOpacity={0.5}
                  />
                </motion.g>
              )}

              {/* Weight text (only show if cell is large enough and not masked) */}
              {!isMasked && cellSize >= 22 && w > 0.01 && (
                <text
                  x={startX + c * cellSize + cellSize / 2}
                  y={startY + r * cellSize + cellSize / 2 + 3}
                  textAnchor="middle"
                  className="text-[6px] font-mono fill-white/50"
                >
                  {w.toFixed(2)}
                </text>
              )}
            </g>
          )
        })
      )}

      {/* Description */}
      <foreignObject
        x={0}
        y={startY + gridSize + 8}
        width={innerWidth}
        height={30}
      >
        <div className="text-[8px] text-text-tertiary text-center px-2 leading-relaxed">
          {description}
        </div>
      </foreignObject>
    </>
  )
}

// ── Generation Walkthrough Panel ──────────────────────────────────────
function GenerationWalkthrough({
  genStep,
}: {
  genStep: number
}) {
  const currentTokens = GEN_TOKENS.slice(0, genStep + 1)
  const weights = useMemo(() => computeGenStepWeights(genStep), [genStep])
  const size = currentTokens.length

  const cellSize = Math.min(30, 200 / Math.max(size, 1))
  const gridSize = cellSize * size
  const labelPad = 50

  const colorScale = d3
    .scaleLinear<string>()
    .domain([0, 0.5])
    .range([NLP.attentionLow, NLP.query])
    .clamp(true)

  return (
    <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border p-4">
      <div className="flex items-center gap-3 mb-3">
        <p className="text-[10px] uppercase tracking-wider text-text-tertiary">
          Autoregressive Generation
        </p>
        <span className="text-[9px] font-mono text-text-secondary px-2 py-0.5 rounded-md bg-obsidian-glass border border-obsidian-border">
          Step {genStep}: generating &quot;{GEN_TOKENS[genStep]}&quot;
        </span>
      </div>

      {/* Token pills showing current sequence */}
      <div className="flex flex-wrap items-center gap-2 mb-4">
        {GEN_TOKENS.map((token, i) => {
          const isGenerated = i <= genStep
          const isCurrent = i === genStep
          const isPending = i > genStep

          return (
            <motion.div
              key={`gen-token-${i}`}
              className="px-2.5 py-1 rounded-full text-[10px] font-mono"
              style={{
                backgroundColor: isCurrent
                  ? NLP.query + '25'
                  : isGenerated
                    ? NLP.token + '15'
                    : 'rgba(255,255,255,0.03)',
                border: `1px solid ${
                  isCurrent
                    ? NLP.query + '80'
                    : isGenerated
                      ? NLP.token + '30'
                      : 'rgba(255,255,255,0.06)'
                }`,
                color: isPending ? '#52525B' : '#A1A1AA',
              }}
              animate={{
                scale: isCurrent ? 1.05 : 1,
              }}
              transition={{ duration: 0.2 }}
            >
              {token}
              {isPending && (
                <span className="ml-1 text-[8px] text-text-tertiary">?</span>
              )}
            </motion.div>
          )
        })}
      </div>

      {/* Growing attention grid */}
      <div className="flex items-start gap-4 overflow-x-auto">
        <svg
          width={labelPad + gridSize + 10}
          height={labelPad + gridSize + 10}
          viewBox={`0 0 ${labelPad + gridSize + 10} ${labelPad + gridSize + 10}`}
        >
          {/* Column labels */}
          {currentTokens.map((token, c) => (
            <text
              key={`gen-col-${c}`}
              x={labelPad + c * cellSize + cellSize / 2}
              y={10}
              textAnchor="middle"
              className="text-[7px] font-mono fill-text-tertiary"
            >
              {token.length > 5 ? token.slice(0, 4) + '..' : token}
            </text>
          ))}

          {/* Row labels */}
          {currentTokens.map((token, r) => (
            <text
              key={`gen-row-${r}`}
              x={labelPad - 4}
              y={18 + r * cellSize + cellSize / 2 + 2}
              textAnchor="end"
              className="text-[7px] font-mono fill-text-tertiary"
            >
              {token.length > 5 ? token.slice(0, 4) + '..' : token}
            </text>
          ))}

          {/* Cells */}
          {weights.map((row, r) =>
            row.map((w, c) => {
              const isMasked = c > r

              return (
                <motion.rect
                  key={`gen-cell-${r}-${c}`}
                  x={labelPad + c * cellSize}
                  y={18 + r * cellSize}
                  width={cellSize - 1}
                  height={cellSize - 1}
                  rx={2}
                  fill={isMasked ? 'rgba(255,255,255,0.02)' : colorScale(w)}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.3, delay: (r + c) * 0.03 }}
                />
              )
            })
          )}

          {/* Causal mask outline */}
          <rect
            x={labelPad}
            y={18}
            width={gridSize}
            height={gridSize}
            fill="none"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth={1}
            rx={2}
          />
        </svg>

        <div className="flex flex-col gap-1 text-[9px] text-text-tertiary py-2 min-w-[140px]">
          <p>
            At step {genStep}, the model sees
            <span className="text-text-secondary font-medium"> {size} token{size > 1 ? 's' : ''}</span>.
          </p>
          <p>
            The attention grid is <span className="font-mono text-text-secondary">{size}x{size}</span>,
            lower-triangular.
          </p>
          {genStep > 0 && (
            <p className="mt-1">
              Each new token can only attend to previous tokens and itself (causal constraint).
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Interactive Mask Builder ──────────────────────────────────────────
function InteractiveMaskBuilder() {
  const [mask, setMask] = useState<boolean[][]>(() => createFullMask(N))

  const weights = useMemo(
    () => computeMaskedWeights(DECODER_LOGITS, mask),
    [mask]
  )

  const toggleCell = useCallback((r: number, c: number) => {
    setMask((prev) => {
      const next = prev.map((row) => [...row])
      next[r][c] = !next[r][c]
      return next
    })
  }, [])

  const applyCausalMask = useCallback(() => {
    setMask(createCausalMask(N))
  }, [])

  const clearMask = useCallback(() => {
    setMask(createFullMask(N))
  }, [])

  const cellSize = 32
  const labelPad = 40
  const gridSize = cellSize * N
  const svgW = labelPad + gridSize + 10
  const svgH = labelPad + gridSize + 10

  const colorScale = d3
    .scaleLinear<string>()
    .domain([0, 0.5])
    .range([NLP.attentionLow, NLP.attentionHigh])
    .clamp(true)

  return (
    <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border p-4">
      <div className="flex items-center justify-between gap-3 mb-3">
        <div>
          <p className="text-[10px] uppercase tracking-wider text-text-tertiary">
            Interactive Mask Builder
          </p>
          <p className="text-[9px] text-text-tertiary mt-0.5">
            Click cells to toggle. Blocked cells become 0 after softmax.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="secondary" size="sm" onClick={applyCausalMask}>
            Apply Causal Mask
          </Button>
          <Button variant="ghost" size="sm" onClick={clearMask}>
            Clear (Full)
          </Button>
        </div>
      </div>

      <div className="flex items-start gap-6 overflow-x-auto">
        {/* Mask grid */}
        <div>
          <p className="text-[9px] text-text-tertiary mb-1 text-center">
            Mask (click to toggle)
          </p>
          <svg width={svgW} height={svgH} viewBox={`0 0 ${svgW} ${svgH}`}>
            {/* Column labels */}
            {TOKENS.map((token, c) => (
              <text
                key={`mb-col-${c}`}
                x={labelPad + c * cellSize + cellSize / 2}
                y={12}
                textAnchor="middle"
                className="text-[8px] font-mono fill-text-tertiary"
              >
                {token}
              </text>
            ))}

            {/* Row labels */}
            {TOKENS.map((token, r) => (
              <text
                key={`mb-row-${r}`}
                x={labelPad - 4}
                y={20 + r * cellSize + cellSize / 2 + 3}
                textAnchor="end"
                className="text-[8px] font-mono fill-text-tertiary"
              >
                {token}
              </text>
            ))}

            {/* Cells */}
            {mask.map((row, r) =>
              row.map((allowed, c) => (
                <g
                  key={`mb-cell-${r}-${c}`}
                  className="cursor-pointer"
                  onClick={() => toggleCell(r, c)}
                >
                  <motion.rect
                    x={labelPad + c * cellSize}
                    y={20 + r * cellSize}
                    width={cellSize - 1}
                    height={cellSize - 1}
                    rx={3}
                    fill={allowed ? `${NLP.attentionHigh}30` : 'rgba(255,255,255,0.02)'}
                    stroke={allowed ? `${NLP.attentionHigh}50` : 'rgba(248,113,113,0.3)'}
                    strokeWidth={1}
                    animate={{
                      fill: allowed ? `${NLP.attentionHigh}30` : 'rgba(255,255,255,0.02)',
                    }}
                    transition={{ duration: 0.2 }}
                  />
                  {!allowed && (
                    <motion.g
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 0.5 }}
                      transition={{ duration: 0.2 }}
                    >
                      <line
                        x1={labelPad + c * cellSize + 5}
                        y1={20 + r * cellSize + 5}
                        x2={labelPad + (c + 1) * cellSize - 6}
                        y2={20 + (r + 1) * cellSize - 6}
                        stroke="#F87171"
                        strokeWidth={1.5}
                      />
                      <line
                        x1={labelPad + (c + 1) * cellSize - 6}
                        y1={20 + r * cellSize + 5}
                        x2={labelPad + c * cellSize + 5}
                        y2={20 + (r + 1) * cellSize - 6}
                        stroke="#F87171"
                        strokeWidth={1.5}
                      />
                    </motion.g>
                  )}
                  {allowed && (
                    <text
                      x={labelPad + c * cellSize + cellSize / 2}
                      y={20 + r * cellSize + cellSize / 2 + 3}
                      textAnchor="middle"
                      className="text-[7px] font-mono fill-white/40 pointer-events-none"
                    >
                      1
                    </text>
                  )}
                </g>
              ))
            )}
          </svg>
        </div>

        {/* Arrow */}
        <div className="flex flex-col items-center justify-center pt-16 text-text-tertiary">
          <span className="text-[9px] font-mono mb-1">softmax</span>
          <span className="text-lg">→</span>
        </div>

        {/* Resulting attention weights */}
        <div>
          <p className="text-[9px] text-text-tertiary mb-1 text-center">
            Attention weights (after softmax)
          </p>
          <svg width={svgW} height={svgH} viewBox={`0 0 ${svgW} ${svgH}`}>
            {/* Column labels */}
            {TOKENS.map((token, c) => (
              <text
                key={`mw-col-${c}`}
                x={labelPad + c * cellSize + cellSize / 2}
                y={12}
                textAnchor="middle"
                className="text-[8px] font-mono fill-text-tertiary"
              >
                {token}
              </text>
            ))}

            {/* Row labels */}
            {TOKENS.map((token, r) => (
              <text
                key={`mw-row-${r}`}
                x={labelPad - 4}
                y={20 + r * cellSize + cellSize / 2 + 3}
                textAnchor="end"
                className="text-[8px] font-mono fill-text-tertiary"
              >
                {token}
              </text>
            ))}

            {/* Cells */}
            {weights.map((row, r) =>
              row.map((w, c) => (
                <g key={`mw-cell-${r}-${c}`}>
                  <motion.rect
                    x={labelPad + c * cellSize}
                    y={20 + r * cellSize}
                    width={cellSize - 1}
                    height={cellSize - 1}
                    rx={3}
                    fill={w === 0 ? 'rgba(255,255,255,0.02)' : colorScale(w)}
                    animate={{
                      fill: w === 0 ? 'rgba(255,255,255,0.02)' : colorScale(w),
                    }}
                    transition={{ duration: 0.3 }}
                  />
                  {w > 0.01 && (
                    <text
                      x={labelPad + c * cellSize + cellSize / 2}
                      y={20 + r * cellSize + cellSize / 2 + 3}
                      textAnchor="middle"
                      className="text-[7px] font-mono fill-white/50"
                    >
                      {w.toFixed(2)}
                    </text>
                  )}
                  {!mask[r][c] && (
                    <text
                      x={labelPad + c * cellSize + cellSize / 2}
                      y={20 + r * cellSize + cellSize / 2 + 3}
                      textAnchor="middle"
                      className="text-[7px] font-mono"
                      fill="#F87171"
                      fillOpacity={0.5}
                    >
                      0
                    </text>
                  )}
                </g>
              ))
            )}
          </svg>
        </div>
      </div>

      <p className="text-[9px] text-text-tertiary mt-2 text-center">
        Masked cells receive -inf before softmax, causing them to become exactly 0. The remaining cells re-normalize to sum to 1.
      </p>
    </div>
  )
}

// ── Cross-Attention Sub-Viz (Encoder-Decoder) ────────────────────────
function CrossAttentionViz() {
  const [activeDecToken, setActiveDecToken] = useState(0)

  const weights = CROSS_ATTENTION_WEIGHTS[activeDecToken]
  const maxW = Math.max(...weights)

  return (
    <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border p-4">
      <div className="flex items-center gap-3 mb-3">
        <p className="text-[10px] uppercase tracking-wider text-text-tertiary">
          Cross-Attention (Encoder-Decoder)
        </p>
        <span className="text-[9px] font-mono text-text-secondary px-2 py-0.5 rounded-md bg-obsidian-glass border border-obsidian-border">
          T5 / mBART style
        </span>
      </div>

      <p className="text-[9px] text-text-tertiary mb-4">
        The decoder asks: &quot;Which parts of the French input are relevant to the English word I&apos;m generating?&quot;
        Queries come from the decoder. Keys and Values come from the encoder.
      </p>

      <div className="flex items-start gap-6 justify-center">
        {/* Encoder side (French) */}
        <div className="flex flex-col items-center gap-2">
          <p className="text-[10px] font-medium" style={{ color: NLP.key }}>
            Encoder (French)
          </p>
          <p className="text-[8px] text-text-tertiary font-mono">Keys + Values</p>
          <div className="flex flex-col gap-1.5">
            {FRENCH_TOKENS.map((token, i) => (
              <motion.div
                key={`fr-${i}`}
                className="px-3 py-1.5 rounded-lg text-[10px] font-mono text-center border"
                style={{
                  backgroundColor: `${NLP.key}${Math.round(10 + weights[i] * 30).toString(16).padStart(2, '0')}`,
                  borderColor: `${NLP.key}${Math.round(20 + weights[i] * 60).toString(16).padStart(2, '0')}`,
                  color: weights[i] > 0.3 ? NLP.key : '#A1A1AA',
                }}
                animate={{
                  scale: weights[i] === maxW ? 1.05 : 1,
                }}
                transition={{ duration: 0.2 }}
              >
                {token}
              </motion.div>
            ))}
          </div>
        </div>

        {/* Attention lines (SVG) */}
        <svg
          width={160}
          height={FRENCH_TOKENS.length * 32 + 40}
          className="shrink-0"
        >
          {FRENCH_TOKENS.map((_, fi) => {
            const w = weights[fi]
            if (w < 0.03) return null
            // French tokens: 7 items, gap 32px each, starting at y=40
            const encY = 40 + fi * 32 + 14
            // English tokens: 6 items, scale to align within the same vertical range
            const decGap = (FRENCH_TOKENS.length - 1) * 32 / Math.max(ENGLISH_TOKENS.length - 1, 1)
            const decY = 40 + activeDecToken * decGap + 14
            return (
              <motion.line
                key={`attn-line-${fi}`}
                x1={0}
                y1={encY}
                x2={160}
                y2={decY}
                stroke={NLP.cross}
                strokeWidth={1 + w * 4}
                strokeOpacity={0.15 + w * 0.7}
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.4, delay: fi * 0.05 }}
              />
            )
          })}
          {/* Label */}
          <text
            x={80}
            y={16}
            textAnchor="middle"
            className="text-[8px] font-mono"
            fill="#71717A"
          >
            attention weights
          </text>
        </svg>

        {/* Decoder side (English) */}
        <div className="flex flex-col items-center gap-2">
          <p className="text-[10px] font-medium" style={{ color: NLP.query }}>
            Decoder (English)
          </p>
          <p className="text-[8px] text-text-tertiary font-mono">Queries</p>
          <div className="flex flex-col gap-1.5">
            {ENGLISH_TOKENS.map((token, i) => (
              <motion.div
                key={`en-${i}`}
                className="px-3 py-1.5 rounded-lg text-[10px] font-mono text-center border cursor-pointer"
                style={{
                  backgroundColor:
                    i === activeDecToken
                      ? `${NLP.query}25`
                      : 'rgba(255,255,255,0.03)',
                  borderColor:
                    i === activeDecToken
                      ? `${NLP.query}80`
                      : 'rgba(255,255,255,0.08)',
                  color:
                    i === activeDecToken ? NLP.query : '#A1A1AA',
                }}
                onClick={() => setActiveDecToken(i)}
                whileHover={{ scale: 1.05 }}
              >
                {token}
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Active token info */}
      <div className="mt-3 text-center">
        <p className="text-[9px] text-text-tertiary">
          Generating <span className="text-text-secondary font-medium">&quot;{ENGLISH_TOKENS[activeDecToken]}&quot;</span>
          {' '}, strongest attention to{' '}
          <span style={{ color: NLP.key }} className="font-medium">
            &quot;{FRENCH_TOKENS[weights.indexOf(maxW)]}&quot;
          </span>
          {' '}({(maxW * 100).toFixed(0)}%)
        </p>
      </div>
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────
export function MaskComparison() {
  const [modelType, setModelType] = useState('both')
  const [maskApplied, setMaskApplied] = useState(true)
  const [genStep, setGenStep] = useState(0)

  const handleToggleMask = useCallback(() => {
    setMaskApplied((prev) => !prev)
  }, [])

  const showEncoder = modelType === 'both' || modelType === 'encoder' || modelType === 'enc-dec'
  const showDecoder = modelType === 'both' || modelType === 'decoder' || modelType === 'enc-dec'
  const showCrossAttention = modelType === 'enc-dec'

  return (
    <div className="space-y-4">
      {/* Token pills */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-[10px] uppercase tracking-wider text-text-tertiary mr-2">
            Tokens:
          </span>
          {TOKENS.map((token, i) => (
            <div
              key={i}
              className="px-3 py-1 rounded-full text-xs font-mono"
              style={{
                backgroundColor: NLP.token + '15',
                border: `1px solid ${NLP.token}30`,
                color: '#A1A1AA',
              }}
            >
              {token}
            </div>
          ))}
        </div>
      </GlassCard>

      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          <Select
            label="Model Type"
            value={modelType}
            options={MODEL_OPTIONS}
            onChange={setModelType}
            className="w-48"
          />

          <div className="h-6 w-px bg-obsidian-border" />

          <Button
            variant="secondary"
            size="sm"
            onClick={handleToggleMask}
            active={maskApplied}
          >
            {maskApplied ? 'Remove Mask' : 'Apply Causal Mask'}
          </Button>
        </div>
      </GlassCard>

      {/* Comparison panels */}
      <div
        className={`grid gap-4 ${
          modelType === 'both' || modelType === 'enc-dec'
            ? 'grid-cols-1 md:grid-cols-2'
            : 'grid-cols-1 max-w-lg mx-auto'
        }`}
      >
        {/* Encoder panel */}
        <AnimatePresence>
          {showEncoder && (
            <motion.div
              key="encoder"
              className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border overflow-hidden"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              <SVGContainer
                aspectRatio={1}
                minHeight={280}
                maxHeight={420}
                padding={{ top: 10, right: 15, bottom: 10, left: 15 }}
              >
                {({ innerWidth, innerHeight }) => (
                  <AttentionPanel
                    title="Encoder: Full Visibility"
                    description="Encoder: full visibility. Every token sees the entire sequence. Used in BERT for bidirectional understanding."
                    weights={ENCODER_WEIGHTS}
                    tokens={TOKENS}
                    maskApplied={false}
                    innerWidth={innerWidth}
                    innerHeight={innerHeight}
                    panelType="encoder"
                  />
                )}
              </SVGContainer>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Decoder panel */}
        <AnimatePresence>
          {showDecoder && (
            <motion.div
              key="decoder"
              className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border overflow-hidden"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.3 }}
            >
              <SVGContainer
                aspectRatio={1}
                minHeight={280}
                maxHeight={420}
                padding={{ top: 10, right: 15, bottom: 10, left: 15 }}
              >
                {({ innerWidth, innerHeight }) => (
                  <AttentionPanel
                    title="Decoder: Causal Mask"
                    description="Decoder: causal mask. Each token only sees itself and what came before. Used in GPT for autoregressive generation."
                    weights={maskApplied ? DECODER_WEIGHTS : ENCODER_WEIGHTS}
                    tokens={TOKENS}
                    maskApplied={maskApplied}
                    innerWidth={innerWidth}
                    innerHeight={innerHeight}
                    panelType="decoder"
                  />
                )}
              </SVGContainer>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Mask explanation */}
      <GlassCard className="p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded" style={{ backgroundColor: NLP.key + '40', border: `1px solid ${NLP.key}60` }} />
              <span className="text-[10px] text-text-secondary font-medium">Encoder (BERT-style)</span>
            </div>
            <p className="text-[9px] text-text-tertiary pl-5">
              Bidirectional attention. All tokens see all other tokens. Best for understanding tasks
              (classification, NER, Q&A).
            </p>
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded" style={{ backgroundColor: NLP.query + '40', border: `1px solid ${NLP.query}60` }} />
              <span className="text-[10px] text-text-secondary font-medium">Decoder (GPT-style)</span>
            </div>
            <p className="text-[9px] text-text-tertiary pl-5">
              Causal (autoregressive) attention. Each token only sees preceding tokens. Required for
              text generation to prevent &quot;seeing the future.&quot;
            </p>
          </div>
        </div>
      </GlassCard>

      {/* Interactive mask builder */}
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <p className="text-xs font-medium text-text-secondary">
            Mask Builder
          </p>
          <span className="text-[9px] text-text-tertiary">
            Click cells to toggle mask on/off and see the effect on attention weights
          </span>
        </div>
        <InteractiveMaskBuilder />
      </div>

      {/* Cross-attention (shown for encoder-decoder mode) */}
      <AnimatePresence>
        {showCrossAttention && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.4 }}
            className="overflow-hidden"
          >
            <CrossAttentionViz />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Generation walkthrough */}
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <p className="text-xs font-medium text-text-secondary">
            Generation Walkthrough
          </p>
          <span className="text-[9px] text-text-tertiary">
            See how the attention grid grows during autoregressive generation
          </span>
        </div>

        <Slider
          label="Generation Step"
          value={genStep}
          min={0}
          max={GEN_TOKENS.length - 1}
          step={1}
          onChange={setGenStep}
          formatValue={(v) => `Step ${v}: "${GEN_TOKENS[v]}"`}
          className="max-w-md"
        />

        <GenerationWalkthrough genStep={genStep} />
      </div>
    </div>
  )
}
