import { useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { motion, AnimatePresence } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Button } from '../../../components/ui/Button'
import { Toggle } from '../../../components/ui/Toggle'
import { TransportControls } from '../../../components/viz/TransportControls'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'
import { AttentionHeatmap } from './AttentionHeatmap'
import { computeScores, applyScaling, applySoftmax } from '../../../lib/algorithms/transformers/attention'

// ── NLP Colors ──────────────────────────────────────────────────────────
const NLP = {
  query: '#F472B6',
  key: '#34D399',
  value: '#FBBF24',
  position: '#38BDF8',
  tokenHighlight: '#818CF8',
  attentionLow: 'rgba(99, 102, 241, 0.05)',
  attentionHigh: '#6366F1',
} as const

// ── Tokens ──────────────────────────────────────────────────────────────
const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat'] as const
const NUM_TOKENS = TOKENS.length

// ── Base pre-computed data (8-dim) ──────────────────────────────────────
const BASE_EMBEDDINGS: number[][] = [
  [ 0.12, -0.34,  0.56, -0.78,  0.23,  0.45, -0.67,  0.89],
  [-0.45,  0.67, -0.12,  0.34, -0.56,  0.78, -0.23,  0.11],
  [ 0.78, -0.23,  0.45, -0.11,  0.67, -0.34,  0.56, -0.45],
  [-0.11,  0.45, -0.67,  0.23, -0.89,  0.12, -0.34,  0.56],
  [ 0.34, -0.56,  0.78, -0.23,  0.11,  0.67, -0.45,  0.89],
  [-0.67,  0.89, -0.23,  0.56, -0.11,  0.34, -0.78,  0.12],
]

const BASE_Q: number[][] = [
  [ 0.31, -0.42,  0.18, -0.55,  0.67, -0.23,  0.44, -0.11],
  [-0.28,  0.53, -0.16,  0.42, -0.37,  0.61, -0.19,  0.33],
  [ 0.55, -0.31,  0.47, -0.12,  0.39, -0.56,  0.22, -0.44],
  [-0.15,  0.38, -0.52,  0.27, -0.63,  0.14, -0.41,  0.36],
  [ 0.42, -0.18,  0.61, -0.33,  0.25, -0.47,  0.53, -0.22],
  [-0.37,  0.62, -0.28,  0.48, -0.14,  0.39, -0.55,  0.17],
]

const BASE_K: number[][] = [
  [ 0.25, -0.38,  0.52, -0.17,  0.43, -0.61,  0.33, -0.28],
  [-0.41,  0.56, -0.22,  0.47, -0.33,  0.18, -0.52,  0.39],
  [ 0.63, -0.19,  0.37, -0.55,  0.28, -0.42,  0.14, -0.47],
  [-0.22,  0.44, -0.58,  0.31, -0.47,  0.23, -0.36,  0.52],
  [ 0.38, -0.27,  0.45, -0.62,  0.19, -0.53,  0.41, -0.15],
  [-0.55,  0.33, -0.41,  0.24, -0.58,  0.47, -0.19,  0.63],
]

const BASE_V: number[][] = [
  [ 0.45, -0.22,  0.67, -0.33,  0.11, -0.55,  0.38, -0.17],
  [-0.33,  0.58, -0.14,  0.41, -0.27,  0.63, -0.45,  0.22],
  [ 0.52, -0.37,  0.28, -0.61,  0.44, -0.18,  0.33, -0.56],
  [-0.17,  0.42, -0.53,  0.19, -0.38,  0.55, -0.22,  0.47],
  [ 0.63, -0.28,  0.41, -0.52,  0.33, -0.44,  0.17, -0.38],
  [-0.44,  0.33, -0.61,  0.28, -0.17,  0.52, -0.38,  0.55],
]

// ── Phase definitions ───────────────────────────────────────────────────
const PHASES = [
  { id: 0, label: 'Projections', short: 'Q K V' },
  { id: 1, label: 'Scores', short: 'QK^T' },
  { id: 2, label: 'Heatmap', short: 'Attn' },
  { id: 3, label: 'Output', short: 'Out' },
] as const

// ── Helpers ─────────────────────────────────────────────────────────────

/** Resize vectors to target dimension (slice or extend deterministically) */
function resizeVectors(vectors: number[][], targetDk: number): number[][] {
  return vectors.map((v) => {
    if (targetDk <= v.length) return v.slice(0, targetDk)
    const ext = [...v]
    for (let i = v.length; i < targetDk; i++) {
      ext.push(v[i % v.length] * 0.75 * (i % 2 === 0 ? 1 : -0.8))
    }
    return ext
  })
}

/** Compute output vector for a given token */
function computeOutputVector(
  tokenIdx: number,
  weights: number[][],
  vVectors: number[][]
): number[] {
  const w = weights[tokenIdx]
  const dk = vVectors[0].length
  const output = new Array(dk).fill(0)
  for (let j = 0; j < NUM_TOKENS; j++) {
    for (let d = 0; d < dk; d++) {
      output[d] += w[j] * vVectors[j][d]
    }
  }
  return output
}

// ── Sub-components ──────────────────────────────────────────────────────

// Token pill row (reused across phases)
function TokenPills({
  selectedToken,
  onSelect,
  innerWidth,
}: {
  selectedToken: number
  onSelect: (idx: number) => void
  innerWidth: number
}) {
  const pillWidth = Math.min(70, (innerWidth - 20) / NUM_TOKENS - 8)
  const pillHeight = 26
  const totalWidth = NUM_TOKENS * (pillWidth + 8) - 8
  const startX = (innerWidth - totalWidth) / 2

  return (
    <g>
      {TOKENS.map((token, i) => {
        const x = startX + i * (pillWidth + 8)
        const isSelected = selectedToken === i
        return (
          <g key={i} onClick={() => onSelect(i)} style={{ cursor: 'pointer' }}>
            <motion.rect
              x={x}
              y={0}
              width={pillWidth}
              height={pillHeight}
              rx={13}
              fill={isSelected ? `${NLP.tokenHighlight}25` : 'rgba(255,255,255,0.04)'}
              stroke={isSelected ? NLP.tokenHighlight : 'rgba(255,255,255,0.12)'}
              strokeWidth={isSelected ? 2 : 1}
              animate={{
                fill: isSelected ? `${NLP.tokenHighlight}25` : 'rgba(255,255,255,0.04)',
                stroke: isSelected ? NLP.tokenHighlight : 'rgba(255,255,255,0.12)',
              }}
              transition={{ duration: 0.2 }}
            />
            <text
              x={x + pillWidth / 2}
              y={pillHeight / 2 + 4}
              textAnchor="middle"
              fontSize={11}
              fontFamily="monospace"
              fill={isSelected ? NLP.tokenHighlight : 'rgba(255,255,255,0.7)'}
              fontWeight={isSelected ? 600 : 400}
              style={{ pointerEvents: 'none' }}
            >
              {token}
            </text>
          </g>
        )
      })}
    </g>
  )
}

// Mini bar chart for vectors
function VectorBars({
  values,
  x,
  y,
  width,
  height,
  color,
  label,
  maxVal = 1,
}: {
  values: number[]
  x: number
  y: number
  width: number
  height: number
  color: string
  label?: string
  maxVal?: number
}) {
  const barWidth = width / values.length - 1
  const yCenter = y + height / 2
  const halfH = height / 2

  return (
    <g>
      {label && (
        <text
          x={x + width / 2}
          y={y - 4}
          textAnchor="middle"
          fontSize={8}
          fontFamily="monospace"
          fill={color}
          opacity={0.8}
        >
          {label}
        </text>
      )}
      <line
        x1={x}
        x2={x + width}
        y1={yCenter}
        y2={yCenter}
        stroke="rgba(255,255,255,0.08)"
        strokeWidth={0.5}
      />
      {values.map((v, i) => {
        const barH = (Math.abs(v) / maxVal) * halfH
        const barY = v >= 0 ? yCenter - barH : yCenter
        return (
          <motion.rect
            key={i}
            x={x + i * (barWidth + 1)}
            width={barWidth}
            rx={1}
            fill={color}
            opacity={0.7}
            initial={{ y: yCenter, height: 0 }}
            animate={{ y: barY, height: barH }}
            transition={{ duration: 0.4, delay: i * 0.03 }}
          />
        )
      })}
    </g>
  )
}

// ── Phase 1: Projections ────────────────────────────────────────────────
function ProjectionsPhase({
  innerWidth,
  selectedToken,
  onSelectToken,
  embeddings,
  qVectors,
  kVectors,
  vVectors,
}: {
  innerWidth: number
  selectedToken: number
  onSelectToken: (i: number) => void
  embeddings: number[][]
  qVectors: number[][]
  kVectors: number[][]
  vVectors: number[][]
}) {
  const tokenPillsY = 10
  const contentY = tokenPillsY + 50
  const colWidth = innerWidth / 5
  const embX = 0
  const matrixX = colWidth * 1.3
  const qkvStartX = colWidth * 2.6
  const vecHeight = 28
  const vecWidth = colWidth * 0.8

  return (
    <motion.g
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
    >
      <TokenPills selectedToken={selectedToken} onSelect={onSelectToken} innerWidth={innerWidth} />

      <text x={innerWidth / 2} y={contentY - 8} textAnchor="middle" fontSize={10} className="fill-text-tertiary">
        Every token gets its own Query, Key, and Value via linear projections.
      </text>

      {/* Embeddings column */}
      <text x={embX + vecWidth / 2} y={contentY + 10} textAnchor="middle" fontSize={9} fontWeight={600} fill="rgba(255,255,255,0.6)">
        Embeddings
      </text>
      {TOKENS.map((token, i) => {
        const yPos = contentY + 22 + i * (vecHeight + 8)
        const isSelected = i === selectedToken
        return (
          <g key={`emb-${i}`} onClick={() => onSelectToken(i)} style={{ cursor: 'pointer' }}>
            <text x={embX - 4} y={yPos + vecHeight / 2 + 3} textAnchor="end" fontSize={8} fontFamily="monospace" fill={isSelected ? NLP.tokenHighlight : 'rgba(255,255,255,0.4)'}>
              {token}
            </text>
            <VectorBars values={embeddings[i]} x={embX} y={yPos} width={vecWidth} height={vecHeight} color={isSelected ? 'rgba(255,255,255,0.8)' : 'rgba(255,255,255,0.3)'} />
          </g>
        )
      })}

      {/* Weight matrices */}
      {([
        { label: 'W_Q', color: NLP.query, yOff: 0 },
        { label: 'W_K', color: NLP.key, yOff: 1 },
        { label: 'W_V', color: NLP.value, yOff: 2 },
      ] as const).map(({ label, color, yOff }) => {
        const matW = colWidth * 0.6
        const matH = 50
        const matY = contentY + 40 + yOff * (matH + 24)
        return (
          <g key={label}>
            <motion.rect x={matrixX} y={matY} width={matW} height={matH} rx={6} fill="rgba(255,255,255,0.02)" stroke={color} strokeWidth={1.5} strokeDasharray="4 2" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 + yOff * 0.15 }} />
            <text x={matrixX + matW / 2} y={matY + matH / 2 + 4} textAnchor="middle" fontSize={11} fontWeight={600} fontFamily="monospace" fill={color}>
              {label}
            </text>
          </g>
        )
      })}

      {/* Arrow from embeddings to matrices */}
      <motion.line x1={embX + vecWidth + 8} x2={matrixX - 8} y1={contentY + 22 + NUM_TOKENS * (vecHeight + 8) / 2} y2={contentY + 22 + NUM_TOKENS * (vecHeight + 8) / 2} stroke="rgba(255,255,255,0.15)" strokeWidth={1} markerEnd="url(#arrowhead)" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.3, duration: 0.4 }} />

      {/* Q, K, V output columns */}
      {([
        { vectors: qVectors, label: 'Q', color: NLP.query, col: 0 },
        { vectors: kVectors, label: 'K', color: NLP.key, col: 1 },
        { vectors: vVectors, label: 'V', color: NLP.value, col: 2 },
      ] as const).map(({ vectors, label, color, col }) => {
        const colX = qkvStartX + col * (vecWidth + 16)
        return (
          <g key={label}>
            <text x={colX + vecWidth / 2} y={contentY + 10} textAnchor="middle" fontSize={10} fontWeight={600} fill={color}>
              {label}
            </text>
            {TOKENS.map((_token, i) => {
              const yPos = contentY + 22 + i * (vecHeight + 8)
              const isSelected = i === selectedToken
              return (
                <motion.g key={`${label}-${i}`} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 + col * 0.15 + i * 0.05 }}>
                  <VectorBars values={vectors[i]} x={colX} y={yPos} width={vecWidth} height={vecHeight} color={isSelected ? color : `${color}55`} />
                </motion.g>
              )
            })}
          </g>
        )
      })}

      <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="rgba(255,255,255,0.3)" />
        </marker>
      </defs>
    </motion.g>
  )
}

// ── Phase 2: Scores ─────────────────────────────────────────────────────
function ScoresPhase({
  innerWidth,
  innerHeight,
  selectedToken,
  onSelectToken,
  showSoftmax,
  showScaling,
  dk,
  rawScores,
  scaledScores,
  attentionWeights,
  unscaledWeights,
}: {
  innerWidth: number
  innerHeight: number
  selectedToken: number
  onSelectToken: (i: number) => void
  showSoftmax: boolean
  showScaling: boolean
  dk: number
  rawScores: number[][]
  scaledScores: number[][]
  attentionWeights: number[][]
  unscaledWeights: number[][]
}) {
  const tokenPillsY = 10
  const contentY = tokenPillsY + 50
  const sqrtDk = Math.sqrt(dk)

  // Choose which matrix to display
  const displayMatrix = useMemo(() => {
    if (showSoftmax) {
      return showScaling ? attentionWeights : unscaledWeights
    }
    return showScaling ? scaledScores : rawScores
  }, [showSoftmax, showScaling, attentionWeights, unscaledWeights, scaledScores, rawScores])

  const cellColorScale = useMemo(() => {
    if (showSoftmax) {
      return d3.scaleLinear<string>().domain([0, 0.5]).range([NLP.attentionLow, NLP.attentionHigh]).clamp(true)
    }
    return d3.scaleLinear<string>().domain([-0.5, 0, 1]).range(['rgba(59,130,246,0.3)', 'rgba(99,102,241,0.05)', NLP.attentionHigh]).clamp(true)
  }, [showSoftmax])

  const matrixSize = Math.min(innerWidth * 0.55, innerHeight - contentY - 40)
  const cellSize = matrixSize / NUM_TOKENS
  const matrixX = (innerWidth - matrixSize) / 2
  const matrixY = contentY + 30

  return (
    <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
      <TokenPills selectedToken={selectedToken} onSelect={onSelectToken} innerWidth={innerWidth} />

      <text x={innerWidth / 2} y={contentY - 4} textAnchor="middle" fontSize={10} className="fill-text-tertiary">
        {showSoftmax
          ? showScaling
            ? 'After softmax: each row becomes a probability distribution summing to 1.'
            : 'Softmax without scaling: notice how the distribution becomes very peaked.'
          : showScaling
            ? `Raw dot products scaled by 1/\u221A${dk} = 1/${sqrtDk.toFixed(2)}`
            : 'Raw dot products (unscaled). Larger d_k means larger values.'}
      </text>

      {/* Column labels (Keys) */}
      {TOKENS.map((token, j) => (
        <text key={`col-${j}`} x={matrixX + j * cellSize + cellSize / 2} y={matrixY - 8} textAnchor="middle" fontSize={9} fontFamily="monospace" fill={NLP.key} opacity={0.8}>
          {token}
        </text>
      ))}

      <text x={matrixX + matrixSize / 2} y={matrixY - 22} textAnchor="middle" fontSize={8} fill={NLP.key} opacity={0.5}>
        Keys (K)
      </text>

      {/* Row labels (Queries) */}
      {TOKENS.map((token, i) => (
        <text key={`row-${i}`} x={matrixX - 8} y={matrixY + i * cellSize + cellSize / 2 + 3} textAnchor="end" fontSize={9} fontFamily="monospace" fill={i === selectedToken ? NLP.query : `${NLP.query}88`} fontWeight={i === selectedToken ? 600 : 400}>
          {token}
        </text>
      ))}

      <text x={matrixX - 12} y={matrixY + matrixSize / 2} textAnchor="middle" fontSize={8} fill={NLP.query} opacity={0.5} transform={`rotate(-90, ${matrixX - 12}, ${matrixY + matrixSize / 2})`}>
        Queries (Q)
      </text>

      {/* Score matrix cells */}
      {displayMatrix.map((row, i) =>
        row.map((val, j) => {
          const cx = matrixX + j * cellSize
          const cy = matrixY + i * cellSize
          const isSelectedRow = i === selectedToken
          return (
            <g key={`cell-${i}-${j}`}>
              <motion.rect x={cx + 1} y={cy + 1} width={cellSize - 2} height={cellSize - 2} rx={4} fill={cellColorScale(val)} stroke={isSelectedRow ? NLP.query : 'rgba(255,255,255,0.06)'} strokeWidth={isSelectedRow ? 1.5 : 0.5} initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: i * 0.05 + j * 0.03 }} />
              <motion.text x={cx + cellSize / 2} y={cy + cellSize / 2 + 3} textAnchor="middle" fontSize={cellSize > 45 ? 10 : 8} fontFamily="monospace" fill={isSelectedRow ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.6)'} fontWeight={isSelectedRow ? 600 : 400} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 + i * 0.05 + j * 0.03 }} style={{ pointerEvents: 'none' }}>
                {val.toFixed(2)}
              </motion.text>
            </g>
          )
        })
      )}

      {/* sqrt(d_k) annotation */}
      {showScaling && !showSoftmax && (
        <motion.g initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}>
          <text x={matrixX + matrixSize + 20} y={matrixY + matrixSize / 2 - 10} fontSize={10} fill="rgba(255,255,255,0.5)" fontFamily="monospace">
            score = Q{'\u00B7'}K^T
          </text>
          <text x={matrixX + matrixSize + 20} y={matrixY + matrixSize / 2 + 6} fontSize={10} fill="rgba(255,255,255,0.5)" fontFamily="monospace">
            / {'\u221A'}{dk} = {sqrtDk.toFixed(2)}
          </text>
        </motion.g>
      )}

      {/* Warning when scaling is off */}
      {!showScaling && showSoftmax && (
        <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}>
          <text x={matrixX + matrixSize + 20} y={matrixY + matrixSize / 2 - 6} fontSize={9} fill="#F87171" fontFamily="monospace">
            No scaling!
          </text>
          <text x={matrixX + matrixSize + 20} y={matrixY + matrixSize / 2 + 8} fontSize={8} fill="rgba(255,255,255,0.4)">
            Softmax saturates
          </text>
        </motion.g>
      )}
    </motion.g>
  )
}

// ── Phase 4: Output ─────────────────────────────────────────────────────
function OutputPhase({
  innerWidth,
  selectedToken,
  onSelectToken,
  attentionWeights,
  vVectors,
}: {
  innerWidth: number
  selectedToken: number
  onSelectToken: (i: number) => void
  attentionWeights: number[][]
  vVectors: number[][]
}) {
  const tokenPillsY = 10
  const contentY = tokenPillsY + 50

  const weights = attentionWeights[selectedToken]
  const outputVec = useMemo(
    () => computeOutputVector(selectedToken, attentionWeights, vVectors),
    [selectedToken, attentionWeights, vVectors]
  )

  const vecWidth = Math.min(120, innerWidth * 0.12)
  const vecHeight = 36
  const gap = 14
  const totalVecH = NUM_TOKENS * (vecHeight + gap) - gap
  const startY = contentY + 30
  const leftX = innerWidth * 0.05
  const rightX = innerWidth * 0.7
  const scaleX = innerWidth * 0.35

  return (
    <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
      <TokenPills selectedToken={selectedToken} onSelect={onSelectToken} innerWidth={innerWidth} />

      <text x={innerWidth / 2} y={contentY - 4} textAnchor="middle" fontSize={10} className="fill-text-tertiary">
        Output = weighted average of all Value vectors, using attention weights as mixing coefficients.
      </text>

      <text x={leftX + vecWidth / 2} y={startY - 12} textAnchor="middle" fontSize={9} fontWeight={600} fill={NLP.value}>
        Value Vectors
      </text>
      <text x={scaleX + vecWidth / 2} y={startY - 12} textAnchor="middle" fontSize={9} fontWeight={600} fill="rgba(255,255,255,0.5)">
        Weight x Value
      </text>

      {TOKENS.map((token, j) => {
        const yPos = startY + j * (vecHeight + gap)
        const weight = weights[j]
        const scaledVec = vVectors[j].map((v) => v * weight)

        return (
          <motion.g key={`val-${j}`} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: j * 0.08 }}>
            <text x={leftX - 6} y={yPos + vecHeight / 2 + 3} textAnchor="end" fontSize={8} fontFamily="monospace" fill="rgba(255,255,255,0.5)">
              {token}
            </text>
            <VectorBars values={vVectors[j]} x={leftX} y={yPos} width={vecWidth} height={vecHeight} color={NLP.value} />
            <text x={leftX + vecWidth + 12} y={yPos + vecHeight / 2 + 4} textAnchor="middle" fontSize={9} fontFamily="monospace" fill="rgba(255,255,255,0.4)">
              {'\u00D7'}
            </text>
            <motion.rect x={leftX + vecWidth + 22} y={yPos + vecHeight / 2 - 9} width={38} height={18} rx={9} fill={`${NLP.attentionHigh}${Math.round(weight * 255).toString(16).padStart(2, '0')}`} stroke={NLP.attentionHigh} strokeWidth={0.5} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.2 + j * 0.08, type: 'spring' }} />
            <text x={leftX + vecWidth + 41} y={yPos + vecHeight / 2 + 3} textAnchor="middle" fontSize={8} fontFamily="monospace" fontWeight={600} fill="rgba(255,255,255,0.9)" style={{ pointerEvents: 'none' }}>
              {weight.toFixed(2)}
            </text>
            <motion.line x1={leftX + vecWidth + 64} x2={scaleX - 4} y1={yPos + vecHeight / 2} y2={yPos + vecHeight / 2} stroke="rgba(255,255,255,0.1)" strokeWidth={1} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.3 + j * 0.08 }} />
            <VectorBars values={scaledVec} x={scaleX} y={yPos} width={vecWidth} height={vecHeight} color={`${NLP.value}${Math.round(Math.max(0.25, weight) * 255).toString(16).padStart(2, '0')}`} />
          </motion.g>
        )
      })}

      <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.8 }}>
        <text x={scaleX + vecWidth + 20} y={startY + totalVecH / 2 + 4} textAnchor="middle" fontSize={18} fill="rgba(255,255,255,0.3)">
          {'\u2211'}
        </text>
        <motion.line x1={scaleX + vecWidth + 35} x2={rightX - 8} y1={startY + totalVecH / 2} y2={startY + totalVecH / 2} stroke="rgba(255,255,255,0.2)" strokeWidth={1.5} strokeDasharray="4 3" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 1.0, duration: 0.5 }} />
      </motion.g>

      <motion.g initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 1.2, type: 'spring' }}>
        <rect x={rightX - 8} y={startY + totalVecH / 2 - 50} width={vecWidth + 16} height={100} rx={10} fill="rgba(99,102,241,0.08)" stroke={NLP.attentionHigh} strokeWidth={1} strokeDasharray="4 2" />
        <text x={rightX + vecWidth / 2} y={startY + totalVecH / 2 - 36} textAnchor="middle" fontSize={9} fontWeight={600} fill={NLP.attentionHigh}>
          Output
        </text>
        <text x={rightX + vecWidth / 2} y={startY + totalVecH / 2 - 24} textAnchor="middle" fontSize={7} fontFamily="monospace" fill="rgba(255,255,255,0.4)">
          for &quot;{TOKENS[selectedToken]}&quot;
        </text>
        <VectorBars values={outputVec} x={rightX} y={startY + totalVecH / 2 - 18} width={vecWidth} height={36} color={NLP.attentionHigh} />
      </motion.g>
    </motion.g>
  )
}

// ── Main Component ──────────────────────────────────────────────────────
export function SelfAttentionWalkthrough() {
  const [selectedToken, setSelectedToken] = useState(2) // "sat"
  const [showSoftmax, setShowSoftmax] = useState(true)
  const [showScaling, setShowScaling] = useState(true)
  const [dk, setDk] = useState(8)

  // Transport controls via useAlgorithmPlayer
  const phaseSnapshots = useMemo(() => [0, 1, 2, 3], [])
  const player = useAlgorithmPlayer({ snapshots: phaseSnapshots, baseFps: 0.4 })
  const phase = player.currentSnapshot

  const handleSelectToken = useCallback((idx: number) => {
    setSelectedToken(idx)
  }, [])

  // Dynamically compute vectors and scores based on d_k
  const embeddings = useMemo(() => resizeVectors(BASE_EMBEDDINGS, dk), [dk])
  const qVectors = useMemo(() => resizeVectors(BASE_Q, dk), [dk])
  const kVectors = useMemo(() => resizeVectors(BASE_K, dk), [dk])
  const vVectors = useMemo(() => resizeVectors(BASE_V, dk), [dk])

  const rawScores = useMemo(() => computeScores(qVectors, kVectors), [qVectors, kVectors])
  const scaledScores = useMemo(() => applyScaling(rawScores, dk), [rawScores, dk])
  const attentionWeights = useMemo(() => applySoftmax(scaledScores), [scaledScores])
  const unscaledWeights = useMemo(() => applySoftmax(rawScores), [rawScores])

  // Dynamic SVG height based on phase
  const aspectRatios: Record<number, number> = { 0: 16 / 11, 1: 16 / 10, 2: 16 / 13, 3: 16 / 11 }
  const minHeights: Record<number, number> = { 0: 400, 1: 380, 2: 480, 3: 400 }

  return (
    <div className="space-y-4">
      {/* Phase selector tabs */}
      <div className="flex flex-wrap gap-2 justify-center">
        {PHASES.map((p) => (
          <Button
            key={p.id}
            variant="secondary"
            size="sm"
            active={phase === p.id}
            onClick={() => player.goToStep(p.id)}
          >
            <span className="hidden sm:inline">{p.label}</span>
            <span className="sm:hidden">{p.short}</span>
          </Button>
        ))}
      </div>

      {/* Main visualization */}
      <GlassCard className="p-4 lg:p-6">
        <SVGContainer
          aspectRatio={aspectRatios[phase]}
          minHeight={minHeights[phase]}
          maxHeight={650}
          padding={{ top: 20, right: 30, bottom: 20, left: 60 }}
        >
          {({ innerWidth, innerHeight }) => (
            <AnimatePresence mode="wait">
              {phase === 0 && (
                <ProjectionsPhase
                  key="projections"
                  innerWidth={innerWidth}
                  selectedToken={selectedToken}
                  onSelectToken={handleSelectToken}
                  embeddings={embeddings}
                  qVectors={qVectors}
                  kVectors={kVectors}
                  vVectors={vVectors}
                />
              )}
              {phase === 1 && (
                <ScoresPhase
                  key="scores"
                  innerWidth={innerWidth}
                  innerHeight={innerHeight}
                  selectedToken={selectedToken}
                  onSelectToken={handleSelectToken}
                  showSoftmax={showSoftmax}
                  showScaling={showScaling}
                  dk={dk}
                  rawScores={rawScores}
                  scaledScores={scaledScores}
                  attentionWeights={attentionWeights}
                  unscaledWeights={unscaledWeights}
                />
              )}
              {phase === 2 && (
                <AttentionHeatmap
                  key="heatmap"
                  tokens={[...TOKENS]}
                  attentionWeights={showScaling ? attentionWeights : unscaledWeights}
                  selectedToken={selectedToken}
                  onSelectToken={handleSelectToken}
                  innerWidth={innerWidth}
                  innerHeight={innerHeight}
                />
              )}
              {phase === 3 && (
                <OutputPhase
                  key="output"
                  innerWidth={innerWidth}
                  selectedToken={selectedToken}
                  onSelectToken={handleSelectToken}
                  attentionWeights={showScaling ? attentionWeights : unscaledWeights}
                  vVectors={vVectors}
                />
              )}
            </AnimatePresence>
          )}
        </SVGContainer>
      </GlassCard>

      {/* Transport controls */}
      <TransportControls
        isPlaying={player.isPlaying}
        isAtStart={player.isAtStart}
        isAtEnd={player.isAtEnd}
        currentStep={player.currentStep}
        totalSteps={player.totalSteps}
        speed={player.speed}
        onPlay={player.play}
        onPause={player.pause}
        onTogglePlay={player.togglePlay}
        onStepForward={player.stepForward}
        onStepBack={player.stepBack}
        onReset={player.reset}
        onSetSpeed={player.setSpeed}
      />

      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap gap-6 items-center">
          {/* Phase info */}
          <div className="flex-1 min-w-[200px]">
            <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-1">
              Phase {phase + 1} of 4
            </p>
            <p className="text-sm text-text-secondary">
              {phase === 0 && 'Each token embedding is projected into Query, Key, and Value vectors using learned weight matrices.'}
              {phase === 1 && `Dot products between Queries and Keys produce the raw attention scores. Scaling by 1/\u221A${dk} prevents large values.`}
              {phase === 2 && 'The attention heatmap shows how much each token attends to every other token. Click a token to explore its attention pattern.'}
              {phase === 3 && 'The final output for each token is a weighted sum of all Value vectors, mixed according to the attention weights.'}
            </p>
          </div>

          {/* Token selector */}
          <div>
            <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-2">Selected Token</p>
            <div className="flex gap-1">
              {TOKENS.map((token, i) => (
                <button
                  key={i}
                  onClick={() => handleSelectToken(i)}
                  className={`px-2 py-1 rounded text-xs font-mono transition-all ${
                    selectedToken === i
                      ? 'bg-[#818CF8]/20 text-[#818CF8] border border-[#818CF8]/40'
                      : 'text-text-tertiary hover:text-text-secondary border border-transparent'
                  }`}
                >
                  {token}
                </button>
              ))}
            </div>
          </div>

          {/* d_k selector */}
          <div>
            <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-2">d_k</p>
            <div className="flex gap-1">
              {[4, 8, 16].map((val) => (
                <button
                  key={val}
                  onClick={() => setDk(val)}
                  className={`px-2 py-1 rounded text-xs font-mono transition-all ${
                    dk === val
                      ? 'bg-accent/20 text-accent border border-accent/40'
                      : 'text-text-tertiary hover:text-text-secondary border border-transparent'
                  }`}
                >
                  {val}
                </button>
              ))}
            </div>
          </div>

          {/* Toggles */}
          <div className="flex flex-col gap-2">
            <Toggle label="Show softmax" checked={showSoftmax} onChange={setShowSoftmax} />
            <Toggle label={`\u221Ad_k scaling`} checked={showScaling} onChange={setShowScaling} />
          </div>
        </div>
      </GlassCard>
    </div>
  )
}
