import { useMemo } from 'react'
import * as d3 from 'd3'
import { motion } from 'framer-motion'

const NLP = {
  query: '#F472B6',
  key: '#34D399',
  attentionLow: 'rgba(99, 102, 241, 0.05)',
  attentionHigh: '#6366F1',
  tokenHighlight: '#818CF8',
} as const

/**
 * Reusable attention heatmap with arc lines.
 * Extracted for use in both Self-Attention and Multi-Head sections.
 */
export function AttentionHeatmap({
  tokens,
  attentionWeights,
  selectedToken,
  onSelectToken,
  innerWidth,
  innerHeight,
}: {
  tokens: string[]
  attentionWeights: number[][]
  selectedToken: number
  onSelectToken: (idx: number) => void
  innerWidth: number
  innerHeight: number
}) {
  const numTokens = tokens.length
  const weights = attentionWeights[selectedToken]

  // Layout: arcs on top, token pills, then heatmap
  const arcSpace = 140
  const tokenPillsY = arcSpace
  const heatmapY = tokenPillsY + 46

  // Token pill sizing
  const pillWidth = Math.min(70, (innerWidth - 20) / numTokens - 8)
  const pillHeight = 26
  const totalPillWidth = numTokens * (pillWidth + 8) - 8
  const pillStartX = (innerWidth - totalPillWidth) / 2

  // Heatmap dimensions
  const availableSize = Math.min(innerWidth * 0.6, innerHeight - heatmapY - 20)
  const cellSize = availableSize / numTokens
  const matrixSize = cellSize * numTokens
  const matrixX = (innerWidth - matrixSize) / 2

  const heatColorScale = useMemo(
    () =>
      d3
        .scaleLinear<string>()
        .domain([0, 0.2, 0.45])
        .range([NLP.attentionLow, `${NLP.attentionHigh}66`, NLP.attentionHigh])
        .clamp(true),
    []
  )

  return (
    <motion.g
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Arc lines from selected token to all others */}
      <g transform={`translate(0, ${tokenPillsY})`}>
        {weights &&
          weights.map((weight, j) => {
            if (weight < 0.02) return null
            const fromX = pillStartX + selectedToken * (pillWidth + 8) + pillWidth / 2
            const toX = pillStartX + j * (pillWidth + 8) + pillWidth / 2
            const arcHeight = Math.max(30, weight * 120 + 20)

            return (
              <motion.path
                key={`arc-${j}`}
                d={`M ${fromX} 0 C ${fromX} ${-arcHeight}, ${toX} ${-arcHeight}, ${toX} 0`}
                fill="none"
                stroke={NLP.attentionHigh}
                strokeWidth={Math.max(1, weight * 5)}
                strokeLinecap="round"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: Math.max(0.2, weight) }}
                transition={{ duration: 0.5, delay: j * 0.08 }}
              />
            )
          })}

        {/* Weight labels on arcs */}
        {weights &&
          weights.map((weight, j) => {
            if (weight < 0.05) return null
            const fromX = pillStartX + selectedToken * (pillWidth + 8) + pillWidth / 2
            const toX = pillStartX + j * (pillWidth + 8) + pillWidth / 2
            const arcHeight = Math.max(30, weight * 120 + 20)
            const labelX = (fromX + toX) / 2
            const midY = -arcHeight * 0.6

            return (
              <motion.text
                key={`arc-label-${j}`}
                x={labelX}
                y={midY}
                textAnchor="middle"
                fontSize={9}
                fontFamily="monospace"
                fill={NLP.attentionHigh}
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.9 }}
                transition={{ delay: 0.3 + j * 0.08 }}
              >
                {weight.toFixed(2)}
              </motion.text>
            )
          })}

        {/* Token pills */}
        {tokens.map((token, i) => {
          const x = pillStartX + i * (pillWidth + 8)
          const isSelected = selectedToken === i
          return (
            <g key={i} onClick={() => onSelectToken(i)} style={{ cursor: 'pointer' }}>
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

      {/* Heatmap title */}
      <text
        x={innerWidth / 2}
        y={heatmapY - 10}
        textAnchor="middle"
        fontSize={10}
        className="fill-text-tertiary"
      >
        Click a token to see what it attends to. Brighter = stronger attention.
      </text>

      {/* Column labels */}
      {tokens.map((token, j) => (
        <text
          key={`hcol-${j}`}
          x={matrixX + j * cellSize + cellSize / 2}
          y={heatmapY + 10}
          textAnchor="middle"
          fontSize={9}
          fontFamily="monospace"
          fill={NLP.key}
          opacity={0.7}
        >
          {token}
        </text>
      ))}

      {/* Row labels */}
      {tokens.map((token, i) => (
        <g key={`hrow-${i}`} onClick={() => onSelectToken(i)} style={{ cursor: 'pointer' }}>
          <text
            x={matrixX - 8}
            y={heatmapY + 22 + i * cellSize + cellSize / 2 + 3}
            textAnchor="end"
            fontSize={9}
            fontFamily="monospace"
            fill={i === selectedToken ? NLP.query : `${NLP.query}66`}
            fontWeight={i === selectedToken ? 700 : 400}
          >
            {token}
          </text>
        </g>
      ))}

      {/* Heatmap cells */}
      {attentionWeights.map((row, i) =>
        row.map((weight, j) => {
          const cx = matrixX + j * cellSize
          const cy = heatmapY + 22 + i * cellSize
          const isSelectedRow = i === selectedToken

          return (
            <g
              key={`hcell-${i}-${j}`}
              onClick={() => onSelectToken(i)}
              style={{ cursor: 'pointer' }}
            >
              <motion.rect
                x={cx + 1.5}
                y={cy + 1.5}
                width={cellSize - 3}
                height={cellSize - 3}
                rx={5}
                fill={heatColorScale(weight)}
                stroke={isSelectedRow ? `${NLP.query}AA` : 'rgba(255,255,255,0.04)'}
                strokeWidth={isSelectedRow ? 2 : 0.5}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{
                  delay: i * 0.04 + j * 0.04,
                  type: 'spring',
                  stiffness: 400,
                  damping: 25,
                }}
              />
              <motion.text
                x={cx + cellSize / 2}
                y={cy + cellSize / 2 + 4}
                textAnchor="middle"
                fontSize={cellSize > 50 ? 11 : 9}
                fontFamily="monospace"
                fontWeight={isSelectedRow ? 700 : 400}
                fill={weight > 0.25 ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.55)'}
                style={{ pointerEvents: 'none' }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 + i * 0.04 + j * 0.04 }}
              >
                {weight.toFixed(2)}
              </motion.text>
            </g>
          )
        })
      )}

      {/* Axis labels */}
      <text
        x={matrixX + matrixSize / 2}
        y={heatmapY + 6}
        textAnchor="middle"
        fontSize={8}
        fill={NLP.key}
        opacity={0.4}
      >
        Attended To (Keys)
      </text>
      <text
        x={matrixX - 14}
        y={heatmapY + 22 + matrixSize / 2}
        textAnchor="middle"
        fontSize={8}
        fill={NLP.query}
        opacity={0.4}
        transform={`rotate(-90, ${matrixX - 14}, ${heatmapY + 22 + matrixSize / 2})`}
      >
        Attending (Queries)
      </text>
    </motion.g>
  )
}
