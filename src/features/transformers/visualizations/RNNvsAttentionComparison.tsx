import { useState, useEffect, useCallback } from 'react'
import * as d3 from 'd3'
import { motion, AnimatePresence } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Button } from '../../../components/ui/Button'

// ── NLP Colors ────────────────────────────────────────────────────────
const NLP = {
  tokenHighlight: '#818CF8',
  attentionLow: 'rgba(99, 102, 241, 0.05)',
  attentionHigh: '#6366F1',
}

// ── Sentence & Attention Data ─────────────────────────────────────────
const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired']

// Realistic pre-defined attention weights for each token attending to all others
// Rows = source (attending token), Cols = target (attended-to token)
// Key pattern: "it" (idx 7) attends strongly to "cat" (idx 1) - coreference resolution
const ATTENTION_WEIGHTS: number[][] = [
  // The   cat   sat    on   the   mat   bec    it   was  tired
  [0.08, 0.25, 0.20, 0.08, 0.12, 0.10, 0.05, 0.04, 0.04, 0.04], // The →
  [0.12, 0.08, 0.30, 0.08, 0.08, 0.15, 0.05, 0.05, 0.04, 0.05], // cat →
  [0.08, 0.35, 0.06, 0.12, 0.04, 0.22, 0.04, 0.03, 0.03, 0.03], // sat →
  [0.06, 0.08, 0.15, 0.05, 0.15, 0.35, 0.05, 0.04, 0.04, 0.03], // on →
  [0.20, 0.08, 0.08, 0.15, 0.05, 0.28, 0.05, 0.04, 0.04, 0.03], // the →
  [0.08, 0.12, 0.25, 0.20, 0.12, 0.05, 0.06, 0.04, 0.04, 0.04], // mat →
  [0.05, 0.10, 0.15, 0.05, 0.05, 0.10, 0.05, 0.15, 0.15, 0.15], // because →
  [0.06, 0.38, 0.10, 0.03, 0.04, 0.08, 0.12, 0.04, 0.08, 0.07], // it → (strong to "cat")
  [0.04, 0.08, 0.06, 0.03, 0.04, 0.05, 0.10, 0.25, 0.05, 0.30], // was →
  [0.04, 0.15, 0.05, 0.03, 0.03, 0.05, 0.08, 0.20, 0.25, 0.12], // tired →
]

// Hidden state simulation: each step accumulates info but early signal fades
// With 10 tokens, the decay is much more dramatic by the time we reach "it"
function generateHiddenStates(step: number, dims: number): number[] {
  const states: number[] = []
  for (let d = 0; d < dims; d++) {
    // Early dimensions fade as more tokens are processed
    const baseSignal = Math.sin((d + 1) * 0.7 + step * 0.4) * 0.8
    // Decay: info from early tokens degrades more aggressively over 10 steps
    const decay = Math.max(0.05, 1 - step * 0.1)
    // Recent tokens dominate
    const recentBoost = Math.sin((d + step) * 1.3) * 0.5 * (step / TOKENS.length)
    states.push(baseSignal * decay + recentBoost)
  }
  return states
}

const HIDDEN_DIMS = 8

// ── RNN Panel ─────────────────────────────────────────────────────────

function RNNPanel({
  innerWidth,
  innerHeight,
  currentStep,
}: {
  innerWidth: number
  innerHeight: number
  currentStep: number
}) {
  const panelW = innerWidth
  const panelH = innerHeight

  const tokenY = 30
  const tokenSpacing = Math.min(panelW / (TOKENS.length + 1), 55)
  const tokensStartX = (panelW - (TOKENS.length - 1) * tokenSpacing) / 2
  const pillWidth = Math.min(tokenSpacing - 4, 44)

  // Hidden state bar area
  const barAreaY = tokenY + 60
  const barAreaH = panelH - barAreaY - 50
  const barWidth = Math.min((panelW - 40) / HIDDEN_DIMS - 2, 16)
  const barsStartX = (panelW - HIDDEN_DIMS * (barWidth + 2)) / 2

  const hiddenStates = generateHiddenStates(currentStep, HIDDEN_DIMS)

  // Color scale for bar fill based on strength
  const barColorScale = d3
    .scaleLinear<string>()
    .domain([-1, 0, 1])
    .range(['#F87171', 'rgba(255,255,255,0.1)', NLP.tokenHighlight])
    .clamp(true)

  return (
    <g>
      {/* Panel title */}
      <text
        x={panelW / 2}
        y={12}
        textAnchor="middle"
        className="text-[11px] font-medium"
        fill="#E4E4E7"
      >
        RNN (Sequential)
      </text>

      {/* Token pills */}
      {TOKENS.map((token, i) => {
        const x = tokensStartX + i * tokenSpacing
        const isProcessed = i < currentStep
        const isCurrent = i === currentStep
        const isPending = i > currentStep

        return (
          <g key={`rnn-token-${i}`}>
            <motion.rect
              x={x - pillWidth / 2}
              y={tokenY - 12}
              width={pillWidth}
              height={24}
              rx={12}
              fill={
                isCurrent
                  ? NLP.tokenHighlight
                  : isProcessed
                    ? 'rgba(129, 140, 248, 0.25)'
                    : 'rgba(255, 255, 255, 0.06)'
              }
              stroke={
                isCurrent
                  ? NLP.tokenHighlight
                  : isProcessed
                    ? 'rgba(129, 140, 248, 0.4)'
                    : 'rgba(255, 255, 255, 0.1)'
              }
              strokeWidth={isCurrent ? 2 : 1}
              animate={{
                fill: isCurrent
                  ? NLP.tokenHighlight
                  : isProcessed
                    ? 'rgba(129, 140, 248, 0.25)'
                    : 'rgba(255, 255, 255, 0.06)',
              }}
              transition={{ duration: 0.3 }}
            />
            <text
              x={x}
              y={tokenY - 1}
              textAnchor="middle"
              dominantBaseline="central"
              className="text-[9px] font-mono"
              fill={isPending ? '#71717A' : '#E4E4E7'}
            >
              {token}
            </text>

            {/* Arrow from token down to hidden state area */}
            {isCurrent && (
              <motion.line
                x1={x}
                y1={tokenY + 14}
                x2={panelW / 2}
                y2={barAreaY - 8}
                stroke={NLP.tokenHighlight}
                strokeWidth={1.5}
                strokeOpacity={0.5}
                strokeDasharray="4,3"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ duration: 0.4 }}
              />
            )}

            {/* Sequential arrows between tokens */}
            {i < TOKENS.length - 1 && (
              <line
                x1={x + pillWidth / 2 + 2}
                y1={tokenY}
                x2={tokensStartX + (i + 1) * tokenSpacing - pillWidth / 2 - 2}
                y2={tokenY}
                stroke={
                  isProcessed
                    ? 'rgba(129, 140, 248, 0.3)'
                    : 'rgba(255, 255, 255, 0.08)'
                }
                strokeWidth={1}
                markerEnd={
                  isProcessed
                    ? 'url(#rnn-seq-arrow-active)'
                    : 'url(#rnn-seq-arrow)'
                }
              />
            )}
          </g>
        )
      })}

      {/* Hidden state label */}
      <text
        x={panelW / 2}
        y={barAreaY - 14}
        textAnchor="middle"
        className="text-[9px] font-mono"
        fill="#A1A1AA"
      >
        Hidden State h_{currentStep}
      </text>

      {/* Hidden state bar chart */}
      {hiddenStates.map((val, d) => {
        const barX = barsStartX + d * (barWidth + 2)
        const normalizedH = Math.abs(val) * barAreaH * 0.7
        const barY = barAreaY + barAreaH / 2

        return (
          <g key={`bar-${d}`}>
            {/* Background bar */}
            <rect
              x={barX}
              y={barAreaY}
              width={barWidth}
              height={barAreaH}
              rx={2}
              fill="rgba(255, 255, 255, 0.03)"
            />
            {/* Value bar */}
            <motion.rect
              x={barX}
              y={val >= 0 ? barY - normalizedH : barY}
              width={barWidth}
              height={normalizedH}
              rx={2}
              fill={barColorScale(val)}
              fillOpacity={0.7}
              animate={{
                y: val >= 0 ? barY - normalizedH : barY,
                height: normalizedH,
                fill: barColorScale(val),
              }}
              transition={{ duration: 0.4, ease: 'easeOut' }}
            />
            {/* Zero line */}
            <line
              x1={barX}
              x2={barX + barWidth}
              y1={barY}
              y2={barY}
              stroke="rgba(255,255,255,0.15)"
              strokeWidth={0.5}
            />
          </g>
        )
      })}

      {/* Info decay indicator */}
      <AnimatePresence>
        {currentStep >= 4 && (
          <motion.g
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.rect
              x={10}
              y={panelH - 32}
              width={panelW - 20}
              height={24}
              rx={6}
              fill="rgba(248, 113, 113, 0.1)"
              stroke="rgba(248, 113, 113, 0.3)"
              strokeWidth={1}
            />
            <motion.text
              x={panelW / 2}
              y={panelH - 18}
              textAnchor="middle"
              className="text-[9px]"
              fill="#F87171"
            >
              {currentStep >= 7
                ? '"cat" was 7 words ago... signal mostly lost'
                : 'Early token info fading... signal compressed'}
            </motion.text>
          </motion.g>
        )}
      </AnimatePresence>

      {/* Bottom label */}
      {currentStep <= 2 && (
        <text
          x={panelW / 2}
          y={panelH - 8}
          textAnchor="middle"
          className="text-[9px]"
          fill="#71717A"
        >
          Sequential processing. Information from early words fades.
        </text>
      )}
    </g>
  )
}

// ── Transformer Panel ─────────────────────────────────────────────────

function TransformerPanel({
  innerWidth,
  innerHeight,
  selectedToken,
  onSelectToken,
}: {
  innerWidth: number
  innerHeight: number
  selectedToken: number
  onSelectToken: (idx: number) => void
}) {
  const panelW = innerWidth
  const panelH = innerHeight

  const tokenY = panelH * 0.5
  const tokenSpacing = Math.min(panelW / (TOKENS.length + 1), 55)
  const tokensStartX = (panelW - (TOKENS.length - 1) * tokenSpacing) / 2
  const pillWidth = Math.min(tokenSpacing - 4, 44)

  // Attention color scale
  const attentionColorScale = d3
    .scaleLinear<string>()
    .domain([0, 0.4])
    .range([NLP.attentionLow, NLP.attentionHigh])
    .clamp(true)

  const attentionWidthScale = d3
    .scaleLinear()
    .domain([0, 0.4])
    .range([0.5, 4])
    .clamp(true)

  const weights = ATTENTION_WEIGHTS[selectedToken]

  return (
    <g>
      {/* Panel title */}
      <text
        x={panelW / 2}
        y={12}
        textAnchor="middle"
        className="text-[11px] font-medium"
        fill="#E4E4E7"
      >
        Transformer (Parallel)
      </text>

      {/* Attention arcs */}
      {TOKENS.map((_, targetIdx) => {
        if (targetIdx === selectedToken) return null
        const weight = weights[targetIdx]

        const srcX = tokensStartX + selectedToken * tokenSpacing
        const tgtX = tokensStartX + targetIdx * tokenSpacing
        const midX = (srcX + tgtX) / 2
        const dist = Math.abs(targetIdx - selectedToken)
        const arcHeight = 15 + dist * 15 // Scaled for 10 tokens
        const arcY = tokenY - 18 - arcHeight

        const pathD = `M ${srcX} ${tokenY - 14} Q ${midX} ${arcY} ${tgtX} ${tokenY - 14}`

        return (
          <motion.path
            key={`arc-${selectedToken}-${targetIdx}`}
            d={pathD}
            fill="none"
            stroke={attentionColorScale(weight)}
            strokeWidth={attentionWidthScale(weight)}
            strokeOpacity={0.5 + weight}
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 0.5, delay: targetIdx * 0.04 }}
          />
        )
      })}

      {/* Weight labels on arcs */}
      {TOKENS.map((_, targetIdx) => {
        if (targetIdx === selectedToken) return null
        const weight = weights[targetIdx]

        const srcX = tokensStartX + selectedToken * tokenSpacing
        const tgtX = tokensStartX + targetIdx * tokenSpacing
        const midX = (srcX + tgtX) / 2
        const dist = Math.abs(targetIdx - selectedToken)
        const arcHeight = 15 + dist * 15
        const labelY = tokenY - 18 - arcHeight * 0.5

        if (weight < 0.12) return null

        return (
          <motion.text
            key={`label-${selectedToken}-${targetIdx}`}
            x={midX}
            y={labelY}
            textAnchor="middle"
            className="text-[8px] font-mono"
            fill={attentionColorScale(weight)}
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.8 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            {weight.toFixed(2)}
          </motion.text>
        )
      })}

      {/* Token pills */}
      {TOKENS.map((token, i) => {
        const x = tokensStartX + i * tokenSpacing
        const isSelected = i === selectedToken
        const weight = weights[i]

        return (
          <g
            key={`tf-token-${i}`}
            style={{ cursor: 'pointer' }}
            onClick={() => onSelectToken(i)}
          >
            <motion.rect
              x={x - pillWidth / 2}
              y={tokenY - 12}
              width={pillWidth}
              height={24}
              rx={12}
              fill={
                isSelected
                  ? NLP.tokenHighlight
                  : `rgba(99, 102, 241, ${0.05 + weight * 0.5})`
              }
              stroke={
                isSelected
                  ? NLP.tokenHighlight
                  : `rgba(129, 140, 248, ${0.2 + weight * 0.6})`
              }
              strokeWidth={isSelected ? 2 : 1}
              whileHover={{ scale: 1.08 }}
              transition={{ duration: 0.2 }}
            />
            <text
              x={x}
              y={tokenY - 1}
              textAnchor="middle"
              dominantBaseline="central"
              className="text-[9px] font-mono pointer-events-none"
              fill="#E4E4E7"
            >
              {token}
            </text>

            {/* Self-attention indicator */}
            {isSelected && (
              <motion.circle
                cx={x}
                cy={tokenY + 20}
                r={3}
                fill={NLP.tokenHighlight}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.2 }}
              />
            )}
          </g>
        )
      })}

      {/* Click hint */}
      <text
        x={panelW / 2}
        y={tokenY + 40}
        textAnchor="middle"
        className="text-[8px]"
        fill="#71717A"
      >
        Click a token to see its attention pattern
      </text>

      {/* Bottom label */}
      <text
        x={panelW / 2}
        y={panelH - 8}
        textAnchor="middle"
        className="text-[9px]"
        fill="#71717A"
      >
        Every word sees every other word directly. No bottleneck.
      </text>
    </g>
  )
}

// ── Main Component ────────────────────────────────────────────────────

export function RNNvsAttentionComparison() {
  const [currentRNNStep, setCurrentRNNStep] = useState(0)
  const [selectedToken, setSelectedToken] = useState(7) // default: "it" - shows coreference to "cat"
  const [isPlaying, setIsPlaying] = useState(true)

  // Auto-play the RNN animation
  useEffect(() => {
    if (!isPlaying) return
    if (currentRNNStep >= TOKENS.length) {
      setIsPlaying(false)
      return
    }

    const timer = setInterval(() => {
      setCurrentRNNStep((prev) => {
        if (prev >= TOKENS.length - 1) {
          setIsPlaying(false)
          return prev
        }
        return prev + 1
      })
    }, 900)

    return () => clearInterval(timer)
  }, [isPlaying, currentRNNStep])

  const handleReset = useCallback(() => {
    setCurrentRNNStep(0)
    setIsPlaying(true)
  }, [])

  const handlePlay = useCallback(() => {
    if (currentRNNStep >= TOKENS.length - 1) {
      setCurrentRNNStep(0)
    }
    setIsPlaying(true)
  }, [currentRNNStep])

  return (
    <div className="space-y-4">
      {/* Two side-by-side panels */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* RNN Panel */}
        <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border overflow-hidden">
          <SVGContainer
            aspectRatio={16 / 12}
            minHeight={280}
            maxHeight={450}
            padding={{ top: 10, right: 15, bottom: 10, left: 15 }}
          >
            {({ innerWidth, innerHeight }) => (
              <>
                <defs>
                  <marker
                    id="rnn-seq-arrow"
                    markerWidth="5"
                    markerHeight="4"
                    refX="5"
                    refY="2"
                    orient="auto"
                  >
                    <path
                      d="M0,0 L5,2 L0,4"
                      fill="rgba(255,255,255,0.1)"
                    />
                  </marker>
                  <marker
                    id="rnn-seq-arrow-active"
                    markerWidth="5"
                    markerHeight="4"
                    refX="5"
                    refY="2"
                    orient="auto"
                  >
                    <path
                      d="M0,0 L5,2 L0,4"
                      fill="rgba(129,140,248,0.4)"
                    />
                  </marker>
                </defs>
                <RNNPanel
                  innerWidth={innerWidth}
                  innerHeight={innerHeight}
                  currentStep={currentRNNStep}
                />
              </>
            )}
          </SVGContainer>
        </div>

        {/* Transformer Panel */}
        <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border overflow-hidden">
          <SVGContainer
            aspectRatio={16 / 12}
            minHeight={280}
            maxHeight={450}
            padding={{ top: 10, right: 15, bottom: 10, left: 15 }}
          >
            {({ innerWidth, innerHeight }) => (
              <TransformerPanel
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                selectedToken={selectedToken}
                onSelectToken={setSelectedToken}
              />
            )}
          </SVGContainer>
        </div>
      </div>

      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={isPlaying ? () => setIsPlaying(false) : handlePlay}
            >
              {isPlaying ? 'Pause' : 'Play'}
            </Button>
            <Button variant="ghost" size="sm" onClick={handleReset}>
              Reset
            </Button>
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          <div className="flex items-center gap-2">
            <span className="text-xs text-text-tertiary">
              RNN step:
            </span>
            <span className="text-xs font-mono text-text-secondary">
              {currentRNNStep + 1} / {TOKENS.length}
            </span>
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          <div className="flex items-center gap-2">
            <span className="text-xs text-text-tertiary">
              Attention from:
            </span>
            <span
              className="text-xs font-mono px-2 py-0.5 rounded-md"
              style={{
                backgroundColor: 'rgba(129, 140, 248, 0.15)',
                color: NLP.tokenHighlight,
              }}
            >
              &quot;{TOKENS[selectedToken]}&quot;
            </span>
          </div>
        </div>
      </GlassCard>
    </div>
  )
}
