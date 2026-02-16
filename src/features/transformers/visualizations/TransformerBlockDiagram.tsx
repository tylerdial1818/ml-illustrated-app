import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'
import { Toggle } from '../../../components/ui/Toggle'
import { TransportControls } from '../../../components/viz/TransportControls'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'

// ── NLP Colors ────────────────────────────────────────────────────────
const NLP = {
  query: '#F472B6',
  key: '#34D399',
  value: '#FBBF24',
  position: '#38BDF8',
  token: '#818CF8',
  attentionHigh: '#6366F1',
}

// ── Block component colors ────────────────────────────────────────────
const BLOCK_COLORS = {
  attention: { bg: 'rgba(99, 102, 241, 0.12)', border: 'rgba(99, 102, 241, 0.35)', label: '#A5B4FC' },
  ffn: { bg: 'rgba(52, 211, 153, 0.10)', border: 'rgba(52, 211, 153, 0.30)', label: '#6EE7B7' },
  norm: { bg: 'rgba(251, 191, 36, 0.08)', border: 'rgba(251, 191, 36, 0.25)', label: '#FCD34D' },
  add: { bg: 'rgba(255, 255, 255, 0.04)', border: 'rgba(255, 255, 255, 0.15)', label: '#A1A1AA' },
}

// ── Stages in a single Transformer block (bottom to top) ──────────────
type StageName = 'input' | 'attention' | 'add1' | 'norm1' | 'ffn' | 'add2' | 'norm2' | 'output'

interface Stage {
  id: StageName
  label: string
  sublabel?: string
  colorKey: 'attention' | 'ffn' | 'norm' | 'add'
  isSkipTarget?: boolean // receives a skip connection
  skipSource?: StageName // where the skip connection originates
}

const STAGES: Stage[] = [
  { id: 'input', label: 'Input Embeddings', colorKey: 'add' },
  { id: 'attention', label: 'Multi-Head Attention', sublabel: 'Q, K, V', colorKey: 'attention' },
  { id: 'add1', label: 'Add', colorKey: 'add', isSkipTarget: true, skipSource: 'input' },
  { id: 'norm1', label: 'Layer Norm', colorKey: 'norm' },
  { id: 'ffn', label: 'Feed-Forward Network', sublabel: 'Dense \u2192 ReLU \u2192 Dense', colorKey: 'ffn' },
  { id: 'add2', label: 'Add', colorKey: 'add', isSkipTarget: true, skipSource: 'norm1' },
  { id: 'norm2', label: 'Layer Norm', colorKey: 'norm' },
  { id: 'output', label: 'Output', colorKey: 'add' },
]

// Simulated vector at each stage (8 dims)
const VECTOR_DIMS = 8
function generateStageVector(stageIdx: number, seed: number): number[] {
  const vals: number[] = []
  for (let d = 0; d < VECTOR_DIMS; d++) {
    // Each stage transforms the vector differently
    const base = Math.sin((d + 1) * 1.3 + seed * 0.7) * 0.6
    const stageEffect = Math.cos((stageIdx + d) * 0.8) * 0.3 * (stageIdx / STAGES.length)
    vals.push(Math.max(-1, Math.min(1, base + stageEffect)))
  }
  return vals
}

// FFN expansion dims (wider intermediate representation)
const FFN_EXPANDED_DIMS = 16
function generateFFNExpanded(seed: number): number[] {
  const vals: number[] = []
  for (let d = 0; d < FFN_EXPANDED_DIMS; d++) {
    vals.push(Math.max(0, Math.sin((d + 1) * 0.9 + seed * 0.5) * 0.8)) // ReLU applied
  }
  return vals
}

// ── Single Block Diagram ──────────────────────────────────────────────
function BlockDiagram({
  innerWidth,
  innerHeight,
  animationStage,
  showResidual,
  showNorm,
}: {
  innerWidth: number
  innerHeight: number
  animationStage: number // 0-7 for each stage in STAGES
  showResidual: boolean
  showNorm: boolean
}) {
  // Filter stages based on toggles
  const visibleStages = STAGES.filter((s) => {
    if (!showNorm && s.id === 'norm1') return false
    if (!showNorm && s.id === 'norm2') return false
    return true
  })

  const stageCount = visibleStages.length
  const blockW = Math.min(innerWidth * 0.55, 220)
  const blockH = Math.min((innerHeight - 60) / stageCount, 40)
  const verticalGap = Math.max(
    (innerHeight - stageCount * blockH) / (stageCount + 1),
    8
  )
  const centerX = innerWidth / 2

  // Map each visible stage to its y position (bottom to top)
  const stagePositions = visibleStages.map((stage, i) => {
    const idx = stageCount - 1 - i // reverse: bottom to top
    const y = verticalGap + idx * (blockH + verticalGap)
    return { ...stage, y, idx: i }
  })

  // Find the mapped index in visible stages for the current animation stage
  const activeStageId = STAGES[Math.min(animationStage, STAGES.length - 1)]?.id
  const activeVisibleIdx = stagePositions.findIndex(
    (s) => s.id === activeStageId
  )

  // Data ball position
  const ballStage = stagePositions[activeVisibleIdx] ?? stagePositions[0]

  // Mini bar chart for the current vector
  const currentVector = generateStageVector(animationStage, 42)
  const ffnExpanded = generateFFNExpanded(42)

  // Skip connection positions
  const skipConnections: Array<{
    fromY: number
    toY: number
    fromId: StageName
    toId: StageName
  }> = []

  if (showResidual) {
    stagePositions.forEach((sp) => {
      if (sp.isSkipTarget && sp.skipSource) {
        const source = stagePositions.find((s) => s.id === sp.skipSource)
        if (source) {
          skipConnections.push({
            fromY: source.y + blockH / 2,
            toY: sp.y + blockH / 2,
            fromId: source.id,
            toId: sp.id,
          })
        }
      }
    })
  }

  // Bar chart dimensions
  const barChartW = Math.min(blockW * 0.35, 80)
  const barChartH = blockH * 0.7
  const barChartX = centerX + blockW / 2 + 20

  return (
    <>
      <defs>
        <marker
          id="block-arrow"
          markerWidth="6"
          markerHeight="5"
          refX="6"
          refY="2.5"
          orient="auto"
        >
          <path d="M0,0 L6,2.5 L0,5" fill="rgba(255,255,255,0.4)" />
        </marker>
        <marker
          id="skip-arrow"
          markerWidth="6"
          markerHeight="5"
          refX="6"
          refY="2.5"
          orient="auto"
        >
          <path d="M0,0 L6,2.5 L0,5" fill="rgba(251,191,36,0.5)" />
        </marker>
      </defs>

      {/* Skip connections (curved bypass arrows) */}
      {skipConnections.map((sc) => {
        const curveX = centerX - blockW / 2 - 45

        return (
          <motion.path
            key={`skip-${sc.fromId}-${sc.toId}`}
            d={`M ${centerX - blockW / 2} ${sc.fromY}
                C ${curveX} ${sc.fromY}, ${curveX} ${sc.toY}, ${centerX - blockW / 2} ${sc.toY}`}
            fill="none"
            stroke="rgba(251, 191, 36, 0.3)"
            strokeWidth={1.5}
            strokeDasharray="4,3"
            markerEnd="url(#skip-arrow)"
            animate={{ opacity: showResidual ? 0.8 : 0.1 }}
            transition={{ duration: 0.3 }}
          />
        )
      })}

      {/* Stage blocks */}
      {stagePositions.map((stage, visIdx) => {
        const colors = BLOCK_COLORS[stage.colorKey]
        const isActive = visIdx <= activeVisibleIdx
        const isCurrent = visIdx === activeVisibleIdx

        return (
          <g key={stage.id}>
            {/* Connection arrow to next stage */}
            {visIdx < stagePositions.length - 1 && (
              <line
                x1={centerX}
                y1={stage.y + blockH}
                x2={centerX}
                y2={stagePositions[visIdx + 1].y - 2}
                stroke="rgba(255,255,255,0.15)"
                strokeWidth={1.5}
                markerEnd="url(#block-arrow)"
              />
            )}

            {/* Block rectangle */}
            <motion.rect
              x={centerX - blockW / 2}
              y={stage.y}
              width={blockW}
              height={blockH}
              rx={8}
              fill={colors.bg}
              stroke={isCurrent ? colors.label : colors.border}
              strokeWidth={isCurrent ? 2 : 1}
              animate={{
                opacity: isActive ? 1 : 0.4,
                fill: isCurrent ? colors.bg.replace(')', ', 0.25)').replace('0.12)', '0.25)').replace('0.10)', '0.22)').replace('0.08)', '0.18)').replace('0.04)', '0.12)') : colors.bg,
              }}
              transition={{ duration: 0.3 }}
            />

            {/* Label */}
            <text
              x={centerX}
              y={stage.y + blockH / 2 - (stage.sublabel ? 3 : 0)}
              textAnchor="middle"
              dominantBaseline="central"
              className="text-[10px] font-medium"
              fill={colors.label}
            >
              {stage.label}
            </text>

            {/* Sublabel */}
            {stage.sublabel && (
              <text
                x={centerX}
                y={stage.y + blockH / 2 + 10}
                textAnchor="middle"
                dominantBaseline="central"
                className="text-[7px] font-mono"
                fill="rgba(255,255,255,0.4)"
              >
                {stage.sublabel}
              </text>
            )}

            {/* Add indicator icon */}
            {stage.id === 'add1' || stage.id === 'add2' ? (
              <text
                x={centerX + blockW / 2 - 12}
                y={stage.y + blockH / 2 + 1}
                textAnchor="middle"
                dominantBaseline="central"
                className="text-[12px] font-mono"
                fill="rgba(255,255,255,0.4)"
              >
                +
              </text>
            ) : null}
          </g>
        )
      })}

      {/* Animated data ball */}
      <motion.circle
        cx={centerX + blockW / 2 + 8}
        cy={ballStage.y + blockH / 2}
        r={5}
        fill={NLP.token}
        animate={{
          cy: ballStage.y + blockH / 2,
          filter: `drop-shadow(0 0 6px ${NLP.token})`,
        }}
        transition={{ duration: 0.4, ease: 'easeInOut' }}
      />

      {/* Mini vector bar chart at current stage */}
      <AnimatePresence mode="wait">
        <motion.g
          key={`bars-${animationStage}`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
        >
          {/* Show expanded bars for FFN stage */}
          {activeStageId === 'ffn' ? (
            <>
              <text
                x={barChartX + barChartW / 2}
                y={ballStage.y - 4}
                textAnchor="middle"
                className="text-[7px] fill-text-tertiary font-mono"
              >
                4x expanded
              </text>
              {ffnExpanded.map((v, d) => {
                const bw = Math.max(barChartW / FFN_EXPANDED_DIMS - 0.5, 1.5)
                const bh = v * barChartH
                return (
                  <motion.rect
                    key={`ffn-bar-${d}`}
                    x={barChartX + d * (bw + 0.5)}
                    y={ballStage.y + blockH / 2 - bh / 2}
                    width={bw}
                    height={Math.max(bh, 1)}
                    rx={0.5}
                    fill={NLP.key}
                    fillOpacity={0.6 + v * 0.3}
                    initial={{ height: 0 }}
                    animate={{ height: Math.max(bh, 1) }}
                    transition={{ duration: 0.3, delay: d * 0.02 }}
                  />
                )
              })}
            </>
          ) : (
            <>
              {currentVector.map((v, d) => {
                const bw = Math.max(barChartW / VECTOR_DIMS - 1, 3)
                const bh = Math.abs(v) * barChartH
                const color = v >= 0 ? NLP.token : NLP.query
                return (
                  <motion.rect
                    key={`vec-bar-${d}`}
                    x={barChartX + d * (bw + 1)}
                    y={
                      v >= 0
                        ? ballStage.y + blockH / 2 - bh
                        : ballStage.y + blockH / 2
                    }
                    width={bw}
                    height={Math.max(bh, 1)}
                    rx={1}
                    fill={color}
                    fillOpacity={0.5 + Math.abs(v) * 0.4}
                    initial={{ height: 0 }}
                    animate={{
                      height: Math.max(bh, 1),
                      y: v >= 0
                        ? ballStage.y + blockH / 2 - bh
                        : ballStage.y + blockH / 2,
                    }}
                    transition={{ duration: 0.3, delay: d * 0.03 }}
                  />
                )
              })}
            </>
          )}
        </motion.g>
      </AnimatePresence>

      {/* Labels for top/bottom */}
      <text
        x={centerX}
        y={stagePositions[stagePositions.length - 1].y - 12}
        textAnchor="middle"
        className="text-[8px] fill-text-tertiary font-mono"
      >
        Output
      </text>
      <text
        x={centerX}
        y={stagePositions[0].y + blockH + 16}
        textAnchor="middle"
        className="text-[8px] fill-text-tertiary font-mono"
      >
        Input tokens
      </text>
    </>
  )
}

// ── Stacking Sub-visualization ────────────────────────────────────────
function StackedBlocks({ numLayers }: { numLayers: number }) {
  const blockH = 28
  const maxDisplay = Math.min(numLayers, 12)

  return (
    <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border p-4">
      <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-3">
        Stacked Transformer Blocks
      </p>
      <div className="flex items-center gap-4">
        <div className="flex flex-col items-center gap-1">
          {/* Output at top */}
          <span className="text-[8px] text-text-tertiary font-mono mb-1">Output</span>

          {Array.from({ length: maxDisplay }).map((_, i) => {
            const layerIdx = maxDisplay - 1 - i
            return (
              <motion.div
                key={i}
                className="rounded-md border flex items-center justify-center"
                style={{
                  width: 140,
                  height: blockH,
                  backgroundColor: 'rgba(99, 102, 241, 0.06)',
                  borderColor: `rgba(99, 102, 241, ${0.1 + (layerIdx / maxDisplay) * 0.15})`,
                }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, delay: i * 0.05 }}
              >
                <span className="text-[8px] font-mono text-text-tertiary">
                  Block {layerIdx + 1}
                </span>
              </motion.div>
            )
          })}

          {/* Input at bottom */}
          <span className="text-[8px] text-text-tertiary font-mono mt-1">Input</span>
        </div>

        <div className="flex flex-col gap-1 text-[9px] text-text-tertiary max-w-[200px]">
          <p>
            <span className="text-text-secondary font-medium">{numLayers} layers</span> stacked.
          </p>
          <p>Data flows through each block sequentially, with residual connections preserving information.</p>
          <div className="mt-2 space-y-0.5 text-[8px]">
            <p><span className="text-text-secondary">BERT-base:</span> N=12</p>
            <p><span className="text-text-secondary">GPT-2:</span> N=12</p>
            <p><span className="text-text-secondary">GPT-3:</span> N=96</p>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── d_model options ──────────────────────────────────────────────────
const D_MODEL_OPTIONS = [64, 128, 256, 512]

// ── Main Component ────────────────────────────────────────────────────
export function TransformerBlockDiagram() {
  const [numLayers, setNumLayers] = useState(6)
  const [showResidual, setShowResidual] = useState(true)
  const [showNorm, setShowNorm] = useState(true)
  const [dModel, setDModel] = useState(256)

  const dFF = dModel * 4

  // Transport controls via useAlgorithmPlayer
  const stageSnapshots = useMemo(
    () => Array.from({ length: STAGES.length }, (_, i) => i),
    []
  )
  const player = useAlgorithmPlayer({ snapshots: stageSnapshots, baseFps: 1.2 })
  const animationStage = player.currentSnapshot

  return (
    <div className="space-y-4">
      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          <Toggle
            label="Residual connections"
            checked={showResidual}
            onChange={setShowResidual}
          />
          <Toggle
            label="Layer normalization"
            checked={showNorm}
            onChange={setShowNorm}
          />

          <div className="h-6 w-px bg-obsidian-border" />

          {/* d_model selector */}
          <div>
            <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-1">d_model</p>
            <div className="flex gap-1">
              {D_MODEL_OPTIONS.map((val) => (
                <button
                  key={val}
                  onClick={() => setDModel(val)}
                  className={`px-2 py-1 rounded text-xs font-mono transition-all ${
                    dModel === val
                      ? 'bg-accent/20 text-accent border border-accent/40'
                      : 'text-text-tertiary hover:text-text-secondary border border-transparent'
                  }`}
                >
                  {val}
                </button>
              ))}
            </div>
          </div>

          {/* d_ff display */}
          <div className="text-right">
            <p className="text-[10px] uppercase tracking-wider text-text-tertiary">d_ff</p>
            <p className="text-sm font-mono text-text-secondary">{dFF}</p>
            <p className="text-[8px] text-text-tertiary">4 x d_model</p>
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          {/* Stage info */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-secondary">
              {STAGES[animationStage].label}
            </span>
          </div>
        </div>
      </GlassCard>

      {/* Main block diagram */}
      <div className="bg-obsidian-surface/40 rounded-xl border border-obsidian-border overflow-hidden">
        <SVGContainer
          aspectRatio={16 / 14}
          minHeight={400}
          maxHeight={600}
          padding={{ top: 25, right: 100, bottom: 30, left: 30 }}
        >
          {({ innerWidth, innerHeight }) => (
            <BlockDiagram
              innerWidth={innerWidth}
              innerHeight={innerHeight}
              animationStage={animationStage}
              showResidual={showResidual}
              showNorm={showNorm}
            />
          )}
        </SVGContainer>
      </div>

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

      {/* Legend */}
      <GlassCard className="p-3">
        <div className="flex flex-wrap items-center gap-4">
          {[
            { label: 'Attention', color: BLOCK_COLORS.attention },
            { label: 'Feed-Forward', color: BLOCK_COLORS.ffn },
            { label: 'Layer Norm', color: BLOCK_COLORS.norm },
            { label: 'Add (residual)', color: BLOCK_COLORS.add },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-1.5">
              <div
                className="w-3 h-3 rounded"
                style={{
                  backgroundColor: item.color.bg,
                  border: `1px solid ${item.color.border}`,
                }}
              />
              <span className="text-[9px] text-text-tertiary">{item.label}</span>
            </div>
          ))}
        </div>
      </GlassCard>

      {/* Stacking sub-viz */}
      <div className="space-y-3">
        <Slider
          label="Number of layers (N)"
          value={numLayers}
          min={1}
          max={12}
          step={1}
          onChange={setNumLayers}
          formatValue={(v) => `${v} layers`}
          className="max-w-xs"
        />
        <StackedBlocks numLayers={numLayers} />
      </div>
    </div>
  )
}
