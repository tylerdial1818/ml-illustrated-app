import { useState } from 'react'
import { motion } from 'framer-motion'
import { SVGContainer } from '../../../components/viz/SVGContainer'
import { GlassCard } from '../../../components/ui/GlassCard'

// ── Colors ────────────────────────────────────────────────────────────
const BERT_COLOR = '#6366F1'
const GPT_COLOR = '#F472B6'
const DIM_TEXT = '#71717A'
const BRIGHT_TEXT = '#E4E4E7'
const MUTED_TEXT = '#A1A1AA'

// ── Attention Heatmap (SVG) ───────────────────────────────────────────

function AttentionHeatmap({
  innerWidth,
  innerHeight,
  causal,
  color,
}: {
  innerWidth: number
  innerHeight: number
  causal: boolean
  color: string
}) {
  const size = 4
  const cellPad = 2
  const gridSize = Math.min(innerWidth, innerHeight) * 0.7
  const cellSize = (gridSize - (size - 1) * cellPad) / size
  const offsetX = (innerWidth - gridSize) / 2
  const offsetY = (innerHeight - gridSize) / 2

  // Simulated attention weights
  const weights = [
    [0.3, 0.25, 0.2, 0.25],
    [0.15, 0.35, 0.3, 0.2],
    [0.2, 0.25, 0.3, 0.25],
    [0.25, 0.2, 0.25, 0.3],
  ]

  const tokens = ['The', 'cat', 'sat', 'on']

  return (
    <>
      {/* Row labels (attending from) */}
      {tokens.map((token, i) => (
        <text
          key={`row-${i}`}
          x={offsetX - 6}
          y={offsetY + i * (cellSize + cellPad) + cellSize / 2}
          textAnchor="end"
          dominantBaseline="central"
          className="text-[8px] font-mono"
          fill={MUTED_TEXT}
        >
          {token}
        </text>
      ))}

      {/* Column labels (attending to) */}
      {tokens.map((token, i) => (
        <text
          key={`col-${i}`}
          x={offsetX + i * (cellSize + cellPad) + cellSize / 2}
          y={offsetY - 6}
          textAnchor="middle"
          className="text-[8px] font-mono"
          fill={MUTED_TEXT}
        >
          {token}
        </text>
      ))}

      {/* Grid cells */}
      {weights.map((row, i) =>
        row.map((weight, j) => {
          const isMasked = causal && j > i
          const x = offsetX + j * (cellSize + cellPad)
          const y = offsetY + i * (cellSize + cellPad)

          return (
            <motion.rect
              key={`cell-${i}-${j}`}
              x={x}
              y={y}
              width={cellSize}
              height={cellSize}
              rx={3}
              fill={
                isMasked
                  ? 'rgba(255, 255, 255, 0.02)'
                  : color
              }
              fillOpacity={isMasked ? 1 : 0.15 + weight * 0.7}
              stroke={
                isMasked
                  ? 'rgba(255, 255, 255, 0.05)'
                  : `${color}40`
              }
              strokeWidth={0.5}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: i * 0.05 + j * 0.05, duration: 0.3 }}
            />
          )
        })
      )}

      {/* Masked cells X markers */}
      {causal &&
        weights.map((row, i) =>
          row.map((_, j) => {
            if (j <= i) return null
            const x = offsetX + j * (cellSize + cellPad) + cellSize / 2
            const y = offsetY + i * (cellSize + cellPad) + cellSize / 2
            return (
              <motion.text
                key={`mask-${i}-${j}`}
                x={x}
                y={y}
                textAnchor="middle"
                dominantBaseline="central"
                className="text-[9px] font-mono"
                fill="rgba(255,255,255,0.15)"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 + i * 0.05 + j * 0.05 }}
              >
                ×
              </motion.text>
            )
          })
        )}

      {/* Label */}
      <text
        x={innerWidth / 2}
        y={innerHeight - 2}
        textAnchor="middle"
        className="text-[9px] font-mono"
        fill={DIM_TEXT}
      >
        {causal ? 'Causal mask: can only attend left' : 'Bidirectional: attends to all tokens'}
      </text>
    </>
  )
}

// ── Architecture Stack (SVG) ──────────────────────────────────────────

function ArchitectureStack({
  innerWidth,
  innerHeight,
  layers,
  label,
  color,
  causal,
}: {
  innerWidth: number
  innerHeight: number
  layers: number
  label: string
  color: string
  causal: boolean
}) {
  const blockH = Math.min(24, (innerHeight - 40) / layers - 4)
  const blockW = innerWidth * 0.65
  const startX = (innerWidth - blockW) / 2
  const totalStackH = layers * (blockH + 4) - 4
  const startY = (innerHeight - totalStackH) / 2

  return (
    <>
      {/* Stack label */}
      <text
        x={innerWidth / 2}
        y={startY - 10}
        textAnchor="middle"
        className="text-[9px] font-mono uppercase tracking-wider"
        fill={MUTED_TEXT}
      >
        {label}
      </text>

      {/* Blocks */}
      {Array.from({ length: layers }).map((_, i) => {
        const y = startY + i * (blockH + 4)
        return (
          <motion.g
            key={i}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.08, duration: 0.3 }}
          >
            <rect
              x={startX}
              y={y}
              width={blockW}
              height={blockH}
              rx={6}
              fill={`${color}15`}
              stroke={`${color}40`}
              strokeWidth={1}
            />
            <text
              x={innerWidth / 2}
              y={y + blockH / 2}
              textAnchor="middle"
              dominantBaseline="central"
              className="text-[7px] font-mono"
              fill={color}
            >
              {causal ? `Decoder Block ${i + 1}` : `Encoder Block ${i + 1}`}
            </text>
          </motion.g>
        )
      })}

      {/* Bottom label */}
      <text
        x={innerWidth / 2}
        y={innerHeight - 2}
        textAnchor="middle"
        className="text-[8px] font-mono"
        fill={DIM_TEXT}
      >
        {layers} layers stacked
      </text>
    </>
  )
}

// ── Context Arrows Demo ───────────────────────────────────────────────

function ContextDemo({
  mode,
  color,
}: {
  mode: 'bert' | 'gpt'
  color: string
}) {
  const isBert = mode === 'bert'
  const sentence = isBert
    ? ['The', '[MASK]', 'sat', 'on', 'the', 'mat']
    : ['The', 'cat', 'sat', 'on', 'the', '___']
  const targetIdx = isBert ? 1 : 5
  const prediction = isBert ? 'cat' : 'mat'

  return (
    <div className="space-y-3">
      {/* Token display */}
      <div className="flex flex-wrap items-center justify-center gap-1.5">
        {sentence.map((token, i) => {
          const isTarget = i === targetIdx
          const isContext = isBert ? i !== targetIdx : i < targetIdx

          return (
            <motion.div
              key={i}
              className="relative"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06 }}
            >
              <span
                className="px-2 py-1 rounded-md text-xs font-mono border inline-block"
                style={{
                  backgroundColor: isTarget
                    ? `${color}25`
                    : isContext
                      ? `${color}10`
                      : 'rgba(255,255,255,0.03)',
                  borderColor: isTarget
                    ? color
                    : isContext
                      ? `${color}40`
                      : 'rgba(255,255,255,0.08)',
                  color: isTarget ? color : isContext ? BRIGHT_TEXT : DIM_TEXT,
                  fontWeight: isTarget ? 700 : 400,
                }}
              >
                {token}
              </span>

              {/* Context arrows */}
              {isContext && (
                <motion.div
                  className="absolute -bottom-2.5 left-1/2 -translate-x-1/2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 0.6 }}
                  transition={{ delay: 0.5 + i * 0.08 }}
                >
                  <svg width="8" height="6" viewBox="0 0 8 6">
                    <path
                      d={
                        isBert
                          ? i < targetIdx
                            ? 'M0,6 L4,0 L8,6' // up arrow (left context)
                            : 'M0,6 L4,0 L8,6' // up arrow (right context)
                          : 'M0,6 L4,0 L8,6'
                      }
                      fill={color}
                      opacity={0.5}
                    />
                  </svg>
                </motion.div>
              )}
            </motion.div>
          )
        })}
      </div>

      {/* Direction indicator */}
      <div className="flex items-center justify-center gap-2">
        {isBert ? (
          <>
            <motion.span
              className="text-xs font-mono"
              style={{ color }}
              animate={{ x: [-2, 2, -2] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              ←←←
            </motion.span>
            <span className="text-[10px] font-mono" style={{ color: MUTED_TEXT }}>
              context flows both ways
            </span>
            <motion.span
              className="text-xs font-mono"
              style={{ color }}
              animate={{ x: [2, -2, 2] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              →→→
            </motion.span>
          </>
        ) : (
          <>
            <motion.span
              className="text-xs font-mono"
              style={{ color }}
              animate={{ x: [2, -2, 2] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              →→→
            </motion.span>
            <span className="text-[10px] font-mono" style={{ color: MUTED_TEXT }}>
              context flows left-to-right only
            </span>
          </>
        )}
      </div>

      {/* Prediction */}
      <div className="flex items-center justify-center gap-2">
        <span className="text-[10px] font-mono" style={{ color: MUTED_TEXT }}>
          {isBert ? 'Predicts:' : 'Next token:'}
        </span>
        <motion.span
          className="px-2 py-0.5 rounded-md text-xs font-mono font-bold border"
          style={{
            backgroundColor: `${color}20`,
            borderColor: `${color}60`,
            color,
          }}
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.8, type: 'spring' }}
        >
          &quot;{prediction}&quot;
        </motion.span>
      </div>
    </div>
  )
}

// ── Use Case Badge ────────────────────────────────────────────────────

function Badge({ label, color }: { label: string; color: string }) {
  return (
    <span
      className="px-2 py-0.5 rounded-full text-[10px] font-mono border"
      style={{
        backgroundColor: `${color}10`,
        borderColor: `${color}30`,
        color: `${color}CC`,
      }}
    >
      {label}
    </span>
  )
}

// ── Model Column ──────────────────────────────────────────────────────

function ModelColumn({
  title,
  subtitle,
  color,
  mode,
  useCases,
  trainingObj,
  numLayers,
}: {
  title: string
  subtitle: string
  color: string
  mode: 'bert' | 'gpt'
  useCases: string[]
  trainingObj: string
  numLayers: number
}) {
  const [activeTab, setActiveTab] = useState<'attention' | 'architecture'>('attention')

  return (
    <GlassCard className="p-5 flex-1">
      {/* Header */}
      <div className="text-center mb-4">
        <motion.h3
          className="text-xl font-bold"
          style={{ color }}
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {title}
        </motion.h3>
        <p className="text-xs font-mono" style={{ color: MUTED_TEXT }}>
          {subtitle}
        </p>
      </div>

      {/* Mini diagram tabs */}
      <div className="flex gap-1 mb-2 justify-center">
        <button
          className="px-2 py-1 rounded text-[10px] font-mono transition-colors"
          style={{
            backgroundColor: activeTab === 'attention' ? `${color}20` : 'transparent',
            color: activeTab === 'attention' ? color : DIM_TEXT,
          }}
          onClick={() => setActiveTab('attention')}
        >
          Attention Pattern
        </button>
        <button
          className="px-2 py-1 rounded text-[10px] font-mono transition-colors"
          style={{
            backgroundColor: activeTab === 'architecture' ? `${color}20` : 'transparent',
            color: activeTab === 'architecture' ? color : DIM_TEXT,
          }}
          onClick={() => setActiveTab('architecture')}
        >
          Architecture
        </button>
      </div>

      {/* SVG diagram area */}
      <div className="rounded-lg overflow-hidden border border-obsidian-border mb-4">
        {activeTab === 'attention' ? (
          <SVGContainer
            aspectRatio={1.2}
            minHeight={160}
            maxHeight={220}
            padding={{ top: 20, right: 15, bottom: 20, left: 35 }}
          >
            {({ innerWidth, innerHeight }) => (
              <AttentionHeatmap
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                causal={mode === 'gpt'}
                color={color}
              />
            )}
          </SVGContainer>
        ) : (
          <SVGContainer
            aspectRatio={1.2}
            minHeight={160}
            maxHeight={220}
            padding={{ top: 20, right: 15, bottom: 20, left: 15 }}
          >
            {({ innerWidth, innerHeight }) => (
              <ArchitectureStack
                innerWidth={innerWidth}
                innerHeight={innerHeight}
                layers={numLayers}
                label={`${title} Architecture`}
                color={color}
                causal={mode === 'gpt'}
              />
            )}
          </SVGContainer>
        )}
      </div>

      {/* Training objective */}
      <div className="text-center mb-3">
        <span className="text-[10px] font-mono uppercase tracking-wider" style={{ color: MUTED_TEXT }}>
          Training Objective
        </span>
        <p className="text-xs font-mono mt-1" style={{ color: BRIGHT_TEXT }}>
          {trainingObj}
        </p>
      </div>

      {/* Interactive context demo */}
      <div
        className="rounded-lg p-3 mb-3 border"
        style={{
          backgroundColor: `${color}08`,
          borderColor: `${color}20`,
        }}
      >
        <ContextDemo mode={mode} color={color} />
      </div>

      {/* Use case badges */}
      <div className="flex flex-wrap gap-1.5 justify-center">
        {useCases.map((uc) => (
          <Badge key={uc} label={uc} color={color} />
        ))}
      </div>
    </GlassCard>
  )
}

// ── Main Component ────────────────────────────────────────────────────

export function BERTvsGPT() {
  return (
    <div className="space-y-4">
      {/* Two-column comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* BERT Column */}
        <ModelColumn
          title="BERT"
          subtitle="Bidirectional Encoder"
          color={BERT_COLOR}
          mode="bert"
          useCases={['Classification', 'NER', 'Question Answering', 'Embeddings']}
          trainingObj="Masked Language Model"
          numLayers={4}
        />

        {/* GPT Column */}
        <ModelColumn
          title="GPT"
          subtitle="Causal Decoder"
          color={GPT_COLOR}
          mode="gpt"
          useCases={['Text Generation', 'Code', 'Reasoning', 'Chat']}
          trainingObj="Next Token Prediction"
          numLayers={4}
        />
      </div>

      {/* Shared insight */}
      <GlassCard className="p-5">
        <div className="text-center space-y-3">
          <p className="text-sm font-medium" style={{ color: BRIGHT_TEXT }}>
            Both use the same core mechanism:{' '}
            <span className="font-mono" style={{ color: '#6366F1' }}>self-attention</span>.
            The difference is{' '}
            <span className="font-mono" style={{ color: '#FBBF24' }}>what they can see</span>.
          </p>

          {/* Key differences table */}
          <div className="grid grid-cols-3 gap-2 max-w-xl mx-auto mt-4">
            {/* Header */}
            <div className="text-[10px] font-mono uppercase tracking-wider" style={{ color: DIM_TEXT }}>
              &nbsp;
            </div>
            <div
              className="text-[10px] font-mono font-bold text-center"
              style={{ color: BERT_COLOR }}
            >
              BERT
            </div>
            <div
              className="text-[10px] font-mono font-bold text-center"
              style={{ color: GPT_COLOR }}
            >
              GPT
            </div>

            {/* Rows */}
            {[
              ['Direction', 'Bidirectional', 'Left-to-right'],
              ['Mask', 'No causal mask', 'Causal mask'],
              ['Objective', 'Fill in blanks', 'Predict next'],
              ['Best for', 'Understanding', 'Generation'],
            ].map(([label, bert, gpt]) => (
              <KeyDiffRow key={label} label={label} bert={bert} gpt={gpt} />
            ))}
          </div>
        </div>
      </GlassCard>
    </div>
  )
}

// ── Helper subcomponent ───────────────────────────────────────────────

function KeyDiffRow({
  label,
  bert,
  gpt,
}: {
  label: string
  bert: string
  gpt: string
}) {
  return (
    <>
      <div
        className="text-[10px] font-mono text-right py-1 pr-2 border-r border-obsidian-border"
        style={{ color: MUTED_TEXT }}
      >
        {label}
      </div>
      <div
        className="text-[10px] font-mono text-center py-1 rounded"
        style={{ color: BRIGHT_TEXT, backgroundColor: `${BERT_COLOR}08` }}
      >
        {bert}
      </div>
      <div
        className="text-[10px] font-mono text-center py-1 rounded"
        style={{ color: BRIGHT_TEXT, backgroundColor: `${GPT_COLOR}08` }}
      >
        {gpt}
      </div>
    </>
  )
}
