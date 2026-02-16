import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Button } from '../../../components/ui/Button'
import { Slider } from '../../../components/ui/Slider'

// ── NLP Colors ────────────────────────────────────────────────────────
const COLORS = {
  token: '#818CF8',
  position: '#38BDF8',
  attention: '#6366F1',
  value: '#FBBF24',
  query: '#F472B6',
  key: '#34D399',
}

// ── Pipeline Stage Data ───────────────────────────────────────────────
const INPUT_TEXT = 'The cat sat on the mat'
const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat']
const TOKEN_IDS = [1996, 4937, 2938, 2006, 1996, 13523]

const STAGES = [
  {
    label: 'Input Text',
    description: 'Raw text enters the model as a string of characters.',
  },
  {
    label: 'Tokenization',
    description: 'Text is split into tokens and mapped to integer IDs from a vocabulary.',
  },
  {
    label: 'Embedding',
    description: 'Each token ID is looked up in an embedding table to produce a dense vector.',
  },
  {
    label: 'Positional Encoding',
    description: 'Position information is added so the model knows word order.',
  },
  {
    label: 'Transformer Blocks',
    description: 'Data passes through stacked layers of self-attention and feed-forward networks.',
  },
  {
    label: 'Output',
    description: 'Final vectors are projected to vocabulary logits to predict the next token.',
  },
] as const

const NUM_STAGES = STAGES.length

// ── Fake embedding vectors (for visual display) ──────────────────────
function generateEmbedding(seed: number, dims: number): number[] {
  const vals: number[] = []
  for (let i = 0; i < dims; i++) {
    vals.push(Math.sin(seed * 0.7 + i * 1.3) * 0.6 + Math.cos(seed * 1.1 + i * 0.9) * 0.4)
  }
  return vals
}

function generatePositionalEncoding(pos: number, dims: number): number[] {
  const vals: number[] = []
  for (let i = 0; i < dims; i++) {
    if (i % 2 === 0) {
      vals.push(Math.sin(pos / Math.pow(10000, i / dims)))
    } else {
      vals.push(Math.cos(pos / Math.pow(10000, (i - 1) / dims)))
    }
  }
  return vals
}

const EMBED_DIMS = 6

// ── Sub-components for each stage ─────────────────────────────────────

function InputTextStage({ active }: { active: boolean }) {
  return (
    <div className="flex items-center justify-center py-3">
      <motion.span
        className="text-lg font-mono tracking-wide"
        style={{ color: active ? '#E4E4E7' : '#71717A' }}
        animate={{ opacity: active ? 1 : 0.5 }}
      >
        &quot;{INPUT_TEXT}&quot;
      </motion.span>
    </div>
  )
}

function TokenizationStage({ active }: { active: boolean }) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-2 py-2">
      {TOKENS.map((token, i) => (
        <motion.div
          key={i}
          className="flex flex-col items-center gap-1"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: active ? 1 : 0.4, y: 0 }}
          transition={{ delay: active ? i * 0.08 : 0, duration: 0.3 }}
        >
          <span
            className="px-3 py-1 rounded-full text-xs font-mono border"
            style={{
              backgroundColor: active ? `${COLORS.token}20` : 'rgba(255,255,255,0.04)',
              borderColor: active ? `${COLORS.token}50` : 'rgba(255,255,255,0.08)',
              color: active ? COLORS.token : '#71717A',
            }}
          >
            {token}
          </span>
          <span
            className="text-[10px] font-mono"
            style={{ color: active ? '#A1A1AA' : '#52525B' }}
          >
            {TOKEN_IDS[i]}
          </span>
        </motion.div>
      ))}
    </div>
  )
}

function EmbeddingStage({ active }: { active: boolean }) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-3 py-2">
      {TOKENS.map((token, i) => {
        const emb = generateEmbedding(TOKEN_IDS[i], EMBED_DIMS)
        return (
          <motion.div
            key={i}
            className="flex flex-col items-center gap-1"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: active ? 1 : 0.4, scale: 1 }}
            transition={{ delay: active ? i * 0.06 : 0, duration: 0.3 }}
          >
            <span
              className="text-[9px] font-mono mb-0.5"
              style={{ color: active ? '#A1A1AA' : '#52525B' }}
            >
              {token}
            </span>
            <div className="flex gap-px">
              {emb.map((v, d) => (
                <motion.div
                  key={d}
                  className="rounded-sm"
                  style={{
                    width: 6,
                    height: 20,
                    backgroundColor: active
                      ? v > 0
                        ? `rgba(129, 140, 248, ${0.2 + Math.abs(v) * 0.6})`
                        : `rgba(251, 191, 36, ${0.2 + Math.abs(v) * 0.6})`
                      : 'rgba(255,255,255,0.06)',
                  }}
                  animate={{
                    scaleY: active ? 0.3 + Math.abs(v) * 0.7 : 0.5,
                  }}
                  transition={{ delay: active ? i * 0.06 + d * 0.02 : 0 }}
                />
              ))}
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}

function PositionalEncodingStage({ active }: { active: boolean }) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-3 py-2">
      {TOKENS.map((token, i) => {
        const emb = generateEmbedding(TOKEN_IDS[i], EMBED_DIMS)
        const pe = generatePositionalEncoding(i, EMBED_DIMS)
        const combined = emb.map((v, d) => v + pe[d])
        return (
          <motion.div
            key={i}
            className="flex flex-col items-center gap-1"
            initial={{ opacity: 0 }}
            animate={{ opacity: active ? 1 : 0.4 }}
            transition={{ delay: active ? i * 0.06 : 0, duration: 0.3 }}
          >
            <span
              className="text-[9px] font-mono mb-0.5"
              style={{ color: active ? '#A1A1AA' : '#52525B' }}
            >
              {token}
            </span>
            <div className="flex items-center gap-1">
              {/* Embedding bars */}
              <div className="flex gap-px">
                {emb.map((v, d) => (
                  <div
                    key={`e-${d}`}
                    className="rounded-sm"
                    style={{
                      width: 4,
                      height: 14,
                      backgroundColor: active
                        ? `rgba(129, 140, 248, ${0.3 + Math.abs(v) * 0.5})`
                        : 'rgba(255,255,255,0.04)',
                    }}
                  />
                ))}
              </div>
              {/* Plus sign */}
              <motion.span
                className="text-[10px] font-bold mx-0.5"
                style={{ color: active ? COLORS.position : '#52525B' }}
                animate={active ? { scale: [1, 1.3, 1] } : {}}
                transition={{ duration: 0.5, delay: i * 0.08 }}
              >
                +
              </motion.span>
              {/* PE bars */}
              <div className="flex gap-px">
                {pe.map((v, d) => (
                  <div
                    key={`p-${d}`}
                    className="rounded-sm"
                    style={{
                      width: 4,
                      height: 14,
                      backgroundColor: active
                        ? `rgba(56, 189, 248, ${0.3 + Math.abs(v) * 0.5})`
                        : 'rgba(255,255,255,0.04)',
                    }}
                  />
                ))}
              </div>
              {/* Equals sign */}
              <span
                className="text-[10px] font-bold mx-0.5"
                style={{ color: active ? '#A1A1AA' : '#52525B' }}
              >
                =
              </span>
              {/* Combined bars */}
              <div className="flex gap-px">
                {combined.map((v, d) => (
                  <motion.div
                    key={`c-${d}`}
                    className="rounded-sm"
                    style={{
                      width: 4,
                      height: 14,
                      backgroundColor: active
                        ? v > 0
                          ? `rgba(99, 102, 241, ${0.3 + Math.min(Math.abs(v), 1) * 0.5})`
                          : `rgba(251, 191, 36, ${0.3 + Math.min(Math.abs(v), 1) * 0.5})`
                        : 'rgba(255,255,255,0.04)',
                    }}
                    animate={active ? { opacity: [0.3, 1] } : {}}
                    transition={{ duration: 0.4, delay: 0.3 + i * 0.06 + d * 0.02 }}
                  />
                ))}
              </div>
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}

function TransformerBlocksStage({ active }: { active: boolean }) {
  const numBlocks = 3
  return (
    <div className="flex items-center justify-center gap-3 py-2">
      {Array.from({ length: numBlocks }).map((_, blockIdx) => (
        <motion.div
          key={blockIdx}
          className="flex flex-col items-center gap-1 rounded-lg border px-4 py-2"
          style={{
            backgroundColor: active
              ? `rgba(99, 102, 241, ${0.08 + blockIdx * 0.04})`
              : 'rgba(255,255,255,0.02)',
            borderColor: active
              ? `rgba(99, 102, 241, ${0.2 + blockIdx * 0.1})`
              : 'rgba(255,255,255,0.06)',
          }}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: active ? 1 : 0.4, x: 0 }}
          transition={{ delay: active ? blockIdx * 0.15 : 0, duration: 0.4 }}
        >
          <span
            className="text-[9px] font-mono uppercase tracking-wider"
            style={{ color: active ? '#A1A1AA' : '#52525B' }}
          >
            Block {blockIdx + 1}
          </span>
          {/* Attention sub-block */}
          <div
            className="w-full rounded px-2 py-1 text-center"
            style={{
              backgroundColor: active ? `${COLORS.attention}15` : 'rgba(255,255,255,0.02)',
            }}
          >
            <span
              className="text-[8px] font-mono"
              style={{ color: active ? COLORS.attention : '#52525B' }}
            >
              Self-Attention
            </span>
          </div>
          {/* Arrow */}
          <span
            className="text-[10px]"
            style={{ color: active ? '#71717A' : '#3F3F46' }}
          >
            ↓
          </span>
          {/* FFN sub-block */}
          <div
            className="w-full rounded px-2 py-1 text-center"
            style={{
              backgroundColor: active ? `${COLORS.value}15` : 'rgba(255,255,255,0.02)',
            }}
          >
            <span
              className="text-[8px] font-mono"
              style={{ color: active ? COLORS.value : '#52525B' }}
            >
              Feed-Forward
            </span>
          </div>
        </motion.div>
      ))}
      {/* Ellipsis for more blocks */}
      <motion.div
        className="flex flex-col items-center justify-center px-2"
        animate={{ opacity: active ? 0.6 : 0.2 }}
      >
        <span
          className="text-lg font-mono"
          style={{ color: active ? '#71717A' : '#3F3F46' }}
        >
          ...
        </span>
        <span
          className="text-[8px] font-mono"
          style={{ color: active ? '#71717A' : '#3F3F46' }}
        >
          ×N
        </span>
      </motion.div>
    </div>
  )
}

function OutputStage({ active }: { active: boolean }) {
  const topPredictions = [
    { token: '.', prob: 0.42 },
    { token: 'today', prob: 0.15 },
    { token: ',', prob: 0.12 },
  ]

  return (
    <div className="flex flex-col items-center gap-2 py-2">
      <div className="flex items-center gap-4">
        {/* Logits representation */}
        <div className="flex flex-col items-center gap-1">
          <span
            className="text-[9px] font-mono"
            style={{ color: active ? '#A1A1AA' : '#52525B' }}
          >
            Vocab Logits
          </span>
          <div className="flex gap-px items-end">
            {[0.1, 0.3, 0.05, 0.8, 0.2, 0.6, 0.15, 0.4].map((v, i) => (
              <motion.div
                key={i}
                className="rounded-sm"
                style={{
                  width: 5,
                  backgroundColor: active
                    ? `rgba(251, 191, 36, ${0.2 + v * 0.6})`
                    : 'rgba(255,255,255,0.06)',
                }}
                animate={{ height: active ? 4 + v * 24 : 8 }}
                transition={{ delay: active ? i * 0.03 : 0 }}
              />
            ))}
          </div>
        </div>

        {/* Arrow */}
        <motion.span
          className="text-sm font-mono"
          style={{ color: active ? COLORS.value : '#52525B' }}
          animate={active ? { x: [0, 4, 0] } : {}}
          transition={{ duration: 1, repeat: Infinity }}
        >
          →
        </motion.span>

        {/* Top predictions */}
        <div className="flex flex-col gap-1">
          <span
            className="text-[9px] font-mono"
            style={{ color: active ? '#A1A1AA' : '#52525B' }}
          >
            Predictions
          </span>
          {topPredictions.map((pred, i) => (
            <motion.div
              key={i}
              className="flex items-center gap-2"
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: active ? 1 : 0.3, x: 0 }}
              transition={{ delay: active ? 0.2 + i * 0.1 : 0 }}
            >
              <span
                className="px-2 py-0.5 rounded text-[10px] font-mono border"
                style={{
                  backgroundColor: i === 0 && active ? `${COLORS.key}20` : 'rgba(255,255,255,0.04)',
                  borderColor: i === 0 && active ? `${COLORS.key}40` : 'rgba(255,255,255,0.08)',
                  color: i === 0 && active ? COLORS.key : '#71717A',
                }}
              >
                {pred.token}
              </span>
              <div
                className="h-1.5 rounded-full"
                style={{
                  width: pred.prob * 60,
                  backgroundColor: active
                    ? i === 0
                      ? COLORS.key
                      : `${COLORS.key}60`
                    : 'rgba(255,255,255,0.08)',
                }}
              />
              <span
                className="text-[9px] font-mono"
                style={{ color: active ? '#A1A1AA' : '#52525B' }}
              >
                {(pred.prob * 100).toFixed(0)}%
              </span>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Connecting Arrow ──────────────────────────────────────────────────

function ConnectingArrow({ active }: { active: boolean }) {
  return (
    <div className="flex justify-center py-0.5">
      <motion.div
        className="flex flex-col items-center"
        animate={{ opacity: active ? 0.7 : 0.2 }}
      >
        <div
          className="w-px h-3"
          style={{
            backgroundColor: active ? COLORS.attention : 'rgba(255,255,255,0.1)',
          }}
        />
        <svg width="8" height="6" viewBox="0 0 8 6">
          <path
            d="M0,0 L4,6 L8,0"
            fill={active ? COLORS.attention : 'rgba(255,255,255,0.1)'}
          />
        </svg>
      </motion.div>
    </div>
  )
}

// ── Stage renderers ───────────────────────────────────────────────────

const STAGE_COMPONENTS = [
  InputTextStage,
  TokenizationStage,
  EmbeddingStage,
  PositionalEncodingStage,
  TransformerBlocksStage,
  OutputStage,
]

// ── Main Component ────────────────────────────────────────────────────

export function FullPipeline() {
  const [currentStage, setCurrentStage] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(2000)

  // Auto-play
  useEffect(() => {
    if (!isPlaying) return

    const timer = setInterval(() => {
      setCurrentStage((prev) => {
        if (prev >= NUM_STAGES - 1) {
          setIsPlaying(false)
          return prev
        }
        return prev + 1
      })
    }, speed)

    return () => clearInterval(timer)
  }, [isPlaying, speed])

  const handleStepForward = useCallback(() => {
    setIsPlaying(false)
    setCurrentStage((prev) => Math.min(prev + 1, NUM_STAGES - 1))
  }, [])

  const handleStepBack = useCallback(() => {
    setIsPlaying(false)
    setCurrentStage((prev) => Math.max(prev - 1, 0))
  }, [])

  const handleReset = useCallback(() => {
    setIsPlaying(false)
    setCurrentStage(0)
  }, [])

  const handleAutoPlay = useCallback(() => {
    setCurrentStage(0)
    setIsPlaying(true)
  }, [])

  return (
    <div className="space-y-3">
      {/* Pipeline stages */}
      <div className="space-y-0">
        {STAGES.map((stage, idx) => {
          const isActive = idx === currentStage
          const isPast = idx < currentStage
          const StageComponent = STAGE_COMPONENTS[idx]

          return (
            <div key={idx}>
              <GlassCard
                className={`px-4 py-2 transition-all duration-500 cursor-pointer ${
                  isActive
                    ? 'ring-1 ring-indigo-500/30'
                    : ''
                }`}
              >
                <motion.div
                  animate={{
                    opacity: isActive ? 1 : isPast ? 0.6 : 0.35,
                  }}
                  transition={{ duration: 0.4 }}
                  onClick={() => {
                    setIsPlaying(false)
                    setCurrentStage(idx)
                  }}
                >
                  {/* Stage header */}
                  <div className="flex items-center gap-3 mb-1">
                    {/* Stage number */}
                    <motion.div
                      className="flex items-center justify-center w-6 h-6 rounded-full text-[10px] font-bold border shrink-0"
                      style={{
                        backgroundColor: isActive
                          ? `${COLORS.attention}30`
                          : isPast
                            ? `${COLORS.key}20`
                            : 'rgba(255,255,255,0.04)',
                        borderColor: isActive
                          ? COLORS.attention
                          : isPast
                            ? `${COLORS.key}50`
                            : 'rgba(255,255,255,0.1)',
                        color: isActive
                          ? COLORS.attention
                          : isPast
                            ? COLORS.key
                            : '#71717A',
                      }}
                      animate={isActive ? { scale: [1, 1.1, 1] } : {}}
                      transition={{ duration: 0.6 }}
                    >
                      {isPast ? '✓' : idx + 1}
                    </motion.div>

                    {/* Stage title and description */}
                    <div className="flex-1 min-w-0">
                      <span
                        className="text-sm font-medium block"
                        style={{
                          color: isActive ? '#E4E4E7' : isPast ? '#A1A1AA' : '#71717A',
                        }}
                      >
                        {stage.label}
                      </span>
                      <AnimatePresence>
                        {isActive && (
                          <motion.span
                            className="text-xs block"
                            style={{ color: '#A1A1AA' }}
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.3 }}
                          >
                            {stage.description}
                          </motion.span>
                        )}
                      </AnimatePresence>
                    </div>
                  </div>

                  {/* Stage visual content */}
                  <AnimatePresence mode="wait">
                    {(isActive || isPast) && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.4 }}
                        className="overflow-hidden"
                      >
                        <StageComponent active={isActive} />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              </GlassCard>

              {/* Connecting arrow between stages */}
              {idx < NUM_STAGES - 1 && (
                <ConnectingArrow active={idx < currentStage} />
              )}
            </div>
          )
        })}
      </div>

      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleStepBack}
              disabled={currentStage === 0}
            >
              ← Back
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={isPlaying ? () => setIsPlaying(false) : handleAutoPlay}
            >
              {isPlaying ? 'Pause' : 'Auto-Play'}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleStepForward}
              disabled={currentStage === NUM_STAGES - 1}
            >
              Next →
            </Button>
            <Button variant="ghost" size="sm" onClick={handleReset}>
              Reset
            </Button>
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          {/* Stage indicator */}
          <div className="flex items-center gap-1.5">
            {STAGES.map((_, idx) => (
              <motion.button
                key={idx}
                className="w-2 h-2 rounded-full transition-colors"
                style={{
                  backgroundColor:
                    idx === currentStage
                      ? COLORS.attention
                      : idx < currentStage
                        ? COLORS.key
                        : 'rgba(255,255,255,0.15)',
                }}
                onClick={() => {
                  setIsPlaying(false)
                  setCurrentStage(idx)
                }}
                whileHover={{ scale: 1.5 }}
              />
            ))}
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          <div className="w-36">
            <Slider
              label="Speed"
              value={speed}
              min={500}
              max={4000}
              step={250}
              onChange={setSpeed}
              formatValue={(v) => `${(v / 1000).toFixed(1)}s`}
            />
          </div>

          <span className="text-xs text-text-tertiary font-mono ml-auto">
            Stage {currentStage + 1} / {NUM_STAGES}
          </span>
        </div>
      </GlassCard>
    </div>
  )
}
