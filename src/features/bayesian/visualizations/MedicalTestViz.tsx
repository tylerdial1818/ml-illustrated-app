import { useState, useMemo, useCallback, useEffect } from 'react'
import { motion } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'
import { Button } from '../../../components/ui/Button'

// ── Seeded random ─────────────────────────────────────────────────────
function seededRandom(seed: number) {
  let s = seed
  return () => {
    s = (s * 16807 + 0) % 2147483647
    return (s - 1) / 2147483646
  }
}

// ── Constants ─────────────────────────────────────────────────────────
const GRID_SIZE = 10
const TOTAL = GRID_SIZE * GRID_SIZE

const COLORS = {
  healthy: '#38BDF8',
  sick: '#F87171',
  falsePositive: '#FBBF24',
  truePositive: '#F87171',
  negative: 'rgba(255,255,255,0.15)',
  posterior: '#6366F1',
}

type AnimStep = 0 | 1 | 2

// ── Icon ─────────────────────────────────────────────────────────────
function PersonIcon({
  x,
  y,
  size,
  color,
  highlighted,
  delay,
}: {
  x: number
  y: number
  size: number
  color: string
  highlighted: boolean
  delay: number
}) {
  return (
    <motion.g
      transform={`translate(${x}, ${y})`}
      initial={{ opacity: 0, scale: 0.5 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay, duration: 0.15 }}
    >
      {/* Head */}
      <circle
        cx={size / 2}
        cy={size * 0.25}
        r={size * 0.18}
        fill={color}
        fillOpacity={highlighted ? 1 : 0.4}
      />
      {/* Body */}
      <rect
        x={size * 0.25}
        y={size * 0.45}
        width={size * 0.5}
        height={size * 0.4}
        rx={size * 0.1}
        fill={color}
        fillOpacity={highlighted ? 1 : 0.4}
      />
      {/* Highlight ring */}
      {highlighted && (
        <motion.rect
          x={0}
          y={0}
          width={size}
          height={size * 0.9}
          rx={size * 0.15}
          fill="none"
          stroke={color}
          strokeWidth={1.5}
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.6 }}
          transition={{ duration: 0.3 }}
        />
      )}
    </motion.g>
  )
}

// ── Main Component ────────────────────────────────────────────────────
export function MedicalTestViz() {
  const [prevalence, setPrevalence] = useState(1)
  const [accuracy, setAccuracy] = useState(90)
  const [step, setStep] = useState<AnimStep>(0)

  // Compute numbers
  const nSick = Math.max(1, Math.round((prevalence / 100) * TOTAL))
  const nHealthy = TOTAL - nSick
  const sensitivity = accuracy / 100 // true positive rate
  const falsePositiveRate = 1 - sensitivity

  const truePositives = Math.round(nSick * sensitivity)
  const falsePositives = Math.round(nHealthy * falsePositiveRate)
  const totalPositives = truePositives + falsePositives
  const posteriorProb = totalPositives > 0 ? truePositives / totalPositives : 0

  // Assign people to grid positions with deterministic sick/healthy
  const people = useMemo(() => {
    const rng = seededRandom(42)
    const indices = Array.from({ length: TOTAL }, (_, i) => i)
    // Shuffle
    for (let i = TOTAL - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1))
      const tmp = indices[i]
      indices[i] = indices[j]
      indices[j] = tmp
    }
    // First nSick in shuffled order are sick
    const sickSet = new Set(indices.slice(0, nSick))

    // Determine test results
    const rng2 = seededRandom(73)
    return Array.from({ length: TOTAL }, (_, i) => {
      const isSick = sickSet.has(i)
      const testedPositive = isSick
        ? rng2() < sensitivity
        : rng2() < falsePositiveRate
      return { index: i, isSick, testedPositive }
    })
  }, [nSick, sensitivity, falsePositiveRate])

  // Auto-reset step when sliders change
  useEffect(() => {
    setStep(0)
  }, [prevalence, accuracy])

  const handleNextStep = useCallback(() => {
    setStep((prev) => Math.min(prev + 1, 2) as AnimStep)
  }, [])

  const handleReset = useCallback(() => {
    setStep(0)
  }, [])

  // Compute icon color and highlight based on step
  const getIconStyle = (person: { isSick: boolean; testedPositive: boolean }) => {
    if (step === 0) {
      return {
        color: person.isSick ? COLORS.sick : COLORS.healthy,
        highlighted: false,
      }
    }
    if (step === 1) {
      if (person.testedPositive) {
        return {
          color: person.isSick ? COLORS.truePositive : COLORS.falsePositive,
          highlighted: true,
        }
      }
      return {
        color: COLORS.negative,
        highlighted: false,
      }
    }
    // Step 2: zoom into positives
    if (person.testedPositive) {
      return {
        color: person.isSick ? COLORS.truePositive : COLORS.falsePositive,
        highlighted: true,
      }
    }
    return {
      color: COLORS.negative,
      highlighted: false,
    }
  }

  const cellSize = 28
  const gap = 3
  const gridWidth = GRID_SIZE * (cellSize + gap)
  const gridHeight = GRID_SIZE * (cellSize + gap)

  return (
    <div className="space-y-4">
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          <Slider
            label="Disease Prevalence"
            value={prevalence}
            min={0.5}
            max={20}
            step={0.5}
            onChange={setPrevalence}
            formatValue={(v) => `${v}%`}
            className="w-44"
          />
          <Slider
            label="Test Accuracy"
            value={accuracy}
            min={50}
            max={99}
            step={1}
            onChange={setAccuracy}
            formatValue={(v) => `${v}%`}
            className="w-44"
          />

          <div className="h-6 w-px bg-obsidian-border" />

          <Button variant="secondary" size="sm" onClick={handleReset}>
            Reset
          </Button>
          <Button variant="primary" size="sm" onClick={handleNextStep} disabled={step >= 2}>
            {step === 0 ? 'Apply Test' : step === 1 ? 'Zoom Into Positives' : 'Done'}
          </Button>
        </div>
      </GlassCard>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Icon grid */}
        <GlassCard className="lg:col-span-2 p-4 overflow-hidden">
          <div className="flex justify-center">
            <svg
              width={gridWidth}
              height={gridHeight}
              viewBox={`0 0 ${gridWidth} ${gridHeight}`}
              className="max-w-full"
            >
              {people.map((person, i) => {
                const col = i % GRID_SIZE
                const row = Math.floor(i / GRID_SIZE)
                const style = getIconStyle(person)
                return (
                  <PersonIcon
                    key={i}
                    x={col * (cellSize + gap)}
                    y={row * (cellSize + gap)}
                    size={cellSize}
                    color={style.color}
                    highlighted={style.highlighted}
                    delay={i * 0.005}
                  />
                )
              })}
            </svg>
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-4 mt-4 justify-center">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded" style={{ backgroundColor: COLORS.sick }} />
              <span className="text-[10px] text-text-secondary">Sick ({nSick})</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded" style={{ backgroundColor: COLORS.healthy }} />
              <span className="text-[10px] text-text-secondary">Healthy ({nHealthy})</span>
            </div>
            {step >= 1 && (
              <>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded border-2" style={{ borderColor: COLORS.falsePositive, backgroundColor: `${COLORS.falsePositive}30` }} />
                  <span className="text-[10px] text-text-secondary">False Positive ({falsePositives})</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded border-2" style={{ borderColor: COLORS.truePositive, backgroundColor: `${COLORS.truePositive}30` }} />
                  <span className="text-[10px] text-text-secondary">True Positive ({truePositives})</span>
                </div>
              </>
            )}
          </div>
        </GlassCard>

        {/* Probability breakdown */}
        <GlassCard className="p-4">
          <h4 className="text-xs font-mono uppercase tracking-wider text-text-secondary mb-4">
            {step === 0 && 'Before the Test'}
            {step === 1 && 'Test Results'}
            {step === 2 && 'The Surprise'}
          </h4>

          <div className="space-y-4">
            {step === 0 && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
                <div>
                  <p className="text-[10px] font-mono text-text-tertiary">Prior probability of being sick</p>
                  <p className="text-2xl font-mono font-bold" style={{ color: COLORS.sick }}>
                    {prevalence}%
                  </p>
                </div>
                <p className="text-[11px] text-text-secondary leading-relaxed">
                  Out of 100 people, about {nSick} have the disease. Now imagine everyone takes a test
                  that is {accuracy}% accurate.
                </p>
              </motion.div>
            )}

            {step === 1 && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
                <div>
                  <p className="text-[10px] font-mono text-text-tertiary">Total positive results</p>
                  <p className="text-2xl font-mono font-bold" style={{ color: COLORS.falsePositive }}>
                    {totalPositives}
                  </p>
                </div>
                <p className="text-[11px] text-text-secondary leading-relaxed">
                  {truePositives} truly sick tested positive. But {falsePositives} healthy people also
                  got false positives. That is a lot of false alarms!
                </p>
              </motion.div>
            )}

            {step === 2 && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
                <div>
                  <p className="text-[10px] font-mono text-text-tertiary">
                    P(sick | positive test)
                  </p>
                  <p className="text-3xl font-mono font-bold" style={{ color: COLORS.posterior }}>
                    {(posteriorProb * 100).toFixed(1)}%
                  </p>
                </div>
                <p className="text-[11px] text-text-secondary leading-relaxed">
                  Even with a {accuracy}% accurate test, a positive result only means a{' '}
                  <strong className="text-text-primary">
                    {(posteriorProb * 100).toFixed(1)}%
                  </strong>{' '}
                  chance of actually being sick. Not {accuracy}%! This is the{' '}
                  <strong className="text-text-primary">base rate fallacy</strong>.
                  The prior matters enormously.
                </p>

                {/* Probability bar */}
                <div className="mt-3">
                  <div className="flex text-[9px] font-mono justify-between mb-1">
                    <span style={{ color: COLORS.sick }}>Actually sick</span>
                    <span style={{ color: COLORS.falsePositive }}>False alarm</span>
                  </div>
                  <div className="h-4 rounded-full overflow-hidden flex">
                    <div
                      className="h-full"
                      style={{
                        width: `${posteriorProb * 100}%`,
                        backgroundColor: COLORS.sick,
                      }}
                    />
                    <div
                      className="h-full"
                      style={{
                        width: `${(1 - posteriorProb) * 100}%`,
                        backgroundColor: COLORS.falsePositive,
                      }}
                    />
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        </GlassCard>
      </div>
    </div>
  )
}
