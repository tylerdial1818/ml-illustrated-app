import { useState, useMemo } from 'react'
import { makeDigitGrid } from '../../../lib/data/nnDataGenerators'
import {
  convolve2D,
  PRESET_KERNELS,
} from '../../../lib/algorithms/neural-networks/convolution'
import { useAlgorithmPlayer } from '../../../hooks/useAlgorithmPlayer'
import { TransportControls } from '../../../components/viz/TransportControls'
import { Select } from '../../../components/ui/Select'
import { Slider } from '../../../components/ui/Slider'
import { GlassCard } from '../../../components/ui/GlassCard'
import { COLORS } from '../../../types'

// ── Kernel options ─────────────────────────────────────────────────────

const KERNEL_OPTIONS = [
  { value: 'horizontal_edge', label: 'Horizontal Edge' },
  { value: 'vertical_edge', label: 'Vertical Edge' },
  { value: 'blur', label: 'Blur' },
  { value: 'sharpen', label: 'Sharpen' },
  { value: 'diagonal', label: 'Diagonal' },
  { value: 'emboss', label: 'Emboss' },
]

const DIGIT_OPTIONS = Array.from({ length: 10 }, (_, i) => ({
  value: String(i),
  label: String(i),
}))

// ── Grid cell component ───────────────────────────────────────────────

function GridCell({
  value,
  size,
  highlighted,
  colorMode,
  showValue,
}: {
  value: number
  size: number
  highlighted?: boolean
  colorMode: 'gray' | 'kernel' | 'output'
  showValue?: boolean
}) {
  let bgColor: string
  let textColor: string

  switch (colorMode) {
    case 'gray': {
      const intensity = Math.min(Math.max(value, 0), 1)
      bgColor = `rgba(255, 255, 255, ${intensity * 0.7 + 0.03})`
      textColor = intensity > 0.5 ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.6)'
      break
    }
    case 'kernel': {
      if (value > 0) {
        bgColor = `rgba(99, 102, 241, ${Math.abs(value) * 0.6 + 0.1})`
      } else if (value < 0) {
        bgColor = `rgba(248, 113, 113, ${Math.abs(value) * 0.6 + 0.1})`
      } else {
        bgColor = 'rgba(255,255,255,0.03)'
      }
      textColor = 'rgba(255,255,255,0.8)'
      break
    }
    case 'output': {
      const absVal = Math.min(Math.abs(value), 4) / 4
      if (value > 0) {
        bgColor = `rgba(52, 211, 153, ${absVal * 0.7 + 0.05})`
      } else if (value < 0) {
        bgColor = `rgba(248, 113, 113, ${absVal * 0.7 + 0.05})`
      } else {
        bgColor = 'rgba(255,255,255,0.03)'
      }
      textColor = 'rgba(255,255,255,0.7)'
      break
    }
  }

  const borderColor = highlighted
    ? COLORS.accent
    : 'rgba(255,255,255,0.08)'
  const borderWidth = highlighted ? 2 : 1

  return (
    <div
      style={{
        width: size,
        height: size,
        backgroundColor: bgColor,
        border: `${borderWidth}px solid ${borderColor}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: size < 36 ? 8 : 10,
        fontFamily: 'monospace',
        color: textColor,
        transition: 'background-color 0.15s ease',
      }}
    >
      {showValue && (typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(1)) : '')}
    </div>
  )
}

// ── Grid renderer ──────────────────────────────────────────────────────

function Grid({
  data,
  cellSize,
  colorMode,
  highlightCells,
  showValues,
  label,
  filledUpTo,
}: {
  data: number[][]
  cellSize: number
  colorMode: 'gray' | 'kernel' | 'output'
  highlightCells?: Set<string>
  showValues?: boolean
  label?: string
  filledUpTo?: { row: number; col: number }
}) {
  return (
    <div>
      {label && (
        <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-2">{label}</p>
      )}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${data[0]?.length || 1}, ${cellSize}px)`,
          gap: 0,
        }}
      >
        {data.map((row, r) =>
          row.map((val, c) => {
            const key = `${r}-${c}`
            const isHighlighted = highlightCells?.has(key) ?? false
            // For output grid, show only cells filled up to current step
            const isFilled = filledUpTo
              ? r < filledUpTo.row || (r === filledUpTo.row && c <= filledUpTo.col)
              : true
            return (
              <GridCell
                key={key}
                value={isFilled ? val : 0}
                size={cellSize}
                highlighted={isHighlighted}
                colorMode={colorMode}
                showValue={showValues && isFilled}
              />
            )
          })
        )}
      </div>
    </div>
  )
}

// ── Main Component ─────────────────────────────────────────────────────

export function CNNViz() {
  const [kernelName, setKernelName] = useState('horizontal_edge')
  const [digit, setDigit] = useState(3)
  const [stride, setStride] = useState(1)

  const inputGrid = useMemo(() => makeDigitGrid(digit), [digit])
  const kernel = useMemo(() => PRESET_KERNELS[kernelName], [kernelName])

  const convResult = useMemo(
    () => convolve2D(inputGrid, kernel, stride),
    [inputGrid, kernel, stride]
  )

  const player = useAlgorithmPlayer({
    snapshots: convResult.steps,
    baseFps: 4,
  })
  const currentStep = player.currentSnapshot

  // Highlighted cells on input grid (kernel position)
  const highlightedInputCells = useMemo(() => {
    const set = new Set<string>()
    if (!currentStep) return set
    const { row, col } = currentStep.position
    const kRows = kernel.length
    const kCols = kernel[0].length
    for (let kr = 0; kr < kRows; kr++) {
      for (let kc = 0; kc < kCols; kc++) {
        set.add(`${row * stride + kr}-${col * stride + kc}`)
      }
    }
    return set
  }, [currentStep, kernel, stride])

  // Highlighted cell on output grid
  const highlightedOutputCells = useMemo(() => {
    const set = new Set<string>()
    if (currentStep) {
      set.add(`${currentStep.position.row}-${currentStep.position.col}`)
    }
    return set
  }, [currentStep])

  // Cell size adapts
  const inputCellSize = 42
  const outputCellSize = 42
  const kernelCellSize = 48

  return (
    <GlassCard className="p-6 lg:p-8">
      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4 mb-6">
        <Select
          label="Digit"
          value={String(digit)}
          options={DIGIT_OPTIONS}
          onChange={(v) => {
            setDigit(parseInt(v, 10))
            player.reset()
          }}
          className="w-24"
        />
        <Select
          label="Kernel"
          value={kernelName}
          options={KERNEL_OPTIONS}
          onChange={(v) => {
            setKernelName(v)
            player.reset()
          }}
          className="w-44"
        />
        <Slider
          label="Stride"
          value={stride}
          min={1}
          max={2}
          step={1}
          onChange={(v) => {
            setStride(v)
            player.reset()
          }}
          className="w-28"
        />
      </div>

      {/* Visualization layout */}
      <div className="flex flex-wrap items-start gap-6 lg:gap-8">
        {/* Input grid */}
        <div className="flex-shrink-0">
          <Grid
            data={inputGrid}
            cellSize={inputCellSize}
            colorMode="gray"
            highlightCells={highlightedInputCells}
            showValues
            label="Input (8x8)"
          />
        </div>

        {/* Convolution symbol */}
        <div className="flex flex-col items-center justify-center self-center">
          <span className="text-2xl text-text-tertiary font-light">*</span>
        </div>

        {/* Kernel */}
        <div className="flex-shrink-0">
          <Grid
            data={kernel}
            cellSize={kernelCellSize}
            colorMode="kernel"
            showValues
            label="Kernel (3x3)"
          />
        </div>

        {/* Equals symbol */}
        <div className="flex flex-col items-center justify-center self-center">
          <span className="text-2xl text-text-tertiary font-light">=</span>
        </div>

        {/* Output grid */}
        <div className="flex-shrink-0">
          <Grid
            data={convResult.outputGrid}
            cellSize={outputCellSize}
            colorMode="output"
            highlightCells={highlightedOutputCells}
            showValues
            label={`Output (${convResult.outputSize.rows}x${convResult.outputSize.cols})`}
            filledUpTo={
              currentStep
                ? { row: currentStep.position.row, col: currentStep.position.col }
                : undefined
            }
          />
        </div>
      </div>

      {/* Step detail panel */}
      {currentStep && (
        <div className="mt-6 bg-obsidian-surface/40 rounded-xl border border-obsidian-border p-4">
          <p className="text-[10px] uppercase tracking-wider text-text-tertiary mb-3">
            Step {player.currentStep + 1}: Position ({currentStep.position.row},{' '}
            {currentStep.position.col})
          </p>
          <div className="flex flex-wrap items-start gap-6">
            {/* Patch */}
            <div>
              <p className="text-[9px] text-text-tertiary mb-1">Input Patch</p>
              <Grid
                data={currentStep.inputPatch}
                cellSize={36}
                colorMode="gray"
                showValues
              />
            </div>

            <div className="self-center text-text-tertiary text-lg font-light">&times;</div>

            {/* Kernel */}
            <div>
              <p className="text-[9px] text-text-tertiary mb-1">Kernel</p>
              <Grid
                data={currentStep.kernel}
                cellSize={36}
                colorMode="kernel"
                showValues
              />
            </div>

            <div className="self-center text-text-tertiary text-lg font-light">=</div>

            {/* Products */}
            <div>
              <p className="text-[9px] text-text-tertiary mb-1">Element-wise Products</p>
              <Grid
                data={currentStep.products}
                cellSize={36}
                colorMode="output"
                showValues
              />
            </div>

            <div className="self-center text-text-tertiary text-lg font-light">&rarr;</div>

            {/* Sum (output value) */}
            <div className="self-center">
              <p className="text-[9px] text-text-tertiary mb-1">Sum</p>
              <div
                className="flex items-center justify-center rounded-lg border border-obsidian-border font-mono text-sm"
                style={{
                  width: 56,
                  height: 56,
                  backgroundColor:
                    currentStep.outputValue > 0
                      ? `rgba(52, 211, 153, ${Math.min(Math.abs(currentStep.outputValue) / 4, 1) * 0.5})`
                      : currentStep.outputValue < 0
                        ? `rgba(248, 113, 113, ${Math.min(Math.abs(currentStep.outputValue) / 4, 1) * 0.5})`
                        : 'rgba(255,255,255,0.03)',
                  color: 'rgba(255,255,255,0.8)',
                }}
              >
                {currentStep.outputValue.toFixed(1)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Transport controls */}
      <div className="mt-4">
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
      </div>
    </GlassCard>
  )
}
