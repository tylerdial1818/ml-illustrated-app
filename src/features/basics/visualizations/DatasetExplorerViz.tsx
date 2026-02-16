import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Select } from '../../../components/ui/Select'
import { Button } from '../../../components/ui/Button'

// ── Dataset definitions ───────────────────────────────────────────────
interface Column {
  name: string
  type: 'numerical-continuous' | 'numerical-discrete' | 'categorical'
  isLabel: boolean
}

interface Dataset {
  name: string
  columns: Column[]
  rows: (string | number)[][]
}

const DATASETS: Record<string, Dataset> = {
  housing: {
    name: 'Housing',
    columns: [
      { name: 'Square Feet', type: 'numerical-continuous', isLabel: false },
      { name: 'Bedrooms', type: 'numerical-discrete', isLabel: false },
      { name: 'Year Built', type: 'numerical-discrete', isLabel: false },
      { name: 'Neighborhood', type: 'categorical', isLabel: false },
      { name: 'Price', type: 'numerical-continuous', isLabel: true },
    ],
    rows: [
      [1400, 3, 1995, 'Downtown', '$285,000'],
      [2100, 4, 2010, 'Suburbs', '$420,000'],
      [950, 2, 1980, 'Midtown', '$195,000'],
      [1800, 3, 2005, 'Suburbs', '$375,000'],
      [3200, 5, 2018, 'Lakefront', '$680,000'],
      [1100, 2, 1990, 'Downtown', '$240,000'],
      [1550, 3, 2000, 'Midtown', '$310,000'],
      [2400, 4, 2015, 'Suburbs', '$485,000'],
      [800, 1, 1975, 'Downtown', '$165,000'],
      [2000, 3, 2012, 'Lakefront', '$520,000'],
    ],
  },
  iris: {
    name: 'Iris Flowers',
    columns: [
      { name: 'Sepal Length', type: 'numerical-continuous', isLabel: false },
      { name: 'Sepal Width', type: 'numerical-continuous', isLabel: false },
      { name: 'Petal Length', type: 'numerical-continuous', isLabel: false },
      { name: 'Petal Width', type: 'numerical-continuous', isLabel: false },
      { name: 'Species', type: 'categorical', isLabel: true },
    ],
    rows: [
      [5.1, 3.5, 1.4, 0.2, 'Setosa'],
      [4.9, 3.0, 1.4, 0.2, 'Setosa'],
      [7.0, 3.2, 4.7, 1.4, 'Versicolor'],
      [6.4, 3.2, 4.5, 1.5, 'Versicolor'],
      [5.9, 3.0, 5.1, 1.8, 'Virginica'],
      [6.3, 3.3, 6.0, 2.5, 'Virginica'],
      [5.8, 2.7, 5.1, 1.9, 'Virginica'],
      [5.0, 3.4, 1.5, 0.2, 'Setosa'],
      [6.7, 3.1, 4.4, 1.4, 'Versicolor'],
      [6.1, 2.8, 4.7, 1.2, 'Versicolor'],
    ],
  },
  students: {
    name: 'Student Grades',
    columns: [
      { name: 'Hours Studied', type: 'numerical-continuous', isLabel: false },
      { name: 'Assignments', type: 'numerical-discrete', isLabel: false },
      { name: 'Major', type: 'categorical', isLabel: false },
      { name: 'Grade', type: 'categorical', isLabel: true },
    ],
    rows: [
      [12.5, 8, 'CS', 'A'],
      [8.0, 6, 'Math', 'B'],
      [3.5, 3, 'CS', 'D'],
      [10.0, 7, 'Physics', 'B+'],
      [15.0, 9, 'Math', 'A'],
      [6.0, 5, 'CS', 'C'],
      [9.5, 7, 'Physics', 'B'],
      [2.0, 2, 'Math', 'F'],
      [11.0, 8, 'CS', 'A-'],
      [7.0, 6, 'Physics', 'C+'],
    ],
  },
}

const DATASET_OPTIONS = [
  { value: 'housing', label: 'Housing' },
  { value: 'iris', label: 'Iris Flowers' },
  { value: 'students', label: 'Student Grades' },
]

type HighlightMode = 'none' | 'features' | 'label' | 'example'

const HIGHLIGHT_OPTIONS: { value: HighlightMode; label: string }[] = [
  { value: 'none', label: 'None' },
  { value: 'features', label: 'Highlight features' },
  { value: 'label', label: 'Highlight label' },
  { value: 'example', label: 'Highlight one example' },
]

// ── Feature type colors ───────────────────────────────────────────────
const FEATURE_COLOR = '#34D399'
const LABEL_COLOR = '#6366F1'
const EXAMPLE_COLOR = '#F472B6'

// ── Feature Types Mini Cards ──────────────────────────────────────────
function FeatureTypeCards({ dataset }: { dataset: Dataset }) {
  const types = {
    'numerical-continuous': {
      label: 'Numerical (continuous)',
      desc: 'Numbers on a continuous scale.',
      color: '#6366F1',
    },
    'numerical-discrete': {
      label: 'Numerical (discrete)',
      desc: 'Countable numbers.',
      color: '#F472B6',
    },
    categorical: {
      label: 'Categorical',
      desc: 'Named categories, not numbers. Must be encoded before most models can use them.',
      color: '#FBBF24',
    },
  }

  // Only show types present in this dataset
  const presentTypes = [...new Set(dataset.columns.map((c) => c.type))]
  const examples: Record<string, string> = {}
  for (const col of dataset.columns) {
    if (!examples[col.type]) examples[col.type] = col.name
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-6">
      {presentTypes.map((type) => {
        const info = types[type]
        return (
          <GlassCard key={type} className="p-4">
            <div className="flex items-center gap-2 mb-1.5">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: info.color }} />
              <span className="text-xs font-medium text-text-primary">{info.label}</span>
            </div>
            <p className="text-[10px] text-text-tertiary leading-relaxed">{info.desc}</p>
            <p className="text-[10px] text-text-secondary mt-1 font-mono">
              e.g. &quot;{examples[type]}&quot;
            </p>
          </GlassCard>
        )
      })}
    </div>
  )
}

// ── Dataset Split Preview ─────────────────────────────────────────────
function DatasetSplitPreview({
  dataset,
  showSplit,
}: {
  dataset: Dataset
  showSplit: boolean
}) {
  const trainCount = Math.round(dataset.rows.length * 0.7)
  const testCount = dataset.rows.length - trainCount

  return (
    <div className="mt-6">
      <div className="flex items-center gap-3 mb-3">
        <p className="text-[10px] uppercase tracking-wider text-text-tertiary">
          Train / Test Split
        </p>
      </div>

      <div className="flex items-center gap-4">
        {/* Training set */}
        <div className="flex-1">
          <motion.div
            className="rounded-lg border p-3"
            style={{
              backgroundColor: showSplit ? '#34D39910' : 'rgba(255,255,255,0.02)',
              borderColor: showSplit ? '#34D39930' : 'rgba(255,255,255,0.06)',
            }}
            animate={{
              backgroundColor: showSplit ? '#34D39910' : 'rgba(255,255,255,0.02)',
            }}
            transition={{ duration: 0.4 }}
          >
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#34D399' }} />
              <span className="text-[10px] font-medium text-text-secondary">
                Training Set ({Math.round((trainCount / dataset.rows.length) * 100)}%)
              </span>
            </div>
            <div className="flex flex-wrap gap-1">
              {dataset.rows.slice(0, trainCount).map((_, i) => (
                <motion.div
                  key={`train-${i}`}
                  className="w-3 h-3 rounded-sm"
                  style={{ backgroundColor: showSplit ? '#34D39940' : '#A1A1AA20' }}
                  initial={false}
                  animate={{
                    x: showSplit ? 0 : 0,
                    backgroundColor: showSplit ? '#34D39940' : '#A1A1AA20',
                  }}
                  transition={{ duration: 0.3, delay: i * 0.03 }}
                />
              ))}
            </div>
          </motion.div>
        </div>

        {/* Divider */}
        <motion.div
          className="text-text-tertiary text-xs font-mono"
          animate={{ opacity: showSplit ? 1 : 0.3 }}
        >
          |
        </motion.div>

        {/* Test set */}
        <div className="flex-shrink-0">
          <motion.div
            className="rounded-lg border p-3"
            style={{
              backgroundColor: showSplit ? '#F472B610' : 'rgba(255,255,255,0.02)',
              borderColor: showSplit ? '#F472B630' : 'rgba(255,255,255,0.06)',
            }}
            animate={{
              backgroundColor: showSplit ? '#F472B610' : 'rgba(255,255,255,0.02)',
            }}
            transition={{ duration: 0.4 }}
          >
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#F472B6' }} />
              <span className="text-[10px] font-medium text-text-secondary">
                Test Set ({Math.round((testCount / dataset.rows.length) * 100)}%)
              </span>
            </div>
            <div className="flex flex-wrap gap-1">
              {dataset.rows.slice(trainCount).map((_, i) => (
                <motion.div
                  key={`test-${i}`}
                  className="w-3 h-3 rounded-sm"
                  style={{ backgroundColor: showSplit ? '#F472B640' : '#A1A1AA20' }}
                  animate={{
                    backgroundColor: showSplit ? '#F472B640' : '#A1A1AA20',
                  }}
                  transition={{ duration: 0.3, delay: 0.2 + i * 0.03 }}
                />
              ))}
            </div>
          </motion.div>
        </div>
      </div>

      <AnimatePresence>
        {showSplit && (
          <motion.p
            className="text-[10px] text-text-tertiary mt-2 leading-relaxed"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            We don't use all our data for training. We hold some back to test whether the model actually
            learned generalizable patterns. More on this in the Overfitting section below.
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────
export function DatasetExplorerViz() {
  const [datasetKey, setDatasetKey] = useState('housing')
  const [highlight, setHighlight] = useState<HighlightMode>('none')
  const [hoveredRow, setHoveredRow] = useState<number | null>(null)
  const [hoveredCol, setHoveredCol] = useState<number | null>(null)
  const [showSplit, setShowSplit] = useState(false)

  const dataset = DATASETS[datasetKey]
  const exampleRow = 2

  const handleToggleSplit = useCallback(() => setShowSplit((p) => !p), [])

  // Determine cell highlight
  const getCellStyle = (rowIdx: number, colIdx: number) => {
    const col = dataset.columns[colIdx]
    const isHoveredRow = hoveredRow === rowIdx
    const isHoveredCol = hoveredCol === colIdx

    let bg = 'transparent'
    let textColor = '#A1A1AA'

    // Highlight modes
    if (highlight === 'features' && !col.isLabel) {
      bg = `${FEATURE_COLOR}10`
      textColor = FEATURE_COLOR
    } else if (highlight === 'label' && col.isLabel) {
      bg = `${LABEL_COLOR}15`
      textColor = LABEL_COLOR
    } else if (highlight === 'example' && rowIdx === exampleRow) {
      bg = `${EXAMPLE_COLOR}10`
      textColor = EXAMPLE_COLOR
    }

    // Hover overrides
    if (isHoveredRow) {
      bg = `${EXAMPLE_COLOR}08`
    }
    if (isHoveredCol) {
      bg = col.isLabel ? `${LABEL_COLOR}12` : `${FEATURE_COLOR}08`
    }

    return { backgroundColor: bg, color: textColor }
  }

  const getHeaderStyle = (colIdx: number) => {
    const col = dataset.columns[colIdx]
    const isHoveredCol = hoveredCol === colIdx

    if (col.isLabel) {
      return {
        backgroundColor: isHoveredCol ? `${LABEL_COLOR}25` : `${LABEL_COLOR}12`,
        color: LABEL_COLOR,
        borderColor: `${LABEL_COLOR}30`,
      }
    }
    return {
      backgroundColor: isHoveredCol ? `${FEATURE_COLOR}20` : `${FEATURE_COLOR}08`,
      color: highlight === 'features' ? FEATURE_COLOR : '#E4E4E7',
      borderColor: `${FEATURE_COLOR}20`,
    }
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-4">
          <Select
            label="Dataset"
            value={datasetKey}
            options={DATASET_OPTIONS}
            onChange={(v) => {
              setDatasetKey(v)
              setHoveredRow(null)
              setHoveredCol(null)
            }}
            className="w-44"
          />

          <div className="h-6 w-px bg-obsidian-border" />

          <Select
            label="Highlight"
            value={highlight}
            options={HIGHLIGHT_OPTIONS}
            onChange={(v) => setHighlight(v as HighlightMode)}
            className="w-48"
          />
        </div>
      </GlassCard>

      {/* Data Table */}
      <GlassCard className="p-5 overflow-x-auto">
        {/* Bracket annotations */}
        <div className="flex items-center gap-3 mb-3">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: FEATURE_COLOR }} />
            <span className="text-[10px] font-mono text-text-secondary">
              Features (d={dataset.columns.filter((c) => !c.isLabel).length})
            </span>
          </div>
          <div className="h-4 w-px bg-obsidian-border" />
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: LABEL_COLOR }} />
            <span className="text-[10px] font-mono text-text-secondary">
              Label (what we predict)
            </span>
          </div>
          <div className="h-4 w-px bg-obsidian-border" />
          <span className="text-[10px] font-mono text-text-tertiary">
            Training Examples (n={dataset.rows.length})
          </span>
        </div>

        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="w-8 py-2 px-1 text-[9px] font-mono text-text-tertiary text-center">#</th>
              {dataset.columns.map((col, ci) => (
                <th
                  key={col.name}
                  className="py-2 px-3 text-left text-[10px] font-mono font-medium rounded-t-md cursor-pointer transition-colors border-b"
                  style={getHeaderStyle(ci)}
                  onMouseEnter={() => setHoveredCol(ci)}
                  onMouseLeave={() => setHoveredCol(null)}
                >
                  {col.name}
                  {col.isLabel && (
                    <span className="ml-1 text-[8px] opacity-60">(label)</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {dataset.rows.map((row, ri) => (
              <motion.tr
                key={ri}
                className="cursor-pointer transition-colors"
                onMouseEnter={() => setHoveredRow(ri)}
                onMouseLeave={() => setHoveredRow(null)}
                initial={{ opacity: 0, x: -5 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: ri * 0.03, duration: 0.3 }}
              >
                <td className="py-1.5 px-1 text-[9px] font-mono text-text-tertiary text-center">
                  {ri + 1}
                </td>
                {row.map((val, ci) => (
                  <td
                    key={ci}
                    className="py-1.5 px-3 text-xs font-mono transition-colors rounded-sm"
                    style={getCellStyle(ri, ci)}
                  >
                    {val}
                  </td>
                ))}
              </motion.tr>
            ))}
          </tbody>
        </table>

        {/* Hover tooltip */}
        <AnimatePresence>
          {hoveredCol !== null && (
            <motion.div
              className="mt-2 text-[10px] text-text-secondary"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <span className="font-mono font-medium" style={{ color: dataset.columns[hoveredCol].isLabel ? LABEL_COLOR : FEATURE_COLOR }}>
                {dataset.columns[hoveredCol].name}
              </span>
              {' '}is {dataset.columns[hoveredCol].isLabel ? 'the label (what we predict)' : `a feature (${dataset.columns[hoveredCol].type})`}
            </motion.div>
          )}
          {hoveredRow !== null && hoveredCol === null && (
            <motion.div
              className="mt-2 text-[10px] text-text-secondary"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              Row {hoveredRow + 1}: one training example
            </motion.div>
          )}
        </AnimatePresence>
      </GlassCard>

      {/* Feature Types */}
      <FeatureTypeCards dataset={dataset} />

      {/* Dataset Split */}
      <GlassCard className="p-4">
        <div className="flex items-center gap-3 mb-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={handleToggleSplit}
            active={showSplit}
          >
            {showSplit ? 'Hide Split' : 'Show Train/Test Split'}
          </Button>
        </div>
        <DatasetSplitPreview dataset={dataset} showSplit={showSplit} />
      </GlassCard>
    </div>
  )
}
