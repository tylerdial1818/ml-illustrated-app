import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'

// ── Colors ──────────────────────────────────────────────────────────
const COLORS = {
  yes: '#4ADE80',
  no: '#F87171',
  bayesian: '#6366F1',
  frequentist: '#FBBF24',
  neutral: '#A1A1AA',
}

interface FlowNode {
  id: string
  question: string
  yes: string
  no: string
  yesResult?: string
  noResult?: string
}

const FLOW: FlowNode[] = [
  {
    id: 'small',
    question: 'Dataset small (< 100 samples)?',
    yes: 'prior',
    no: 'uncertainty',
    yesResult: 'Bayesian helps: priors regularize small samples.',
  },
  {
    id: 'prior',
    question: 'Have genuine prior knowledge?',
    yes: 'done-bayes-1',
    no: 'uncertainty',
    yesResult: 'Bayesian can encode it directly.',
  },
  {
    id: 'uncertainty',
    question: 'Need calibrated uncertainty?',
    yes: 'safety',
    no: 'scale',
  },
  {
    id: 'safety',
    question: 'Downstream decisions depend on confidence? (medical, financial)',
    yes: 'done-bayes-2',
    no: 'scale',
    yesResult: 'Bayesian is built for this.',
  },
  {
    id: 'scale',
    question: 'Large dataset, just need point prediction?',
    yes: 'done-freq',
    no: 'done-bayes-3',
    yesResult: 'Frequentist is faster and sufficient.',
    noResult: 'Consider Bayesian for richer inference.',
  },
]

const TERMINAL: Record<string, { label: string; color: string; desc: string }> = {
  'done-bayes-1': { label: 'Go Bayesian', color: COLORS.bayesian, desc: 'Small data + prior knowledge is the sweet spot for Bayesian methods.' },
  'done-bayes-2': { label: 'Go Bayesian', color: COLORS.bayesian, desc: 'When decisions depend on confidence, Bayesian uncertainty is invaluable.' },
  'done-bayes-3': { label: 'Consider Bayesian', color: COLORS.bayesian, desc: 'Bayesian offers richer inference even without strong prior knowledge.' },
  'done-freq': { label: 'Frequentist OK', color: COLORS.frequentist, desc: 'With lots of data and no need for uncertainty, frequentist methods are fast and effective.' },
}

// ── Comparison table ────────────────────────────────────────────────
const TABLE_ROWS = [
  { category: 'Philosophy', freq: 'Parameters are fixed unknowns', bayes: 'Parameters have probability distributions' },
  { category: 'Parameters', freq: 'Single point estimate (MLE)', bayes: 'Full posterior distribution' },
  { category: 'Predictions', freq: 'Point predictions', bayes: 'Predictive distributions with uncertainty' },
  { category: 'Uncertainty', freq: 'Confidence intervals (frequency interpretation)', bayes: 'Credible intervals (probability interpretation)' },
  { category: 'Small data', freq: 'Prone to overfitting', bayes: 'Prior acts as regularization' },
  { category: 'Large data', freq: 'Efficient, scales well', bayes: 'Prior washes out; more compute for same answer' },
  { category: 'Cost', freq: 'Usually O(n) to O(n²)', bayes: 'O(n²) to O(n³) for exact inference' },
  { category: 'Prior knowledge', freq: 'Cannot encode directly', bayes: 'Priors encode domain expertise' },
]

export function BayesianFlowchart() {
  const [path, setPath] = useState<string[]>(['small'])
  const [expandedRow, setExpandedRow] = useState<string | null>(null)

  const currentId = path[path.length - 1]
  const currentNode = FLOW.find((n) => n.id === currentId)
  const terminal = TERMINAL[currentId]

  const handleAnswer = (answer: 'yes' | 'no') => {
    if (!currentNode) return
    const next = answer === 'yes' ? currentNode.yes : currentNode.no
    setPath((prev) => [...prev, next])
  }

  const handleReset = () => setPath(['small'])

  return (
    <div className="space-y-8">
      {/* Flowchart */}
      <div>
        <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-4">
          When to Go Bayesian
        </h4>
        <GlassCard className="p-5">
          <div className="space-y-3 max-w-lg">
            {/* Previous answers */}
            {path.slice(0, -1).map((nodeId, i) => {
              const node = FLOW.find((n) => n.id === nodeId)
              if (!node) return null
              const answeredYes = path[i + 1] === node.yes
              const result = answeredYes ? node.yesResult : node.noResult
              return (
                <motion.div
                  key={nodeId}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 0.5, x: 0 }}
                  className="flex items-start gap-3"
                >
                  <span
                    className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded mt-0.5"
                    style={{
                      color: answeredYes ? COLORS.yes : COLORS.no,
                      backgroundColor: answeredYes ? `${COLORS.yes}15` : `${COLORS.no}15`,
                    }}
                  >
                    {answeredYes ? 'YES' : 'NO'}
                  </span>
                  <div>
                    <p className="text-xs text-text-tertiary line-through">{node.question}</p>
                    {result && <p className="text-[10px] text-text-tertiary mt-0.5">{result}</p>}
                  </div>
                </motion.div>
              )
            })}

            {/* Current question or terminal */}
            <AnimatePresence mode="wait">
              {terminal ? (
                <motion.div
                  key="terminal"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4"
                >
                  <div
                    className="inline-block px-4 py-2 rounded-lg text-sm font-bold font-mono"
                    style={{ color: terminal.color, backgroundColor: `${terminal.color}18` }}
                  >
                    {terminal.label}
                  </div>
                  <p className="text-sm text-text-secondary mt-2">{terminal.desc}</p>
                  <button onClick={handleReset} className="text-xs text-accent mt-3 hover:underline">
                    Start over
                  </button>
                </motion.div>
              ) : currentNode ? (
                <motion.div
                  key={currentNode.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="mt-2"
                >
                  <p className="text-sm text-text-primary font-medium mb-3">{currentNode.question}</p>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleAnswer('yes')}
                      className="px-4 py-1.5 text-xs font-mono font-bold rounded-lg transition-colors"
                      style={{ color: COLORS.yes, backgroundColor: `${COLORS.yes}15`, border: `1px solid ${COLORS.yes}30` }}
                    >
                      Yes
                    </button>
                    <button
                      onClick={() => handleAnswer('no')}
                      className="px-4 py-1.5 text-xs font-mono font-bold rounded-lg transition-colors"
                      style={{ color: COLORS.no, backgroundColor: `${COLORS.no}15`, border: `1px solid ${COLORS.no}30` }}
                    >
                      No
                    </button>
                  </div>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </div>
        </GlassCard>
      </div>

      {/* Comparison table */}
      <div>
        <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-4">
          Frequentist vs. Bayesian Summary
        </h4>
        <GlassCard className="overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-obsidian-border">
                <th className="text-left p-3 text-[10px] font-mono uppercase tracking-wider text-text-tertiary w-28">Aspect</th>
                <th className="text-left p-3 text-[10px] font-mono uppercase tracking-wider" style={{ color: COLORS.frequentist }}>Frequentist</th>
                <th className="text-left p-3 text-[10px] font-mono uppercase tracking-wider" style={{ color: COLORS.bayesian }}>Bayesian</th>
              </tr>
            </thead>
            <tbody>
              {TABLE_ROWS.map((row) => (
                <tr
                  key={row.category}
                  className="border-b border-obsidian-border/50 cursor-pointer hover:bg-white/[0.02] transition-colors"
                  onClick={() => setExpandedRow(expandedRow === row.category ? null : row.category)}
                >
                  <td className="p-3 text-xs font-medium text-text-secondary">{row.category}</td>
                  <td className="p-3 text-xs text-text-tertiary">{row.freq}</td>
                  <td className="p-3 text-xs text-text-tertiary">{row.bayes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </GlassCard>
      </div>
    </div>
  )
}
