import { useState, useMemo, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Button } from '../../../components/ui/Button'
import { NaiveBayesClassifier, EXAMPLE_EMAILS } from '../../../lib/algorithms/bayesian/naiveBayes'
import type { SpamWordData, ClassificationResult } from '../../../lib/algorithms/bayesian/naiveBayes'
import spamData from '../../../lib/data/bayesian/spam-word-probs.json'

// ── Colors ──────────────────────────────────────────────────────────
const COLORS = {
  spam: '#F472B6',
  notSpam: '#34D399',
  neutral: '#A1A1AA',
  posterior: '#6366F1',
}

type PipelineStep = 0 | 1 | 2 | 3 | 4

// ── Main Component ──────────────────────────────────────────────────
export function NaiveBayesPipeline() {
  const [step, setStep] = useState<PipelineStep>(0)
  const [selectedEmail, setSelectedEmail] = useState<'spam' | 'professional' | 'ambiguous'>('spam')
  const [highlightedWord, setHighlightedWord] = useState<string | null>(null)

  const emailText = EXAMPLE_EMAILS[selectedEmail]

  const classifier = useMemo(
    () => new NaiveBayesClassifier(spamData as SpamWordData),
    []
  )

  const result: ClassificationResult = useMemo(
    () => classifier.classify(emailText),
    [classifier, emailText]
  )

  const tokens = useMemo(
    () => classifier.tokenize(emailText),
    [classifier, emailText]
  )

  const topWords = useMemo(
    () => classifier.getTopWords(20),
    [classifier]
  )

  const handleNext = useCallback(() => {
    setStep((prev) => Math.min(prev + 1, 4) as PipelineStep)
  }, [])

  const handleReset = useCallback(() => {
    setStep(0)
    setHighlightedWord(null)
  }, [])

  const handleEmailChange = useCallback((type: 'spam' | 'professional' | 'ambiguous') => {
    setSelectedEmail(type)
    setStep(0)
    setHighlightedWord(null)
  }, [])

  const stepLabels = ['Show Email', 'Tokenize', 'Word Probabilities', 'Multiply & Prior', 'Final Verdict']

  // Get contribution color for a word
  const getWordColor = (word: string) => {
    const contrib = result.perWordContributions.find((c) => c.word === word)
    if (!contrib) return COLORS.neutral
    if (contrib.contributionDirection === 'spam') return COLORS.spam
    if (contrib.contributionDirection === 'notSpam') return COLORS.notSpam
    return COLORS.neutral
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <GlassCard className="p-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex gap-2">
            {(['spam', 'professional', 'ambiguous'] as const).map((type) => (
              <Button
                key={type}
                variant={selectedEmail === type ? 'primary' : 'secondary'}
                size="sm"
                onClick={() => handleEmailChange(type)}
              >
                {type === 'spam' ? 'Spam-like' : type === 'professional' ? 'Professional' : 'Ambiguous'}
              </Button>
            ))}
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          <Button variant="secondary" size="sm" onClick={handleReset}>Reset</Button>
          <Button variant="primary" size="sm" onClick={handleNext} disabled={step >= 4}>
            {step < 4 ? stepLabels[step + 1] : 'Done'}
          </Button>

          {/* Step indicator */}
          <div className="flex gap-1.5 ml-auto">
            {stepLabels.map((label, i) => (
              <div
                key={label}
                className={`w-2 h-2 rounded-full transition-colors ${
                  i <= step ? 'bg-accent' : 'bg-obsidian-hover'
                }`}
                title={label}
              />
            ))}
          </div>
        </div>
      </GlassCard>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Pipeline (2 cols) */}
        <GlassCard className="lg:col-span-2 p-5 space-y-5">
          {/* Step 0: Show email */}
          <div>
            <p className="text-[10px] font-mono uppercase tracking-wider text-text-tertiary mb-2">Input Email</p>
            <div className="bg-obsidian-surface rounded-lg p-4 text-sm text-text-secondary leading-relaxed">
              {step >= 1 ? (
                <span className="flex flex-wrap gap-1">
                  {tokens.map((word, i) => {
                    const color = step >= 2 ? getWordColor(word) : COLORS.neutral
                    const isHighlighted = highlightedWord === word
                    return (
                      <motion.span
                        key={i}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.02, duration: 0.15 }}
                        className="px-1.5 py-0.5 rounded text-xs font-mono cursor-pointer transition-all"
                        style={{
                          backgroundColor: `${color}${isHighlighted ? '40' : '15'}`,
                          color: color,
                          border: `1px solid ${color}${isHighlighted ? '60' : '20'}`,
                        }}
                        onMouseEnter={() => setHighlightedWord(word)}
                        onMouseLeave={() => setHighlightedWord(null)}
                      >
                        {word}
                      </motion.span>
                    )
                  })}
                </span>
              ) : (
                emailText
              )}
            </div>
          </div>

          {/* Step 2: Word probability bars */}
          <AnimatePresence>
            {step >= 2 && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <p className="text-[10px] font-mono uppercase tracking-wider text-text-tertiary mb-2">Word Likelihoods</p>
                <div className="space-y-1.5 max-h-52 overflow-y-auto pr-2">
                  {result.perWordContributions
                    .filter((c, i, arr) => arr.findIndex((a) => a.word === c.word) === i)
                    .slice(0, 15)
                    .map((contrib, i) => {
                    const maxLog = Math.max(
                      Math.abs(contrib.logLikelihoodSpam),
                      Math.abs(contrib.logLikelihoodNotSpam)
                    )
                    const spamWidth = maxLog > 0 ? (Math.abs(contrib.logLikelihoodSpam) / 8) * 100 : 0
                    const notSpamWidth = maxLog > 0 ? (Math.abs(contrib.logLikelihoodNotSpam) / 8) * 100 : 0
                    const isHighlighted = highlightedWord === contrib.word

                    return (
                      <motion.div
                        key={`${contrib.word}-${i}`}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.03 }}
                        className={`flex items-center gap-2 py-0.5 px-1 rounded transition-colors ${isHighlighted ? 'bg-white/5' : ''}`}
                        onMouseEnter={() => setHighlightedWord(contrib.word)}
                        onMouseLeave={() => setHighlightedWord(null)}
                      >
                        <span className="text-[10px] font-mono text-text-tertiary w-20 text-right truncate">{contrib.word}</span>
                        <div className="flex-1 flex gap-1">
                          <div className="flex-1 flex justify-end">
                            <div
                              className="h-3 rounded-l"
                              style={{
                                width: `${Math.min(spamWidth, 100)}%`,
                                backgroundColor: COLORS.spam,
                                opacity: isHighlighted ? 0.9 : 0.5,
                              }}
                            />
                          </div>
                          <div className="flex-1">
                            <div
                              className="h-3 rounded-r"
                              style={{
                                width: `${Math.min(notSpamWidth, 100)}%`,
                                backgroundColor: COLORS.notSpam,
                                opacity: isHighlighted ? 0.9 : 0.5,
                              }}
                            />
                          </div>
                        </div>
                      </motion.div>
                    )
                  })}
                </div>
                <div className="flex justify-between text-[9px] font-mono text-text-tertiary mt-1 px-1">
                  <span style={{ color: COLORS.spam }}>P(word|spam)</span>
                  <span style={{ color: COLORS.notSpam }}>P(word|not spam)</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Step 3: Prior × Likelihood */}
          <AnimatePresence>
            {step >= 3 && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <p className="text-[10px] font-mono uppercase tracking-wider text-text-tertiary mb-2">Log Scores</p>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-obsidian-surface rounded-lg p-3">
                    <p className="text-[9px] font-mono text-text-tertiary mb-1">log P(spam) + Σ log P(wᵢ|spam)</p>
                    <p className="text-lg font-mono font-bold" style={{ color: COLORS.spam }}>
                      {result.logPosteriorSpam.toFixed(2)}
                    </p>
                  </div>
                  <div className="bg-obsidian-surface rounded-lg p-3">
                    <p className="text-[9px] font-mono text-text-tertiary mb-1">log P(¬spam) + Σ log P(wᵢ|¬spam)</p>
                    <p className="text-lg font-mono font-bold" style={{ color: COLORS.notSpam }}>
                      {result.logPosteriorNotSpam.toFixed(2)}
                    </p>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Step 4: Final verdict */}
          <AnimatePresence>
            {step >= 4 && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <p className="text-[10px] font-mono uppercase tracking-wider text-text-tertiary mb-2">Posterior</p>
                <div className="space-y-2">
                  <div className="h-6 rounded-full overflow-hidden flex">
                    <motion.div
                      className="h-full"
                      style={{ backgroundColor: COLORS.spam }}
                      initial={{ width: '50%' }}
                      animate={{ width: `${result.posteriorSpam * 100}%` }}
                      transition={{ duration: 0.6, ease: 'easeOut' }}
                    />
                    <motion.div
                      className="h-full"
                      style={{ backgroundColor: COLORS.notSpam }}
                      initial={{ width: '50%' }}
                      animate={{ width: `${result.posteriorNotSpam * 100}%` }}
                      transition={{ duration: 0.6, ease: 'easeOut' }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] font-mono">
                    <span style={{ color: COLORS.spam }}>Spam: {(result.posteriorSpam * 100).toFixed(1)}%</span>
                    <span style={{ color: COLORS.notSpam }}>Not spam: {(result.posteriorNotSpam * 100).toFixed(1)}%</span>
                  </div>
                  <div className="text-center">
                    <span
                      className="text-sm font-bold font-mono px-3 py-1 rounded"
                      style={{
                        color: result.prediction === 'spam' ? COLORS.spam : COLORS.notSpam,
                        backgroundColor: result.prediction === 'spam' ? `${COLORS.spam}20` : `${COLORS.notSpam}20`,
                      }}
                    >
                      {result.prediction.toUpperCase()}
                    </span>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </GlassCard>

        {/* Feature probability explorer */}
        <GlassCard className="p-4">
          <p className="text-[10px] font-mono uppercase tracking-wider text-text-tertiary mb-3">Top Discriminative Words</p>
          <div className="space-y-1.5 max-h-[420px] overflow-y-auto">
            {topWords.map((entry) => {
              const isHighlighted = highlightedWord === entry.word
              const isSpammy = entry.logRatio > 0
              const barWidth = Math.min(Math.abs(entry.logRatio) / 6 * 100, 100)
              return (
                <div
                  key={entry.word}
                  className={`flex items-center gap-2 py-0.5 px-1 rounded cursor-pointer transition-colors ${isHighlighted ? 'bg-white/5' : ''}`}
                  onMouseEnter={() => setHighlightedWord(entry.word)}
                  onMouseLeave={() => setHighlightedWord(null)}
                >
                  <span className="text-[10px] font-mono text-text-tertiary w-20 text-right truncate">{entry.word}</span>
                  <div className="flex-1 relative h-3">
                    <div
                      className="absolute h-full rounded"
                      style={{
                        width: `${barWidth}%`,
                        backgroundColor: isSpammy ? COLORS.spam : COLORS.notSpam,
                        opacity: isHighlighted ? 0.8 : 0.4,
                        left: isSpammy ? undefined : undefined,
                      }}
                    />
                  </div>
                  <span className="text-[8px] font-mono text-text-tertiary w-10">
                    {entry.logRatio > 0 ? '+' : ''}{entry.logRatio.toFixed(1)}
                  </span>
                </div>
              )
            })}
          </div>
          <p className="text-[9px] text-text-tertiary mt-2">
            Log ratio: log(P(w|spam) / P(w|not spam)). Positive = spammy, negative = professional.
          </p>
        </GlassCard>
      </div>
    </div>
  )
}
