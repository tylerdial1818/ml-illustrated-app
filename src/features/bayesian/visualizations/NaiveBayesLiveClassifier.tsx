import { useState, useMemo, useCallback } from 'react'
import { motion } from 'framer-motion'
import { GlassCard } from '../../../components/ui/GlassCard'
import { Slider } from '../../../components/ui/Slider'
import { Toggle } from '../../../components/ui/Toggle'
import { Button } from '../../../components/ui/Button'
import { NaiveBayesClassifier, EXAMPLE_EMAILS } from '../../../lib/algorithms/bayesian/naiveBayes'
import type { SpamWordData } from '../../../lib/algorithms/bayesian/naiveBayes'
import spamData from '../../../lib/data/bayesian/spam-word-probs.json'

// ── Colors ──────────────────────────────────────────────────────────
const COLORS = {
  spam: '#F472B6',
  notSpam: '#34D399',
  neutral: '#A1A1AA',
}

export function NaiveBayesLiveClassifier() {
  const [text, setText] = useState(EXAMPLE_EMAILS.spam)
  const [priorSpam, setPriorSpam] = useState(35)
  const [smoothing, setSmoothing] = useState(1.0)
  const [showContribs, setShowContribs] = useState(true)
  const [showLogProbs, setShowLogProbs] = useState(false)

  const classifier = useMemo(() => {
    const data = { ...spamData, priorSpam: priorSpam / 100, priorNotSpam: 1 - priorSpam / 100 } as SpamWordData
    const c = new NaiveBayesClassifier(data, smoothing)
    return c
  }, [priorSpam, smoothing])

  const result = useMemo(
    () => classifier.classify(text),
    [classifier, text]
  )

  const tokens = useMemo(
    () => classifier.tokenize(text),
    [classifier, text]
  )

  const handleExample = useCallback((type: keyof typeof EXAMPLE_EMAILS) => {
    setText(EXAMPLE_EMAILS[type])
  }, [])

  // Tug-of-war position: 0 = fully not spam, 1 = fully spam
  const tugPosition = result.posteriorSpam

  return (
    <div className="space-y-4">
      {/* Text input + controls */}
      <GlassCard className="p-4 space-y-4">
        <div>
          <label className="text-[10px] font-mono uppercase tracking-wider text-text-tertiary mb-2 block">Type or edit email text</label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full h-24 bg-obsidian-surface border border-obsidian-border rounded-lg p-3 text-sm text-text-primary resize-none focus:outline-none focus:ring-2 focus:ring-accent/50 focus:border-accent/50"
            placeholder="Type an email here..."
          />
        </div>

        <div className="flex flex-wrap items-end gap-4">
          <div className="flex gap-2">
            <Button variant="secondary" size="sm" onClick={() => handleExample('spam')}>Spam</Button>
            <Button variant="secondary" size="sm" onClick={() => handleExample('professional')}>Professional</Button>
            <Button variant="secondary" size="sm" onClick={() => handleExample('ambiguous')}>Ambiguous</Button>
          </div>

          <div className="h-6 w-px bg-obsidian-border" />

          <Slider label="P(spam) prior" value={priorSpam} min={10} max={90} step={5} onChange={setPriorSpam} formatValue={(v) => `${v}%`} className="w-36" />
          <Slider label="Smoothing α" value={smoothing} min={0} max={2} step={0.1} onChange={setSmoothing} formatValue={(v) => v.toFixed(1)} className="w-28" />

          <div className="h-6 w-px bg-obsidian-border" />

          <Toggle label="Contributions" checked={showContribs} onChange={setShowContribs} />
          <Toggle label="Log probs" checked={showLogProbs} onChange={setShowLogProbs} />
        </div>
      </GlassCard>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Tug-of-war + classification */}
        <GlassCard className="lg:col-span-2 p-5 space-y-4">
          {/* Tug of war bar */}
          <div>
            <p className="text-[10px] font-mono uppercase tracking-wider text-text-tertiary mb-2">Classification Tug-of-War</p>
            <div className="relative h-8 rounded-full overflow-hidden bg-obsidian-surface">
              <motion.div
                className="absolute inset-y-0 left-0"
                style={{ backgroundColor: COLORS.spam }}
                animate={{ width: `${tugPosition * 100}%` }}
                transition={{ type: 'spring', stiffness: 200, damping: 25 }}
              />
              <motion.div
                className="absolute inset-y-0 right-0"
                style={{ backgroundColor: COLORS.notSpam }}
                animate={{ width: `${(1 - tugPosition) * 100}%` }}
                transition={{ type: 'spring', stiffness: 200, damping: 25 }}
              />
              {/* Indicator */}
              <motion.div
                className="absolute top-0 bottom-0 w-0.5 bg-white"
                animate={{ left: `${tugPosition * 100}%` }}
                transition={{ type: 'spring', stiffness: 200, damping: 25 }}
              />
            </div>
            <div className="flex justify-between text-[10px] font-mono mt-1">
              <span style={{ color: COLORS.spam }}>Spam {(result.posteriorSpam * 100).toFixed(1)}%</span>
              <span
                className="font-bold px-2 py-0.5 rounded"
                style={{
                  color: result.prediction === 'spam' ? COLORS.spam : COLORS.notSpam,
                  backgroundColor: result.prediction === 'spam' ? `${COLORS.spam}20` : `${COLORS.notSpam}20`,
                }}
              >
                {result.prediction.toUpperCase()}
              </span>
              <span style={{ color: COLORS.notSpam }}>Not Spam {(result.posteriorNotSpam * 100).toFixed(1)}%</span>
            </div>
          </div>

          {/* Word pills with contribution coloring */}
          {showContribs && tokens.length > 0 && (
            <div>
              <p className="text-[10px] font-mono uppercase tracking-wider text-text-tertiary mb-2">Per-Word Coloring</p>
              <div className="flex flex-wrap gap-1">
                {result.perWordContributions.map((contrib, i) => {
                  const color =
                    contrib.contributionDirection === 'spam' ? COLORS.spam :
                    contrib.contributionDirection === 'notSpam' ? COLORS.notSpam :
                    COLORS.neutral
                  return (
                    <span
                      key={i}
                      className="px-1.5 py-0.5 rounded text-xs font-mono"
                      style={{
                        color,
                        backgroundColor: `${color}15`,
                        border: `1px solid ${color}30`,
                      }}
                      title={
                        showLogProbs
                          ? `log P(w|spam)=${contrib.logLikelihoodSpam.toFixed(2)}, log P(w|¬spam)=${contrib.logLikelihoodNotSpam.toFixed(2)}`
                          : `${contrib.contributionDirection}`
                      }
                    >
                      {contrib.word}
                      {showLogProbs && (
                        <span className="text-[8px] ml-0.5 opacity-60">
                          {(contrib.logLikelihoodSpam - contrib.logLikelihoodNotSpam).toFixed(1)}
                        </span>
                      )}
                    </span>
                  )
                })}
              </div>
            </div>
          )}

          {/* Smoothing demo */}
          {smoothing === 0 && (
            <div className="bg-error/10 border border-error/20 rounded-lg p-3 text-xs text-error">
              Warning: With α=0, any unseen word zeroes out the entire class probability. Try typing an unusual word to see the zero-probability catastrophe.
            </div>
          )}

          {/* Log scores */}
          {showLogProbs && (
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-obsidian-surface rounded-lg p-3">
                <p className="text-[9px] font-mono text-text-tertiary mb-1">Log score (spam)</p>
                <p className="text-sm font-mono font-bold" style={{ color: COLORS.spam }}>{result.logPosteriorSpam.toFixed(3)}</p>
              </div>
              <div className="bg-obsidian-surface rounded-lg p-3">
                <p className="text-[9px] font-mono text-text-tertiary mb-1">Log score (not spam)</p>
                <p className="text-sm font-mono font-bold" style={{ color: COLORS.notSpam }}>{result.logPosteriorNotSpam.toFixed(3)}</p>
              </div>
            </div>
          )}
        </GlassCard>

        {/* Stats panel */}
        <GlassCard className="p-4 space-y-4">
          <div>
            <p className="text-[10px] font-mono uppercase tracking-wider text-text-tertiary mb-2">Stats</p>
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-text-tertiary">Words</span>
                <span className="font-mono text-text-secondary">{tokens.length}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-text-tertiary">Unique</span>
                <span className="font-mono text-text-secondary">{new Set(tokens).size}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-text-tertiary">Spam prior</span>
                <span className="font-mono text-text-secondary">{priorSpam}%</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-text-tertiary">Smoothing</span>
                <span className="font-mono text-text-secondary">α = {smoothing.toFixed(1)}</span>
              </div>
            </div>
          </div>

          <div>
            <p className="text-[10px] font-mono uppercase tracking-wider text-text-tertiary mb-2">Top Spam Signals</p>
            <div className="space-y-1">
              {result.perWordContributions
                .filter((c) => c.contributionDirection === 'spam')
                .sort((a, b) => (b.logLikelihoodSpam - b.logLikelihoodNotSpam) - (a.logLikelihoodSpam - a.logLikelihoodNotSpam))
                .slice(0, 5)
                .map((c, i) => (
                  <div key={i} className="flex justify-between text-[10px]">
                    <span className="font-mono" style={{ color: COLORS.spam }}>{c.word}</span>
                    <span className="font-mono text-text-tertiary">+{(c.logLikelihoodSpam - c.logLikelihoodNotSpam).toFixed(2)}</span>
                  </div>
                ))
              }
            </div>
          </div>

          <div>
            <p className="text-[10px] font-mono uppercase tracking-wider text-text-tertiary mb-2">Top Not-Spam Signals</p>
            <div className="space-y-1">
              {result.perWordContributions
                .filter((c) => c.contributionDirection === 'notSpam')
                .sort((a, b) => (a.logLikelihoodSpam - a.logLikelihoodNotSpam) - (b.logLikelihoodSpam - b.logLikelihoodNotSpam))
                .slice(0, 5)
                .map((c, i) => (
                  <div key={i} className="flex justify-between text-[10px]">
                    <span className="font-mono" style={{ color: COLORS.notSpam }}>{c.word}</span>
                    <span className="font-mono text-text-tertiary">{(c.logLikelihoodSpam - c.logLikelihoodNotSpam).toFixed(2)}</span>
                  </div>
                ))
              }
            </div>
          </div>
        </GlassCard>
      </div>
    </div>
  )
}
