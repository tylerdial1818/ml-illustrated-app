// ── Naive Bayes Classifier ────────────────────────────────────────────

export interface WordProbs {
  spam: number
  notSpam: number
}

export interface SpamWordData {
  priorSpam: number
  priorNotSpam: number
  words: Record<string, WordProbs>
}

export interface WordContribution {
  word: string
  logLikelihoodSpam: number
  logLikelihoodNotSpam: number
  contributionDirection: 'spam' | 'notSpam' | 'neutral'
}

export interface ClassificationResult {
  prediction: 'spam' | 'not spam'
  posteriorSpam: number
  posteriorNotSpam: number
  logPosteriorSpam: number
  logPosteriorNotSpam: number
  perWordContributions: WordContribution[]
}

export class NaiveBayesClassifier {
  wordProbs: Map<string, WordProbs>
  priorSpam: number
  smoothingAlpha: number
  vocabSize: number

  constructor(data: SpamWordData, smoothingAlpha = 1.0) {
    this.wordProbs = new Map(Object.entries(data.words))
    this.priorSpam = data.priorSpam
    this.smoothingAlpha = smoothingAlpha
    this.vocabSize = this.wordProbs.size
  }

  tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter((w) => w.length > 0)
  }

  getWordProbability(word: string, isSpam: boolean): number {
    const probs = this.wordProbs.get(word)

    if (probs) {
      const raw = isSpam ? probs.spam : probs.notSpam
      // Apply Laplace smoothing
      if (this.smoothingAlpha > 0) {
        // Smoothed: (raw * vocabSize + alpha) / (vocabSize + alpha * vocabSize)
        // Simplified: we treat raw as already normalized, just blend with uniform
        const uniform = 1 / this.vocabSize
        const smoothed =
          (raw + this.smoothingAlpha * uniform) / (1 + this.smoothingAlpha)
        return smoothed
      }
      return raw
    }

    // Unknown word
    if (this.smoothingAlpha > 0) {
      return this.smoothingAlpha / (this.vocabSize + this.smoothingAlpha * this.vocabSize)
    }
    return 0
  }

  classify(text: string): ClassificationResult {
    const tokens = this.tokenize(text)

    let logSpam = Math.log(this.priorSpam)
    let logNotSpam = Math.log(1 - this.priorSpam)

    const contributions: WordContribution[] = []

    for (const word of tokens) {
      const pSpam = this.getWordProbability(word, true)
      const pNotSpam = this.getWordProbability(word, false)

      const logPSpam = pSpam > 0 ? Math.log(pSpam) : -20
      const logPNotSpam = pNotSpam > 0 ? Math.log(pNotSpam) : -20

      logSpam += logPSpam
      logNotSpam += logPNotSpam

      const diff = logPSpam - logPNotSpam
      contributions.push({
        word,
        logLikelihoodSpam: logPSpam,
        logLikelihoodNotSpam: logPNotSpam,
        contributionDirection:
          Math.abs(diff) < 0.1 ? 'neutral' : diff > 0 ? 'spam' : 'notSpam',
      })
    }

    // Normalize to probabilities (log-sum-exp trick)
    const maxLog = Math.max(logSpam, logNotSpam)
    const logSum =
      maxLog + Math.log(Math.exp(logSpam - maxLog) + Math.exp(logNotSpam - maxLog))
    const posteriorSpam = Math.exp(logSpam - logSum)
    const posteriorNotSpam = Math.exp(logNotSpam - logSum)

    return {
      prediction: posteriorSpam > posteriorNotSpam ? 'spam' : 'not spam',
      posteriorSpam,
      posteriorNotSpam,
      logPosteriorSpam: logSpam,
      logPosteriorNotSpam: logNotSpam,
      perWordContributions: contributions,
    }
  }

  // Get top discriminative words (sorted by |log ratio|)
  getTopWords(n = 20): { word: string; spam: number; notSpam: number; logRatio: number }[] {
    const entries: { word: string; spam: number; notSpam: number; logRatio: number }[] = []

    for (const [word, probs] of this.wordProbs.entries()) {
      const ratio = Math.log((probs.spam + 1e-6) / (probs.notSpam + 1e-6))
      entries.push({ word, spam: probs.spam, notSpam: probs.notSpam, logRatio: ratio })
    }

    entries.sort((a, b) => Math.abs(b.logRatio) - Math.abs(a.logRatio))
    return entries.slice(0, n)
  }

  setPrior(spamPrior: number): void {
    this.priorSpam = spamPrior
  }

  setSmoothing(alpha: number): void {
    this.smoothingAlpha = alpha
  }
}

// ── Example emails ───────────────────────────────────────────────────
export const EXAMPLE_EMAILS = {
  spam: 'Congratulations! You have won a free prize. Click here to claim your money now. Limited time offer, act fast!',
  professional:
    'Hi team, please review the attached report before our meeting tomorrow. The project deadline is Friday and we need to finalize the budget.',
  ambiguous:
    'Dear customer, your account update is ready. Please click the link below to review your recent order and confirm delivery details.',
}
