import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function NaiveBayesMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Posterior via Bayes' Theorem</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="P(C \mid x_1, x_2, \ldots, x_n) \propto P(C) \prod_{i=1}^{n} P(x_i \mid C)" display />
        </div>
        <div className="mt-3 space-y-1.5 text-sm text-text-secondary">
          <p><Eq tex="C" /> is the class label (spam or not spam).</p>
          <p><Eq tex="x_1, \ldots, x_n" /> are the observed features (words in the email).</p>
          <p>The <strong className="text-text-primary">naive</strong> assumption: all features are conditionally independent given the class.</p>
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Log-Space Decision Rule</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\hat{y} = \operatorname*{argmax}_{C} \left[ \log P(C) + \sum_{i=1}^{n} \log P(x_i \mid C) \right]" display />
        </div>
        <p className="text-sm text-text-secondary mt-3">
          We work in log space to avoid multiplying many small probabilities together (which causes
          numerical underflow). The classification decision is equivalent: the class with the highest
          log-posterior wins.
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Laplace Smoothing</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="P(x_i \mid C) = \frac{\text{count}(x_i, C) + \alpha}{\text{count}(C) + \alpha \cdot |V|}" display />
        </div>
        <p className="text-sm text-text-secondary mt-3">
          Without smoothing, a single unseen word gives <Eq tex="P(x_i \mid C) = 0" />, which zeros out
          the entire product. Laplace smoothing adds <Eq tex="\alpha" /> pseudocounts (typically 1) to
          every word. <Eq tex="|V|" /> is the vocabulary size.
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Normalizing to Probabilities</p>
        <div className="bg-obsidian-surface rounded-lg p-4 space-y-3 text-center">
          <Eq tex="P(\text{spam} \mid \mathbf{x}) = \frac{e^{s_{\text{spam}}}}{e^{s_{\text{spam}}} + e^{s_{\text{not spam}}}}" display />
        </div>
        <p className="text-sm text-text-secondary mt-3">
          Where <Eq tex="s_C = \log P(C) + \sum_i \log P(x_i \mid C)" />. We use the log-sum-exp trick
          to compute this stably.
        </p>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          The independence assumption is almost never true (for example, "free" and "money" are correlated in
          spam). Despite this, Naive Bayes often classifies correctly because the argmax decision only
          needs the correct ordering of class probabilities, not well-calibrated values.
        </p>
      </div>
    </div>
  )
}
