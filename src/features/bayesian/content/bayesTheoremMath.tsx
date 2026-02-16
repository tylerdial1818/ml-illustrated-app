import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function BayesTheoremMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Bayes' Theorem</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="P(\theta \mid D) = \frac{P(D \mid \theta) \, P(\theta)}{P(D)}" display />
        </div>
        <div className="mt-3 space-y-1.5 text-sm text-text-secondary">
          <p><Eq tex="P(\theta \mid D)" /> = <strong className="text-[#6366F1]">Posterior</strong> (what we believe after seeing data)</p>
          <p><Eq tex="P(D \mid \theta)" /> = <strong className="text-[#FBBF24]">Likelihood</strong> (how probable is the data given these parameters)</p>
          <p><Eq tex="P(\theta)" /> = <strong className="text-[#A1A1AA]">Prior</strong> (what we believed before seeing data)</p>
          <p><Eq tex="P(D)" /> = <strong className="text-text-tertiary">Evidence</strong> (normalizing constant so posterior integrates to 1)</p>
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">In words</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Coin Flip: Conjugate Prior</p>
        <p className="text-sm text-text-secondary mb-3">
          When the prior is a Beta distribution and the data is binomial (coin flips), the posterior
          is also a Beta distribution. This is called a <strong className="text-text-primary">conjugate prior</strong>:
          the prior and posterior belong to the same family.
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 space-y-3 text-center">
          <Eq tex="\text{Prior: } \theta \sim \text{Beta}(\alpha, \beta)" display />
          <Eq tex="\text{Data: } h \text{ heads}, t \text{ tails}" display />
          <Eq tex="\text{Posterior: } \theta \mid D \sim \text{Beta}(\alpha + h, \beta + t)" display />
        </div>
        <p className="text-sm text-text-secondary mt-3">
          Each observed head adds 1 to <Eq tex="\alpha" />. Each tail adds 1 to <Eq tex="\beta" />.
          The posterior parameters are simply the prior parameters plus the counts. This makes
          updating instant and exact.
        </p>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          Not all models have conjugate priors. When they do not, we use numerical methods like
          Markov Chain Monte Carlo (MCMC) or variational inference to approximate the posterior.
          The coin flip example is special because we can compute the answer exactly.
        </p>
      </div>
    </div>
  )
}
