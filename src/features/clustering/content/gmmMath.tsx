import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function GMMMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Mixture model</p>
        <p className="text-sm text-text-secondary mb-3">
          The probability of observing a data point is a weighted sum of Gaussians:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">E-step (Expectation)</p>
        <p className="text-sm text-text-secondary mb-3">
          Compute the responsibility of each component for each data point:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">M-step (Maximization)</p>
        <p className="text-sm text-text-secondary mb-3">
          Update parameters using the responsibilities:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 space-y-2 text-center">
          <Eq tex="\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}" display />
          <Eq tex="\pi_k = \frac{1}{N} \sum_i \gamma_{ik}" display />
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          <Eq tex="\pi_k" /> is how big each cloud is (its weight). <Eq tex="\mu_k" /> is its center.{' '}
          <Eq tex="\Sigma_k" /> is its shape — the ellipse you see. The E-step figures out which cloud
          each point probably came from; the M-step adjusts the clouds to match.
        </p>
      </div>
    </div>
  )
}
