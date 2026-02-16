import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function BayesianRegressionMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">The Model</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="y = X\beta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)" display />
        </div>
        <div className="mt-3 space-y-1.5 text-sm text-text-secondary">
          <p><Eq tex="\beta = [\beta_0, \beta_1]^\top" /> are the intercept and slope parameters.</p>
          <p><Eq tex="X" /> is the design matrix with rows <Eq tex="[1, x_i]" />.</p>
          <p><Eq tex="\sigma^2" /> is the known noise variance.</p>
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Prior</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\beta \sim \mathcal{N}(\mu_0, \Sigma_0)" display />
        </div>
        <p className="text-sm text-text-secondary mt-3">
          We place a <strong className="text-[#A1A1AA]">Gaussian prior</strong> on the weights. An uninformed
          prior uses <Eq tex="\mu_0 = 0" /> and large <Eq tex="\Sigma_0" />, expressing no preference
          for any particular slope or intercept.
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Posterior (Analytic)</p>
        <div className="bg-obsidian-surface rounded-lg p-4 space-y-3 text-center">
          <Eq tex="\beta \mid D \sim \mathcal{N}(\mu_n, \Sigma_n)" display />
          <Eq tex="\Sigma_n = \left(\Sigma_0^{-1} + \sigma^{-2} X^\top X\right)^{-1}" display />
          <Eq tex="\mu_n = \Sigma_n \left(\Sigma_0^{-1} \mu_0 + \sigma^{-2} X^\top y\right)" display />
        </div>
        <p className="text-sm text-text-secondary mt-3">
          The <strong className="text-[#6366F1]">posterior</strong> is also Gaussian. This is a conjugate
          model: Gaussian prior + Gaussian likelihood = Gaussian posterior. No sampling needed.
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Predictive Distribution</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="p(y_* \mid x_*, D) = \mathcal{N}\!\left(\mu_n^\top \phi_*, \; \phi_*^\top \Sigma_n \phi_* + \sigma^2\right)" display />
        </div>
        <p className="text-sm text-text-secondary mt-3">
          For a new input <Eq tex="x_*" />, the <strong className="text-[#34D399]">predictive</strong> has
          two sources of uncertainty: <Eq tex="\phi_*^\top \Sigma_n \phi_*" /> from parameter uncertainty
          (wide where data is sparse) and <Eq tex="\sigma^2" /> from observation noise (irreducible).
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Connection to Ridge Regression</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\beta \sim \mathcal{N}(0, \tau^2 I) \;\Longrightarrow\; \text{MAP} = \text{Ridge with } \lambda = \frac{\sigma^2}{\tau^2}" display />
        </div>
        <p className="text-sm text-text-secondary mt-3">
          When the prior is a zero-mean isotropic Gaussian, the MAP estimate is exactly Ridge regression.
          The Ridge penalty <Eq tex="\lambda" /> is the ratio of noise to prior variance. A tighter
          prior (smaller <Eq tex="\tau^2" />) means more regularization.
        </p>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          Unlike OLS which gives a single point estimate, the Bayesian posterior gives a full distribution.
          We can sample weight vectors from <Eq tex="\mathcal{N}(\mu_n, \Sigma_n)" /> to draw plausible
          regression lines. The spread of these lines reflects our remaining uncertainty.
        </p>
      </div>
    </div>
  )
}
