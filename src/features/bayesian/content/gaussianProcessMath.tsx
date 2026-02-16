import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function GaussianProcessMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">GP Prior</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="f \sim \mathcal{GP}\big(m(x),\; k(x, x')\big)" display />
        </div>
        <div className="mt-3 space-y-1.5 text-sm text-text-secondary">
          <p><Eq tex="m(x)" /> is the mean function (usually 0).</p>
          <p><Eq tex="k(x, x')" /> is the <strong className="text-[#FBBF24]">kernel</strong> (covariance function). It controls smoothness, periodicity, and scale.</p>
          <p>A GP defines a distribution over <em>functions</em>. Any finite set of outputs is jointly Gaussian.</p>
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">RBF Kernel (Squared Exponential)</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="k(x, x') = \sigma^2 \exp\!\left(-\frac{|x - x'|^2}{2\ell^2}\right)" display />
        </div>
        <p className="text-sm text-text-secondary mt-3">
          <Eq tex="\ell" /> is the <strong className="text-text-primary">length scale</strong>: short = wiggly, long = smooth.
          <Eq tex="\sigma^2" /> is the signal variance (vertical scale).
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Posterior Predictive</p>
        <div className="bg-obsidian-surface rounded-lg p-4 space-y-3 text-center">
          <Eq tex="\mu_* = \mathbf{k}_*^\top \big[\mathbf{K} + \sigma_n^2 \mathbf{I}\big]^{-1} \mathbf{y}" display />
          <Eq tex="\sigma_*^2 = k(x_*, x_*) - \mathbf{k}_*^\top \big[\mathbf{K} + \sigma_n^2 \mathbf{I}\big]^{-1} \mathbf{k}_*" display />
        </div>
        <div className="mt-3 space-y-1.5 text-sm text-text-secondary">
          <p><Eq tex="\mathbf{k}_* = k(x_*, \mathbf{X})" /> measures similarity of the new point to every training point.</p>
          <p><Eq tex="\mathbf{K} = k(\mathbf{X}, \mathbf{X})" /> is the kernel matrix capturing pairwise training similarity.</p>
          <p><Eq tex="\mu_*" /> is a weighted combination of training outputs, weighted by kernel similarity.</p>
          <p><Eq tex="\sigma_*^2" /> is large when the new point is far from all training points (high uncertainty).</p>
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Other Kernels</p>
        <div className="bg-obsidian-surface rounded-lg p-4 space-y-3 text-center">
          <div>
            <p className="text-[10px] text-text-tertiary font-mono mb-1">Matérn 3/2</p>
            <Eq tex="k(x,x') = \sigma^2\left(1 + \frac{\sqrt{3}\,|x-x'|}{\ell}\right)\exp\!\left(-\frac{\sqrt{3}\,|x-x'|}{\ell}\right)" display />
          </div>
          <div>
            <p className="text-[10px] text-text-tertiary font-mono mb-1">Periodic</p>
            <Eq tex="k(x,x') = \sigma^2 \exp\!\left(-\frac{2\sin^2(\pi|x-x'|/p)}{\ell^2}\right)" display />
          </div>
          <div>
            <p className="text-[10px] text-text-tertiary font-mono mb-1">Linear</p>
            <Eq tex="k(x,x') = \sigma_b^2 + \sigma^2 (x-c)(x'-c)" display />
          </div>
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Computational Cost</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\mathcal{O}(n^3) \text{ training}, \quad \mathcal{O}(n^2) \text{ per prediction}" display />
        </div>
        <p className="text-sm text-text-secondary mt-3">
          The cubic cost comes from inverting (or Cholesky-decomposing) the <Eq tex="n \times n" /> kernel
          matrix. This limits standard GPs to roughly 10,000 training points. Sparse and variational
          approximations exist for larger datasets.
        </p>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          The key insight: GPs are non-parametric. Instead of learning a fixed number of parameters,
          the model complexity grows with the data. The kernel encodes your assumptions about what
          kind of functions are plausible. Different kernels, different assumptions, different fits.
        </p>
      </div>
    </div>
  )
}
