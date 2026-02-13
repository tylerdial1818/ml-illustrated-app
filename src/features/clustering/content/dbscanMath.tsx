import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function DBSCANMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">ε-neighborhood</p>
        <p className="text-sm text-text-secondary mb-3">
          The set of all points within distance ε of point p:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="N_\varepsilon(p) = \{q \in D \mid \text{dist}(p, q) \leq \varepsilon\}" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Core point condition</p>
        <p className="text-sm text-text-secondary mb-3">
          A point is a core point if its ε-neighborhood contains at least minPts points:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="|N_\varepsilon(p)| \geq \text{minPts}" display />
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          <Eq tex="\varepsilon" /> is the circle radius you see around each point.{' '}
          <Eq tex="\text{minPts}" /> is how many dots need to be inside that circle for the point to be
          considered "core." Border points are within ε of a core point but don't have enough neighbors
          themselves. Everything else is noise.
        </p>
      </div>
    </div>
  )
}
