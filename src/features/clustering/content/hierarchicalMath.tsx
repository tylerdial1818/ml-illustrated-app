import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function HierarchicalMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Linkage criteria</p>
        <p className="text-sm text-text-secondary mb-3">
          The linkage method determines how distance between clusters is measured:
        </p>
        <div className="space-y-3">
          <div className="bg-obsidian-surface rounded-lg p-4">
            <p className="text-xs text-text-tertiary mb-1">Single linkage (minimum)</p>
            <Eq tex="d(A, B) = \min_{a \in A, b \in B} \|a - b\|" display />
          </div>
          <div className="bg-obsidian-surface rounded-lg p-4">
            <p className="text-xs text-text-tertiary mb-1">Complete linkage (maximum)</p>
            <Eq tex="d(A, B) = \max_{a \in A, b \in B} \|a - b\|" display />
          </div>
          <div className="bg-obsidian-surface rounded-lg p-4">
            <p className="text-xs text-text-tertiary mb-1">Average linkage</p>
            <Eq tex="d(A, B) = \frac{1}{|A||B|} \sum_{a \in A} \sum_{b \in B} \|a - b\|" display />
          </div>
          <div className="bg-obsidian-surface rounded-lg p-4">
            <p className="text-xs text-text-tertiary mb-1">Ward's method (minimize variance)</p>
            <Eq tex="d(A, B) = \sqrt{\frac{2|A||B|}{|A|+|B|}} \|\mu_A - \mu_B\|" display />
          </div>
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          Single linkage finds the shortest distance between any pair of points across two clusters.
          Ward's method minimizes the total within-cluster variance after merging. It tends to produce
          the most balanced, compact clusters.
        </p>
      </div>
    </div>
  )
}
