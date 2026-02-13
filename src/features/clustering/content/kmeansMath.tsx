import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function KMeansMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm text-text-secondary mb-3">
          K-Means minimizes the total squared distance from each point to its assigned centroid:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Assignment step</p>
        <p className="text-sm text-text-secondary mb-2">
          Each point is assigned to the cluster with the nearest centroid:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="c_i = \arg\min_k \|x_i - \mu_k\|^2" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Update step</p>
        <p className="text-sm text-text-secondary mb-2">
          Each centroid moves to the mean of its assigned points:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i" display />
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          <Eq tex="\mu_k" /> is the centroid you see moving. <Eq tex="\|x_i - \mu_k\|^2" /> is the
          squared length of the line from each point to its centroid. The algorithm alternates between
          assigning points and updating centroids until convergence.
        </p>
      </div>
    </div>
  )
}
