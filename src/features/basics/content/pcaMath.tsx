import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function PCAMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Center the Data</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\mathbf{X}_c = \mathbf{X} - \boldsymbol{\mu}" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Covariance Matrix</p>
        <p className="text-sm text-text-secondary mb-3">
          Captures how every pair of features varies together:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\mathbf{C} = \frac{1}{n} \mathbf{X}_c^\top \mathbf{X}_c" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Eigendecomposition</p>
        <p className="text-sm text-text-secondary mb-3">
          The eigenvectors of <Eq tex="\mathbf{C}" /> are the principal component directions.
          The eigenvalues tell you how much variance is in each direction.
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\mathbf{C} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^\top" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Projection</p>
        <p className="text-sm text-text-secondary mb-3">
          Keep only the top <Eq tex="k" /> eigenvectors (the most informative directions):
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\mathbf{X}_{\text{proj}} = \mathbf{X}_c \mathbf{V}_k" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Variance Explained</p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\text{Variance explained by PC}_j = \frac{\lambda_j}{\sum_i \lambda_i}" display />
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          The covariance matrix <Eq tex="\mathbf{C}" /> captures how every pair of features varies
          together. Its eigenvectors point in the directions of maximum variance. Its eigenvalues tell
          you how much variance is in each direction. PCA keeps the top <Eq tex="k" /> eigenvectors
          (the most informative directions) and discards the rest.
        </p>
      </div>
    </div>
  )
}
