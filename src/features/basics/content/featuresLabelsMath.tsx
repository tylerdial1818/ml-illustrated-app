import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function FeaturesLabelsMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm text-text-secondary mb-3">
          A dataset is a collection of <Eq tex="n" /> examples, where each example pairs
          a feature vector with a label:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n)\}" display />
        </div>
      </div>

      <div>
        <p className="text-sm text-text-secondary mb-3">
          Each <Eq tex="\mathbf{x}_i" /> is a <strong className="text-text-primary">feature vector</strong> with{' '}
          <Eq tex="d" /> dimensions:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\mathbf{x}_i = [x_{i1}, x_{i2}, \ldots, x_{id}]" display />
        </div>
      </div>

      <div>
        <p className="text-sm text-text-secondary mb-3">
          Each <Eq tex="y_i" /> is a <strong className="text-text-primary">label</strong>:
          a real number for regression, or a category for classification.
        </p>
      </div>

      <div>
        <p className="text-sm text-text-secondary mb-3">
          The entire feature matrix is:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\mathbf{X} \in \mathbb{R}^{n \times d}" display />
        </div>
        <p className="text-sm text-text-secondary mt-2">
          That is <Eq tex="n" /> rows (examples) and <Eq tex="d" /> columns (features).
        </p>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          This notation appears everywhere in ML. <Eq tex="\mathbf{X}" /> is the data
          matrix, <Eq tex="\mathbf{y}" /> is the label vector, <Eq tex="n" /> is
          the number of examples, <Eq tex="d" /> is the number of features. When other
          pages write things like <Eq tex="\mathbf{y} = \mathbf{X}\boldsymbol{\beta}" />,
          they mean: multiply the feature matrix by some weights to predict the labels.
        </p>
      </div>
    </div>
  )
}
