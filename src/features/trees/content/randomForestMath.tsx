import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function RandomForestMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Bootstrap Aggregating (Bagging)</p>
        <p className="text-sm text-text-secondary mb-3">
          Each tree is trained on a bootstrap sample — a random sample of size n drawn with replacement:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="D_b = \{(x_i, y_i)\}_{i=1}^{n}, \quad x_i \sim \text{Uniform}(D)" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          On average, each bootstrap sample contains about 63.2% of the original data points
          (the rest are "out-of-bag" samples useful for validation).
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Feature Subsampling</p>
        <p className="text-sm text-text-secondary mb-3">
          At each split, only a random subset of features is considered:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="m = \lfloor \sqrt{p} \rfloor \quad \text{(classification)} \qquad m = \lfloor p/3 \rfloor \quad \text{(regression)}" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          <Eq tex="p" /> is the total number of features. By limiting each split to <Eq tex="m" /> random
          features, trees are forced to be different from one another — this decorrelation is what makes
          the ensemble stronger than any individual tree.
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Ensemble Prediction</p>
        <p className="text-sm text-text-secondary mb-3">
          The final prediction aggregates all B trees by majority vote (classification) or averaging (regression):
        </p>
        <div className="space-y-3">
          <div className="bg-obsidian-surface rounded-lg p-4">
            <p className="text-xs text-text-tertiary mb-1">Classification (majority vote)</p>
            <Eq tex="\hat{y} = \text{mode}\{h_b(x)\}_{b=1}^{B}" display />
          </div>
          <div className="bg-obsidian-surface rounded-lg p-4">
            <p className="text-xs text-text-tertiary mb-1">Regression (average)</p>
            <Eq tex="\hat{y} = \frac{1}{B} \sum_{b=1}^{B} h_b(x)" display />
          </div>
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          <Eq tex="h_b(x)" /> is the prediction of tree <Eq tex="b" />. The variance of the ensemble
          prediction decreases roughly as <Eq tex="1/B" />, which is why more trees almost always
          help — unlike a single decision tree, a Random Forest rarely overfits by adding more trees.
        </p>
      </div>
    </div>
  )
}
