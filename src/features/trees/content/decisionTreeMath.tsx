import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function DecisionTreeMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Gini Impurity</p>
        <p className="text-sm text-text-secondary mb-3">
          Measures how often a randomly chosen element would be misclassified:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="G(t) = 1 - \sum_{k=1}^{K} p_k^2" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          <Eq tex="p_k" /> is the proportion of class <Eq tex="k" /> in node <Eq tex="t" />.
          Gini = 0 means the node is perfectly pure (all one class).
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Entropy</p>
        <p className="text-sm text-text-secondary mb-3">
          An alternative purity measure from information theory:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="H(t) = -\sum_{k=1}^{K} p_k \log_2(p_k)" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          Higher entropy means more disorder. A pure node has entropy 0.
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Information Gain</p>
        <p className="text-sm text-text-secondary mb-3">
          The reduction in impurity achieved by splitting on a feature:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\text{IG}(t, f) = H(t) - \frac{|t_L|}{|t|} H(t_L) - \frac{|t_R|}{|t|} H(t_R)" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          The tree greedily picks the feature <Eq tex="f" /> and threshold that maximize information gain
          at each step. <Eq tex="t_L" /> and <Eq tex="t_R" /> are the left and right child nodes after splitting.
        </p>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          CART (Classification and Regression Trees) uses Gini impurity by default. For regression,
          the splitting criterion becomes variance reduction: pick the split that minimizes the
          weighted sum of variances in the two child nodes.
        </p>
      </div>
    </div>
  )
}
