import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function PerceptronMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Perceptron output</p>
        <p className="text-sm text-text-secondary mb-3">
          The perceptron computes a weighted sum of inputs, adds a bias, and passes the result through an activation function:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\hat{y} = \sigma(w_1 x_1 + w_2 x_2 + b)" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Sigmoid activation</p>
        <p className="text-sm text-text-secondary mb-3">
          The sigmoid squashes any real number into the range (0, 1):
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\sigma(z) = \frac{1}{1 + e^{-z}}" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Weight update rule</p>
        <p className="text-sm text-text-secondary mb-3">
          After each prediction, the weights are nudged in the direction that reduces the error:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="w_i \leftarrow w_i + \eta \, (y - \hat{y}) \, x_i" display />
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          <Eq tex="\eta" /> is the learning rate. This controls how big a step each update takes.{' '}
          <Eq tex="(y - \hat{y})" /> is the error: the difference between the true label and the prediction.
          Each weight <Eq tex="w_i" /> is adjusted proportionally to both the error and the input <Eq tex="x_i" />.
        </p>
      </div>
    </div>
  )
}
