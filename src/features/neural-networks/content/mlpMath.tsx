import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function MLPMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Forward pass</p>
        <p className="text-sm text-text-secondary mb-3">
          Each layer transforms its input through a linear map followed by a non-linear activation:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 space-y-2 text-center">
          <Eq tex="z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}" display />
          <Eq tex="a^{(l)} = \sigma(z^{(l)})" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Binary cross-entropy loss</p>
        <p className="text-sm text-text-secondary mb-3">
          Measures how far the predicted probability is from the true label:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\bigl[y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i)\bigr]" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Backpropagation</p>
        <p className="text-sm text-text-secondary mb-3">
          Gradients flow backward through the network using the chain rule:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 space-y-2 text-center">
          <Eq tex="\delta^{(L)} = \hat{y} - y" display />
          <Eq tex="\delta^{(l)} = (W^{(l+1)})^\top \delta^{(l+1)} \odot \sigma'(z^{(l)})" display />
          <Eq tex="\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^\top" display />
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          <Eq tex="W^{(l)}" /> is the weight matrix for layer <Eq tex="l" />.{' '}
          <Eq tex="\delta^{(l)}" /> is the error signal at that layer. It tells each neuron how much
          it contributed to the final mistake. The <Eq tex="\odot" /> symbol means element-wise multiplication.
          Gradients chain from the output back to the input, one layer at a time.
        </p>
      </div>
    </div>
  )
}
