import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function PositionalEncodingMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Sinusoidal Position Encoding</p>
        <p className="text-sm text-text-secondary mb-3">
          Even dimensions use sine, odd dimensions use cosine:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center space-y-2">
          <Eq tex="PE(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)" display />
          <Eq tex="PE(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          The 10000 base creates a geometric progression of frequencies. Each position gets a unique
          combination of sine and cosine values across dimensions.
        </p>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Adding Position to Embedding</p>
        <p className="text-sm text-text-secondary mb-3">
          The final input to the Transformer is the sum of the token embedding and the positional encoding:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="x = \text{embedding}(\text{token}) + PE(\text{pos})" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          We simply add the two vectors element-wise. The model learns to separate positional
          information from semantic information during training.
        </p>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          Why sine and cosine? Because for any fixed offset <Eq tex="k" />, the encoding
          at position <Eq tex="\text{pos} + k" /> can be written as a linear function of the encoding
          at position <Eq tex="\text{pos}" />. This lets the model easily learn to attend to relative
          positions.
        </p>
      </div>
    </div>
  )
}
