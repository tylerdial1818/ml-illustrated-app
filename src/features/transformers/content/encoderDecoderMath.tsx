import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function EncoderDecoderMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Standard (Bidirectional) Attention</p>
        <p className="text-sm text-text-secondary mb-3">
          Every token can attend to every other token. Used in encoders:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Masked (Causal) Attention</p>
        <p className="text-sm text-text-secondary mb-3">
          A mask prevents tokens from attending to future positions. Used in decoders:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T + M}{\sqrt{d_k}}\right) V" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">The Causal Mask</p>
        <p className="text-sm text-text-secondary mb-3">
          The mask matrix sets illegal positions to negative infinity:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center space-y-2">
          <Eq tex="M_{ij} = 0 \quad \text{if } i \geq j \quad \text{(can attend)}" display />
          <Eq tex="M_{ij} = -\infty \quad \text{if } i < j \quad \text{(blocked)}" display />
        </div>
        <p className="mt-2 text-xs text-text-tertiary">
          The mask is applied BEFORE softmax. Setting a score
          to <Eq tex="-\infty" /> guarantees softmax assigns it weight 0.
        </p>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          The lower-triangular mask ensures each position can only attend to earlier positions and
          itself. This is essential for autoregressive generation: when predicting the next word, the
          model must not peek at future words that don't exist yet.
        </p>
      </div>
    </div>
  )
}
