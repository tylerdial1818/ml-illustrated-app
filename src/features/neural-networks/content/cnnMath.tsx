import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function CNNMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">2D convolution</p>
        <p className="text-sm text-text-secondary mb-3">
          A filter slides over the input, computing a dot product at each position:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="(I * K)(i,j) = \sum_{m}\sum_{n} I(i+m,\, j+n)\, K(m,n)" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Output size</p>
        <p className="text-sm text-text-secondary mb-3">
          Given input size W, filter size F, padding P, and stride S:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="W_{\text{out}} = \frac{W - F + 2P}{S} + 1" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Max pooling</p>
        <p className="text-sm text-text-secondary mb-3">
          Downsample by taking the maximum value in each window:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="\text{pool}(i,j) = \max_{(m,n) \in R_{ij}} \, a(m,n)" display />
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          <Eq tex="I" /> is the input image (or feature map), <Eq tex="K" /> is the filter (kernel).
          The convolution operation is what detects features. Small filters learn edges, larger
          receptive fields learn shapes. Pooling reduces spatial dimensions while keeping the
          strongest activations.
        </p>
      </div>
    </div>
  )
}
