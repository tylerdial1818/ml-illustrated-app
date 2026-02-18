import katex from 'katex'

function Eq({ tex, display = false }: { tex: string; display?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: display, throwOnError: false })
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

export function FeatureScalingMath() {
  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Min-Max Scaling</p>
        <p className="text-sm text-text-secondary mb-3">
          Squeezes values into the range <Eq tex="[0, 1]" />:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Standardization (Z-score)</p>
        <p className="text-sm text-text-secondary mb-3">
          Centers at zero with unit variance:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="x' = \frac{x - \mu}{\sigma}" display />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Robust Scaling</p>
        <p className="text-sm text-text-secondary mb-3">
          Uses the median and interquartile range, which are not affected by extreme outliers:
        </p>
        <div className="bg-obsidian-surface rounded-lg p-4 text-center">
          <Eq tex="x' = \frac{x - \text{median}}{\text{IQR}}" display />
        </div>
      </div>

      <div className="text-sm text-text-tertiary border-l-2 border-obsidian-border pl-4">
        <p>
          Min-Max squeezes values into <Eq tex="[0, 1]" />. Standardization centers at 0 with a std dev of 1 (most data ranges between -3 to 3).
          Robust uses the median and interquartile range, which are not affected by extreme outliers.
          Most models (especially gradient-based ones like neural networks) benefit from scaling.
          Tree-based models are an exception since they only care about the order of values, not their magnitude.
        </p>
      </div>
    </div>
  )
}
