import { ModelSection } from '../../../components/ui/ModelSection'
import { CNNViz } from '../visualizations/CNNViz'
import { CNNMath } from '../content/cnnMath'

export function CNNSection() {
  return (
    <ModelSection
      id="cnn"
      title="Convolutional Neural Network (CNN)"
      subtitle="Slide small filters across an image to detect patterns (edges, textures, shapes) then combine them into recognition."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            Imagine sliding a small <strong className="text-text-primary">magnifying glass</strong> across
            an image, one patch at a time. At each position, the magnifying glass checks for a specific
            pattern (examples: a vertical edge, or a color gradient). You use many different magnifying glasses
            (filters), each looking for a different pattern.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            Early layers detect simple features like edges and corners. Deeper layers combine those into
            textures, then parts, then whole objects. A CNN learns <em>what</em> to look for and{' '}
            <em>where</em> to look, and the same filter works regardless of where the pattern appears
            in the image. This is called <em>translation invariance</em>.
          </p>
        </div>
      }
      mechanism={<CNNViz />}
      math={<CNNMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Excellent for image and spatial data</li>
                <li>Parameter sharing via filters</li>
                <li>Learns hierarchical features automatically</li>
                <li>Translation invariant, detects patterns anywhere</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Not well-suited for tabular or sequential data</li>
                <li>Requires large datasets to train well</li>
                <li>Computationally expensive (many parameters)</li>
                <li>Fixed input size without extra engineering</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
