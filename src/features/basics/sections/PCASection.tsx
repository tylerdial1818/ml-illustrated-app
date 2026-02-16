import { ModelSection } from '../../../components/ui/ModelSection'
import { PCAViz } from '../visualizations/PCAViz'
import { PCAMath } from '../content/pcaMath'

export function PCASection() {
  return (
    <ModelSection
      id="pca"
      title="PCA (Principal Component Analysis)"
      subtitle="Find the directions in your data that carry the most information and compress everything else away."
      intuition={
        <div className="max-w-2xl space-y-3">
          <p className="text-text-secondary leading-relaxed">
            Look at a cloud of data points in 2D. The cloud is shaped like an elongated oval, tilted
            at an angle. PCA rotates your axes to align with the oval. The first new axis (the first
            principal component) points along the longest direction of the cloud, where the data varies
            the most. The second axis is perpendicular to it, capturing whatever variation is left.
          </p>
          <p className="text-text-secondary leading-relaxed">
            Why does this matter? If most of the variation in your data is along one direction, you
            can drop the other direction and go from 2D to 1D with minimal information loss. You
            have compressed your data. In high dimensions (hundreds of features), PCA can often
            compress the data down to a handful of components that capture 90%+ of the variation.
            This makes visualization possible, speeds up training, and removes noise.
          </p>
        </div>
      }
      mechanism={<PCAViz />}
      math={<PCAMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">When PCA Helps</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>High-dimensional data with correlated features</li>
                <li>Visualization: project high-D data to 2D or 3D</li>
                <li>Denoising: remove low-variance components (often noise)</li>
                <li>Preprocessing before models sensitive to dimensionality</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Only captures linear relationships (use t-SNE/UMAP for nonlinear)</li>
                <li>Circular or uncorrelated data cannot be compressed</li>
                <li>PCA components may not be interpretable</li>
                <li>Sensitive to feature scaling. Always scale first.</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
