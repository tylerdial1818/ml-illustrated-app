import { ModelSection } from '../../../components/ui/ModelSection'
import { DBSCANViz } from '../visualizations/DBSCANViz'
import { DBSCANMath } from '../content/dbscanMath'

export function DBSCANSection() {
  return (
    <ModelSection
      id="dbscan"
      title="DBSCAN"
      subtitle="Density-Based Spatial Clustering of Applications with Noise."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            A point is in a cluster if it has <strong className="text-text-primary">enough neighbors nearby</strong>.
            Start from any dense point, keep expanding outward as long as density holds. Anything left over is noise.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            Unlike K-Means, DBSCAN doesn't need you to specify the number of clusters. It discovers them
            automatically based on density, and it can find clusters of any shape, not just spheres.
          </p>
        </div>
      }
      mechanism={<DBSCANViz />}
      math={<DBSCANMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Finds arbitrary-shaped clusters</li>
                <li>No need to specify number of clusters</li>
                <li>Naturally detects noise/outliers</li>
                <li>Robust to cluster shape</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Sensitive to ε parameter</li>
                <li>Struggles with varying densities</li>
                <li>Not great for high dimensions</li>
                <li>Doesn't assign probabilities</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
