import { ModelSection } from '../../../components/ui/ModelSection'
import { KMeansViz } from '../visualizations/KMeansViz'
import { KMeansMath } from '../content/kmeansMath'

export function KMeansSection() {
  return (
    <ModelSection
      id="kmeans"
      title="K-Means"
      subtitle="Partition data into k groups by minimizing distance to cluster centers."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            Imagine dropping <strong className="text-text-primary">k pins</strong> randomly onto a map of your data.
            Then repeatedly: (1) each point walks to its nearest pin, (2) each pin moves to the center of its crowd.
            Repeat until nobody moves.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            K-Means is greedy. It finds <em>a</em> solution fast, but not necessarily <em>the best</em> one.
            That's why initialization matters, and why you might want to run it multiple times.
          </p>
        </div>
      }
      mechanism={<KMeansViz />}
      math={<KMeansMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Fast and scalable</li>
                <li>Works well with spherical clusters</li>
                <li>Easy to interpret</li>
                <li>Good when you know (or can estimate) k</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-2 text-sm text-text-secondary">
                <li>Assumes spherical, equally-sized clusters</li>
                <li>Must specify k in advance</li>
                <li>Sensitive to initialization</li>
                <li>Fails on non-convex shapes</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
