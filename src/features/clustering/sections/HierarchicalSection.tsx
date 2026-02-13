import { ModelSection } from '../../../components/ui/ModelSection'
import { HierarchicalViz } from '../visualizations/HierarchicalViz'
import { HierarchicalMath } from '../content/hierarchicalMath'

export function HierarchicalSection() {
  return (
    <ModelSection
      id="hierarchical"
      title="Hierarchical Clustering"
      subtitle="Build a tree of nested clusters by merging the closest pairs."
      intuition={
        <div className="max-w-2xl">
          <p className="text-text-secondary leading-relaxed">
            Start with every point as its own cluster. Repeatedly{' '}
            <strong className="text-text-primary">merge the two closest clusters</strong> until
            everything is one big group. The dendrogram records every merge — you cut it at the
            height where the grouping looks right.
          </p>
          <p className="mt-3 text-text-secondary leading-relaxed">
            The beauty of hierarchical clustering is that you get all possible groupings at once. Move the
            cut line up for fewer clusters, down for more — without re-running the algorithm.
          </p>
        </div>
      }
      mechanism={<HierarchicalViz />}
      math={<HierarchicalMath />}
      whenToUse={
        <div className="max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h5 className="text-sm font-medium text-success mb-2">Strengths</h5>
              <ul className="space-y-1 text-sm text-text-secondary">
                <li>No need to specify k in advance</li>
                <li>Reveals hierarchical relationships</li>
                <li>Explore multiple granularities</li>
                <li>Dendrogram gives rich information</li>
              </ul>
            </div>
            <div>
              <h5 className="text-sm font-medium text-error mb-2">Limitations</h5>
              <ul className="space-y-1 text-sm text-text-secondary">
                <li>O(n²) memory, O(n³) time</li>
                <li>Greedy — no backtracking</li>
                <li>Not suitable for large datasets</li>
                <li>Sensitive to linkage choice</li>
              </ul>
            </div>
          </div>
        </div>
      }
    />
  )
}
