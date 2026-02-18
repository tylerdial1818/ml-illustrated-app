import { SectionNav } from '../../components/layout/SectionNav'
import { ClusteringOverview } from './sections/ClusteringOverview'
import { KMeansSection } from './sections/KMeansSection'
import { DBSCANSection } from './sections/DBSCANSection'
import { HierarchicalSection } from './sections/HierarchicalSection'
import { GMMSection } from './sections/GMMSection'
import { ClusteringComparison } from './sections/ClusteringComparison'

const SECTIONS = [
  { id: 'clustering-overview', label: 'Overview' },
  { id: 'kmeans', label: 'K-Means' },
  { id: 'dbscan', label: 'DBSCAN' },
  { id: 'hierarchical', label: 'Hierarchical' },
  { id: 'gmm', label: 'GMM' },
  { id: 'clustering-comparison', label: 'Comparison' },
]

export function ClusteringPage() {
  return (
    <div className="relative">
      <SectionNav sections={SECTIONS} />

      <div>
        {/* Page Header */}
        <div className="pt-12 lg:pt-16 pb-10">
          <h1 className="text-4xl lg:text-5xl font-bold text-text-primary">Clustering</h1>
          <p className="mt-4 text-lg text-text-secondary max-w-2xl leading-relaxed">
            Grouping data without labels. Discover hidden structure through algorithms
            that find patterns you didn't know were there.
          </p>
        </div>

        <ClusteringOverview />

        <div className="border-t border-obsidian-border" />
        <KMeansSection />

        <div className="border-t border-obsidian-border" />
        <DBSCANSection />

        <div className="border-t border-obsidian-border" />
        <HierarchicalSection />

        <div className="border-t border-obsidian-border" />
        <GMMSection />

        <div className="border-t border-obsidian-border" />
        <ClusteringComparison />

        <div className="h-20" />
      </div>
    </div>
  )
}
