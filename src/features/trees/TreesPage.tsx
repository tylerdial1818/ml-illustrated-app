import { SectionNav } from '../../components/layout/SectionNav'
import { TreesOverview } from './sections/TreesOverview'
import { DecisionTreeSection } from './sections/DecisionTreeSection'
import { RandomForestSection } from './sections/RandomForestSection'
import { GradientBoostedSection } from './sections/GradientBoostedSection'
import { TreesComparison } from './sections/TreesComparison'

const SECTIONS = [
  { id: 'trees-overview', label: 'Overview' },
  { id: 'decision-tree', label: 'Decision Tree' },
  { id: 'random-forest', label: 'Random Forest' },
  { id: 'gradient-boosted', label: 'Gradient Boosted' },
  { id: 'trees-comparison', label: 'Comparison' },
]

export function TreesPage() {
  return (
    <div className="relative">
      <SectionNav sections={SECTIONS} />

      <div>
        {/* Page Header */}
        <div className="pt-12 lg:pt-16 pb-10">
          <h1 className="text-4xl lg:text-5xl font-bold text-text-primary">Tree-Based Models</h1>
          <p className="mt-4 text-lg text-text-secondary max-w-2xl leading-relaxed">
            From simple decision rules to powerful ensembles — learn how trees partition data
            through sequences of yes/no questions.
          </p>
        </div>

        <TreesOverview />

        <div className="border-t border-obsidian-border" />
        <DecisionTreeSection />

        <div className="border-t border-obsidian-border" />
        <RandomForestSection />

        <div className="border-t border-obsidian-border" />
        <GradientBoostedSection />

        <div className="border-t border-obsidian-border" />
        <TreesComparison />

        <div className="h-20" />
      </div>
    </div>
  )
}
