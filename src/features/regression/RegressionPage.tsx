import { SectionNav } from '../../components/layout/SectionNav'
import { RegressionOverview } from './sections/RegressionOverview'
import { LinearRegressionSection } from './sections/LinearRegressionSection'
import { LogisticRegressionSection } from './sections/LogisticRegressionSection'
import { RidgeSection } from './sections/RidgeSection'
import { LassoSection } from './sections/LassoSection'
import { ElasticNetSection } from './sections/ElasticNetSection'
import { RegressionComparison } from './sections/RegressionComparison'

const SECTIONS = [
  { id: 'regression-overview', label: 'Overview' },
  { id: 'linear-regression', label: 'Linear' },
  { id: 'logistic-regression', label: 'Logistic' },
  { id: 'ridge', label: 'Ridge' },
  { id: 'lasso', label: 'Lasso' },
  { id: 'elasticnet', label: 'ElasticNet' },
  { id: 'regression-comparison', label: 'Comparison' },
]

export function RegressionPage() {
  return (
    <div className="relative">
      <SectionNav sections={SECTIONS} />

      <div className="max-w-5xl mx-auto px-6">
        {/* Page Header */}
        <div className="pt-16 pb-8">
          <h1 className="text-5xl font-bold text-text-primary">Regression</h1>
          <p className="mt-4 text-xl text-text-secondary max-w-2xl">
            Drawing the best line through data — predicting continuous outcomes by minimizing
            error with increasing sophistication.
          </p>
        </div>

        <RegressionOverview />

        <div className="border-t border-obsidian-border" />
        <LinearRegressionSection />

        <div className="border-t border-obsidian-border" />
        <LogisticRegressionSection />

        <div className="border-t border-obsidian-border" />
        <RidgeSection />

        <div className="border-t border-obsidian-border" />
        <LassoSection />

        <div className="border-t border-obsidian-border" />
        <ElasticNetSection />

        <div className="border-t border-obsidian-border" />
        <RegressionComparison />

        <div className="h-20" />
      </div>
    </div>
  )
}
