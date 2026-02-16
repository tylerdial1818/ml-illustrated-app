import { SectionNav } from '../../components/layout/SectionNav'
import { BayesianOverview } from './sections/BayesianOverview'
import { BayesTheoremSection } from './sections/BayesTheoremSection'
import { BayesianRegressionSection } from './sections/BayesianRegressionSection'
import { NaiveBayesSection } from './sections/NaiveBayesSection'
import { GaussianProcessSection } from './sections/GaussianProcessSection'
import { BayesianComparisonSection } from './sections/BayesianComparisonSection'

const SECTIONS = [
  { id: 'overview', label: 'Overview' },
  { id: 'bayes-theorem', label: "Bayes' Theorem" },
  { id: 'bayesian-regression', label: 'Bayesian Regression' },
  { id: 'naive-bayes', label: 'Naive Bayes' },
  { id: 'gaussian-processes', label: 'Gaussian Processes' },
  { id: 'comparison', label: 'Comparison' },
]

export function BayesianPage() {
  return (
    <div className="relative">
      <SectionNav sections={SECTIONS} />

      <div>
        {/* Page Header */}
        <div className="pt-12 lg:pt-16 pb-10">
          <h1 className="text-4xl lg:text-5xl font-bold text-text-primary">Bayesian ML</h1>
          <p className="mt-4 text-lg text-text-secondary max-w-2xl leading-relaxed">
            Instead of finding one best answer, maintain a full probability distribution over
            all possible answers and update it as you see more data. Every prediction comes
            with built-in uncertainty.
          </p>
        </div>

        <BayesianOverview />

        <div className="border-t border-obsidian-border" />
        <BayesTheoremSection />

        <div className="border-t border-obsidian-border" />
        <BayesianRegressionSection />

        <div className="border-t border-obsidian-border" />
        <NaiveBayesSection />

        <div className="border-t border-obsidian-border" />
        <GaussianProcessSection />

        <div className="border-t border-obsidian-border" />
        <BayesianComparisonSection />

        <div className="h-20" />
      </div>
    </div>
  )
}
