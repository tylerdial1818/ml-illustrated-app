import { SectionNav } from '../../components/layout/SectionNav'
import { WhatIsMLSection } from './sections/WhatIsMLSection'
import { FeaturesLabelsSection } from './sections/FeaturesLabelsSection'
import { LossFunctionsSection } from './sections/LossFunctionsSection'
import { GradientDescentSection } from './sections/GradientDescentSection'
import { OverfittingSection } from './sections/OverfittingSection'
import { FeatureScalingSection } from './sections/FeatureScalingSection'
import { PCASection } from './sections/PCASection'
import { WhereToNextSection } from './sections/WhereToNextSection'

const SECTIONS = [
  { id: 'what-is-ml', label: 'What Is ML?' },
  { id: 'features-labels', label: 'Features & Labels' },
  { id: 'loss-functions', label: 'Loss Functions' },
  { id: 'gradient-descent', label: 'Gradient Descent' },
  { id: 'overfitting', label: 'Overfitting' },
  { id: 'feature-scaling', label: 'Feature Scaling' },
  { id: 'pca', label: 'PCA' },
  { id: 'where-next', label: 'Where to Next' },
]

export function BasicsPage() {
  return (
    <div className="relative">
      <SectionNav sections={SECTIONS} />

      <div>
        {/* Page Header */}
        <div className="pt-12 lg:pt-16 pb-10">
          <h1 className="text-4xl lg:text-5xl font-bold text-text-primary">ML Basics</h1>
          <p className="mt-4 text-lg text-text-secondary max-w-2xl leading-relaxed">
            The prerequisite concepts every ML model depends on. Start here if you have high school
            math and want to understand machine learning from the ground up.
          </p>
        </div>

        <WhatIsMLSection />

        <div className="border-t border-obsidian-border" />
        <FeaturesLabelsSection />

        <div className="border-t border-obsidian-border" />
        <LossFunctionsSection />

        <div className="border-t border-obsidian-border" />
        <GradientDescentSection />

        <div className="border-t border-obsidian-border" />
        <OverfittingSection />

        <div className="border-t border-obsidian-border" />
        <FeatureScalingSection />

        <div className="border-t border-obsidian-border" />
        <PCASection />

        <div className="border-t border-obsidian-border" />
        <WhereToNextSection />

        <div className="h-20" />
      </div>
    </div>
  )
}
