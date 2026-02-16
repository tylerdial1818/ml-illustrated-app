import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { CollapsibleSection } from '../../../components/ui/CollapsibleSection'
import { MedicalTestViz } from '../visualizations/MedicalTestViz'
import { CoinFlipUpdater } from '../visualizations/CoinFlipUpdater'
import { BayesTheoremMath } from '../content/bayesTheoremMath'

export function BayesTheoremSection() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: true })

  return (
    <section id="bayes-theorem" ref={ref} className="py-16 lg:py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      >
        <h2 className="text-2xl lg:text-3xl font-semibold text-text-primary">Bayes' Theorem</h2>
        <p className="text-base lg:text-lg text-text-secondary mt-2 max-w-2xl leading-relaxed">
          Start with a belief, observe evidence, update your belief. That is the entire framework
          in one equation.
        </p>

        <div className="mt-10 space-y-14">
          {/* Intuition */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-5">
              The Intuition
            </h4>
            <div className="max-w-2xl space-y-3">
              <p className="text-text-secondary leading-relaxed">
                Imagine you are at home and you hear a soft sound from the kitchen. Before
                investigating, your brain already has priors: there is probably a 60% chance it is
                your cat, a 35% chance it is the fridge making noise, and a 5% chance it is a
                raccoon that got in. Then you hear a soft meow. That is new evidence. Your brain
                instantly updates: it is almost certainly the cat now. The raccoon hypothesis drops
                to near zero. The fridge hypothesis drops too, because fridges do not meow.
              </p>
              <p className="text-text-secondary leading-relaxed">
                This is Bayes' theorem at work. You started with a belief (the prior), observed
                evidence (the meow), and updated your belief (the posterior). The strength of the
                update depends on how likely the evidence is under each hypothesis.
              </p>
            </div>
          </div>

          {/* Part A: Medical Test */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-2">
              The Mechanism: Part A
            </h4>
            <p className="text-sm text-text-secondary mb-4 max-w-2xl">
              The base rate fallacy. A disease affects 1% of the population. A test is 90%
              accurate. You test positive. What is the probability you are actually sick?
            </p>
            <MedicalTestViz />
          </div>

          {/* Part B: Coin Flip */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-2">
              The Mechanism: Part B
            </h4>
            <p className="text-sm text-text-secondary mb-4 max-w-2xl">
              Sequential updating with a coin. Start with a flat belief about the coin's bias.
              Flip one coin at a time and watch your belief sharpen. After many flips, the
              posterior concentrates near the true bias. Try a strong prior and see how many flips
              it takes to override.
            </p>
            <CoinFlipUpdater />
          </div>

          {/* Math */}
          <CollapsibleSection title="The Math">
            <BayesTheoremMath />
          </CollapsibleSection>

          {/* When to Use */}
          <div>
            <h4 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-tertiary mb-5">
              When to Use
            </h4>
            <div className="max-w-2xl">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <h5 className="text-sm font-medium text-success mb-2">Key Takeaways</h5>
                  <ul className="space-y-2 text-sm text-text-secondary">
                    <li>Bayes' theorem is the foundation of all Bayesian inference, not a model you "choose"</li>
                    <li>Priors encode domain knowledge. With enough data, the prior washes out</li>
                    <li>The posterior is always a compromise between prior and data</li>
                    <li>Conjugate priors (like Beta-Binomial) give exact, instant updates</li>
                  </ul>
                </div>
                <div>
                  <h5 className="text-sm font-medium text-error mb-2">Watch Out For</h5>
                  <ul className="space-y-2 text-sm text-text-secondary">
                    <li>The base rate fallacy: ignoring priors leads to wildly wrong conclusions</li>
                    <li>Strong priors require a lot of data to override. Choose priors carefully.</li>
                    <li>For most real models, exact posteriors are intractable. We need approximations.</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </section>
  )
}
