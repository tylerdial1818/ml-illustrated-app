import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { Link } from 'react-router-dom'
import { GlassCard } from '../../../components/ui/GlassCard'

const NEXT_PAGES = [
  {
    path: '/clustering',
    label: 'Clustering',
    icon: '⊕',
    description: 'Group similar data points together without labels. K-Means, DBSCAN, GMM.',
    color: '#6366F1',
  },
  {
    path: '/regression',
    label: 'Regression',
    icon: '⟋',
    description: 'Predict a continuous number or classify into categories. Linear Regression, Logistic Regression, Ridge/Lasso.',
    color: '#F472B6',
  },
  {
    path: '/trees',
    label: 'Tree-Based Models',
    icon: '⊞',
    description: 'Make predictions by learning a series of yes/no questions. Decision Trees, Random Forests, Gradient Boosting.',
    color: '#34D399',
  },
  {
    path: '/bayesian',
    label: 'Bayesian ML',
    icon: '⊘',
    description: "Quantify uncertainty by maintaining probability distributions over parameters. Bayes' Theorem, Gaussian Processes, Naive Bayes.",
    color: '#FBBF24',
  },
  {
    path: '/neural-networks',
    label: 'Neural Networks',
    icon: '◎',
    description: 'Learn complex patterns by stacking layers of simple computations. Perceptrons, MLPs, CNNs, RNNs.',
    color: '#F87171',
  },
  {
    path: '/transformers',
    label: 'Transformers',
    icon: '⧫',
    description: 'The architecture behind modern large language models. Self-Attention, Multi-Head Attention, BERT vs. GPT.',
    color: '#38BDF8',
  },
]

export function WhereToNextSection() {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: true })

  return (
    <section id="where-next" ref={ref} className="py-16 lg:py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={isInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5 }}
      >
        <h2 className="text-2xl lg:text-3xl font-semibold text-text-primary">Where to Go Next</h2>
        <p className="text-base lg:text-lg text-text-secondary mt-2 max-w-2xl leading-relaxed">
          You now have the foundations: what ML is, how data is structured, what loss functions
          measure, how gradient descent optimizes, why overfitting happens, when to scale features,
          and how PCA compresses information. Every model on this site builds on these ideas.
          Pick a category that interests you and dive in.
        </p>

        <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {NEXT_PAGES.map((page, i) => (
            <Link key={page.path} to={page.path} className="block group">
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.4, delay: i * 0.08 }}
              >
                <GlassCard className="p-5 h-full transition-all duration-200 group-hover:border-white/15 group-hover:bg-obsidian-hover">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-lg" style={{ color: page.color }}>{page.icon}</span>
                    <h3
                      className="text-sm font-semibold group-hover:text-text-primary transition-colors"
                      style={{ color: page.color }}
                    >
                      {page.label}
                    </h3>
                    <span className="text-text-tertiary text-xs ml-auto group-hover:translate-x-1 transition-transform">
                      →
                    </span>
                  </div>
                  <p className="text-xs text-text-secondary leading-relaxed">
                    {page.description}
                  </p>
                </GlassCard>
              </motion.div>
            </Link>
          ))}
        </div>
      </motion.div>
    </section>
  )
}
