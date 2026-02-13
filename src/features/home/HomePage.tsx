import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { GlassCard } from '../../components/ui/GlassCard'

const categories = [
  {
    title: 'Clustering',
    path: '/clustering',
    description: 'Discover how algorithms find hidden groups in unlabeled data.',
    models: ['K-Means', 'DBSCAN', 'Hierarchical', 'GMM'],
    color: '#6366F1',
    icon: (
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <circle cx="10" cy="10" r="3" fill="#6366F1" opacity="0.8" />
        <circle cx="22" cy="8" r="2.5" fill="#F472B6" opacity="0.8" />
        <circle cx="14" cy="22" r="3.5" fill="#34D399" opacity="0.8" />
        <circle cx="24" cy="20" r="2" fill="#FBBF24" opacity="0.8" />
        <circle cx="8" cy="16" r="2" fill="#6366F1" opacity="0.5" />
        <circle cx="18" cy="14" r="1.5" fill="#F472B6" opacity="0.5" />
      </svg>
    ),
  },
  {
    title: 'Regression',
    path: '/regression',
    description: 'See how models learn to predict continuous outcomes from data.',
    models: ['Linear', 'Logistic', 'Ridge', 'Lasso', 'ElasticNet'],
    color: '#F472B6',
    icon: (
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <line x1="4" y1="26" x2="28" y2="6" stroke="#F472B6" strokeWidth="2" strokeLinecap="round" />
        <circle cx="8" cy="20" r="2" fill="#6366F1" opacity="0.6" />
        <circle cx="14" cy="18" r="2" fill="#6366F1" opacity="0.6" />
        <circle cx="18" cy="14" r="2" fill="#6366F1" opacity="0.6" />
        <circle cx="24" cy="10" r="2" fill="#6366F1" opacity="0.6" />
        <circle cx="11" cy="22" r="1.5" fill="#6366F1" opacity="0.4" />
        <circle cx="21" cy="8" r="1.5" fill="#6366F1" opacity="0.4" />
      </svg>
    ),
  },
  {
    title: 'Trees',
    path: '/trees',
    description: 'Decision boundaries, random forests, and gradient boosting.',
    models: ['Decision Trees', 'Random Forests', 'XGBoost'],
    color: '#34D399',
    available: false,
    icon: (
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <path d="M16 4v8M16 12l-8 8M16 12l8 8" stroke="#34D399" strokeWidth="2" strokeLinecap="round" opacity="0.5" />
        <circle cx="16" cy="4" r="2" fill="#34D399" opacity="0.5" />
        <circle cx="8" cy="22" r="2" fill="#34D399" opacity="0.3" />
        <circle cx="24" cy="22" r="2" fill="#34D399" opacity="0.3" />
      </svg>
    ),
  },
  {
    title: 'Neural Networks',
    path: '/neural-networks',
    description: 'Neurons, layers, activation functions, and backpropagation.',
    models: ['Perceptrons', 'MLPs', 'Backprop'],
    color: '#FBBF24',
    available: false,
    icon: (
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <circle cx="6" cy="10" r="2" fill="#FBBF24" opacity="0.4" />
        <circle cx="6" cy="22" r="2" fill="#FBBF24" opacity="0.4" />
        <circle cx="16" cy="8" r="2" fill="#FBBF24" opacity="0.4" />
        <circle cx="16" cy="16" r="2" fill="#FBBF24" opacity="0.4" />
        <circle cx="16" cy="24" r="2" fill="#FBBF24" opacity="0.4" />
        <circle cx="26" cy="16" r="2" fill="#FBBF24" opacity="0.4" />
      </svg>
    ),
  },
]

export function HomePage() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Hero */}
      <section className="flex-1 flex flex-col items-center justify-center px-6 py-20 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }}
          className="max-w-3xl"
        >
          <h1 className="text-5xl md:text-6xl font-bold text-text-primary leading-tight">
            ML{' '}
            <span className="bg-gradient-to-r from-cluster-1 via-cluster-2 to-cluster-3 bg-clip-text text-transparent">
              Illustrated
            </span>
          </h1>
          <p className="mt-6 text-xl text-text-secondary leading-relaxed max-w-2xl mx-auto">
            The visual intuition engine for machine learning. Interactive animations that show you
            how models actually work — from K-Means to Transformers.
          </p>
          <p className="mt-4 text-sm text-text-tertiary">
            No black boxes. No hand-waving. Just clear, animated explanations you can play with.
          </p>
        </motion.div>
      </section>

      {/* Category Grid */}
      <section className="px-6 pb-20 max-w-5xl mx-auto w-full">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {categories.map((cat, i) => {
            const isAvailable = cat.available !== false
            const content = (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 + i * 0.1 }}
              >
                <GlassCard hover={isAvailable} className={`p-6 h-full ${!isAvailable ? 'opacity-50' : ''}`}>
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0 mt-1">{cat.icon}</div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h3 className="text-lg font-semibold text-text-primary">{cat.title}</h3>
                        {!isAvailable && (
                          <span className="text-[10px] uppercase tracking-wider text-text-tertiary bg-obsidian-hover px-1.5 py-0.5 rounded">
                            Coming Soon
                          </span>
                        )}
                      </div>
                      <p className="mt-1 text-sm text-text-secondary">{cat.description}</p>
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {cat.models.map((model) => (
                          <span
                            key={model}
                            className="text-xs px-2 py-0.5 rounded-full bg-obsidian-hover text-text-tertiary"
                          >
                            {model}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </GlassCard>
              </motion.div>
            )

            return isAvailable ? (
              <Link key={cat.title} to={cat.path} className="block">
                {content}
              </Link>
            ) : (
              <div key={cat.title}>{content}</div>
            )
          })}
        </div>
      </section>
    </div>
  )
}
