import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { GlassCard } from '../../components/ui/GlassCard'

interface Category {
  title: string
  path: string
  description: string
  models: string[]
  color: string
  available?: boolean
  icon: React.ReactNode
}

const categories: Category[] = [
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
    description: 'From simple decision rules to powerful ensembles that dominate tabular data.',
    models: ['Decision Trees', 'Random Forest', 'Gradient Boosted Trees'],
    color: '#34D399',
    icon: (
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <path d="M16 4v8M16 12l-8 8M16 12l8 8" stroke="#34D399" strokeWidth="2" strokeLinecap="round" opacity="0.8" />
        <circle cx="16" cy="4" r="2.5" fill="#34D399" opacity="0.8" />
        <circle cx="8" cy="22" r="2" fill="#34D399" opacity="0.6" />
        <circle cx="24" cy="22" r="2" fill="#34D399" opacity="0.6" />
        <circle cx="4" cy="28" r="1.5" fill="#34D399" opacity="0.4" />
        <circle cx="12" cy="28" r="1.5" fill="#34D399" opacity="0.4" />
        <circle cx="20" cy="28" r="1.5" fill="#34D399" opacity="0.4" />
        <circle cx="28" cy="28" r="1.5" fill="#34D399" opacity="0.4" />
      </svg>
    ),
  },
  {
    title: 'Neural Networks',
    path: '/neural-networks',
    description: 'Layers of simple transformations that compose into powerful learned functions.',
    models: ['Perceptron', 'MLP', 'CNN', 'RNN/LSTM', 'GANs'],
    color: '#FBBF24',
    icon: (
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <circle cx="6" cy="10" r="2.5" fill="#FBBF24" opacity="0.7" />
        <circle cx="6" cy="22" r="2.5" fill="#FBBF24" opacity="0.7" />
        <circle cx="16" cy="8" r="2.5" fill="#FBBF24" opacity="0.7" />
        <circle cx="16" cy="16" r="2.5" fill="#FBBF24" opacity="0.7" />
        <circle cx="16" cy="24" r="2.5" fill="#FBBF24" opacity="0.7" />
        <circle cx="26" cy="16" r="2.5" fill="#FBBF24" opacity="0.7" />
        <line x1="8" y1="10" x2="14" y2="8" stroke="#FBBF24" strokeWidth="0.8" opacity="0.3" />
        <line x1="8" y1="10" x2="14" y2="16" stroke="#FBBF24" strokeWidth="0.8" opacity="0.3" />
        <line x1="8" y1="22" x2="14" y2="16" stroke="#FBBF24" strokeWidth="0.8" opacity="0.3" />
        <line x1="8" y1="22" x2="14" y2="24" stroke="#FBBF24" strokeWidth="0.8" opacity="0.3" />
        <line x1="18" y1="8" x2="24" y2="16" stroke="#FBBF24" strokeWidth="0.8" opacity="0.3" />
        <line x1="18" y1="16" x2="24" y2="16" stroke="#FBBF24" strokeWidth="0.8" opacity="0.3" />
        <line x1="18" y1="24" x2="24" y2="16" stroke="#FBBF24" strokeWidth="0.8" opacity="0.3" />
      </svg>
    ),
  },
  {
    title: 'Transformers & NLP',
    path: '/transformers',
    description: 'The architecture behind GPT, BERT, and every modern LLM — built up from first principles.',
    models: ['Attention', 'Positional Encoding', 'Multi-Head', 'BERT vs GPT'],
    color: '#38BDF8',
    icon: (
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <rect x="4" y="12" width="5" height="8" rx="1" fill="#F472B6" opacity="0.7" />
        <rect x="13.5" y="12" width="5" height="8" rx="1" fill="#34D399" opacity="0.7" />
        <rect x="23" y="12" width="5" height="8" rx="1" fill="#FBBF24" opacity="0.7" />
        <path d="M6.5 12 C6.5 6, 16 6, 16 12" stroke="#38BDF8" strokeWidth="1.2" opacity="0.5" fill="none" />
        <path d="M6.5 12 C6.5 4, 25.5 4, 25.5 12" stroke="#38BDF8" strokeWidth="0.8" opacity="0.3" fill="none" />
        <path d="M16 12 C16 8, 25.5 8, 25.5 12" stroke="#38BDF8" strokeWidth="1" opacity="0.4" fill="none" />
      </svg>
    ),
  },
]

export function HomePage() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Hero */}
      <section className="flex-1 flex flex-col items-center justify-center py-20 text-center">
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
            how models work from cluster models to deep neural networks.
          </p>
          <p className="mt-4 text-sm text-text-tertiary">
            No black boxes. Minimal equations. We're focused on clear, animated explanations you can play with.
          </p>
        </motion.div>
      </section>

      {/* Category Grid */}
      <section className="pb-20 max-w-4xl mx-auto w-full">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {categories.map((cat, i) => {
            const isAvailable = cat.available !== false
            const content = (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 + i * 0.1 }}
              >
                <GlassCard hover={isAvailable} className={`p-8 h-full ${!isAvailable ? 'opacity-50' : ''}`}>
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
