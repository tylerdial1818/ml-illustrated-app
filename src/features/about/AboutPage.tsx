import { motion } from 'framer-motion'
import { GlassCard } from '../../components/ui/GlassCard'

export function AboutPage() {
  return (
    <div className="max-w-3xl mx-auto py-16 lg:py-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="text-4xl font-bold text-text-primary">About ML Illustrated</h1>
        <p className="mt-6 text-lg text-text-secondary leading-relaxed">
          ML Illustrated is an interactive visualization platform that helps
          students build deep intuition for how machine learning models work through
          beautiful, animated explanations paired with hands-on parameter exploration.
        </p>

        <GlassCard className="mt-10 p-8">
          <h2 className="text-xl font-semibold text-text-primary mb-4">Why This Exists</h2>
          <p className="text-text-secondary leading-relaxed">
            Most ML education falls into two camps: dense textbooks full of math proofs, or passive
            YouTube videos. ML Illustrated hopefully fills the gap between them. You can see the
            algorithm running step by step, tweak the parameters, and watch what happens. The goal
            is that "aha" moment when the math clicks because you can <em>see</em> it.
          </p>
        </GlassCard>

        <GlassCard className="mt-6 p-8">
          <h2 className="text-xl font-semibold text-text-primary mb-4">How It Works</h2>
          <p className="text-text-secondary leading-relaxed">
            Every visualization runs entirely in your browser. All algorithms are implemented from
            scratch in TypeScript so we can expose every intermediate step. This means there are no hidden states, no
            API calls, no loading spinners.
          </p>
        </GlassCard>

        <GlassCard className="mt-6 p-8">
          <h2 className="text-xl font-semibold text-text-primary mb-4">Built With</h2>
          <div className="flex flex-wrap gap-2">
            {['React', 'TypeScript', 'D3.js', 'Framer Motion', 'Tailwind CSS', 'KaTeX', 'Vite'].map(
              (tech) => (
                <span
                  key={tech}
                  className="text-sm px-3 py-1 rounded-full bg-obsidian-hover text-text-secondary border border-obsidian-border"
                >
                  {tech}
                </span>
              )
            )}
          </div>
        </GlassCard>

        <div className="mt-10 text-sm text-text-tertiary">
          <p>Built by Tyler Dial as a project for the Northwestern MSDS Program.</p>
        </div>
      </motion.div>
    </div>
  )
}
