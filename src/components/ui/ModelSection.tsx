import { type ReactNode, useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { CollapsibleSection } from './CollapsibleSection'

interface ModelSectionProps {
  id: string
  title: string
  subtitle: string
  intuition: ReactNode
  mechanism: ReactNode
  math: ReactNode
  whenToUse: ReactNode
}

function Subsection({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <h4 className="text-xs font-semibold uppercase tracking-[0.1em] text-text-tertiary mb-4">
        {label}
      </h4>
      {children}
    </div>
  )
}

export function ModelSection({
  id,
  title,
  subtitle,
  intuition,
  mechanism,
  math,
  whenToUse,
}: ModelSectionProps) {
  const ref = useRef<HTMLElement>(null)
  const isInView = useInView(ref, { amount: 0.1, once: false })

  return (
    <section id={id} ref={ref} className="py-20 scroll-mt-20">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
        transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
      >
        <h2 className="text-3xl font-semibold text-text-primary">{title}</h2>
        <p className="text-lg text-text-secondary mt-2 max-w-2xl">{subtitle}</p>

        <div className="mt-12 space-y-16">
          <Subsection label="The Intuition">{intuition}</Subsection>
          <Subsection label="The Mechanism">{mechanism}</Subsection>
          <CollapsibleSection title="The Math">{math}</CollapsibleSection>
          <Subsection label="When to Use">{whenToUse}</Subsection>
        </div>
      </motion.div>
    </section>
  )
}
