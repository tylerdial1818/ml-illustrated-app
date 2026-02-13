import { useState, type ReactNode } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface CollapsibleSectionProps {
  title: string
  children: ReactNode
  defaultOpen?: boolean
  className?: string
}

export function CollapsibleSection({
  title,
  children,
  defaultOpen = false,
  className = '',
}: CollapsibleSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className={className}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-3 w-full text-left group py-2"
        aria-expanded={isOpen}
      >
        <motion.svg
          width="12"
          height="12"
          viewBox="0 0 12 12"
          className="text-text-tertiary group-hover:text-accent transition-colors flex-shrink-0"
          animate={{ rotate: isOpen ? 90 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <path d="M4 2l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </motion.svg>
        <span className="text-sm font-semibold uppercase tracking-widest text-text-tertiary group-hover:text-text-secondary transition-colors">
          {title}
        </span>
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            className="overflow-hidden"
          >
            <div className="pt-5 pb-3">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
