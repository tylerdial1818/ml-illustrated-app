import { type ReactNode } from 'react'

interface GlassCardProps {
  children: ReactNode
  className?: string
  hover?: boolean
}

export function GlassCard({ children, className = '', hover = false }: GlassCardProps) {
  return (
    <div
      className={`
        bg-obsidian-surface/60 backdrop-blur-xl
        border border-obsidian-border rounded-xl
        ${hover ? 'transition-all duration-200 hover:bg-obsidian-hover hover:border-white/10 hover:shadow-lg hover:shadow-black/20' : ''}
        ${className}
      `}
    >
      {children}
    </div>
  )
}
