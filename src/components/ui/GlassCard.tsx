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
        bg-obsidian-glass backdrop-blur-xl
        border border-obsidian-border rounded-xl
        ${hover ? 'transition-all duration-200 hover:bg-obsidian-hover hover:border-white/12 hover:scale-[1.01]' : ''}
        ${className}
      `}
    >
      {children}
    </div>
  )
}
