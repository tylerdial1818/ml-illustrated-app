import { type ReactNode, type ButtonHTMLAttributes } from 'react'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  children: ReactNode
  active?: boolean
}

export function Button({
  variant = 'secondary',
  size = 'md',
  children,
  active = false,
  className = '',
  ...props
}: ButtonProps) {
  const baseStyles = 'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-accent/50'

  const variants = {
    primary: 'bg-cluster-1 text-white hover:bg-cluster-1/90 active:bg-cluster-1/80',
    secondary: `bg-obsidian-glass border border-obsidian-border text-text-secondary hover:text-text-primary hover:bg-obsidian-hover hover:border-white/12 ${
      active ? 'bg-obsidian-hover text-text-primary border-white/12' : ''
    }`,
    ghost: 'text-text-secondary hover:text-text-primary hover:bg-obsidian-hover/50',
  }

  const sizes = {
    sm: 'px-2.5 py-1.5 text-xs gap-1.5',
    md: 'px-3.5 py-2 text-sm gap-2',
    lg: 'px-5 py-2.5 text-base gap-2.5',
  }

  return (
    <button
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      {...props}
    >
      {children}
    </button>
  )
}
