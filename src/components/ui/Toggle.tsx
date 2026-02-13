interface ToggleProps {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  className?: string
}

export function Toggle({ label, checked, onChange, className = '' }: ToggleProps) {
  return (
    <label className={`flex items-center gap-2.5 cursor-pointer group ${className}`}>
      <div className="relative">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="sr-only"
        />
        <div
          className={`w-9 h-5 rounded-full transition-colors duration-200 ${
            checked ? 'bg-accent' : 'bg-obsidian-hover'
          }`}
        />
        <div
          className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform duration-200 ${
            checked ? 'translate-x-4' : 'translate-x-0'
          }`}
        />
      </div>
      <span className="text-xs font-medium text-text-secondary group-hover:text-text-primary transition-colors">
        {label}
      </span>
    </label>
  )
}
