import { useState, useRef, useCallback } from 'react'

interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
  formatValue?: (value: number) => string
  className?: string
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  formatValue = (v) => String(v),
  className = '',
}: SliderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const trackRef = useRef<HTMLInputElement>(null)

  const percentage = ((value - min) / (max - min)) * 100

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange(parseFloat(e.target.value))
    },
    [onChange]
  )

  return (
    <div className={`flex flex-col gap-2 ${className}`}>
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-text-secondary uppercase tracking-wider">
          {label}
        </label>
        <span
          className={`text-xs font-mono tabular-nums transition-colors ${
            isDragging ? 'text-accent' : 'text-text-tertiary'
          }`}
        >
          {formatValue(value)}
        </span>
      </div>
      <div className="relative">
        <div
          className="absolute top-1/2 -translate-y-1/2 h-1 rounded-full bg-accent/40 pointer-events-none"
          style={{ width: `${percentage}%` }}
        />
        <input
          ref={trackRef}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={handleChange}
          onMouseDown={() => setIsDragging(true)}
          onMouseUp={() => setIsDragging(false)}
          onTouchStart={() => setIsDragging(true)}
          onTouchEnd={() => setIsDragging(false)}
          className="w-full h-6 relative z-10"
        />
      </div>
    </div>
  )
}
