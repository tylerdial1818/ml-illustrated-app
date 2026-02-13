import { useRef, useState, useEffect, type ReactNode } from 'react'

interface SVGContainerProps {
  aspectRatio?: number
  minHeight?: number
  maxHeight?: number
  padding?: { top: number; right: number; bottom: number; left: number }
  children: (dimensions: { width: number; height: number; innerWidth: number; innerHeight: number }) => ReactNode
  className?: string
}

const DEFAULT_PADDING = { top: 20, right: 20, bottom: 40, left: 50 }

export function SVGContainer({
  aspectRatio = 16 / 10,
  minHeight = 300,
  maxHeight = 600,
  padding = DEFAULT_PADDING,
  children,
  className = '',
}: SVGContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 })

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const width = entry.contentRect.width
        const height = Math.min(Math.max(width / aspectRatio, minHeight), maxHeight)
        setDimensions({ width, height })
      }
    })

    observer.observe(container)
    return () => observer.disconnect()
  }, [aspectRatio, minHeight, maxHeight])

  const innerWidth = dimensions.width - padding.left - padding.right
  const innerHeight = dimensions.height - padding.top - padding.bottom

  return (
    <div ref={containerRef} className={`w-full ${className}`}>
      <svg
        width={dimensions.width}
        height={dimensions.height}
        viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
        className="overflow-visible"
      >
        <g transform={`translate(${padding.left}, ${padding.top})`}>
          {children({
            width: dimensions.width,
            height: dimensions.height,
            innerWidth,
            innerHeight,
          })}
        </g>
      </svg>
    </div>
  )
}
