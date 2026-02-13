import { useScrollSpy } from '../../hooks/useScrollSpy'

interface SectionNavProps {
  sections: { id: string; label: string }[]
}

export function SectionNav({ sections }: SectionNavProps) {
  const activeId = useScrollSpy(sections.map((s) => s.id))

  const handleClick = (id: string) => {
    const element = document.getElementById(id)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <nav className="hidden xl:block fixed right-6 top-1/2 -translate-y-1/2 z-30">
      <div className="flex flex-col gap-3">
        {sections.map((section) => (
          <button
            key={section.id}
            onClick={() => handleClick(section.id)}
            className="group flex items-center gap-3 relative"
            aria-label={`Jump to ${section.label}`}
          >
            <span
              className={`block w-2 h-2 rounded-full transition-all duration-300 ${
                activeId === section.id
                  ? 'bg-accent scale-125'
                  : 'bg-text-tertiary/40 group-hover:bg-text-tertiary'
              }`}
            />
            <span
              className={`absolute right-6 text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none ${
                activeId === section.id ? 'text-accent' : 'text-text-secondary'
              }`}
            >
              {section.label}
            </span>
          </button>
        ))}
      </div>
    </nav>
  )
}
