import { NavLink } from 'react-router-dom'
import { NAV_ITEMS } from '../../types'

export function Sidebar() {
  const mainItems = NAV_ITEMS.filter((item) => item.label !== 'About')
  const secondaryItems = NAV_ITEMS.filter((item) => item.label === 'About')

  return (
    <aside className="hidden lg:flex lg:flex-col lg:w-64 lg:fixed lg:inset-y-0 bg-obsidian-surface border-r border-obsidian-border z-40">
      <div className="flex items-center h-16 px-6 border-b border-obsidian-border">
        <NavLink to="/" className="flex items-center gap-3 group">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cluster-1 to-cluster-2 flex items-center justify-center text-white text-xs font-bold shadow-lg shadow-cluster-1/20">
            ML
          </div>
          <span className="text-text-primary font-semibold tracking-tight">
            ML Illustrated
          </span>
        </NavLink>
      </div>

      <nav className="flex-1 px-3 py-4 overflow-y-auto">
        <p className="px-3 mb-2 text-[10px] font-semibold uppercase tracking-widest text-text-tertiary">
          Topics
        </p>
        <div className="space-y-0.5">
          {mainItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.available ? item.path : '#'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200 ${
                  !item.available
                    ? 'opacity-40 cursor-not-allowed'
                    : isActive
                    ? 'bg-accent/10 text-accent'
                    : 'text-text-secondary hover:text-text-primary hover:bg-obsidian-hover/60'
                }`
              }
              onClick={(e) => !item.available && e.preventDefault()}
            >
              <span className="text-base w-5 text-center flex-shrink-0">{item.icon}</span>
              <div className="flex-1 min-w-0">
                <div className="font-medium leading-snug">{item.label}</div>
                <div className="text-[11px] text-text-tertiary truncate leading-snug">{item.description}</div>
              </div>
              {!item.available && (
                <span className="text-[9px] uppercase tracking-wider text-text-tertiary bg-obsidian-hover px-1.5 py-0.5 rounded font-medium">
                  Soon
                </span>
              )}
            </NavLink>
          ))}
        </div>

        <div className="my-4 mx-3 border-t border-obsidian-border" />

        <div className="space-y-0.5">
          {secondaryItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200 ${
                  isActive
                    ? 'bg-accent/10 text-accent'
                    : 'text-text-secondary hover:text-text-primary hover:bg-obsidian-hover/60'
                }`
              }
            >
              <span className="text-base w-5 text-center flex-shrink-0">{item.icon}</span>
              <span className="font-medium">{item.label}</span>
            </NavLink>
          ))}
        </div>
      </nav>

      <div className="px-5 py-4 border-t border-obsidian-border">
        <p className="text-[11px] text-text-tertiary">
          Built by Tyler Dial
        </p>
      </div>
    </aside>
  )
}
