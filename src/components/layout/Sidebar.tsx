import { NavLink } from 'react-router-dom'
import { NAV_ITEMS } from '../../types'

export function Sidebar() {
  return (
    <aside className="hidden lg:flex lg:flex-col lg:w-64 lg:fixed lg:inset-y-0 bg-obsidian-surface border-r border-obsidian-border z-40">
      <div className="flex items-center h-16 px-6 border-b border-obsidian-border">
        <NavLink to="/" className="flex items-center gap-3 group">
          <div className="w-8 h-8 rounded-lg bg-cluster-1 flex items-center justify-center text-white text-sm font-bold">
            ML
          </div>
          <span className="text-text-primary font-semibold tracking-tight">
            ML Illustrated
          </span>
        </NavLink>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.available ? item.path : '#'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200 ${
                !item.available
                  ? 'opacity-40 cursor-not-allowed'
                  : isActive
                  ? 'bg-obsidian-hover text-text-primary'
                  : 'text-text-secondary hover:text-text-primary hover:bg-obsidian-hover/50'
              }`
            }
            onClick={(e) => !item.available && e.preventDefault()}
          >
            <span className="text-lg w-6 text-center">{item.icon}</span>
            <div className="flex-1 min-w-0">
              <div className="font-medium">{item.label}</div>
              <div className="text-xs text-text-tertiary truncate">{item.description}</div>
            </div>
            {!item.available && (
              <span className="text-[10px] uppercase tracking-wider text-text-tertiary bg-obsidian-hover px-1.5 py-0.5 rounded">
                Soon
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="px-4 py-4 border-t border-obsidian-border">
        <p className="text-xs text-text-tertiary">
          Built by Tyler Dial
        </p>
      </div>
    </aside>
  )
}
