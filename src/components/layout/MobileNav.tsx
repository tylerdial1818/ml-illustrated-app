import { NavLink } from 'react-router-dom'
import { useNavigationStore } from '../../stores/useNavigationStore'
import { NAV_ITEMS } from '../../config/navItems'

export function MobileNav() {
  const { sidebarOpen, setSidebarOpen } = useNavigationStore()

  return (
    <>
      {/* Mobile top bar */}
      <div className="lg:hidden fixed top-0 left-0 right-0 h-14 bg-obsidian-surface/80 backdrop-blur-xl border-b border-obsidian-border z-50 flex items-center px-4">
        <button
          onClick={() => setSidebarOpen(true)}
          className="p-2 text-text-secondary hover:text-text-primary transition-colors"
          aria-label="Open navigation"
        >
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M3 5h14M3 10h14M3 15h14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </button>
        <NavLink to="/" className="ml-3 flex items-center gap-2">
          <div className="w-6 h-6 rounded bg-cluster-1 flex items-center justify-center text-white text-[10px] font-bold">
            ML
          </div>
          <span className="text-text-primary font-semibold text-sm">ML Illustrated</span>
        </NavLink>
      </div>

      {/* Mobile drawer overlay */}
      {sidebarOpen && (
        <div className="lg:hidden fixed inset-0 z-50">
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setSidebarOpen(false)}
          />
          <div className="absolute left-0 top-0 bottom-0 w-72 bg-obsidian-surface border-r border-obsidian-border p-4 overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <span className="text-text-primary font-semibold">ML Illustrated</span>
              <button
                onClick={() => setSidebarOpen(false)}
                className="p-1 text-text-secondary hover:text-text-primary"
                aria-label="Close navigation"
              >
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                  <path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </button>
            </div>
            <nav className="space-y-1">
              {NAV_ITEMS.filter((item) => item.available).map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  onClick={() => setSidebarOpen(false)}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                      isActive
                        ? 'bg-obsidian-hover text-text-primary'
                        : 'text-text-secondary hover:text-text-primary hover:bg-obsidian-hover/50'
                    }`
                  }
                >
                  <span className="text-lg">{item.icon}</span>
                  <span className="font-medium">{item.label}</span>
                </NavLink>
              ))}
            </nav>
          </div>
        </div>
      )}

      {/* Spacer for mobile top bar */}
      <div className="lg:hidden h-14" />
    </>
  )
}
