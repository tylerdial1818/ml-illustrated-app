import { create } from 'zustand'

interface NavigationState {
  activeSection: string
  sidebarOpen: boolean
  setActiveSection: (section: string) => void
  setSidebarOpen: (open: boolean) => void
  toggleSidebar: () => void
}

export const useNavigationStore = create<NavigationState>((set) => ({
  activeSection: '',
  sidebarOpen: false,
  setActiveSection: (section) => set({ activeSection: section }),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
}))
