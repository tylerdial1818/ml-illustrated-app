import { type ReactNode } from 'react'
import { Sidebar } from './Sidebar'
import { MobileNav } from './MobileNav'

interface AppShellProps {
  children: ReactNode
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="min-h-screen bg-obsidian-bg">
      <Sidebar />
      <MobileNav />
      <main className="lg:pl-64 min-h-screen">
        <div className="w-full">
          {children}
        </div>
      </main>
    </div>
  )
}
