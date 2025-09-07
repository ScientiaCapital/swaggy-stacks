"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export function Sidebar() {
  const pathname = usePathname();

  const menuItems = [
    { 
      href: '/', 
      label: 'Dashboard', 
      icon: '📊',
      description: 'Trading overview and metrics'
    },
    { 
      href: '/ai', 
      label: 'AI Assistant', 
      icon: '🤖',
      description: 'Chat with AI trading agents'
    },
    { 
      href: '/trading', 
      label: 'Live Trading', 
      icon: '📈',
      description: 'Active trading positions'
    },
    { 
      href: '/analysis', 
      label: 'Market Analysis', 
      icon: '🔍',
      description: 'Technical and fundamental analysis'
    },
    { 
      href: '/portfolio', 
      label: 'Portfolio', 
      icon: '💼',
      description: 'Portfolio performance and allocation'
    }
  ];

  return (
    <aside className="w-64 bg-gray-50 min-h-screen border-r">
      <div className="p-6">
        <nav className="space-y-2">
          {menuItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`block px-4 py-3 rounded-lg transition-colors ${
                pathname === item.href
                  ? 'bg-blue-100 text-blue-700 border-l-4 border-blue-500'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <div className="flex items-center space-x-3">
                <span className="text-lg">{item.icon}</span>
                <div>
                  <div className="font-medium">{item.label}</div>
                  <div className="text-xs text-gray-500 mt-1">{item.description}</div>
                </div>
              </div>
            </Link>
          ))}
        </nav>
      </div>
    </aside>
  );
}