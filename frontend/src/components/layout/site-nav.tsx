"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  {
    href: "/",
    label: "Dashboard",
  },
  {
    href: "/profile",
    label: "Profile",
  },
  {
    href: "/instruments",
    label: "Instruments",
  },
  {
    href: "/portfolio",
    label: "Portfolio",
  },
  {
    href: "/recommendations",
    label: "Recommendations",
  },
  {
    href: "/explanations",
    label: "Explanations",
  },
];

function isActivePath(currentPath: string, href: string) {
  if (href === "/") {
    return currentPath === "/";
  }

  return currentPath.startsWith(href);
}

export function SiteNav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-950/95 backdrop-blur">
      <nav className="mx-auto flex max-w-7xl flex-col gap-4 px-6 py-4 text-white md:flex-row md:items-center md:justify-between">
        <Link href="/" className="group">
          <div>
            <p className="text-lg font-bold tracking-tight">
              Stock Analysis Tool
            </p>
            <p className="text-xs text-slate-400 group-hover:text-emerald-300">
              Educational investment recommendation platform
            </p>
          </div>
        </Link>

        <div className="flex flex-wrap gap-2">
          {navItems.map((item) => {
            const isActive = isActivePath(pathname, item.href);

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`rounded-lg px-3 py-2 text-sm font-medium transition ${
                  isActive
                    ? "bg-emerald-500 text-slate-950"
                    : "border border-slate-700 text-slate-300 hover:border-emerald-400 hover:text-emerald-300"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </div>
      </nav>
    </header>
  );
}
