"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

type NavItem = {
  href: string;
  label: string;
};

type NavGroup = {
  label: string;
  items: NavItem[];
};

const navGroups: NavGroup[] = [
  {
    label: "Main",
    items: [
      {
        href: "/",
        label: "Dashboard",
      },
    ],
  },
  {
    label: "Setup",
    items: [
      {
        href: "/profile",
        label: "Profile",
      },
      {
        href: "/instruments",
        label: "Instruments",
      },
    ],
  },
  {
    label: "Portfolio",
    items: [
      {
        href: "/portfolio",
        label: "Holdings",
      },
      {
        href: "/upload",
        label: "Upload Statement",
      },
    ],
  },
  {
  label: "AI Workflow",
  items: [
    {
      href: "/recommendations",
      label: "Recommendations",
    },
    {
      href: "/explanations",
      label: "Explanations",
    },
    {
      href: "/research",
      label: "Research",
    },
  ],
  },
  {
    label: "History",
    items: [
      {
        href: "/recommendations/history",
        label: "Recommendation History",
      },
      {
        href: "/explanations/history",
        label: "Explanation History",
      },
    ],
  },
];

function isActivePath(currentPath: string, href: string) {
  if (href === "/") {
    return currentPath === "/";
  }

  return currentPath === href || currentPath.startsWith(`${href}/`);
}

export function SiteNav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-950/95 backdrop-blur">
      <nav className="mx-auto flex max-w-7xl flex-col gap-5 px-6 py-4 text-white">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
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

          <div className="rounded-full border border-emerald-500/30 bg-emerald-950/30 px-4 py-2 text-xs font-medium text-emerald-200">
            AI + Portfolio Analysis MVP
          </div>
        </div>

        <div className="flex flex-wrap gap-4">
          {navGroups.map((group) => (
            <div key={group.label} className="space-y-2">
              <p className="px-1 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                {group.label}
              </p>

              <div className="flex flex-wrap gap-2">
                {group.items.map((item) => {
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
            </div>
          ))}
        </div>
      </nav>
    </header>
  );
}
