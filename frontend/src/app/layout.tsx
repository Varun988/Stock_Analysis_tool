import type { Metadata } from "next";

import { SiteNav } from "@/components/layout/site-nav";

import "./globals.css";

export const metadata: Metadata = {
  title: "Stock Analysis Tool",
  description:
    "Educational investment recommendation and portfolio analysis tool.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <SiteNav />
        {children}
      </body>
    </html>
  );
}