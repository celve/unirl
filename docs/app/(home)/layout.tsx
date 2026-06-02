import "../global.css";

import type { ReactNode } from "react";

export const metadata = {
  title: "UniRL Docs",
  description: "Agent-first documentation for UniRL.",
};

export default function HomeLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="flex min-h-screen flex-col">{children}</body>
    </html>
  );
}
