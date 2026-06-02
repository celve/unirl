import "../global.css";

import { isLanguage, languages } from "@/lib/i18n";
import { translations } from "@/lib/layout.shared";
import { withBasePath } from "@/lib/site-paths";
import { i18nProvider } from "fumadocs-ui/i18n";
import { RootProvider } from "fumadocs-ui/provider/next";
import { notFound } from "next/navigation";
import type { ReactNode } from "react";

export const metadata = {
  title: "UniRL Docs",
  description: "Agent-first documentation for UniRL.",
};

type LanguageLayoutProps = {
  children: ReactNode;
  params: Promise<{
    lang: string;
  }>;
};

export function generateStaticParams() {
  return languages.map((lang) => ({ lang }));
}

export default async function LanguageLayout({ children, params }: LanguageLayoutProps) {
  const { lang } = await params;
  if (!isLanguage(lang)) notFound();

  return (
    <html lang={lang} suppressHydrationWarning>
      <body className="flex min-h-screen flex-col">
        <RootProvider
          i18n={i18nProvider(translations, lang)}
          search={{ options: { api: withBasePath("/api/search.json"), type: "static" } }}
        >
          {children}
        </RootProvider>
      </body>
    </html>
  );
}
