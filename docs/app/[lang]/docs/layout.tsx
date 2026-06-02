import { baseOptions } from "@/lib/layout.shared";
import { source } from "@/lib/source";
import { DocsLayout } from "fumadocs-ui/layouts/docs";
import { notFound } from "next/navigation";
import type { ReactNode } from "react";

type DocsLayoutProps = {
  children: ReactNode;
  params: Promise<{
    lang: string;
  }>;
};

export default async function Layout({ children, params }: DocsLayoutProps) {
  const { lang } = await params;
  const tree = source.getPageTree(lang);
  if (!tree) notFound();

  return (
    <DocsLayout {...baseOptions(lang)} tree={tree}>
      {children}
    </DocsLayout>
  );
}
