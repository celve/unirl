import { i18n } from "@/lib/i18n";
import { uiTranslations } from "fumadocs-ui/i18n";
import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";

export const translations = i18n
  .translations()
  .extend(uiTranslations())
  .add("ui", {
    en: {
      displayName: "English",
    },
    zh: {
      displayName: "中文",
      search: "搜索文档",
      searchNoResult: "没有找到结果",
      toc: "目录",
      chooseLanguage: "选择语言",
      nextPage: "下一页",
      previousPage: "上一页",
    },
  });

export function baseOptions(locale: string): BaseLayoutProps {
  const docsPrefix = `/${locale}/docs`;

  return {
    nav: {
      title: locale === "zh" ? "UniRL 文档" : "UniRL",
      url: docsPrefix,
    },
    links: [
      {
        text: locale === "zh" ? "Agent 索引" : "Agent Index",
        url: `${docsPrefix}/agents`,
      },
    ],
  };
}
