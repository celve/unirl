import { defineI18n } from "fumadocs-core/i18n";

export const languages = ["en", "zh"] as const;
export type Language = (typeof languages)[number];

export const i18n = defineI18n({
  defaultLanguage: "en",
  languages: [...languages],
  fallbackLanguage: "en",
  parser: "dir",
});

export function isLanguage(lang: string): lang is Language {
  return (languages as readonly string[]).includes(lang);
}
