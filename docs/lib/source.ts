import { docs } from "collections/server";
import { i18n } from "@/lib/i18n";
import { loader } from "fumadocs-core/source";

export const source = loader({
  baseUrl: "/docs",
  i18n,
  source: docs.toFumadocsSource(),
});
